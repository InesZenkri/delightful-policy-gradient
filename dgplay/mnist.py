"""MNIST as a one-step contextual bandit with a small neural policy (PyTorch).

Follows the diagnostic in Osband, "Delightful Policy Gradient" (arXiv:2603.14608,
Section 3 and Appendix B): given an image, the learner samples a digit from its policy
and observes only whether the sample was correct (reward 1) or not (reward 0). The label
is never revealed to the learner. The paper trains a two-layer ReLU MLP (hidden width
100) with Adam (lr 1e-3) on batches of B=100 images for 10,000 episodes; the defaults
here match that, scaled down only where the caller asks for it.

Methods: "pg" (REINFORCE), "dg" (delight-gated, Algorithm 1), "ce" (supervised
cross-entropy oracle that sees the label; an upper reference, not a bandit method).

Baselines for the advantage U = R - b: "zero" (b=0), "constant" (b=0.5, the paper's
middle option), "batch_mean" (mean reward of the batch, matching dgplay/contextual.py).

MNIST is loaded from the IDX files (no torchvision dependency), downloaded once from the
ossci-datasets S3 mirror and cached under data/mnist/ (gitignored).
"""
from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

MNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mnist"


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, magic
        return np.frombuffer(f.read(rows * cols * n), dtype=np.uint8).reshape(n, rows * cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, magic
        return np.frombuffer(f.read(n), dtype=np.uint8).astype(np.int64)


def load_mnist(data_dir: Path | str = DATA_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_train, y_train, X_test, y_test); images float32 in [0, 1], flattened."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in MNIST_URLS.items():
        p = data_dir / url.rsplit("/", 1)[1]
        if not p.exists():
            print(f"downloading {url} -> {p}")
            urllib.request.urlretrieve(url, p)
        paths[name] = p
    Xtr = _read_idx_images(paths["train_images"]).astype(np.float32) / 255.0
    ytr = _read_idx_labels(paths["train_labels"])
    Xte = _read_idx_images(paths["test_images"]).astype(np.float32) / 255.0
    yte = _read_idx_labels(paths["test_labels"])
    return Xtr, ytr, Xte, yte


def build_policy(width: int = 100, seed: int = 0):
    """Two-layer ReLU MLP 784 -> width -> 10, as in the paper's Appendix B."""
    import torch

    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(28 * 28, width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, 10),
    )


def train(method: str, Xtr, ytr, Xte, yte, episodes: int = 10_000, batch: int = 100,
          lr: float = 1e-3, eta: float = 1.0, baseline: str = "constant",
          width: int = 100, seed: int = 0, eval_every: int = 200, eval_n: int = 2_000):
    """Train one method on the MNIST contextual bandit. Returns a dict of histories.

    eval_every: episodes between evaluations. eval_n: size of the held-out subset used
    for the test-error curve (first eval_n test images, fixed across methods/seeds).
    """
    import torch

    assert method in ("pg", "dg", "ce")
    assert baseline in ("zero", "constant", "batch_mean")
    # build_policy seeds torch with `seed` for the init; the same stream then drives
    # sampling, so identical seeds give identical runs and all methods share one init.
    policy = build_policy(width=width, seed=seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)
    Xte_t = torch.from_numpy(Xte[:eval_n])
    yte_t = torch.from_numpy(yte[:eval_n])

    hist_ep, hist_train_acc, hist_test_err = [], [], []
    n = len(Xtr_t)
    for t in range(episodes):
        idx = torch.randint(0, n, (batch,))
        x, y = Xtr_t[idx], ytr_t[idx]
        logits = policy(x)
        log_pi = torch.log_softmax(logits, dim=1)
        if method == "ce":
            loss = -(log_pi[torch.arange(batch), y]).mean()
            batch_acc = float((logits.argmax(1) == y).float().mean())
        else:
            pi = log_pi.exp()
            a = torch.multinomial(pi, 1).squeeze(1)
            r = (a == y).float()
            if baseline == "zero":
                b = 0.0
            elif baseline == "constant":
                b = 0.5
            else:
                b = float(r.mean())
            U = r - b
            log_pi_a = log_pi[torch.arange(batch), a]
            if method == "dg":
                surprisal = -log_pi_a
                chi = U * surprisal
                w = torch.sigmoid(chi / eta)
                loss = -(w * U * log_pi_a).mean()
            else:
                loss = -(U * log_pi_a).mean()
            batch_acc = float(r.mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if t % eval_every == 0 or t == episodes - 1:
            with torch.no_grad():
                test_err = 1.0 - float((policy(Xte_t).argmax(1) == yte_t).float().mean())
            hist_ep.append(t)
            hist_train_acc.append(batch_acc)
            hist_test_err.append(test_err)
    return {
        "episode": np.array(hist_ep),
        "batch_train_acc": np.array(hist_train_acc),
        "test_err": np.array(hist_test_err),
        "final_test_err": hist_test_err[-1],
        "method": method,
        "seed": seed,
        "baseline": baseline,
        "eta": eta,
    }
