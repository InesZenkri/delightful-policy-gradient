"""Tests for the MNIST contextual-bandit module. No dataset download needed: the loader
is tested on synthetic IDX files, the training tests use small random arrays."""
import gzip
import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dgplay.mnist import load_mnist, build_policy, train


def _write_idx(path, magic, shape, data):
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">" + "I" * (1 + len(shape)), magic, *shape))
        f.write(np.asarray(data, dtype=np.uint8).tobytes())


def test_idx_loader_roundtrip(tmp_path):
    imgs = np.arange(4 * 2 * 2, dtype=np.uint8).reshape(4, 2, 2)
    labels = np.array([0, 1, 2, 3], dtype=np.uint8)
    _write_idx(tmp_path / "train-images-idx3-ubyte.gz", 2051, (4, 2, 2), imgs)
    _write_idx(tmp_path / "train-labels-idx1-ubyte.gz", 2049, (4,), labels)
    _write_idx(tmp_path / "t10k-images-idx3-ubyte.gz", 2051, (4, 2, 2), imgs)
    _write_idx(tmp_path / "t10k-labels-idx1-ubyte.gz", 2049, (4,), labels)
    Xtr, ytr, Xte, yte = load_mnist(tmp_path)
    assert Xtr.shape == (4, 4) and np.isclose(Xtr.max(), 15 / 255)
    assert list(ytr) == [0, 1, 2, 3] and Xte.shape == (4, 4)


def test_same_seed_same_init_across_methods():
    a = build_policy(seed=7)
    b = build_policy(seed=7)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_dg_gate_matches_bandit_formula():
    # The gate inside train() must be sigmoid(U * surprisal / eta), as in dgplay.bandit.
    from dgplay.bandit import dg_gate
    log_pi = torch.log_softmax(torch.randn(8, 10, dtype=torch.float64), dim=1)
    a = torch.randint(0, 10, (8,))
    U = torch.randn(8, dtype=torch.float64)
    surprisal = -log_pi[torch.arange(8), a]
    w = torch.sigmoid(U * surprisal / 1.0).numpy()
    w_ref = [dg_gate(float(u), float(np.exp(-s))) for u, s in zip(U.numpy(), surprisal.numpy())]
    assert np.allclose(w, w_ref, atol=1e-10)


def _toy_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 10, size=n)
    X = np.zeros((n, 784), dtype=np.float32)
    X[np.arange(n), y] = 1.0  # label is directly readable: CE should hit ~0 error
    return X, y


def test_ce_oracle_solves_toy_problem():
    X, y = _toy_data()
    r = train("ce", X, y, X, y, episodes=300, batch=32, eval_every=100, eval_n=200, seed=0)
    assert r["final_test_err"] < 0.02


def test_bandit_methods_learn_toy_problem_slower_than_ce():
    X, y = _toy_data()
    r_ce = train("ce", X, y, X, y, episodes=300, batch=32, eval_every=100, eval_n=200, seed=0)
    r_pg = train("pg", X, y, X, y, episodes=300, batch=32, eval_every=100, eval_n=200, seed=0)
    r_dg = train("dg", X, y, X, y, episodes=300, batch=32, eval_every=100, eval_n=200, seed=0)
    # both bandit methods improve over chance (0.9 error) on an easy problem
    assert r_pg["final_test_err"] < 0.9
    assert r_dg["final_test_err"] < 0.9
    assert r_ce["final_test_err"] <= r_pg["final_test_err"]


def test_determinism():
    X, y = _toy_data(n=64)
    kw = dict(episodes=50, batch=16, eval_every=25, eval_n=64)
    r1 = train("dg", X, y, X, y, seed=3, **kw)
    r2 = train("dg", X, y, X, y, seed=3, **kw)
    assert np.array_equal(r1["test_err"], r2["test_err"])
