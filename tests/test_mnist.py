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

def test_n_samples_and_entropy_run_and_record_config():
    X, y = _toy_data(n=64)
    kw = dict(episodes=50, batch=16, eval_every=25, eval_n=64)
    r = train("dg", X, y, X, y, seed=3, n_samples=4, entropy_coef=0.01, **kw)
    assert r["n_samples"] == 4 and r["entropy_coef"] == 0.01
    assert np.all(np.isfinite(r["test_err"]))


def test_entropy_coef_changes_the_update():
    # a large entropy weight keeps the policy near uniform on the toy problem,
    # while plain PG solves it - so the entropy term must change the trajectory
    X, y = _toy_data()
    kw = dict(episodes=300, batch=32, eval_every=100, eval_n=200, seed=0)
    r_plain = train("pg", X, y, X, y, **kw)
    r_ent = train("pg", X, y, X, y, entropy_coef=100.0, **kw)
    assert r_ent["eval_entropy"][-1] > r_plain["eval_entropy"][-1]


def test_default_config_unchanged_by_new_options():
    # n_samples=1, entropy_coef=0 must reproduce the original update exactly
    X, y = _toy_data(n=64)
    kw = dict(episodes=50, batch=16, eval_every=25, eval_n=64)
    r1 = train("dg", X, y, X, y, seed=3, **kw)
    r2 = train("dg", X, y, X, y, seed=3, n_samples=1, entropy_coef=0.0, **kw)
    assert np.array_equal(r1["test_err"], r2["test_err"])


def _manual_dg_grad(logits, actions, U, eta, through_gate):
    """Closed-form gradient of the DG loss w.r.t. logits (batch mean, k=1).

    Detached gate (paper Algorithm 1): coefficient w*U.
    Autograd through the gate (pre-fix): coefficient w*U - U^2 * log pi(A) * w(1-w)/eta.
    """
    B, K = logits.shape
    pi = torch.softmax(logits, dim=1)
    lp_a = log_pi = None
    log_pi = torch.log_softmax(logits, dim=1)
    lp_a = log_pi[torch.arange(B), actions]
    w = torch.sigmoid(U * (-lp_a) / eta)
    coef = U * w
    if through_gate:
        coef = coef - U**2 * lp_a * w * (1 - w) / eta
    onehot = torch.zeros_like(logits)
    onehot[torch.arange(B), actions] = 1.0
    return -(coef.unsqueeze(1) * (onehot - pi)) / B


def test_detached_gate_gradient_matches_algorithm1():
    # The default loss must give exactly w * U * grad log pi(A) per sample:
    # the gradient autograd computes must equal the closed form with a fixed gate.
    from dgplay.mnist import gated_loss
    torch.manual_seed(0)
    B, K = 64, 10
    logits = torch.randn(B, K, dtype=torch.float64, requires_grad=True)
    actions = torch.randint(0, K, (B,))
    U = torch.randint(0, 2, (B,), dtype=torch.float64) - 0.5  # rewards 0/1 minus b=0.5
    loss = gated_loss(torch.log_softmax(logits, 1), actions.unsqueeze(1), U.unsqueeze(1), eta=1.0)
    (g,) = torch.autograd.grad(loss, logits)
    expected = _manual_dg_grad(logits.detach(), actions, U, eta=1.0, through_gate=False)
    assert torch.allclose(g, expected, atol=1e-12)


def test_autograd_through_gate_variant_matches_its_closed_form():
    # The pre-fix variant (kept as the "autograd through the gate" ablation) adds the
    # term -U^2 * log pi(A) * w(1-w)/eta per sample; pin both that it differs from
    # Algorithm 1 and that it equals its own closed form.
    from dgplay.mnist import gated_loss
    torch.manual_seed(1)
    B, K = 64, 10
    logits = torch.randn(B, K, dtype=torch.float64, requires_grad=True)
    actions = torch.randint(0, K, (B,))
    U = torch.randint(0, 2, (B,), dtype=torch.float64) - 0.5
    loss = gated_loss(torch.log_softmax(logits, 1), actions.unsqueeze(1), U.unsqueeze(1),
                      eta=1.0, detach_gate=False)
    (g,) = torch.autograd.grad(loss, logits)
    expected = _manual_dg_grad(logits.detach(), actions, U, eta=1.0, through_gate=True)
    assert torch.allclose(g, expected, atol=1e-12)
    algo1 = _manual_dg_grad(logits.detach(), actions, U, eta=1.0, through_gate=False)
    assert not torch.allclose(g, algo1, atol=1e-6)


def test_train_dg_detaches_gate_by_default():
    # train(..., "dg") defaults to the Algorithm 1 (detached) gate and records the choice.
    X, y = _toy_data(n=64)
    kw = dict(episodes=5, batch=16, eval_every=5, eval_n=64)
    assert train("dg", X, y, X, y, seed=0, **kw)["detach_gate"] is True
    assert train("dg", X, y, X, y, seed=0, detach_gate=False, **kw)["detach_gate"] is False
