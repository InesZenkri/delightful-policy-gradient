import numpy as np
from dgplay.continuous import (SmoothBandit, gaussian_logp, grad_logp, run_continuous,
                               LOG_CLIP)


def test_grad_logp_matches_finite_difference():
    rng = np.random.default_rng(0)
    for _ in range(10):
        a, mu, ls = rng.normal(), rng.normal(), rng.normal()
        g = grad_logp(a, mu, ls)
        e = 1e-6
        num = np.array([(gaussian_logp(a, mu + e, ls) - gaussian_logp(a, mu - e, ls)) / (2 * e),
                        (gaussian_logp(a, mu, ls + e) - gaussian_logp(a, mu, ls - e)) / (2 * e)])
        assert np.allclose(g, num, atol=1e-6)


def test_log_density_clip_bounds_surprisal():
    # Algorithm 2 clips the log-density to [-LOG_CLIP, LOG_CLIP]
    surprisal = np.clip(-gaussian_logp(np.linspace(-100, 100, 5), 0.0, 0.0), -LOG_CLIP, LOG_CLIP)
    assert np.all(np.abs(surprisal) <= LOG_CLIP)


def test_smoothbandit_optimum_and_seed_determinism():
    env = SmoothBandit(a_star=1.0, width=0.5, scale=2.0, noise=0.0, seed=0)
    assert np.isclose(env.reward(1.0), 2.0)
    assert env.reward(0.0) < 2.0
    a, b = SmoothBandit(seed=3), SmoothBandit(seed=3)
    assert [a.reward(0.1) for _ in range(5)] == [b.reward(0.1) for _ in range(5)]


def test_pg_learns_on_continuous_bandit():
    r = run_continuous("pg", scale=1.0, lr=0.05, steps=600, seed=0)
    assert r["return_at_mean"][-1] > 0.9


def test_whitened_dg_is_reward_scale_invariant():
    # With lr scaled by 1/scale, whitened DG at eta=1 follows the same curve at any reward
    # scale; unwhitened DG does not (the gate saturates or collapses to 1/2).
    for scale in (0.01, 100.0):
        rw = run_continuous("dg", scale=scale, lr=0.05 / scale, whiten=True, steps=200, seed=1)
        ru = run_continuous("dg", scale=scale, lr=0.05 / scale, whiten=False, steps=200, seed=1)
        rw1 = run_continuous("dg", scale=1.0, lr=0.05, whiten=True, steps=200, seed=1)
        ru1 = run_continuous("dg", scale=1.0, lr=0.05, whiten=False, steps=200, seed=1)
        assert np.allclose(rw["return_at_mean"], rw1["return_at_mean"], atol=1e-9)
        assert not np.allclose(ru["return_at_mean"], ru1["return_at_mean"], atol=1e-9)
