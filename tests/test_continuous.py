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


def test_multidim_pg_is_action_scale_invariant_by_design():
    # Control: with sigma_i = w_i and per-dim lr * w_i^2, PG's trajectory in z-units is the
    # same at any per-dimension action scale, up to floating-point rounding.
    from dgplay.continuous import run_continuous_multidim
    a = run_continuous_multidim("pg", widths=(1.0, 1.0, 1.0), steps=200, seed=3)["return_at_mean"]
    b = run_continuous_multidim("pg", widths=(0.02, 0.02, 0.02), steps=200, seed=3)["return_at_mean"]
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_multidim_only_log_scale_sum_enters_the_gate():
    # Mechanism: mixed widths whose log-scales sum to the same value give the same DG
    # trajectory as the uniform setting (per-dim heterogeneity per se does not matter).
    from dgplay.continuous import run_continuous_multidim
    a = run_continuous_multidim("dg", widths=(1.0, 1.0, 1.0), steps=200, seed=4)["return_at_mean"]
    b = run_continuous_multidim("dg", widths=(0.05, 1.0, 20.0), steps=200, seed=4)["return_at_mean"]
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
    c = run_continuous_multidim("dg", widths=(0.05, 0.2, 20.0), steps=200, seed=4)["return_at_mean"]
    assert not np.allclose(a, c, rtol=1e-10, atol=1e-12)  # different sum -> different run


def test_multidim_raw_surprisal_offset_goes_negative_at_small_scales():
    # The scale-dependent offset: at small action scales the Gaussian log-density surprisal
    # is negative on average (the gate inverts); at large scales it rails the +clip bound.
    from dgplay.continuous import gaussian_logp_diag
    rng = np.random.default_rng(0)
    for w, expect_negative in ((0.02, True), (50.0, False)):
        widths = np.full(3, w)
        mu, log_sigma = -1.5 * widths, np.log(widths)
        acts = mu + np.exp(log_sigma) * rng.normal(size=(256, 3))
        raw = -gaussian_logp_diag(acts, mu, log_sigma)
        if expect_negative:
            assert raw.mean() < 0.0
        else:
            assert (np.abs(raw) >= LOG_CLIP).mean() > 0.99


def test_multidim_mahalanobis_surprisal_is_scale_free():
    # Our comparator (not the paper's): 0.5 * z^2 depends only on z-scores, is nonnegative,
    # and is zero at the mode, at any action scale.
    from dgplay.continuous import mahalanobis_surprisal
    rng = np.random.default_rng(1)
    z = rng.normal(size=(64, 3))
    vals = []
    for w in (0.02, 1.0, 50.0):
        widths = np.full(3, w)
        mu, log_sigma = -1.5 * widths, np.log(widths)
        acts = mu + np.exp(log_sigma) * z
        vals.append(mahalanobis_surprisal(acts, mu, log_sigma))
    np.testing.assert_allclose(vals[0], vals[1], rtol=1e-12)
    np.testing.assert_allclose(vals[1], vals[2], rtol=1e-12)
    assert np.all(vals[0] >= 0.0)
    assert mahalanobis_surprisal(np.zeros((1, 3)), np.zeros(3), np.zeros(3))[0] == 0.0


def test_multidim_batch_whitening_is_affine_invariant_only():
    # Why the paper's whitening fixes reward scaling but not action scaling here: it cancels
    # exactly affine transforms chi -> a*chi + b (a > 0). An action-scale change adds a
    # constant to the surprisal, so chi -> chi + C*adv is not affine in chi and survives it.
    def whiten(chi):
        return (chi - chi.mean()) / (chi.std() + 1e-8)
    rng = np.random.default_rng(2)
    adv, l = rng.normal(size=128), rng.uniform(0.5, 3.0, size=128)
    chi = adv * l
    np.testing.assert_allclose(whiten(7.0 * chi - 3.0), whiten(chi), rtol=1e-6)
    shifted = adv * (l + 5.0)  # chi + 5 * adv: what an action-scale offset does
    assert not np.allclose(whiten(shifted), whiten(chi), rtol=1e-6)
