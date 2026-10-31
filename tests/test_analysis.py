import numpy as np
from dgplay import softmax, dg_direction, run_agent
from dgplay.bandit import grad_log_pi
from dgplay.analysis import (sigmoid, visit_coefficient, direction_bias,
                             expected_direction, simulate_expected,
                             stall_equilibrium)


def test_visit_coefficient_matches_exact_enumeration():
    # c(pi; p, b, eta) is the per-visit average of w * U over the Bernoulli reward
    # draw, so pi_a * (c_a - sum_j pi_j c_j) must equal the enumerated expected
    # single-sample direction exactly.
    logits = np.array([0.7, -0.2, 1.3, 0.0, -1.1])
    means = np.array([0.2, 0.5, 0.7, 0.9, 0.35])
    b, eta = 0.48, 0.8
    pi = softmax(logits)
    enum = np.zeros(5)
    for a in range(5):
        for r, pr in ((1.0, means[a]), (0.0, 1 - means[a])):
            u = r - b
            chi = u * (-np.log(pi[a]))
            w = 1.0 / (1.0 + np.exp(-chi / eta))
            enum += pi[a] * pr * w * u * grad_log_pi(logits, a)
    c = visit_coefficient(pi, means, b, eta)
    assert np.allclose(pi * (c - pi @ c), enum, atol=1e-12)
    assert np.allclose(expected_direction(logits, means, b, eta, "dg"), enum, atol=1e-12)


def test_expected_direction_matches_monte_carlo():
    logits = np.array([0.4, -0.1, 2.0, -1.0])
    means = np.array([0.3, 0.6, 0.55, 0.8])
    b, eta = 0.5, 1.0
    pi = softmax(logits)
    rng = np.random.default_rng(0)
    n = 200_000
    actions = rng.choice(4, size=n, p=pi)
    rewards = (rng.random(n) < means[actions]).astype(float)
    adv = rewards - b
    chi = adv * (-np.log(pi[actions]))
    w = sigmoid(chi / eta)
    coef = w * adv
    mc = (np.bincount(actions, weights=coef, minlength=4) - pi * coef.sum()) / n
    assert np.allclose(expected_direction(logits, means, b, eta), mc, atol=0.01)


def test_direction_bias_closed_form_at_matched_baseline():
    # With b = p the bias is p(1-p)[sigmoid(p l/eta) + sigmoid((1-p) l/eta) - 1] >= 0
    # and grows as the arm gets rarer (larger l = smaller pi).
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        for eta in (0.5, 1.0, 3.0):
            pis = np.logspace(-4, -0.001, 50)
            l = -np.log(pis)
            closed = p * (1 - p) * (sigmoid(p * l / eta) + sigmoid((1 - p) * l / eta) - 1)
            bias = direction_bias(pis, p, p, eta)
            assert np.allclose(bias, closed, atol=1e-12)
            assert np.all(bias >= -1e-12)
            assert np.all(np.diff(bias) < 0)   # larger for rarer arms


def test_bias_analysis_covers_existing_unit_test_case():
    # The case of test_dg_has_upward_bias_on_rare_arm_with_noisy_equal_arms,
    # through the general formula: equal Bernoulli(0.5) arms, b = 0.5, one rare arm.
    logits = np.array([2.0, 2.0, 2.0, 2.0, -1.0])
    means = np.full(5, 0.5)
    d = expected_direction(logits, means, 0.5, 1.0, "dg")
    assert d[4] > 1e-3
    assert np.allclose(expected_direction(logits, means, 0.5, method="pg"), 0, atol=1e-12)


def test_balance_equation_matches_expected_dynamics_plateau():
    # Canonical noisy 20-arm setting: the un-normalized expected dynamics converges
    # to the balance-equation fixed point.
    k, p, gap = 20, 0.5, 0.1
    means = np.array([p] * (k - 1) + [p + gap])
    eq = stall_equilibrium(k, p, gap, eta=1.0)
    r = simulate_expected(means, "dg", steps=30_000, lr=1.0, eta=1.0,
                          baseline_decay=0.9, normalize=False)
    assert abs(r["probs"][-1, -1] - eq["pi_good"]) < 0.02
    assert abs((p + gap) - r["expected_reward"][-1] - eq["regret"]) < 0.01


def test_balance_equation_is_stable_fixed_point():
    # Perturbing pi_best off the fixed point creates a restoring sign of f = c_g - c_r:
    # above the fixed point f < 0 (pi_best pushed back down), below it f > 0.
    k, p, gap = 20, 0.5, 0.1
    eq = stall_equilibrium(k, p, gap)
    from dgplay.analysis import visit_coefficient as vc
    for delta, sign in ((-0.03, 1), (0.03, -1)):
        pi_g = eq["pi_good"] + delta
        pi_r = (1 - pi_g) / (k - 1)
        b = p + pi_g * gap
        f = float(vc(pi_g, p + gap, b) - vc(pi_r, p, b))
        assert np.sign(f) == sign


def test_large_eta_removes_stall_in_prediction():
    # Flattening the gate (large eta) moves the fixed point to the optimum,
    # matching the sweep's observation that eta >= 4 removes the stall.
    stall_1 = stall_equilibrium(20, 0.5, 0.1, eta=1.0)["regret"]
    stall_8 = stall_equilibrium(20, 0.5, 0.1, eta=8.0)["regret"]
    assert stall_1 > 0.05
    assert stall_8 < 0.2 * stall_1


def test_pg_has_no_interior_fixed_point():
    # PG's coefficient p - b is independent of pi, so its expected direction keeps
    # pushing the best arm up: the expected dynamics solves the canonical task.
    means = np.array([0.5] * 19 + [0.6])
    r = simulate_expected(means, "pg", steps=1500, lr=0.1, normalize=True)
    assert 0.6 - r["expected_reward"][-1] < 0.02


def test_simulate_expected_is_deterministic():
    means = np.array([0.5] * 19 + [0.6])
    a = simulate_expected(means, "dg", steps=200)
    b = simulate_expected(means, "dg", steps=200)
    assert np.array_equal(a["probs"], b["probs"])
