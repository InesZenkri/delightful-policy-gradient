import numpy as np
from dgplay import softmax, dg_gate, pg_direction, dg_direction, run_agent
from dgplay.bandit import grad_log_pi


def test_softmax_sums_to_one_and_stable():
    p = softmax(np.array([1000.0, 0.0, -1000.0]))
    assert np.isclose(p.sum(), 1.0) and np.all(np.isfinite(p))


def test_grad_log_pi_matches_finite_difference():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=5)
    for a in range(5):
        g = grad_log_pi(logits, a)
        num = np.zeros(5)
        for i in range(5):
            e = np.zeros(5); e[i] = 1e-6
            num[i] = (np.log(softmax(logits + e)[a]) - np.log(softmax(logits - e)[a])) / 2e-6
        assert np.allclose(g, num, atol=1e-6)


def test_gate_values():
    assert np.isclose(dg_gate(0.0, 0.3), 0.5)                      # zero advantage -> 0.5
    p = 0.01
    assert dg_gate(1.0, p) > 0.98                                   # rare success amplified
    assert dg_gate(-1.0, p) < 0.02                                  # rare failure suppressed
    assert np.isclose(dg_gate(1.0, 0.5), 1 / (1 + np.exp(-np.log(2))))  # hand computed


def test_gate_matches_formula():
    u, p, eta = 0.7, 0.2, 1.0
    chi = u * -np.log(p)
    assert np.isclose(dg_gate(u, p, eta), 1 / (1 + np.exp(-chi / eta)))


def test_zero_advantage_gives_zero_update():
    logits = np.array([0.1, -0.2, 0.3])
    assert np.allclose(dg_direction(logits, [0, 1, 2], [0, 0, 0]), 0)
    assert np.allclose(pg_direction(logits, [0, 1, 2], [0, 0, 0]), 0)


def test_dg_downweights_rare_failure_vs_pg():
    logits = np.array([3.0, 0.0, 0.0, 0.0])   # arm 3 rare
    pg = pg_direction(logits, [3], [-1.0])
    dg = dg_direction(logits, [3], [-1.0])
    assert np.linalg.norm(dg) < 0.1 * np.linalg.norm(pg)


def test_direction_is_ascent_for_expected_reward_in_expectation():
    # With exact baseline = expected reward, the DG direction (expected over actions)
    # should still correlate positively with the true reward gradient.
    means = np.array([0.1, 0.5, 0.9])
    logits = np.array([0.5, 0.0, -0.5])
    pi = softmax(logits)
    v = pi @ means
    true_grad = pi * (means - v)           # d E[r] / d logits
    exp_dir = np.zeros(3)
    for a in range(3):
        for r, pr in ((1.0, means[a]), (0.0, 1 - means[a])):
            exp_dir += pi[a] * pr * dg_direction(logits, [a], [r - v])
    assert exp_dir @ true_grad > 0


def test_learning_improves_and_is_deterministic():
    means = [0.2, 0.4, 0.8]
    a = run_agent(means, "pg", steps=200, seed=1)
    b = run_agent(means, "pg", steps=200, seed=1)
    assert np.array_equal(a["probs"], b["probs"])
    assert a["expected_reward"][-1] > a["expected_reward"][0] + 0.1
    c = run_agent(means, "dg", steps=200, seed=1)
    assert c["expected_reward"][-1] > c["expected_reward"][0] + 0.1


def test_contextual_gradient_matches_finite_difference_for_ce():
    from dgplay.contextual import make_data, step_direction, _softmax_rows
    X, y = make_data(n=32, k=4, d=5, seed=3)
    rng = np.random.default_rng(0)
    W = rng.normal(size=(5, 4)) * 0.3
    g, _ = step_direction(W, X, y, "ce", rng)

    def ll(W_):
        p = _softmax_rows(X @ W_)
        return np.log(p[np.arange(len(y)), y]).mean()
    num = np.zeros_like(W)
    for i in range(5):
        for j in range(4):
            e = np.zeros_like(W); e[i, j] = 1e-6
            num[i, j] = (ll(W + e) - ll(W - e)) / 2e-6
    assert np.allclose(g, num, atol=1e-6)


def test_contextual_pg_equals_dg_when_gate_is_one_half_scaled():
    # With huge eta the gate -> 0.5 for every sample, so DG direction = 0.5 * PG direction.
    from dgplay.contextual import make_data, step_direction
    X, y = make_data(n=64, k=4, d=5, seed=4)
    W = np.random.default_rng(1).normal(size=(5, 4)) * 0.3
    g_pg, _ = step_direction(W, X, y, "pg", np.random.default_rng(7))
    g_dg, _ = step_direction(W, X, y, "dg", np.random.default_rng(7), eta=1e9)
    assert np.allclose(g_dg, 0.5 * g_pg, atol=1e-6)


def test_contextual_methods_learn():
    from dgplay.contextual import make_data, train
    X, y = make_data(n=2000, seed=0); Xt, yt = make_data(n=1000, seed=1)
    for m in ("pg", "dg", "ce"):
        h, _ = train(m, X, y, Xt, yt, steps=300, seed=0)
        assert h[-1, 1] > 0.3, m  # chance is 0.1


def test_normalized_steps_have_equal_logit_movement():
    means = np.zeros(10); means[3] = 1.0
    for m in ("pg", "dg"):
        r = run_agent(means, m, steps=1, batch=50, lr=0.1, normalize=True, fixed_baseline=0.5, seed=2)
        assert np.isclose(np.linalg.norm(r["final_logits"]), 0.1, atol=1e-9)


def test_dg_has_upward_bias_on_rare_arm_with_noisy_equal_arms():
    """Documented limitation: with Bernoulli(0.5) rewards on all arms and baseline 0.5,
    PG expected direction is zero, but DG pushes probability toward the rarest arm."""
    K = 5
    logits = np.array([2.0, 2.0, 2.0, 2.0, -1.0])   # last arm is rare
    pi = softmax(logits)
    exp_dg, exp_pg = np.zeros(K), np.zeros(K)
    for a in range(K):
        for r, pr in ((1.0, 0.5), (0.0, 0.5)):
            exp_dg += pi[a] * pr * dg_direction(logits, [a], [r - 0.5])
            exp_pg += pi[a] * pr * pg_direction(logits, [a], [r - 0.5])
    assert np.allclose(exp_pg, 0, atol=1e-12)
    assert exp_dg[4] > 1e-3          # rare arm logit pushed up
