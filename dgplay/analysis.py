"""Exact expected-update analysis for the delight gate on Bernoulli bandits.

For a Bernoulli arm with success probability p, policy probability pi, baseline b
and gate temperature eta, DG's per-visit update averages (over the reward draw) to
a scalar coefficient on grad log pi(a):

    c(pi; p, b, eta) = p (1-b) sigmoid((1-b) l / eta) - (1-p) b sigmoid(-b l / eta),
        with l = -log pi,

so the expected DG direction over both the action draw and the rewards is

    E[d]_a = pi_a * (c_a - sum_j pi_j c_j).

PG is the same formula with the gate fixed at 1, i.e. c_a = p_a - b. DG's
coefficient minus PG's is the gate's direction bias; at a matched baseline (b = p)
it has the closed form

    bias = p (1-p) [ sigmoid(p l / eta) + sigmoid((1-p) l / eta) - 1 ] >= 0,

which grows as the arm gets rarer (larger l). Because c is strictly decreasing in
pi_a, every arm's probability moves toward the allocation where all coefficients
are equal (c_a = weighted mean). For PG that balance only holds at the optimum;
for DG it generally holds at an interior allocation - the noisy-reward stall of
README results 3 and 5. `stall_equilibrium` solves that balance condition.
"""
from __future__ import annotations

import numpy as np

from .bandit import softmax


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def visit_coefficient(probs, p, baseline: float, eta: float = 1.0):
    """Expected per-visit coefficient c(pi; p, b, eta) on grad log pi(a).

    `probs` may be a scalar or an array (one arm's probability, or one per arm);
    `p` broadcasts the same way (that arm's success probability).
    """
    probs = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    l = -np.log(probs)
    return (p * (1 - baseline) * sigmoid((1 - baseline) * l / eta)
            - (1 - p) * baseline * sigmoid(-baseline * l / eta))


def direction_bias(probs, p, baseline: float, eta: float = 1.0):
    """DG's expected per-visit coefficient minus PG's (p - baseline), same shape."""
    return visit_coefficient(probs, p, baseline, eta) - (p - baseline)


def expected_direction(logits, means, baseline: float, eta: float = 1.0,
                       method: str = "dg") -> np.ndarray:
    """Exact E[direction] over the action draw and Bernoulli rewards.

    For un-normalized updates this is the exact per-step expected direction (the
    batch average concentrates on it). For normalized steps it is the direction
    the normalized update follows on average, up to the E[d/||d||] vs
    E[d]/||E[d]|| gap, which the experiment checks empirically.
    """
    assert method in ("pg", "dg")
    means = np.asarray(means, dtype=float)
    pi = softmax(logits)
    if method == "pg":
        c = means - baseline
    else:
        c = visit_coefficient(pi, means, baseline, eta)
    return pi * (c - pi @ c)


def simulate_expected(means, method: str, steps: int = 1500, lr: float = 0.1,
                      eta: float = 1.0, baseline_decay: float = 0.9,
                      normalize: bool = True, fixed_baseline: float | None = None):
    """Deterministic expected-update counterpart of run_agent (no sampling).

    Mirrors run_agent's options; the EMA baseline tracks the expected batch reward
    instead of the sampled one. Returns the same keys.
    """
    assert method in ("pg", "dg")
    means = np.asarray(means, dtype=float)
    k = len(means)
    logits = np.zeros(k)
    baseline = 0.0 if fixed_baseline is None else float(fixed_baseline)
    probs_hist, exp_reward = [], []
    for _ in range(steps):
        pi = softmax(logits)
        probs_hist.append(pi.copy())
        exp_reward.append(float(pi @ means))
        d = expected_direction(logits, means, baseline, eta, method)
        if normalize:
            n = np.linalg.norm(d)
            logits = logits + (lr * d / n if n > 0 else 0.0)
        else:
            logits = logits + lr * d
        if fixed_baseline is None:
            baseline = baseline_decay * baseline + (1 - baseline_decay) * exp_reward[-1]
    return {"probs": np.array(probs_hist), "expected_reward": np.array(exp_reward),
            "final_logits": logits}


def stall_equilibrium(k: int, p_rest: float, gap: float, eta: float = 1.0,
                      n_good: int = 1, tol: float = 1e-10):
    """Solve the gate-balance condition for a two-group Bernoulli bandit.

    n_good arms have success probability p_rest + gap, the other k - n_good arms
    p_rest. At an interior symmetric fixed point all visit coefficients are equal
    and the baseline equals the expected reward:

        c(pi_g; p_g, b) = c(pi_r; p_r, b),   b = p_r + n_good * pi_g * gap,
        n_good * pi_g + (k - n_good) * pi_r = 1.

    pi_g increases whenever c_g > c_r (pi_a moves up iff c_a exceeds the
    probability-weighted mean coefficient, which here lies between c_r and c_g),
    and c is strictly decreasing in pi, so the first sign change of
    f(pi_g) = c_g - c_r above the uniform point is the stable fixed point.

    Returns dict(pi_good, pi_rest, baseline, regret, uniform_regret). `regret` is
    the expected regret at the fixed point; the prediction "no stall" corresponds
    to regret ~ 0, i.e. pi_good ~ 1 / n_good.
    """
    p_g = p_rest + gap
    n_bad = k - n_good

    def f(pi_g):
        pi_r = (1 - n_good * pi_g) / n_bad
        b = p_rest + n_good * pi_g * gap
        return float(visit_coefficient(pi_g, p_g, b, eta)
                     - visit_coefficient(pi_r, p_rest, b, eta))

    lo, hi = 1.0 / k, (1 - 1e-9) / n_good   # per-arm: uniform .. all mass on good arms
    flo = f(lo)
    # scan for the first + to - sign change above uniform
    grid = np.linspace(lo, hi, 2001)[1:]
    prev_x, prev_f = lo, flo
    root = None
    for x in grid:
        fx = f(x)
        if prev_f > 0 >= fx:
            a, b_, fa, fb = prev_x, x, prev_f, fx
            for _ in range(200):
                m = 0.5 * (a + b_)
                fm = f(m)
                if fa > 0 >= fm:
                    b_, fb = m, fm
                else:
                    a, fa = m, fm
                if b_ - a < tol:
                    break
            root = 0.5 * (a + b_)
            break
        prev_x, prev_f = x, fx
    if root is None:          # no interior crossing: predict convergence, no stall
        root = hi
    pi_r = (1 - n_good * root) / n_bad
    b = p_rest + n_good * root * gap
    regret = float(gap * (1 - n_good * root))
    return {"pi_good": float(root), "pi_rest": float(pi_r), "baseline": float(b),
            "regret": regret, "uniform_regret": float(gap * (1 - n_good / k)),
            "f_at_uniform": flo}
