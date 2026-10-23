"""K-armed bandit with a softmax policy, trained by plain policy gradient (PG) or a
delight-gated variant (DG).

DG follows Algorithm 1 (discrete actions) of Osband, "Delightful Policy Gradient",
arXiv:2603.14608: for each sample, surprisal l = -log pi(A), delight chi = U * l
(U = advantage), gate w = sigmoid(chi / eta), update += w * U * grad log pi(A).

This is an independent small implementation for learning purposes. It is NOT a
reproduction of the paper's experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


def grad_log_pi(logits: np.ndarray, action: int) -> np.ndarray:
    """Gradient of log softmax(logits)[action] w.r.t. logits: onehot(a) - pi."""
    g = -softmax(logits)
    g[action] += 1.0
    return g


def dg_gate(advantage: float, prob: float, eta: float = 1.0) -> float:
    """Sigmoid of delight: delight = advantage * surprisal, surprisal = -log prob."""
    chi = advantage * (-np.log(max(prob, 1e-12)))
    return float(1.0 / (1.0 + np.exp(-chi / eta)))


def pg_direction(logits, actions, advantages) -> np.ndarray:
    d = np.zeros_like(logits, dtype=float)
    for a, u in zip(actions, advantages):
        d += u * grad_log_pi(logits, a)
    return d


def dg_direction(logits, actions, advantages, eta: float = 1.0) -> np.ndarray:
    pi = softmax(logits)
    d = np.zeros_like(logits, dtype=float)
    for a, u in zip(actions, advantages):
        d += dg_gate(u, pi[a], eta) * u * grad_log_pi(logits, a)
    return d


def advantage_ema_update(baseline: float, rewards, decay: float = 0.9) -> float:
    return decay * baseline + (1 - decay) * float(np.mean(rewards))


@dataclass
class BanditEnv:
    means: np.ndarray            # Bernoulli success probability per arm
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    @property
    def k(self) -> int:
        return len(self.means)

    def pull(self, arm: int) -> float:
        return float(self.rng.random() < self.means[arm])


def run_agent(means, method: str, steps: int = 400, batch: int = 8, lr: float = 0.5,
              eta: float = 1.0, seed: int = 0, baseline_decay: float = 0.9,
              normalize: bool = False, fixed_baseline: float | None = None):
    """Run one agent. Returns dict with per-step policy probs and expected reward.

    method: "pg" or "dg". Advantage = reward - baseline, where baseline is an EMA of batch
    mean reward, or a constant if fixed_baseline is given.
    normalize: take unit-norm steps (logits += lr * d/||d||), the "normalized step" model
    the paper uses in its bandit analysis (arXiv:2603.14608, Section 4). Equalises step
    size so PG and DG differ only in direction.
    """
    assert method in ("pg", "dg")
    means = np.asarray(means, dtype=float)
    env = BanditEnv(means, seed=seed)
    rng = np.random.default_rng(seed + 10_000)
    logits = np.zeros(env.k)
    baseline = 0.0 if fixed_baseline is None else float(fixed_baseline)
    probs_hist, exp_reward = [], []
    for _ in range(steps):
        pi = softmax(logits)
        probs_hist.append(pi.copy())
        exp_reward.append(float(pi @ means))
        actions = rng.choice(env.k, size=batch, p=pi)
        rewards = np.array([env.pull(a) for a in actions])
        adv = rewards - baseline
        direction = (dg_direction if method == "dg" else pg_direction)(
            logits, actions, adv, *(([eta]) if method == "dg" else []))
        if normalize:
            n = np.linalg.norm(direction)
            logits = logits + (lr * direction / n if n > 0 else 0.0)
        else:
            logits = logits + lr * direction / batch
        if fixed_baseline is None:
            baseline = advantage_ema_update(baseline, rewards, baseline_decay)
    return {"probs": np.array(probs_hist), "expected_reward": np.array(exp_reward),
            "final_logits": logits}
