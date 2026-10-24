"""1-D continuous-action bandit with a Gaussian policy.

The paper's continuous-action variant (arXiv:2603.14608, Algorithm 2 in Appendix E.1) uses
density-based surprisal l = -log pi(a), with the log-density clipped to [-10, 10], and notes
that chi = U * l is sensitive to scaling, so one can whiten the batch of delights,
chi_tilde = (chi - mean) / (std + eps), before gating. This module is a minimal testbed for
that suggestion: a smooth 1-D reward and a Gaussian policy with learnable mean and
log-standard-deviation.

Not a reproduction of the paper's DeepMind Control Suite experiments.
"""
from __future__ import annotations
import numpy as np

LOG_CLIP = 10.0  # log-density clip from the paper's Algorithm 2


def gaussian_logp(a, mu: float, log_sigma: float):
    """Log density of N(mu, exp(log_sigma)^2) at a (broadcasts over arrays)."""
    return -0.5 * np.log(2 * np.pi) - log_sigma - 0.5 * ((np.asarray(a) - mu) / np.exp(log_sigma)) ** 2


def grad_logp(a, mu: float, log_sigma: float):
    """Gradient of log pi(a) w.r.t. [mu, log_sigma]. Returns shape (2,) or (len(a), 2)."""
    a = np.asarray(a, dtype=float)
    s2 = np.exp(2 * log_sigma)
    g_mu = (a - mu) / s2
    g_ls = (a - mu) ** 2 / s2 - 1.0
    return np.stack([g_mu, g_ls], axis=-1)


class SmoothBandit:
    """Reward(a) = scale * exp(-(a - a*)^2 / (2 width^2)) + Gaussian noise."""

    def __init__(self, a_star: float = 1.0, width: float = 0.5, scale: float = 1.0,
                 noise: float = 0.1, seed: int = 0):
        self.a_star, self.width, self.scale, self.noise = a_star, width, scale, noise
        self.rng = np.random.default_rng(seed)

    def reward(self, a: float) -> float:
        clean = self.scale * np.exp(-0.5 * ((a - self.a_star) / self.width) ** 2)
        return float(clean + self.noise * self.rng.normal())

    def clean_reward_at(self, a: float) -> float:
        return float(self.scale * np.exp(-0.5 * ((a - self.a_star) / self.width) ** 2))


def run_continuous(method: str = "pg", scale: float = 1.0, steps: int = 600, batch: int = 32,
                   lr: float = 0.05, eta: float = 1.0, whiten: bool = False,
                   baseline_decay: float = 0.9, seed: int = 0):
    """Run one agent on SmoothBandit. method: "pg" or "dg".

    lr is the learning rate for [mu, log_sigma] jointly; callers should scale it by
    1/scale to keep updates comparable across reward scales (the update is linear in the
    advantage). Returns per-step clean reward at the current mean action, normalized by
    scale so 1.0 is the optimum.
    """
    assert method in ("pg", "dg")
    env = SmoothBandit(scale=scale, width=0.5, noise=0.3 * scale, seed=seed)
    rng = np.random.default_rng(seed + 5_000)
    mu, log_sigma = -1.0, 0.0
    baseline = 0.0
    out = []
    for _ in range(steps):
        sigma = np.exp(log_sigma)
        acts = mu + sigma * rng.normal(size=batch)
        rews = np.array([env.reward(a) for a in acts])
        adv = rews - baseline
        if method == "dg":
            surprisal = np.clip(-gaussian_logp(acts, mu, log_sigma), -LOG_CLIP, LOG_CLIP)
            chi = adv * surprisal
            if whiten:
                chi = (chi - chi.mean()) / (chi.std() + 1e-8)
            w = 1.0 / (1.0 + np.exp(-np.clip(chi / eta, -500, 500)))
        else:
            w = np.ones(batch)
        g = ((w * adv)[:, None] * grad_logp(acts, mu, log_sigma)).mean(0)
        mu += lr * g[0]
        log_sigma = float(np.clip(log_sigma + lr * g[1], -3.0, 2.0))
        baseline = baseline_decay * baseline + (1 - baseline_decay) * float(rews.mean())
        out.append(env.clean_reward_at(mu) / scale)
    return {"return_at_mean": np.array(out), "mu": mu, "log_sigma": log_sigma}
