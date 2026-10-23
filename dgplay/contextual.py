"""Synthetic MNIST-style one-step contextual bandit with a linear softmax policy.

Scaled-down analogue of the paper's MNIST diagnostic (a classifier trained from reward
only: reward 1 if the sampled class is right). Uses Gaussian class clusters, not MNIST.
Methods: "pg" (REINFORCE with batch-mean baseline), "dg" (delight-gated), "ce" (supervised
cross-entropy oracle that sees the label).
"""
from __future__ import annotations
import numpy as np


def make_data(n=4000, k=10, d=20, noise=2.5, seed=0, center_seed=12345):
    """seed controls samples; center_seed fixes the class centers (shared by train/test)."""
    centers = np.random.default_rng(center_seed).normal(size=(k, d))
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, size=n)
    X = centers[y] + noise * rng.normal(size=(n, d))
    return X, y


def _softmax_rows(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def step_direction(W, X, y, method, rng, eta=1.0):
    """Returns (grad ascent direction for W, mean reward of the sampled batch)."""
    B, k = len(X), W.shape[1]
    pi = _softmax_rows(X @ W)
    if method == "ce":
        onehot = np.eye(k)[y]
        return X.T @ (onehot - pi) / B, float((pi[np.arange(B), y]).mean())
    a = np.array([rng.choice(k, p=p) for p in pi])
    r = (a == y).astype(float)
    U = r - r.mean()                              # batch-mean baseline
    onehot_a = np.eye(k)[a]
    if method == "dg":
        surprisal = -np.log(np.maximum(pi[np.arange(B), a], 1e-12))
        w = 1.0 / (1.0 + np.exp(-(U * surprisal) / eta))
    else:
        w = np.ones(B)
    coef = (w * U)[:, None] * (onehot_a - pi)     # per-sample grad wrt logits
    return X.T @ coef / B, float(r.mean())


def accuracy(W, X, y):
    return float(((X @ W).argmax(1) == y).mean())


def train(method, X, y, Xt, yt, steps=600, batch=64, lr=0.5, eta=1.0, seed=0, eval_every=20):
    rng = np.random.default_rng(seed)
    k = int(max(y.max(), yt.max())) + 1
    W = np.zeros((X.shape[1], k))
    hist = []
    for t in range(steps):
        idx = rng.integers(0, len(X), size=batch)
        g, _ = step_direction(W, X[idx], y[idx], method, rng, eta)
        W = W + lr * g
        if t % eval_every == 0 or t == steps - 1:
            hist.append((t, accuracy(W, Xt, yt)))
    return np.array(hist), W
