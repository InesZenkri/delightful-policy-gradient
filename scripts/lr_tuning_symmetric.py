"""Per-method learning-rate sweep for result 1 (symmetric bandit, normalized steps).

Result 1 compares PG and DG at the paper's alpha=0.1 only. This script sweeps the
step size separately for each method to check that DG's faster convergence is not a
tuning artifact. Setting: K=100, one correct arm, B=100, fixed baseline 0.5,
normalized steps, 300 steps, 30 seeds (as scripts/symmetric_normalized.py).

    PYTHONPATH=. python scripts/lr_tuning_symmetric.py

Writes docs/results_lr_tuning_symmetric.json and docs/lr_tuning_symmetric.png.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dgplay import run_agent

K, B, STEPS = 100, 100, 300
SEEDS = range(30)
ALPHAS = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
THRESH = 0.1

means = np.zeros(K)
means[0] = 1.0
DOCS = Path(__file__).resolve().parent.parent / "docs"

out = {"K": K, "B": B, "steps": STEPS, "n_seeds": len(SEEDS), "threshold": THRESH,
       "alphas": ALPHAS, "methods": {}}
for m in ("pg", "dg"):
    rows = []
    for a in ALPHAS:
        err = np.array([1 - run_agent(means, m, steps=STEPS, batch=B, lr=a, seed=s,
                                      normalize=True, fixed_baseline=0.5)["probs"][:, 0]
                        for s in SEEDS])
        mu = err.mean(0)
        crossing = int(np.argmax(mu < THRESH)) if (mu < THRESH).any() else None
        rows.append({
            "alpha": a,
            "first_step_below_threshold": crossing,
            "err_at_100": float(mu[100]),
            "err_final_mean": float(err[:, -1].mean()),
            "err_final_se": float(err[:, -1].std(ddof=1) / np.sqrt(len(SEEDS))),
        })
        print(m, rows[-1])
    out["methods"][m] = rows

(DOCS / "results_lr_tuning_symmetric.json").write_text(json.dumps(out, indent=1))

fig, ax = plt.subplots(figsize=(6, 4))
for m, c in (("pg", "tab:blue"), ("dg", "tab:red")):
    xs = [r["alpha"] for r in out["methods"][m] if r["first_step_below_threshold"] is not None]
    ys = [r["first_step_below_threshold"] for r in out["methods"][m] if r["first_step_below_threshold"] is not None]
    ax.plot(xs, ys, "o-", color=c, label=m.upper())
ax.set_xscale("log")
ax.set_xlabel("step size alpha (normalized steps, log scale)")
ax.set_ylabel("first step with mean error < 0.1")
ax.set_title(f"Symmetric bandit (K={K}, B={B}, {len(SEEDS)} seeds):\nmatched step-size sweep", fontsize=10)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(DOCS / "lr_tuning_symmetric.png", dpi=110)
print(f"wrote {DOCS/'lr_tuning_symmetric.png'}")
