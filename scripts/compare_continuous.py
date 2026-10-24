"""Continuous-action testbed for the paper's Algorithm 2 suggestion: density-based surprisal
with clipped log-densities, and whitening delight before gating.

Runs PG, DG (eta=1), DG with whitened delight, and DG with eta retuned to the reward scale,
on a smooth 1-D bandit at three reward scales (0.01, 1, 100). Learning rates are divided by
the scale so PG is exactly scale-invariant; the question is whether DG's gate is.

Writes docs/continuous_whitening.png and docs/results_continuous.json. About 30 s on a CPU.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dgplay.continuous import run_continuous

SCALES = (0.01, 1.0, 100.0)
STEPS, BATCH, SEEDS, BASE_LR = 600, 32, range(20), 0.05
CONFIGS = [
    ("PG", dict(method="pg")),
    ("DG eta=1", dict(method="dg", eta=1.0)),
    ("DG whitened", dict(method="dg", eta=1.0, whiten=True)),
    ("DG eta=scale", None),  # eta set per scale
]
out = {}
fig, axes = plt.subplots(1, len(SCALES), figsize=(13, 4), sharey=True)
for j, scale in enumerate(SCALES):
    lr = BASE_LR / scale
    ax = axes[j]
    for name, kw in CONFIGS:
        kw = dict(kw) if kw else dict(method="dg", eta=scale)
        runs = np.array([run_continuous(scale=scale, steps=STEPS, batch=BATCH, lr=lr, seed=s, **kw)["return_at_mean"]
                         for s in SEEDS])
        m, sem = runs.mean(0), runs.std(0) / np.sqrt(len(SEEDS))
        auc = runs.mean(1)
        t90 = [int(np.argmax(r > 0.9)) if (r > 0.9).any() else None for r in runs]
        key = f"scale={scale:g}|{name}"
        out[key] = {"mean_return_over_run": float(auc.mean()),
                    "mean_return_sem": float(auc.std() / np.sqrt(len(SEEDS))),
                    "final_return": float(m[-1]),
                    "median_steps_to_0.9": (float(np.median([t for t in t90 if t is not None]))
                                            if any(t is not None for t in t90) else None)}
        ax.plot(m, label=name)
        ax.fill_between(range(STEPS), m - sem, m + sem, alpha=0.2)
    ax.set_title(f"reward scale = {scale:g}", fontsize=10)
    ax.set_xlabel("step")
    if j == 0:
        ax.set_ylabel("clean reward at mean action / max")
        ax.legend(fontsize=8)
fig.suptitle("Gaussian policy on a smooth 1-D bandit (mu0=-1, a*=1, width=0.5, noise=0.3*scale), "
             "lr=0.05/scale, 20 seeds, band = SEM", fontsize=9)
fig.tight_layout()
fig.savefig("docs/continuous_whitening.png", dpi=110)
json.dump(out, open("docs/results_continuous.json", "w"), indent=1)
for k, v in out.items():
    print(f"{k:28s} auc={v['mean_return_over_run']:.3f}±{v['mean_return_sem']:.3f} "
          f"final={v['final_return']:.3f} t90={v['median_steps_to_0.9']}")
