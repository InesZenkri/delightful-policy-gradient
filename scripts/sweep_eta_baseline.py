"""Sweep DG's temperature eta and the EMA baseline strength (baseline_decay) in two bandit
settings where DG behaves differently, plus a PG reference per decay and a whitened-DG row.

Scenario A "symmetric": K=100, one correct arm, normalized steps (the setting where DG was
faster; scripts/symmetric_normalized.py used a fixed baseline of 0.5 - here the baseline is
an EMA, so the sweep covers how much that choice matters).
Scenario B "noisy_20arm": 20 near-equal Bernoulli arms with one slightly better arm,
normalized steps (the setting where DG stalled).

Writes docs/sweep_eta_baseline.png and docs/results_sweep.json. A few minutes on a CPU.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dgplay import run_agent

ETAS = (0.25, 0.5, 1.0, 2.0, 4.0)
DECAYS = (0.0, 0.5, 0.9, 0.99)   # 0.0 = plain batch-mean baseline, no memory
SEEDS = range(10)

SCENARIOS = {
    "symmetric": dict(means=[1.0] + [0.0] * 99, steps=300, batch=100, lr=0.1),
    "noisy_20arm": dict(means=[0.5] * 19 + [0.6], steps=1500, batch=8, lr=0.1),
}


def run_cfg(scen, method, eta=None, decay=0.9, whiten=False):
    cfg = SCENARIOS[scen]
    means = np.array(cfg["means"])
    best = means.max()
    finals, aucs, t90 = [], [], []
    for s in SEEDS:
        r = run_agent(means, method, steps=cfg["steps"], batch=cfg["batch"], lr=cfg["lr"],
                      eta=eta if eta is not None else 1.0, seed=s, baseline_decay=decay,
                      normalize=True, whiten=whiten)
        if scen == "symmetric":
            err = 1 - r["probs"][:, 0]          # error = 1 - pi(correct arm)
            metric = err
        else:
            metric = best - r["expected_reward"]  # expected regret
        finals.append(metric[-1])
        aucs.append(metric.mean())
        below = np.where(metric < 0.05)[0]
        t90.append(int(below[0]) if len(below) else None)
    n = len(SEEDS)
    return {"final_mean": float(np.mean(finals)), "final_sem": float(np.std(finals) / np.sqrt(n)),
            "mean_over_run": float(np.mean(aucs)), "mean_over_run_sem": float(np.std(aucs) / np.sqrt(n)),
            "median_first_step_below_0.05": (float(np.median([t for t in t90 if t is not None]))
                                             if any(t is not None for t in t90) else None)}


out, grids = {}, {}
for scen in SCENARIOS:
    grid = np.zeros((len(DECAYS), len(ETAS)))
    for i, d in enumerate(DECAYS):
        out[f"{scen}|pg|decay={d}"] = run_cfg(scen, "pg", decay=d)
        out[f"{scen}|dg_whitened|eta=1|decay={d}"] = run_cfg(scen, "dg", eta=1.0, decay=d, whiten=True)
        for k, e in enumerate(ETAS):
            r = run_cfg(scen, "dg", eta=e, decay=d)
            out[f"{scen}|dg|eta={e}|decay={d}"] = r
            # final error is 0 everywhere in symmetric; use run-average there
            grid[i, k] = r["final_mean"] if scen == "noisy_20arm" else r["mean_over_run"]
    grids[scen] = grid

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (scen, grid) in zip(axes, grids.items()):
    im = ax.imshow(grid, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(ETAS)), [str(e) for e in ETAS])
    ax.set_yticks(range(len(DECAYS)), [str(d) for d in DECAYS])
    ax.set_xlabel("eta"); ax.set_ylabel("baseline EMA decay")
    for i in range(len(DECAYS)):
        for k in range(len(ETAS)):
            ax.text(k, i, f"{grid[i, k]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if grid[i, k] < grid.max() * 0.7 else "black")
    fld = "final_mean" if scen == "noisy_20arm" else "mean_over_run"
    pg_ref = "; ".join(f"{d}: {out[f'{scen}|pg|decay={d}'][fld]:.3f}" for d in DECAYS)
    wh_ref = "; ".join(f"{d}: {out[f'{scen}|dg_whitened|eta=1|decay={d}'][fld]:.3f}" for d in DECAYS)
    ax.set_title(f"{scen}: DG grid (10 seeds)\nPG by decay - {pg_ref}\nDG whitened eta=1 - {wh_ref}",
                 fontsize=7.5)
    fig.colorbar(im, ax=ax)
fig.suptitle("Normalized steps. Metric: mean error over run (symmetric) / final expected regret (noisy_20arm), mean over seeds")
fig.tight_layout()
fig.savefig("docs/sweep_eta_baseline.png", dpi=110)
json.dump(out, open("docs/results_sweep.json", "w"), indent=1)
for k, v in out.items():
    print(f"{k:42s} final={v['final_mean']:.4f}±{v['final_sem']:.4f} avg={v['mean_over_run']:.4f}")
