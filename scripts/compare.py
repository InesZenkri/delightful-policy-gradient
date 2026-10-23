"""Multi-seed PG vs DG comparison across a few bandit settings. Writes docs/results.json and docs/compare.png."""
import json, sys, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dgplay import run_agent

SCENARIOS = {
    "easy_3_arms": dict(means=[0.2, 0.4, 0.8], steps=300),
    "hard_20_arms_small_gap": dict(means=[0.5] * 19 + [0.6], steps=1500),
    "sparse_reward_20_arms": dict(means=[0.02] * 19 + [0.15], steps=1500),
}
ap = argparse.ArgumentParser(); ap.add_argument("--normalize", action="store_true"); A = ap.parse_args()
TAG = "_normalized" if A.normalize else ""
SEEDS = range(30)
LRS = (0.05, 0.1, 0.3) if A.normalize else (0.1, 0.5, 2.0)
out = {}
fig, axes = plt.subplots(len(LRS), len(SCENARIOS), figsize=(14, 8), sharex="col")
for j, (name, cfg) in enumerate(SCENARIOS.items()):
    means = np.array(cfg["means"]); best = means.max()
    for i, lr in enumerate(LRS):
        ax = axes[i, j]
        for method, color in (("pg", "tab:blue"), ("dg", "tab:red")):
            runs = np.array([run_agent(means, method, steps=cfg["steps"], lr=lr, seed=s, normalize=A.normalize)["expected_reward"]
                             for s in SEEDS])
            gap = best - runs                    # per-step expected regret
            m, sd = gap.mean(0), gap.std(0)
            ax.plot(m, color=color, label=method.upper())
            ax.fill_between(range(len(m)), m - sd / np.sqrt(len(SEEDS)), m + sd / np.sqrt(len(SEEDS)), color=color, alpha=0.2)
            out[f"{name}|lr={lr}|{method}"] = {
                "final_gap_mean": float(gap[:, -1].mean()),
                "final_gap_sem": float(gap[:, -1].std() / np.sqrt(len(SEEDS))),
                "mean_gap_over_run": float(gap.mean()),
            }
        ax.set_title(f"{name}  lr={lr}", fontsize=9)
        if j == 0: ax.set_ylabel("expected regret")
        if i == len(LRS) - 1: ax.set_xlabel("step")
        if i == 0 and j == 0: ax.legend()
fig.suptitle(("NORMALIZED STEPS. " if A.normalize else "") + "PG (REINFORCE, EMA baseline) vs DG, mean over 30 seeds, band = SEM")
fig.tight_layout()
Path("docs").mkdir(exist_ok=True)
fig.savefig(f"docs/compare{TAG}.png", dpi=110)
json.dump(out, open(f"docs/results{TAG}.json", "w"), indent=1)
for k, v in out.items():
    print(f"{k:50s} final={v['final_gap_mean']:.4f}±{v['final_gap_sem']:.4f} avg={v['mean_gap_over_run']:.4f}")
