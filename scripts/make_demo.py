"""Animated side-by-side view of PG vs DG learning on a 5-arm bandit (same seed/environment)."""
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from dgplay import run_agent

ap = argparse.ArgumentParser()
ap.add_argument("--means", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.4])
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--lr", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="docs/demo.gif")
a = ap.parse_args()
means = np.array(a.means)
runs = {m: run_agent(means, m, steps=a.steps, lr=a.lr, seed=a.seed) for m in ("pg", "dg")}
k = len(means)
fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), gridspec_kw={"height_ratios": [3, 2]})
colors = {"pg": "tab:blue", "dg": "tab:red"}
names = {"pg": "PG (REINFORCE + EMA baseline)", "dg": "DG (delight-gated)"}
bars = {}
for col, m in enumerate(("pg", "dg")):
    ax = axes[0, col]
    bars[m] = ax.bar(range(k), np.ones(k) / k, color=colors[m])
    ax.set_ylim(0, 1); ax.set_xticks(range(k))
    ax.set_xticklabels([f"arm {i}\np={p:.2f}" for i, p in enumerate(means)], fontsize=8)
    ax.set_title(names[m], fontsize=10); ax.set_ylabel("policy probability" if col == 0 else "")
ax = axes[1, 0]; ax.axis("off")
axr = axes[1, 1]
best = means.max()
for m in ("pg", "dg"):
    axr.plot(best - runs[m]["expected_reward"], color=colors[m], alpha=0.25)
lines = {m: axr.plot([], [], color=colors[m], label=m.upper())[0] for m in ("pg", "dg")}
axr.set_xlim(0, a.steps); axr.set_ylim(0, best - runs["pg"]["expected_reward"][0] + 0.02)
axr.set_xlabel("step"); axr.set_ylabel("expected regret"); axr.legend()
txt = ax.text(0, 0.5, "", fontsize=10, va="center", family="monospace")
fig.suptitle("Independent small-scale demo of DG (arXiv:2603.14608) vs PG. Not a paper reproduction.", fontsize=9)
fig.tight_layout()
frames = list(range(0, a.steps, 5)) + [a.steps - 1]

def update(t):
    for m in ("pg", "dg"):
        for b, h in zip(bars[m], runs[m]["probs"][t]):
            b.set_height(h)
        lines[m].set_data(range(t + 1), (best - runs[m]["expected_reward"])[: t + 1])
    txt.set_text(f"step {t}\nexpected reward\n  PG {runs['pg']['expected_reward'][t]:.3f}\n  DG {runs['dg']['expected_reward'][t]:.3f}\n  best arm {best:.2f}")
    return []

FuncAnimation(fig, update, frames=frames).save(a.out, writer=PillowWriter(fps=12))
update(a.steps - 1); fig.savefig(a.out.replace(".gif", "_final.png"), dpi=110)
print("wrote", a.out)
