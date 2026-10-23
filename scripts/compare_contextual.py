import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dgplay.contextual import make_data, train

X, y = make_data(n=30000, noise=2.0, seed=0); Xt, yt = make_data(n=5000, noise=2.0, seed=1)
LRS = (0.1, 0.5, 2.0); SEEDS = range(10); STEPS = 1500
fig, axes = plt.subplots(1, len(LRS), figsize=(14, 4), sharey=True)
out = {}
for ax, lr in zip(axes, LRS):
    for m, c in (("ce", "black"), ("pg", "tab:blue"), ("dg", "tab:red")):
        hs = np.array([train(m, X, y, Xt, yt, steps=STEPS, lr=lr, seed=s)[0][:, 1] for s in SEEDS])
        t = train(m, X, y, Xt, yt, steps=STEPS, lr=lr, seed=0)[0][:, 0]
        mean, sem = hs.mean(0), hs.std(0) / np.sqrt(len(SEEDS))
        ax.plot(t, mean, color=c, label=m.upper()); ax.fill_between(t, mean - sem, mean + sem, color=c, alpha=0.2)
        out[f"lr={lr}|{m}"] = {"final_acc": float(mean[-1]), "sem": float(sem[-1]), "acc_at_step_200": float(mean[np.searchsorted(t, 200)])}
    ax.set_title(f"lr={lr}"); ax.set_xlabel("step")
axes[0].set_ylabel("test accuracy"); axes[0].legend()
fig.suptitle("Synthetic 10-class contextual bandit (noise 2.0, reward = correct class), 10 seeds")
fig.tight_layout(); Path("docs").mkdir(exist_ok=True)
fig.savefig("docs/contextual.png", dpi=110); json.dump(out, open("docs/results_contextual.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
