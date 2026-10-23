"""Symmetric bandit with normalized steps: one correct arm (reward 1), K-1 wrong arms (reward 0).
My reading of the paper's Figure 5 setup (K=100, B=100, alpha=0.1, fixed baseline 0.5). Not an
exact replication: the paper's exact protocol (exact vs sampled gradients, baseline) is not
fully reproduced here."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dgplay import run_agent

K, B, ALPHA, STEPS, SEEDS = 100, 100, 0.1, 300, range(30)
means = np.zeros(K); means[0] = 1.0
fig, ax = plt.subplots(figsize=(6, 4)); out = {}
for m, c in (("pg", "tab:blue"), ("dg", "tab:red")):
    err = np.array([1 - run_agent(means, m, steps=STEPS, batch=B, lr=ALPHA, seed=s,
                                  normalize=True, fixed_baseline=0.5)["probs"][:, 0] for s in SEEDS])
    mu, sem = err.mean(0), err.std(0) / np.sqrt(len(SEEDS))
    ax.plot(mu, color=c, label=m.upper()); ax.fill_between(range(STEPS), mu - sem, mu + sem, color=c, alpha=0.2)
    out[m] = {"error_at_100": float(mu[100]), "error_at_299": float(mu[-1]), "sem_at_299": float(sem[-1]),
              "first_step_error_below_0.1": int(np.argmax(mu < 0.1)) if (mu < 0.1).any() else None}
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("error = 1 - pi(correct arm)")
ax.set_title(f"Symmetric bandit, normalized steps\n(K={K}, B={B}, alpha={ALPHA})", fontsize=10); ax.legend()
fig.tight_layout(); fig.savefig("docs/symmetric_normalized.png", dpi=110)
json.dump(out, open("docs/results_symmetric.json", "w"), indent=1); print(json.dumps(out, indent=1))
