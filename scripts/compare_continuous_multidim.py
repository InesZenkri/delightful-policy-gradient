"""Multi-dimensional continuous testbed: per-dimension action scales and the gate.

The paper's Algorithm 2 (Appendix E.1) uses a diagonal Gaussian policy with log-densities
clipped to [-10, 10], and Section 2.2 suggests whitening the batch of delights,
chi_tilde = (chi - mean) / (std + eps), "when action scales vary substantially across
dimensions". Result 6 tested whitening against reward scaling in 1-D, where it works. This
script tests it against per-dimension ACTION scaling, the case the paper motivates it with.

Design: a d=3 smooth bandit whose per-dimension widths set the action scale. The policy
starts at each dimension's scale (sigma_i = w_i) and per-dimension learning rates
lr * w_i^2 make the update scale-free in z-units, so PG is exactly invariant to the widths
by construction - a designed control showing the task itself is unchanged. What remains is
the gate's sensitivity to the scale-dependent offset in the Gaussian log-density.

Methods: PG, DG (eta=1, clipped log-density), DG with the paper's batch whitening, and DG
with a mode-relative (Mahalanobis 0.5*z^2) surprisal - our scale-free comparator, not the
paper's variant. Also two mixed-width settings whose log-scales sum to the same value as a
uniform setting: if the offset story is right, they must match the uniform setting.

Writes docs/continuous_multidim.png and docs/results_continuous_multidim.json.
About 1 minute on a CPU.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dgplay.continuous import run_continuous_multidim, gaussian_logp_diag, LOG_CLIP

SCALE_SETS = [
    ("all w=0.02", (0.02, 0.02, 0.02)),
    ("all w=0.2", (0.2, 0.2, 0.2)),
    ("all w=1", (1.0, 1.0, 1.0)),
    ("all w=5", (5.0, 5.0, 5.0)),
    ("all w=50", (50.0, 50.0, 50.0)),
    ("mixed (0.05,1,20)", (0.05, 1.0, 20.0)),
    ("mixed (0.05,0.2,20)", (0.05, 0.2, 20.0)),
]
CONFIGS = [
    ("PG", dict(method="pg")),
    ("DG eta=1", dict(method="dg", eta=1.0)),
    ("DG whitened (paper)", dict(method="dg", eta=1.0, whiten=True)),
    ("DG mahalanobis", dict(method="dg", eta=1.0, surprisal="mahalanobis")),
]
STEPS, BATCH, SEEDS, LR, NOISE = 1200, 32, range(20), 0.05, 0.05


def surprisal_probe(widths, steps=100, batch=32, seed=0):
    """Mean and clipped fraction of the raw clipped log-density surprisal early in training."""
    w = np.asarray(widths, float)
    rng = np.random.default_rng(seed + 5_000)
    mu, log_sigma = -1.5 * w.copy(), np.log(w.copy())
    means, clips = [], []
    for _ in range(steps):
        sigma = np.exp(log_sigma)
        acts = mu + sigma * rng.normal(size=(batch, len(w)))
        raw = -gaussian_logp_diag(acts, mu, log_sigma)
        means.append(float(raw.mean()))
        clips.append(float((np.abs(raw) >= LOG_CLIP).mean()))
    return {"mean_surprisal": float(np.mean(means)), "clip_fraction": float(np.mean(clips))}


out, probes = {}, {}
fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
for j, (sname, widths) in enumerate(SCALE_SETS):
    ax = axes.flat[j]
    probes[sname] = surprisal_probe(widths)
    for name, kw in CONFIGS:
        runs = np.array([run_continuous_multidim(widths=widths, steps=STEPS, batch=BATCH,
                                                 lr=LR, noise=NOISE, seed=s, **kw)["return_at_mean"]
                         for s in SEEDS])
        m, sem = runs.mean(0), runs.std(0) / np.sqrt(len(list(SEEDS)))
        auc = runs.mean(1)
        out[f"{sname}|{name}"] = {
            "mean_return_over_run": float(auc.mean()),
            "mean_return_sem": float(auc.std() / np.sqrt(len(list(SEEDS)))),
            "final_return": float(m[-1]),
            "final_return_sem": float(sem[-1]),
        }
        ax.plot(m, label=name)
        ax.fill_between(range(STEPS), m - sem, m + sem, alpha=0.2)
    ax.set_title(sname, fontsize=10)
    ax.set_xlabel("step")
    if j in (0, 4):
        ax.set_ylabel("clean reward at mean action")
    if j == 0:
        ax.legend(fontsize=8)
# summary panel: final return per setting
ax = axes.flat[7]
x = np.arange(len(SCALE_SETS))
for k, (name, kw) in enumerate(CONFIGS):
    finals = [out[f"{sname}|{name}"]["final_return"] for sname, _ in SCALE_SETS]
    ax.plot(x, finals, marker="o", label=name)
ax.set_xticks(x)
ax.set_xticklabels([s for s, _ in SCALE_SETS], rotation=45, ha="right", fontsize=7)
ax.set_title("final return per setting", fontsize=10)
ax.legend(fontsize=8)
fig.suptitle("Per-dimension action scales: smooth 3-D bandit, diagonal Gaussian policy (sigma_i = w_i), "
             "per-dim lr 0.05*w_i^2, noise 0.05, 20 seeds, band = SEM. PG is scale-free by construction (control).",
             fontsize=9)
fig.tight_layout()
fig.savefig("docs/continuous_multidim.png", dpi=110)
json.dump({"results": out, "surprisal_probes": probes,
           "config": {"steps": STEPS, "batch": BATCH, "lr": LR, "noise": NOISE,
                      "seeds": len(list(SEEDS)), "log_clip": LOG_CLIP}},
          open("docs/results_continuous_multidim.json", "w"), indent=1)
for k, v in out.items():
    print(f"{k:38s} auc={v['mean_return_over_run']:.3f}+-{v['mean_return_sem']:.3f} "
          f"final={v['final_return']:.3f}+-{v['final_return_sem']:.3f}")
print(json.dumps(probes, indent=1))
