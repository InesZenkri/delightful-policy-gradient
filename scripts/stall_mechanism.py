"""Characterize the noisy-reward stall mechanism of the DG gate (issue #5).

The sweep (README result 5) showed DG stalls on the noisy 20-arm setting at every
baseline strength and pointed at the gate's direction bias. dgplay/analysis.py
derives that bias in closed form: DG's expected per-visit coefficient on arm a is
c(pi_a; p_a) (module docstring there), strictly decreasing in pi_a, so the
expected dynamics has an interior fixed point where all arms' coefficients are
equal - the stall. This script tests whether that derivation actually predicts
the stall, and maps where the prediction holds.

Four parts, all normalized steps, batch 8, lr 0.1, EMA decay 0.9 (the canonical
setting of results 3 and 5):

A. Canonical noisy 20-arm run: stochastic PG/DG trajectories vs the deterministic
   expected-update ODE vs the balance-equation plateau.
B. Coefficient curves at the canonical equilibrium (the mechanism picture).
C. Generality grid: K in {2,5,10,20,50} x gap in {0.05,0.1,0.2,0.4} x p_rest in
   {0.3,0.5,0.7} (5 seeds per config, DG and PG): observed plateau regret vs the
   balance-equation prediction. Also reports where the prediction fails.
D. Robustness: eta scan on the canonical setting (extends result 5) and a scan
   over the number of good arms, observed vs predicted.

Metrics are mean expected regret over the last 500 of 1500 steps (the normalized
update has constant step size, so trajectories oscillate around the plateau
rather than converging to a point).

Writes docs/stall_mechanism.png and docs/results_stall_mechanism.json.
A few minutes on a CPU.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dgplay import run_agent
from dgplay.analysis import (visit_coefficient, stall_equilibrium,
                             simulate_expected)

STEPS, LATE, BATCH, LR, DECAY = 1500, 500, 8, 0.1, 0.9
CANON = dict(k=20, p=0.5, gap=0.1)
GRID_K = (2, 5, 10, 20, 50)
GRID_GAP = (0.05, 0.1, 0.2, 0.4)
GRID_P = (0.3, 0.5, 0.7)
ETAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
N_GOODS = (1, 2, 5, 10)


def means_two_groups(k, p, gap, n_good=1):
    return np.array([p + gap] * n_good + [p] * (k - n_good))


def late_regret(result, best):
    return float(best - result["expected_reward"][-LATE:].mean())


def observed(means, method, seeds, eta=1.0):
    seeds = list(seeds)
    best = means.max()
    vals = [late_regret(run_agent(means, method, steps=STEPS, batch=BATCH, lr=LR,
                                  eta=eta, seed=s, baseline_decay=DECAY,
                                  normalize=True), best)
            for s in seeds]
    return {"mean": float(np.mean(vals)), "sem": float(np.std(vals) / np.sqrt(len(vals))),
            "runs": [round(v, 4) for v in vals]}


def predicted(means, eta=1.0, n_good=1):
    k = len(means)
    p_rest, gap = means.min(), means.max() - means.min()
    eq = stall_equilibrium(k, float(p_rest), float(gap), eta=eta, n_good=n_good)
    ode = late_regret(simulate_expected(means, "dg", steps=STEPS, lr=LR, eta=eta,
                                        baseline_decay=DECAY, normalize=True),
                      means.max())
    return {"balance_regret": eq["regret"], "ode_regret": ode,
            "pi_good": eq["pi_good"], "pi_rest": eq["pi_rest"],
            "baseline": eq["baseline"]}


out = {"protocol": {"steps": STEPS, "late_window": LATE, "batch": BATCH, "lr": LR,
                    "baseline_decay": DECAY, "normalize": True}}

# --- A: canonical trajectories -------------------------------------------------
means = means_two_groups(**CANON)
seeds = range(10)
traj = {}
for method in ("pg", "dg"):
    runs = [run_agent(means, method, steps=STEPS, batch=BATCH, lr=LR, seed=s,
                      baseline_decay=DECAY, normalize=True) for s in seeds]
    traj[method] = {"seeds": [CANON["p"] + CANON["gap"] - r["expected_reward"] for r in runs],
                    "ode": CANON["p"] + CANON["gap"]
                           - simulate_expected(means, method, steps=STEPS, lr=LR,
                                               baseline_decay=DECAY,
                                               normalize=True)["expected_reward"]}
out["canonical"] = {"dg_observed": observed(means, "dg", seeds),
                    "pg_observed": observed(means, "pg", seeds),
                    "dg_predicted": predicted(means)}
print("canonical: observed DG {:.4f}+-{:.4f} | balance {:.4f} | ODE {:.4f} | PG {:.4f}".format(
    out["canonical"]["dg_observed"]["mean"], out["canonical"]["dg_observed"]["sem"],
    out["canonical"]["dg_predicted"]["balance_regret"],
    out["canonical"]["dg_predicted"]["ode_regret"],
    out["canonical"]["pg_observed"]["mean"]))

# --- C: generality grid ---------------------------------------------------------
grid = []
for k in GRID_K:
    for p in GRID_P:
        for gap in GRID_GAP:
            if p + gap > 0.95:
                continue
            m = means_two_groups(k, p, gap)
            row = {"k": k, "p_rest": p, "gap": gap,
                   "dg_observed": observed(m, "dg", range(5)),
                   "pg_observed": observed(m, "pg", range(5)),
                   "dg_predicted": predicted(m)}
            grid.append(row)
            print("grid K={:3d} p={} gap={}: observed {:.4f} | balance {:.4f} | PG {:.4f}".format(
                k, p, gap, row["dg_observed"]["mean"],
                row["dg_predicted"]["balance_regret"], row["pg_observed"]["mean"]))
out["grid"] = grid
err = np.array([g["dg_observed"]["mean"] - g["dg_predicted"]["balance_regret"] for g in grid])
noisy = np.array([(g["gap"] <= 0.1 and g["p_rest"] >= 0.5) for g in grid])
out["grid_fit"] = {
    "noisy_regime": "gap <= 0.1 and p_rest >= 0.5 (reward noise large vs gap)",
    "n_noisy": int(noisy.sum()),
    "max_abs_err_noisy": float(np.abs(err[noisy]).max()),
    "mean_abs_err_noisy": float(np.abs(err[noisy]).mean()),
    "max_abs_err_all": float(np.abs(err).max()),
    "escape_configs": [{"k": g["k"], "p_rest": g["p_rest"], "gap": g["gap"],
                        "observed": g["dg_observed"]["mean"],
                        "predicted": g["dg_predicted"]["balance_regret"]}
                       for g in grid
                       if g["dg_predicted"]["balance_regret"] > 0.05
                       and g["dg_observed"]["mean"] < 0.5 * g["dg_predicted"]["balance_regret"]]}

# --- D: eta scan and number-of-good-arms scan ------------------------------------
eta_scan = []
for eta in ETAS:
    row = {"eta": eta, "dg_observed": observed(means, "dg", seeds, eta=eta),
           "dg_predicted": predicted(means, eta=eta)}
    eta_scan.append(row)
    print("eta={}: observed {:.4f} | balance {:.4f}".format(
        eta, row["dg_observed"]["mean"], row["dg_predicted"]["balance_regret"]))
out["eta_scan"] = eta_scan

ng_scan = []
for n in N_GOODS:
    m = means_two_groups(CANON["k"], CANON["p"], CANON["gap"], n_good=n)
    row = {"n_good": n, "dg_observed": observed(m, "dg", seeds),
           "dg_predicted": predicted(m, n_good=n)}
    ng_scan.append(row)
    print("n_good={}: observed {:.4f} | balance {:.4f}".format(
        n, row["dg_observed"]["mean"], row["dg_predicted"]["balance_regret"]))
out["n_good_scan"] = ng_scan

# --- controls: stall vs convergence lag -----------------------------------------
# At 1500 steps a config can sit above zero either because it is at the predicted
# stall or because it has not converged yet (PG at K=50, or the easy n_good>=5
# settings). Re-run those at 4x the steps: a stall stays, a lag resolves.
control = []
for label, m, method, eta in (
        [("K=50 p=0.5 gap=0.1", means_two_groups(50, 0.5, 0.1), "dg", 1.0),
         ("K=50 p=0.5 gap=0.1", means_two_groups(50, 0.5, 0.1), "pg", 1.0)]
        + [(f"n_good={n}", means_two_groups(CANON["k"], CANON["p"], CANON["gap"], n_good=n),
            "dg", 1.0) for n in N_GOODS]):
    vals = [late_regret(run_agent(m, method, steps=4 * STEPS, batch=BATCH, lr=LR,
                                  eta=eta, seed=s, baseline_decay=DECAY,
                                  normalize=True), m.max())
            for s in range(5)]
    control.append({"config": label, "method": method, "steps": 4 * STEPS,
                    "regret_mean": float(np.mean(vals)),
                    "regret_sem": float(np.std(vals) / np.sqrt(5))})
    print("control {} {} @{} steps: {:.4f}".format(label, method, 4 * STEPS, np.mean(vals)))
out["long_run_controls"] = control

# --- figure ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
t = np.arange(STEPS)
for s in traj["dg"]["seeds"]:
    ax.plot(t, s, color="tab:blue", alpha=0.15, lw=0.8)
ax.plot(t, np.mean(traj["dg"]["seeds"], axis=0), color="tab:blue", lw=1.8,
        label="DG, mean of 10 seeds")
ax.plot(t, np.mean(traj["pg"]["seeds"], axis=0), color="tab:green", lw=1.8,
        label="PG, mean of 10 seeds")
ax.plot(t, traj["dg"]["ode"], color="black", ls="--", lw=1.4, label="DG expected-update ODE")
ax.plot(t, traj["pg"]["ode"], color="tab:green", ls="--", lw=1.0, alpha=0.6,
        label="PG expected-update ODE")
ax.axhline(out["canonical"]["dg_predicted"]["balance_regret"], color="tab:red",
           ls=":", lw=1.4, label="balance-equation plateau")
ax.set_xlabel("step"); ax.set_ylabel("expected regret")
ax.set_title("A. Canonical noisy 20-arm: derivation tracks the runs")
ax.legend(fontsize=8)

ax = axes[0, 1]
eq = stall_equilibrium(CANON["k"], CANON["p"], CANON["gap"])
b = eq["baseline"]
pi = np.logspace(np.log10(1e-3), np.log10(0.999), 400)
ax.plot(pi, visit_coefficient(pi, CANON["p"] + CANON["gap"], b), color="tab:blue",
        label="best arm (p=0.6)")
ax.plot(pi, visit_coefficient(pi, CANON["p"], b), color="tab:orange",
        label="normal arm (p=0.5)")
c_star = float(visit_coefficient(eq["pi_good"], CANON["p"] + CANON["gap"], b))
ax.plot([eq["pi_rest"], eq["pi_good"]], [c_star, c_star], "k--", lw=1.2)
ax.plot([eq["pi_good"]], [c_star], "ko"); ax.plot([eq["pi_rest"]], [c_star], "ko")
ax.annotate(f"stall: pi_best={eq['pi_good']:.2f}", (eq["pi_good"], c_star),
            textcoords="offset points", xytext=(8, 8), fontsize=8)
ax.annotate(f"pi_rest={eq['pi_rest']:.3f}", (eq["pi_rest"], c_star),
            textcoords="offset points", xytext=(8, -14), fontsize=8)
ax.axhline(CANON["p"] + CANON["gap"] - b, color="tab:blue", ls=":", lw=1.0, alpha=0.7)
ax.axhline(CANON["p"] - b, color="tab:orange", ls=":", lw=1.0, alpha=0.7)
ax.text(1.1e-3, CANON["p"] + CANON["gap"] - b, "PG coefficient, best arm", fontsize=7,
        va="bottom", color="tab:blue")
ax.text(1.1e-3, CANON["p"] - b, "PG coefficient, normal arm", fontsize=7,
        va="bottom", color="tab:orange")
ax.set_xscale("log"); ax.set_xlabel("arm probability pi"); ax.set_ylabel("visit coefficient c(pi)")
ax.set_title("B. Mechanism: coefficients balance at an interior point")
ax.legend(fontsize=8)

ax = axes[1, 0]
for gap, marker in zip(GRID_GAP, ("o", "s", "^", "D")):
    xs = [g["dg_predicted"]["balance_regret"] for g in grid if g["gap"] == gap]
    ys = [g["dg_observed"]["mean"] for g in grid if g["gap"] == gap]
    ax.scatter(xs, ys, marker=marker, s=35, alpha=0.8, label=f"gap={gap}")
lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, lim], [0, lim], "k-", lw=0.8, alpha=0.5)
ax.set_xlabel("predicted plateau regret (balance equation)")
ax.set_ylabel("observed regret (5 seeds)")
ax.set_title(f"C. Predicted vs observed across {len(grid)} settings\n"
             f"(noisy regime: max |err| {out['grid_fit']['max_abs_err_noisy']:.3f}; large gaps escape)",
             fontsize=9)
ax.legend(fontsize=8)

ax = axes[1, 1]
es = [r["eta"] for r in eta_scan]
obs = [r["dg_observed"]["mean"] for r in eta_scan]
sem = [r["dg_observed"]["sem"] for r in eta_scan]
pred = [r["dg_predicted"]["balance_regret"] for r in eta_scan]
ode = [r["dg_predicted"]["ode_regret"] for r in eta_scan]
ax.errorbar(es, obs, yerr=sem, marker="o", capsize=3, label="observed (10 seeds)")
ax.plot(es, pred, "s--", label="balance equation")
ax.plot(es, ode, "^:", label="expected-update ODE")
ax.set_xscale("log", base=2); ax.set_xticks(es, [str(e) for e in es])
ax.set_xlabel("eta"); ax.set_ylabel("expected regret")
ax.set_title("D. Eta scan on the canonical setting")
ax.legend(fontsize=8)

fig.suptitle("The noisy-reward stall is the gate's interior balance point "
             "(normalized steps; regret = mean over last 500 of 1500 steps)", fontsize=11)
fig.tight_layout()
fig.savefig("docs/stall_mechanism.png", dpi=110)
json.dump(out, open("docs/results_stall_mechanism.json", "w"), indent=1, default=float)
print("wrote docs/stall_mechanism.png and docs/results_stall_mechanism.json")
