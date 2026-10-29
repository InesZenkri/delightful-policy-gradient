"""Appendix B ablations for the MNIST one-step contextual bandit (arXiv:2603.14608).

Extends scripts/compare_mnist.py along the three axes the paper varies in its
Appendix B, around the base config (Adam lr 1e-3, width 100, B=100, 10k episodes,
eta=1, constant baseline b=0.5, 1 action sample per image, no entropy term):

  1. baselines:        "zero" (b=0) and "batch_mean" against the default "constant"
  2. action samples:   k = 4 and 16 samples per image against the default k = 1
  3. entropy:          entropy regularization coef 0.01 and 0.1 against the default 0

Only one axis is varied at a time; the (constant, k=1, no-entropy) center is shared.
5 seeds per config (the paper uses 100; CPU budget), so 70 runs of ~10 s each.

    PYTHONPATH=. python scripts/ablations_mnist.py --run          # all configs/seeds
    PYTHONPATH=. python scripts/ablations_mnist.py --plot         # aggregate + figure

Per-run files go to docs/results_mnist_ablations/<name>_seed<seed>.json (resumable:
existing files are skipped), --plot merges them into docs/results_mnist_ablations.json
(summary) and docs/mnist_ablations.svg.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dgplay.mnist import load_mnist, train

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "results_mnist_ablations"
METHODS = ["pg", "dg"]

# (config name, baseline, n_samples, entropy_coef); the center (constant, 1, 0.0)
# is shared by all three axes.
AXES = {
    "baseline": [("constant", 1, 0.0), ("zero", 1, 0.0), ("batch_mean", 1, 0.0)],
    "samples": [("constant", 1, 0.0), ("constant", 4, 0.0), ("constant", 16, 0.0)],
    "entropy": [("constant", 1, 0.0), ("constant", 1, 0.01), ("constant", 1, 0.1)],
}
AXIS_LABELS = {
    "baseline": ["b=0.5 (paper)", "b=0", "b=batch mean"],
    "samples": ["1 sample (paper)", "4 samples", "16 samples"],
    "entropy": ["no entropy (paper)", "entropy 0.01", "entropy 0.1"],
}


def configs() -> list[tuple[str, str, str, int, float]]:
    """(name, method, baseline, n_samples, entropy_coef), deduplicated across axes."""
    seen, out = set(), []
    for axis, variants in AXES.items():
        for (baseline, k, ent), label in zip(variants, AXIS_LABELS[axis]):
            for m in METHODS:
                key = (m, baseline, k, ent)
                if key in seen:
                    continue  # shared center config lives under the baseline axis
                seen.add(key)
                out.append((f"{m}_{axis}_{baseline}_{k}_{ent}", m, baseline, k, ent))
    return out


def run_all(seeds: list[int], episodes: int, eval_every: int, max_runs: int | None = None):
    data = load_mnist()
    todo = [(n, m, b, k, e, s) for (n, m, b, k, e) in configs() for s in seeds
            if not (OUT_DIR / f"{n}_seed{s}.json").exists()]
    if max_runs:
        todo = todo[:max_runs]
    print(f"{len(todo)} runs this call", flush=True)
    for name, m, b, k, e, s in todo:
        r = train(m, *data, episodes=episodes, eval_every=eval_every, seed=s,
                  baseline=b, n_samples=k, entropy_coef=e)
        p = OUT_DIR / f"{name}_seed{s}.json"
        p.write_text(json.dumps({k2: (v.tolist() if isinstance(v, np.ndarray) else v)
                                 for k2, v in r.items()}))
        print(f"wrote {p}  final test err {r['final_test_err']*100:.2f}%", flush=True)


def aggregate(seeds: list[int]) -> dict:
    summary = {"episodes": None, "axes": {}}
    for axis, variants in AXES.items():
        summary["axes"][axis] = {"labels": AXIS_LABELS[axis], "variants": []}
        for (baseline, k, ent), label in zip(variants, AXIS_LABELS[axis]):
            for m in METHODS:
                name = f"{m}_{axis}_{baseline}_{k}_{ent}"
                run_name = name if (baseline, k, ent) != ("constant", 1, 0.0) else f"{m}_baseline_constant_1_0.0"
                runs = []
                for s in seeds:
                    p = OUT_DIR / f"{run_name}_seed{s}.json"
                    if p.exists():
                        runs.append(json.loads(p.read_text()))
                if not runs:
                    continue
                ep = np.array(runs[0]["episode"])
                summary["episodes"] = ep.tolist()
                errs = np.array([r["test_err"] for r in runs])
                ents = np.array([r["eval_entropy"] for r in runs]) if "eval_entropy" in runs[0] else None
                summary["axes"][axis]["variants"].append({
                    "name": name, "method": m, "label": label,
                    "baseline": baseline, "n_samples": k, "entropy_coef": ent,
                    "n_seeds": len(runs),
                    "test_err_mean": np.round(errs.mean(0), 4).tolist(),
                    "test_err_se": (np.round(errs.std(0, ddof=1) / np.sqrt(len(runs)), 4)).tolist() if len(runs) > 1 else [0.0] * len(ep),
                    "final_test_err_mean": round(float(errs[:, -1].mean()), 4),
                    "final_test_err_se": round(float(errs[:, -1].std(ddof=1) / np.sqrt(len(runs))), 4) if len(runs) > 1 else 0.0,
                    "final_test_err_all": np.round(errs[:, -1], 4).tolist(),
                })
                if ents is not None:
                    summary["axes"][axis]["variants"][-1]["eval_entropy_mean"] = np.round(ents.mean(0), 4).tolist()
    if summary["episodes"] is None:
        raise SystemExit(f"no per-run files in {OUT_DIR}")
    return summary


def make_plot(summary: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = np.array(summary["episodes"])
    colors = {"pg": "tab:blue", "dg": "tab:red"}
    styles = ["-", "--", ":"]
    titles = {"baseline": "baseline ablation", "samples": "action samples per image",
              "entropy": "entropy regularization"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (axis, block) in zip(axes, summary["axes"].items()):
        for i, v in enumerate(block["variants"]):
            mean = np.array(v["test_err_mean"]) * 100
            se = np.array(v["test_err_se"]) * 100
            lab = f"{v['method'].upper()}, {v['label']}"
            ax.plot(ep, mean, label=lab, color=colors[v["method"]], linestyle=styles[i % 3])
            ax.fill_between(ep, mean - se, mean + se, color=colors[v["method"]],
                            alpha=0.15, linewidth=0)
        ax.set_xlabel("episode (batch of 100)")
        ax.set_title(titles[axis])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("test error (%)")
    fig.suptitle("MNIST bandit ablations (paper Appendix B; mean +/- 1 SE over 5 seeds)")
    fig.tight_layout()
    fig.savefig(path)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--episodes", type=int, default=10_000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--max-runs", type=int, default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.plot:
        summary = aggregate(args.seeds)
        out = Path(__file__).resolve().parent.parent / "docs" / "results_mnist_ablations.json"
        out.write_text(json.dumps(summary, indent=2))
        print(f"wrote {out}")
        make_plot(summary, out.parent / "mnist_ablations.svg")
        for axis, block in summary["axes"].items():
            print(f"[{axis}]")
            for v in block["variants"]:
                print(f"  {v['method']:3s} {v['label']:18s} final test err "
                      f"{v['final_test_err_mean']*100:.2f}% +/- {v['final_test_err_se']*100:.2f}")
        return
    if not args.run:
        ap.error("--run or --plot required")
    run_all(args.seeds, args.episodes, args.eval_every, args.max_runs)


if __name__ == "__main__":
    main()
