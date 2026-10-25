"""PG vs DG vs supervised cross-entropy on MNIST as a one-step contextual bandit.

Setup follows the paper's Appendix B (arXiv:2603.14608): two-layer ReLU MLP, hidden
width 100, Adam lr 1e-3, batches of B=100 images, 10,000 episodes, advantage U = R - b
with a constant baseline b = 0.5. The paper averages over 100 seeds; this script's
defaults use 5 (CPU budget). One (method, seed) per invocation so runs can be chunked:

    PYTHONPATH=. python scripts/compare_mnist.py --method dg --seed 0
    PYTHONPATH=. python scripts/compare_mnist.py --plot      # aggregate + figures

Writes docs/results_mnist/<method>_seed<seed>.json per run, then --plot merges them into
docs/results_mnist.json (summary) and docs/mnist_curve.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dgplay.mnist import load_mnist, train

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "results_mnist"
METHODS = ["ce", "pg", "dg"]


def run_one(method: str, seed: int, episodes: int, eval_every: int) -> dict:
    data = load_mnist()
    r = train(method, *data, episodes=episodes, eval_every=eval_every, seed=seed)
    return r


def aggregate(seeds: list[int]) -> dict:
    runs = {m: [] for m in METHODS}
    for m in METHODS:
        for s in seeds:
            p = OUT_DIR / f"{m}_seed{s}.json"
            if p.exists():
                runs[m].append(json.loads(p.read_text()))
    if not any(runs.values()):
        raise SystemExit(f"no per-run files in {OUT_DIR}")
    ep = np.array(runs[[m for m in METHODS if runs[m]][0]][0]["episode"])
    summary = {"episodes": ep.tolist(), "methods": {}}
    for m in METHODS:
        if not runs[m]:
            continue
        errs = np.array([r["test_err"] for r in runs[m]])  # seeds x checkpoints
        summary["methods"][m] = {
            "n_seeds": len(runs[m]),
            "test_err_mean": errs.mean(0).tolist(),
            "test_err_se": (errs.std(0, ddof=1) / np.sqrt(len(runs[m]))).tolist() if len(runs[m]) > 1 else [0.0] * len(ep),
            "final_test_err_mean": float(errs[:, -1].mean()),
            "final_test_err_se": float(errs[:, -1].std(ddof=1) / np.sqrt(len(runs[m]))) if len(runs[m]) > 1 else 0.0,
            "final_test_err_all": errs[:, -1].tolist(),
        }
    return summary


def make_plot(summary: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"ce": "CE (supervised, sees labels)", "pg": "PG (REINFORCE)", "dg": "DG (eta=1)"}
    colors = {"ce": "gray", "pg": "tab:blue", "dg": "tab:red"}
    ep = np.array(summary["episodes"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in METHODS:
        d = summary["methods"].get(m)
        if d is None:
            continue
        mean = np.array(d["test_err_mean"]) * 100
        se = np.array(d["test_err_se"]) * 100
        ax.plot(ep, mean, label=labels[m], color=colors[m])
        ax.fill_between(ep, mean - se, mean + se, color=colors[m], alpha=0.2, linewidth=0)
    ax.set_xlabel("episode (batch of 100)")
    ax.set_ylabel("test error (%)")
    ax.set_title("MNIST as a one-step contextual bandit (mean +/- 1 SE over seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--episodes", type=int, default=10_000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.plot:
        summary = aggregate(args.seeds)
        out = Path(__file__).resolve().parent.parent / "docs" / "results_mnist.json"
        out.write_text(json.dumps(summary, indent=2))
        print(f"wrote {out}")
        make_plot(summary, out.parent / "mnist_curve.png")
        for m, d in summary["methods"].items():
            print(f"{m:3s} final test err {d['final_test_err_mean']*100:.2f}% "
                  f"+/- {d['final_test_err_se']*100:.2f} (seeds={d['n_seeds']})")
        return
    if args.method is None:
        ap.error("--method is required unless --plot")
    r = run_one(args.method, args.seed, args.episodes, args.eval_every)
    p = OUT_DIR / f"{args.method}_seed{args.seed}.json"
    p.write_text(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}))
    print(f"wrote {p}  final test err {r['final_test_err']*100:.2f}%")


if __name__ == "__main__":
    main()
