"""Smoke tests: every script runs end-to-end with tiny settings and writes its outputs.

Scripts with CLI args get tiny argv; constant-driven scripts are monkeypatched to a few
seeds/steps. Each runs in a tmp cwd so real docs/ artifacts are never touched.
"""
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_compare(workdir, monkeypatch):
    m = importlib.import_module("scripts.compare")
    monkeypatch.setattr(m, "SEEDS", range(2))
    m.main([])
    assert (workdir / "docs" / "results.json").exists()
    assert (workdir / "docs" / "compare.png").exists()


def test_symmetric_normalized(workdir, monkeypatch):
    m = importlib.import_module("scripts.symmetric_normalized")
    monkeypatch.setattr(m, "SEEDS", range(2))
    m.main()
    assert (workdir / "docs" / "results_symmetric.json").exists()


def test_compare_contextual(workdir, monkeypatch):
    m = importlib.import_module("scripts.compare_contextual")
    monkeypatch.setattr(m, "N_TRAIN", 1000)
    monkeypatch.setattr(m, "N_TEST", 200)
    monkeypatch.setattr(m, "SEEDS", range(2))
    m.main()
    assert (workdir / "docs" / "results_contextual.json").exists()


def test_sweep_eta_baseline(workdir, monkeypatch):
    m = importlib.import_module("scripts.sweep_eta_baseline")
    monkeypatch.setattr(m, "ETAS", (1.0,))
    monkeypatch.setattr(m, "DECAYS", (0.9,))
    monkeypatch.setattr(m, "SEEDS", range(2))
    monkeypatch.setattr(m, "SCENARIOS", {
        k: dict(v, steps=30) for k, v in m.SCENARIOS.items()})
    m.main()
    assert (workdir / "docs" / "results_sweep.json").exists()


def test_compare_continuous(workdir, monkeypatch):
    m = importlib.import_module("scripts.compare_continuous")
    monkeypatch.setattr(m, "SEEDS", range(2))
    m.main()
    assert (workdir / "docs" / "results_continuous.json").exists()


def test_compare_continuous_multidim(workdir, monkeypatch):
    m = importlib.import_module("scripts.compare_continuous_multidim")
    monkeypatch.setattr(m, "SEEDS", range(2))
    m.main()
    assert (workdir / "docs" / "results_continuous_multidim.json").exists()


def test_stall_mechanism(workdir, monkeypatch):
    m = importlib.import_module("scripts.stall_mechanism")
    monkeypatch.setattr(m, "STEPS", 60)
    monkeypatch.setattr(m, "LATE", 20)
    monkeypatch.setattr(m, "GRID_K", (2,))
    monkeypatch.setattr(m, "GRID_GAP", (0.1,))
    monkeypatch.setattr(m, "GRID_P", (0.5,))
    monkeypatch.setattr(m, "ETAS", (1.0,))
    monkeypatch.setattr(m, "N_GOODS", (1,))
    monkeypatch.setattr(m, "CANON_SEEDS", range(2))
    monkeypatch.setattr(m, "CONTROL_SEEDS", range(2))
    monkeypatch.setattr(m, "ESCAPE_STEPS", 100)
    m.main()
    assert (workdir / "docs" / "results_stall_mechanism.json").exists()


def test_make_demo(workdir):
    import os
    out = workdir / "demo.gif"
    env = dict(os.environ, PYTHONPATH=str(REPO))
    subprocess.run([sys.executable, str(REPO / "scripts" / "make_demo.py"),
                    "--steps", "5", "--out", str(out)],
                   check=True, capture_output=True, cwd=workdir, env=env)
    assert out.exists()


def _fake_mnist(n_train=256, n_test=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n_train, 784), dtype=np.float32)
    y = rng.integers(0, 10, size=n_train)
    Xt = rng.random((n_test, 784), dtype=np.float32)
    yt = rng.integers(0, 10, size=n_test)
    return X, y, Xt, yt


def test_compare_mnist(workdir, monkeypatch):
    pytest.importorskip("torch")
    m = importlib.import_module("scripts.compare_mnist")
    out_dir = workdir / "results_mnist"
    monkeypatch.setattr(m, "OUT_DIR", out_dir)
    monkeypatch.setattr(m, "load_mnist", lambda: _fake_mnist())
    monkeypatch.setattr(sys, "argv", ["compare_mnist.py", "--method", "dg", "--seed", "0",
                                      "--episodes", "30", "--eval-every", "15"])
    m.main()
    assert (out_dir / "dg_seed0.json").exists()


def test_ablations_mnist(workdir, monkeypatch):
    pytest.importorskip("torch")
    m = importlib.import_module("scripts.ablations_mnist")
    out_dir = workdir / "results_mnist_ablations"
    monkeypatch.setattr(m, "OUT_DIR", out_dir)
    monkeypatch.setattr(m, "load_mnist", lambda: _fake_mnist())
    monkeypatch.setattr(sys, "argv", ["ablations_mnist.py", "--run", "--seeds", "0",
                                      "--episodes", "30", "--eval-every", "15",
                                      "--max-runs", "2"])
    m.main()
    assert len(list(out_dir.glob("*.json"))) == 2
