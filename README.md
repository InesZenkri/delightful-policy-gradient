# Delightful Policy Gradient: a small bandit playground

A small NumPy playground that compares plain policy gradient (REINFORCE) with the **Delightful Policy Gradient (DG)** from:

> Ian Osband (Google DeepMind), *Delightful Policy Gradient*, arXiv:2603.14608 (March 2026), accepted at the Reinforcement Learning Journal / RLC 2026.
> https://arxiv.org/abs/2603.14608

This is an independent, small-scale implementation written from the paper's Algorithm 1. **It is not a reproduction of the paper's experiments** and is not affiliated with the author or Google. The paper's results use much larger setups (MNIST, transformer sequence tasks, 28 DeepMind Control Suite environments at 10M steps), none of which are run here.

## The method in one paragraph

For each sampled action `A` with advantage `U` and policy probability `pi(A)`:

```
surprisal  l   = -log pi(A)
delight    chi = U * l
gate       w   = sigmoid(chi / eta)        (eta = 1)
update    += w * U * grad log pi(A)
```

A rare action with positive advantage (a "breakthrough") keeps almost its full weight; a rare action with negative advantage (a "blunder") is damped. Actions the policy already favors get a gate near 0.5. See `dgplay/bandit.py`.

## What is in the repo

| Path | What it does |
| --- | --- |
| `dgplay/bandit.py` | Softmax-policy K-armed bandit, PG and DG updates, optional unit-norm ("normalized") steps, optional delight whitening |
| `dgplay/contextual.py` | Synthetic 10-class contextual bandit with a linear softmax policy (reward = correct class) |
| `dgplay/continuous.py` | 1-D continuous-action bandit with a Gaussian policy (density-based surprisal, clipped log-density) |
| `scripts/make_demo.py` | Side-by-side animated demo (`docs/demo.gif`) |
| `scripts/compare.py` | Multi-seed PG vs DG over three bandit settings, optionally with normalized steps |
| `scripts/symmetric_normalized.py` | One-correct-arm bandit with normalized steps (my reading of the paper's symmetric-bandit setup) |
| `scripts/compare_contextual.py` | PG vs DG vs supervised cross-entropy on the synthetic contextual bandit |
| `scripts/sweep_eta_baseline.py` | Sweep of DG's temperature eta and the EMA baseline strength in two bandit settings |
| `scripts/compare_continuous.py` | PG vs DG with clipped, whitened, and retuned delight on the continuous bandit at three reward scales |
| `tests/` | Finite-difference gradient checks, gate formulas, and behavior tests |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
PYTHONPATH=. python scripts/make_demo.py            # writes docs/demo.gif
PYTHONPATH=. python scripts/symmetric_normalized.py
PYTHONPATH=. python scripts/compare.py              # about 40 s on a CPU
PYTHONPATH=. python scripts/compare.py --normalize
PYTHONPATH=. python scripts/compare_contextual.py
PYTHONPATH=. python scripts/sweep_eta_baseline.py     # about 30 s on a CPU
PYTHONPATH=. python scripts/compare_continuous.py     # about 20 s on a CPU
```

Everything runs on CPU with NumPy and Matplotlib.

## Results from this prototype

All numbers come from `docs/results*.json`, produced by the scripts above. Seeds: 30 for the original bandit runs, 10 for the contextual task and the sweep, 20 for the continuous runs. One configuration per setting, no tuning beyond the learning rates shown.

| Setting | PG | DG (eta=1) | Takeaway |
| --- | --- | --- | --- |
| Symmetric bandit, normalized steps | error < 0.1 at step 101 | step 74 | DG faster at equal step size (matches paper's direction) |
| Noisy Bernoulli bandits, un-normalized | faster in all 3 settings | slower | step-size effect: the gate (<= 1) shrinks DG's steps |
| Noisy 20-arm, normalized, eta=1 | final regret ~0.001 | stalls at ~0.084 | gate's direction bias; see sweep below |
| Synthetic contextual bandit | ~0.67 accuracy | ~0.67 (+0.002 to 0.007 over PG) | roughly tied, DG slightly ahead |
| Eta/baseline sweep (normalized) | solves both scenarios | symmetric: robust; noisy: stall unless eta >= 4 | stall is in the gate, not the baseline |
| Continuous 1-D bandit | identical at any reward scale | changes with reward scale; whitening fixes that | whitening makes eta unit-free, not faster |

![Demo](docs/demo_final.png)

**1. Symmetric bandit with normalized steps (K=100, one correct arm, B=100, alpha=0.1, fixed baseline 0.5).** DG gets to error below 0.1 at step 74 versus step 101 for PG (mean over 30 seeds). This matches the direction of the paper's claim that DG converges faster at equal step size. ![symmetric](docs/symmetric_normalized.png)

**2. Noisy Bernoulli bandits, un-normalized steps.** DG was slower than PG in all three settings and all learning rates I tried (`docs/compare.png`). Part of this is that the gate is near 0.5 for most samples, so DG takes smaller steps. That is a step-size effect, not evidence about direction.

**3. Noisy Bernoulli bandits, normalized steps (`docs/compare_normalized.png`).** On the 3-arm and sparse-reward 20-arm settings DG and PG were about equal (DG slightly slower on the 3-arm case). On a 20-arm setting with equal rewards of 0.5 and one arm at 0.6, DG stalled at an expected regret of about 0.08 while PG kept improving. A unit test (`tests/test_bandit.py::test_dg_has_upward_bias_on_rare_arm_with_noisy_equal_arms`) computes the exact expected update for one specific case: equal arms with Bernoulli(0.5) rewards, baseline 0.5, one rare arm. There PG's expected update is zero while DG's pushes probability toward the rare arm. That is a derivation for that case only; I have not shown it explains the stall in the 20-arm run or that it applies generally. The paper analyses deterministic-reward bandits and lists sparse-reward and other settings as open questions.

**4. Synthetic contextual bandit (`docs/contextual.png`).** PG, DG and supervised cross-entropy all end near the same accuracy (about 0.67 at the best learning rate). DG was slightly higher than PG at every learning rate (by 0.002 to 0.007, standard errors about 0.001 to 0.002), and cross-entropy degraded at high learning rates. This is not a replication of the paper's MNIST result.

**5. Sweep of eta and the baseline (`docs/sweep_eta_baseline.png`, 10 seeds, normalized steps).** On the symmetric bandit, DG is insensitive to both: mean error over the run stays between 0.163 and 0.165 for eta in [0.25, 4] and EMA decay in {0, 0.5, 0.9, 0.99}, the same range as PG. On the noisy 20-arm setting the picture is different. The stall from result 3 is not a baseline problem: DG at eta=1 stalls at a final regret of about 0.082 to 0.087 at every baseline strength, while PG solves the task at all of them (final regret 0.000 to 0.005). Raising eta to 4, which flattens the gate toward a constant, removes the stall (final regret 0.003 at decay 0.99); lowering eta to 0.25 only softens it (about 0.06). In other words, the stall disappears exactly when the gate stops gating, which supports the direction-bias explanation from result 3 rather than a baseline or tuning issue. Whitening the delight (below) does not fix the stall either (whitened DG, eta=1: 0.081 to 0.086).

**6. Whitening delight on a continuous-action bandit (`docs/continuous_whitening.png`, 20 seeds).** The paper suggests whitening the batch of delights, chi_tilde = (chi - mean) / (std + eps), before gating, because density-based surprisal makes chi sensitive to scaling (its Algorithm 2 clips the log-density to [-10, 10]; both are implemented in `dgplay/continuous.py`). On a smooth 1-D bandit with a Gaussian policy, scaling the rewards by 0.01 or 100 changes how unwhitened DG at eta=1 behaves: at scale 0.01 the gate collapses to about 0.5 and DG becomes a half-magnitude PG (mean return over the run 0.448 versus 0.466 at scale 1); at scale 100 the gate saturates to a hard 0/1 filter, which happened to be faster here (0.552). Whitened DG follows the identical curve at every scale (0.478), and retuning eta to the reward scale also restores the scale-1 behavior (0.466). So whitening does what the paper says: it makes eta unit-free. Two honest caveats: PG is exactly scale-invariant here by construction (its update is linear in the advantage and the learning rate was divided by the scale), and in this task every DG variant is slower than PG (0.721), the same step-size effect as result 2 - this testbed uses un-normalized steps. Whitening bought consistency, not speed.

## Limitations

- Linear and tabular policies only; no neural networks, no MNIST, no continuous control.
- The normalized-step bandit follows my reading of the paper's Section 4, not its exact protocol.
- Small number of settings and seeds. Treat the comparisons as a study of behavior, not benchmarks.

## Possible extensions

- Add a small neural policy on MNIST (needs PyTorch).
- Investigate the noisy-reward stall mechanism further: the sweep (result 5) points at the gate's direction bias, but a general characterization is missing.
- Try whitening in a setting where action scales vary across dimensions, the case the paper motivates it with; the 1-D testbed here only varies reward scale.

## License

MIT. See `LICENSE`.
