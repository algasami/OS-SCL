# Ensemble + back-end — can scoring alone take OS-SCL past the paper?

> **Status: pre-registered, nothing confirmatory run yet.** Everything below
> *Results* was written on 2026-10-12, before any eval-set score under the rules
> tested here and before the seed-2026/2027 checkpoints existed. The only data
> already seen is the dev-set exploration in *What has already been seen*.
> Waits for [`../tfgram-linear`](../tfgram-linear) stage 2 (seed 2027 due
> ~2026-10-13).

## Why this exists

Where this reproduction stands against the paper (arXiv:2509.13853):

```
                                  dev AUC / pAUC / mAUC       eval mean(AUC, pAUC)
paper, TFSTgram (OS-SCL)          95.71 / 90.23 / 91.23       93.22
ours, f1, 4 seeds (seed-sweep)    94.90 / 89.76 / 88.66       93.32
```

The eval set already matches. The dev gap is 0.81 AUC, about six seed
standard deviations (sd 0.14), so rerolling seeds will not close it. Per
machine it is ToyConveyor −2.05, pump −1.70, ToyCar −0.71. The other three are
within 0.2.

Eight ablations have since closed the cheap model-side routes. TFgram trained,
frozen or deleted are indistinguishable at n=4 (`../seed-sweep`). A
decorrelated TFgram is null at n=1 (`../tfgram-linear`). No rescoring of the 41
ArcFace logits beats eval.py's rule R0 (`../scoring-probe`). Two scoring-side
routes are untested:

1. **Embedding back-ends.** The DCASE literature often scores by distance to
   normal training embeddings (kNN, Mahalanobis, GMM) instead of the classifier's
   softmax (Wilkinghoff & Kurth 2024; Zhang et al. 2023, where Mahalanobis is the
   best single score). `../scoring-probe` only re-read the logits.
2. **Score-level ensembles** of independently trained checkpoints. ToyConveyor
   swings >4 points between seeds (`../seed-sweep`). A large share of the error
   is variance, and averaging removes variance.

## What has already been seen

An exploratory dev-only probe on 2026-10-12 (`exploration/`, outputs in
`scores/exploration_dev_*.txt`) used 4 checkpoints: `tfl_relu_max_s2024/s2025`
and `tfl_linear_avg_s2024/s2025`. R0 reproduced the committed CSV
(`relu_max_s2024` dev 94.97 vs 0.94973). **The eval set was deliberately not
scored.**

Single-model rules, mean Δ vs R0 over the 4 checkpoints (dev):

```
rule                    AUC     pAUC    mAUC    AUC wins
R0z (per-ID z of R0)   -0.01   -0.12   +0.00    2/4
kNN, post-head         -0.20   -0.21   +0.32    2/4
kNN, pre-head          -0.33   -0.16   +0.43    0/4
Mahalanobis, post      -0.83   -1.16   -0.01    0/4
Mahalanobis, pre       -1.71   -4.73   -1.29    0/4
z(R0) + z(kNN pre)     +0.44   +0.27   +1.20    4/4    <- ToyConveyor +1.70, ToyCar +1.02
```

Ensembles of z-scored R0, mean over all subsets of each size (dev):

```
n    AUC     pAUC    mAUC         with + z(kNN pre):  AUC     pAUC    mAUC
1    94.90   89.77   89.29                            95.34   90.16   90.49
2    95.54   90.52   90.37                            95.76   90.76   91.14
3    95.80   90.79   90.82                            95.92   91.00   91.39
4    95.93   90.92   91.04                            95.99   91.10   91.48
```

At n=4 the per-machine AUC is fan 98.70, pump 94.96, slider 99.71, ToyCar
96.73, ToyConveyor 85.51, valve 99.95. The ensemble gain does not depend on the
normalisation: a raw sum of R0 gives 95.81 and a sum of log R0 gives 95.84.

**How much this is worth.** The fusion rule was picked from 8 variants after
looking at dev, so its +0.44 is optimistic and should shrink. The ensemble
effect involved no such choice: it rises with n across every subset. The 4
models mix both `tfgram-linear` arms, so the exploration never measured a
homogeneous ensemble of the paper's own model.

**Seen:** dev scores of those 4 checkpoints under the rules above. Single-model
R0 eval CSVs for seeds 2024/2025 also exist, committed by `../tfgram-linear`;
R0 is the baseline here, not a rule under test.
**Not seen:** any eval score under any rule other than R0, and any score at all
from the seed-2026/2027 checkpoints.

## Design

### Checkpoints

The 8 `../tfgram-linear` checkpoints: `relu_max` (the paper's TFSTgram) and
`linear_avg`, seeds 2024–2027. All were trained on smartlab1 (torch 2.6,
`--num_workers 8`, 300 epochs). EMA weights (`model.pth`) as in `eval.py`.

### Confirmation sets

| set | checkpoints | test clips | role |
|-----|-------------|------------|------|
| **C-eval** | all 8 | eval | **primary**: no rule here has been scored on it |
| C-dev-fresh | seeds 2026, 2027 (4) | dev | secondary: new models, though the dev clips were seen via other models |
| C-dev-all | all 8 | dev | reported, not decision-bearing on its own (half of it is the exploration) |

### Scores

For checkpoint *m*, test clip *x* and its known machine ID *i*:

- **R0(x)** is eval.py's score: CE of the margin ArcFace logits against *i*, over all 41 classes.
- **K(x)** = 1 − max cos(e(x), e(r)) over r ∈ Ref_i. Here e is the L2-normalised
  **pre-head** MobileFaceNet embedding (128-d), and Ref_i is ID *i*'s clips in
  the **training part** of that seed's split (the 95% the model trained on).
  k = 1.
- **Normalisation statistics** are the mean and sd of each score over ID *i*'s
  clips in that seed's **held-out validation part** (the 5% never trained on).
  For K these are the held-out clips' distances to Ref_i, which excludes them by
  construction. That is ~44 clips per ID (1,815 held out over 41 IDs, varying
with the random split), enough for a mean and sd. The statistics are fixed
before any test clip is scored:
  zR0 = (R0 − μ) / σ and likewise zK.

This differs from the exploration, which took its statistics from training
clips and its kNN statistics leave-one-out. Training clips are fit
optimistically: their median R0 is 0.20–0.21 against 0.21–0.25 for unseen
normal test clips (`scores/exploration_dev_normalisation.txt`). The held-out
version is the principled one, and the exploration's choice is kept as a
sensitivity check (P6).

### Rules

| rule | definition | kind |
|------|------------|------|
| S0 | R0 | single model, **baseline** |
| S1 | zR0 + zK | single model, fusion |
| E0 | Σ_m R0 | ensemble, raw (sensitivity) |
| E1 | Σ_m zR0 | ensemble |
| E2 | Σ_m (zR0 + zK) | ensemble + fusion |
| S1′, E1′ | as S1, E1 with training-part statistics | sensitivity (P6) |

Ensemble compositions:

- **`relu4`**: the four `relu_max` seeds, i.e. the paper's architecture, a
  homogeneous ensemble. **E1[relu4] is the primary configuration, fixed now.**
- `lin4`: the four `linear_avg` seeds.
- `mix8`: all eight.
- Size curve: E1 over every subset of the 8, n = 1…8.

### Metrics

`eval.py`'s metrics. Per machine: pooled AUC, pAUC (max FPR 0.1), and mAUC
(minimum per-ID AUC), each averaged over the six machines. Also the eval-set
mean of AUC and pAUC, the single number the paper reports for eval.

Uncertainty comes from a stratified bootstrap over test clips: resample within
machine × ID × normal/anomalous, 2000 replicates, and apply the same resample
to both sides of any paired difference. This captures test-set sampling only,
not training-seed variance. Seed variance is what the "beats every member" and
"k of 8 checkpoints" conditions are for.

## Predictions

Written before any confirmatory score.

**P0. Reproduction.** Batched eval-mode R0 matches every committed per-ID AUC
within 1e-4. The replayed validation split reproduces the checkpoint's own
best `Valid_loss` in `curve.csv` to 1e-4 (see *Verification*).

**P1. The ensemble beats a single model.** E1[relu4] beats the best of its own
four members (S0) in overall AUC, on dev **and** eval. Expected gain over the
members' mean: +0.6 to +1.2 dev, +0.4 to +1.0 eval. Eval has less room because
its single models already score higher. In the size curve, mean AUC is
non-decreasing in n on both sets. Confidence: high.

**P2. The ensemble beats the paper.** E1[relu4] reaches dev AUC ≥ 95.71 and pAUC
≥ 90.23 (moderate confidence: the exploration's 95.93 was a mixed ensemble), and
eval mean(AUC, pAUC) ≥ 93.22 (high confidence: one model is already at 93.3).
Dev mAUC ≥ 91.23 is a coin flip.

**P3. Pump is not fixed.** E1[relu4] dev pump AUC stays below the paper's 96.25
(exploration: 94.96). If the ensemble beats the paper overall, it does so
through ToyConveyor, fan and ToyCar, not pump. Confidence: high.

**P4. Fusion helps a single model, but less than it did on dev.** S1 − S0
overall AUC is > 0 for ≥ 7 of 8 checkpoints on C-eval and 4 of 4 on C-dev-fresh.
Mean +0.1 to +0.4 AUC, against the exploration's +0.44, because selection
inflates it. mAUC mean +0.5 to +1.5. Confidence: moderate.

**P5. Fusion adds little on top of an ensemble.** E2[relu4] − E1[relu4] lies
in [−0.1, +0.3] AUC on both sets.

**P6. Where the statistics come from does not matter.** |S1 − S1′| and
|E1 − E1′| ≤ 0.15 overall AUC, and |E1 − E0| ≤ 0.3.

**P7. Mixing arms: no prediction of direction.** For E1, |mix 2+2 − relu4| <
0.3 AUC averaged over the 2+2 subsets. E1[mix8] ≥ E1[relu4] is expected only
because it has twice the members.

## Decision rules

Fixed now. Eval is the test set, so a configuration is never chosen for scoring
best on it.

- **D1, adopt the ensemble.** E1[relu4] becomes this repo's reported *ensemble*
  system iff it beats every member's S0 overall AUC on eval **and** on dev.
- **D2, claim against the paper.** Allowed per metric only where E1[relu4]'s
  point estimate exceeds the paper's number. It is always worded as an
  ensemble: "4 × OS-SCL, 4 × 1.38 M parameters". It is reported with the
  bootstrap 95% CI, and "clearly exceeds" is used only where the lower bound
  clears the paper. If `mix8` or E2 scores higher on eval, that is reported, but
  it does **not** replace E1[relu4] as the headline.
- **D3, adopt the fusion as the single-model rule.** S1 replaces R0 iff P4's
  sign condition holds (≥ 7/8 on C-eval and 4/4 on C-dev-fresh) **and** the
  mean Δ AUC ≥ +0.2 on both. Otherwise R0 stays, as in `../scoring-probe`.
- Single-model mAUC deltas under ~2 points are not read (`CLAUDE.md`).

## What this cannot answer

- **It is not a single-model improvement**, except through S1. An ensemble costs
  4× training and 4× inference. That is legitimate (several top DCASE 2020
  systems were ensembles), but it is not the comparison the paper's table makes.
  The single-model route this opens, if D1 holds, is distilling the ensemble
  into one network (a separate ablation).
- **It does not touch pump**, if P3 holds. That points to model-side work
  aimed at multi-modal normal data, e.g. sub-cluster ArcFace
  (`ArcMarginProduct(sub=k)` already exists and has never been used).
- The paper's numbers come from other hardware, a different torch version and
  an unknown seed. "Beats the paper" means beats its published numbers, not
  beats a paired rerun.

## Verification (planned, before scoring)

1. **Split replay.** `random_seed(seed)` → `train_dataset` → `dataset_split`,
   the order `train.py` uses and `../tfgram-gradient/probe.py` already replays.
   Check: the replayed validation part has 1,815 of the 36,283 training clips.
   Recomputing the EMA's validation CE on it, with
   `trainer.valid()`'s batching (64, no shuffle, `train=True`), reproduces the
   best-epoch `Valid_loss` in `curve.csv`. A match proves both the split and the
   checkpoint.
2. **R0 reproduction**, as P0, for every checkpoint and both sets.
3. **Clip alignment.** Every per-run array is in the same clip order, which is
   asserted by path before any ensemble sum.

## Planned files

```
extract.py      GPU, once per checkpoint: R0, 41 cosines, pre-head embedding for
                train / held-out valid / dev test / eval test clips
score.py        CPU, float64: S0-S1, E0-E2, size curve, bootstrap, reproduction
run.sh          end to end, resumable; PROBE_REPO points at the checkout with data/ and check_points/
make_plots.py   reads scores/ only
scores/         rule_metrics.csv, ensemble_curve.csv, bootstrap.json, reproduction.json,
                exploration_dev_*.txt (the pre-registration's own provenance)
exploration/    the 2026-10-12 dev-only scripts, kept as run (paths are scratch-relative)
results/        per-clip arrays, gitignored
```

## Planned figures

| figure | will show |
|--------|-----------|
| `plots/fig1_ensemble_size.png` | AUC vs ensemble size n (every subset, mean and spread), dev and eval, with the paper's line |
| `plots/fig2_per_machine.png` | E1[relu4] vs mean single model vs paper, per machine: where the gain comes from, and whether pump moves |
| `plots/fig3_fusion_delta.png` | S1 − S0 per checkpoint, dev-fresh and eval, against the D3 threshold |
| `plots/fig4_vs_paper.png` | bootstrap 95% CIs of E1[relu4] against the paper's three dev numbers and its eval number |

## Results

*Not run.*
