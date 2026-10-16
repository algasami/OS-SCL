# Ensemble + back-end — can scoring alone take OS-SCL past the paper?

> **Superseded in part by `../confirm-seeds` (2026-10-16, 4 fresh seeds).** S1′ was
> confirmed there and is now the single-model rule; E2′ = Σ S1′ replaced E1 as the
> reported ensemble. E1[relu4]'s dev AUC claim below did **not** replicate on
> fresh seeds (E1[N4] 95.64 vs the paper's 95.71). Numbers here are left as
> written, with inline **[confirm-seeds]** notes.

> **Status: done (2026-10-13), 8 checkpoints, dev and eval.** **D1 holds:** a
> sum of the 4 paper-model seeds (E1[relu4]) beats every member on both sets
> (+0.96 dev, +0.54 eval AUC over the mean member). **D2:** it exceeds the
> paper on 3 of 4 numbers by point estimate (dev AUC 95.79, dev pAUC 90.57,
> eval 93.70; not dev mAUC 90.47), and every 95% interval still includes the
> paper's value. **D3 fails:** the pre-registered fusion S1 is positive on only
> 4/8 eval checkpoints. Its sensitivity variant S1′ (training-clip statistics,
> the exploration's own choice) is positive in 16/16 cells. That is the
> follow-up, not a result here. Everything above *Results* was written on
> 2026-10-12, before any confirmatory score, and is left unedited apart from
> the dated addendum.

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

**Addendum, 2026-10-13, before any confirmatory score.** `../tfgram-linear`
stage 2 has since committed single-model R0 (S0) CSVs for seeds 2026/2027, so
the baseline is now known. On this host, `relu_max` at n=4 scores dev
94.82 ± 0.23 and eval 95.47 ± 0.30. Per machine on dev against the paper:
ToyConveyor −3.03, pump −1.53, ToyCar −0.67, the rest within 0.1. Clearing the
paper's dev AUC therefore needs E1[relu4] to add ≥ +0.89 over the mean member,
against the exploration's +1.0 at n=4, so P2's dev half is closer to a coin flip
than "moderate". The predictions above are left as committed. No rule under test
(S1, E0–E2) has been scored on any new checkpoint or on eval.

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

## Figures

Regenerate with `/home/f74134118/anaconda3/bin/python ablation/ensemble-backend/make_plots.py`
(matplotlib lives in `base`). Reads `scores/` only. The headline ensemble is
blue, the training-statistics variant orange; single models and the paper
are references, drawn achromatic.

| figure | shows |
|--------|-----------|
| `plots/fig1_ensemble_size.png` | score vs ensemble size n over every subset of the 8 checkpoints: the mean rises at every step; E1[relu4] sits above the paper's dev AUC and eval number |
| `plots/fig2_per_machine.png` | per machine, mean single model → E1[relu4], with the paper on dev: ToyConveyor carries the gain, pump stays 1.3 short, eval valve drops through calibration |
| `plots/fig3_fusion_delta.png` | fusion − R0 per checkpoint for S1 (held-out statistics) and S1′ (training statistics): 4/8 vs 8/8 positive on eval |
| `plots/fig4_vs_paper.png` | E1[relu4] with bootstrap 95% intervals against the paper's three dev numbers and its eval number |

## Results

8 checkpoints (`relu_max`, `linear_avg` × seeds 2024–2027), dev and eval. One
pass over every clip on one A4000: extraction 12 min, scoring and bootstrap
4 min. Full scorecard in `scores/scorecard.txt`.

### Verification

- **Split replay: holds.** For all 8 checkpoints, the EMA's validation CE on
  the replayed 5% split (1,815 clips, `trainer.valid()`'s batching)
  reproduces the best-epoch `Valid_loss` in `curve.csv` to ≤ 1.2×10⁻⁵. So
  the held-out clips are exactly the ones each model never trained on, and
  the weights are the ones that were scored.
- **P0, score reproduction: misses its tolerance, harmlessly.** Per-ID AUC
  matches the committed CSVs to ≤ 3.6×10⁻⁴ (`relu_max_s2026` dev). 4 of 16
  cells exceed the 1×10⁻⁴ I wrote. The cause is float32: batched inference
  uses different kernels from eval.py's batch of 1, which reorders a few
  near-tied clip pairs. `../scoring-probe`'s own batch-1 rerun differs from
  the same CSVs by up to 1.8×10⁻⁴. No overall AUC moves at the second
  decimal.

### Headline: E1[relu4]

```
                          AUC     pAUC    mAUC    mean(AUC,pAUC)
dev   single (mean of 4)  94.82   89.74   89.03   92.28
      E1[relu4]           95.79   90.57   90.47   93.18
      paper               95.71   90.23   91.23   92.97
eval  single (mean of 4)  95.46   90.61   91.32   93.04
      E1[relu4]           96.01   91.39   92.55   93.70
      paper                 —       —       —     93.22
```

The members' overall AUC is 94.97 / 94.59 / 95.06 / 94.66 on dev and
95.20 / 95.53 / 95.27 / 95.86 on eval. Note that a **single** `relu_max`
model on this host is *below* the paper's eval number (93.04 vs 93.22). The
seed-sweep's 93.32 was measured on the other host.

Bootstrap 95% intervals (test-clip sampling only):

```
dev AUC    95.79 [95.42, 96.12]   paper 95.71   65% of draws above
dev pAUC   90.57 [89.90, 91.19]   paper 90.23   84%
dev mAUC   90.47 [89.27, 91.40]   paper 91.23    6%
eval mean  93.70 [93.19, 94.23]   paper 93.22   96%
best member − E1, AUC:  dev −0.73 [−0.93, −0.54]   eval −0.14 [−0.40, +0.11]
```

### Against the predictions

- **P1: holds.** E1[relu4] beats the best member on both sets: dev 95.79 vs
  95.06, eval 96.01 vs 95.86. The gain over the mean member is +0.96 dev
  (predicted +0.6 to +1.2) and +0.54 eval (predicted +0.4 to +1.0). The size
  curve rises at every n on both sets: dev 94.62 → 95.33 → 95.61 → 95.76 →
  95.86 → 95.93 → 95.98 → 96.02. On eval the margin over the *best* member
  is inside test-set noise (CI −0.40 to +0.11). On dev it is not.
- **P2: dev AUC, dev pAUC and eval hold; dev mAUC does not** (90.47 vs 91.23,
  called a coin flip). No interval excludes the paper's value. The eval lower
  bound misses by 0.03.
- **P3: holds.** Dev pump 94.96 vs the paper's 96.25. Per machine on dev
  against the paper: ToyConveyor +1.45, fan +0.42, valve +0.14, slider −0.02,
  ToyCar −0.24, pump −1.29. Against the mean single model, ToyConveyor gains
  +4.48 on dev and +2.68 on eval. Most of the ensemble's gain is removing
  ToyConveyor's seed variance (`../seed-sweep`).
- **P4: fails.** S1 − S0 is positive on 4/8 eval checkpoints (needed ≥ 7) and
  3/4 fresh dev checkpoints (needed 4/4). The means are +0.13 eval and +0.20
  dev-fresh, inside the predicted +0.1 to +0.4. mAUC improves by +1.3 on both,
  but single-model mAUC deltas under 2 points are not read.
- **P5: dev holds, eval misses high.** E2 − E1 [relu4] is +0.18 dev and +0.35
  eval (CI +0.21 to +0.51), above the predicted ceiling of +0.3. On eval,
  adding kNN inside the ensemble helps in all three compositions (+0.35,
  +0.23, +0.19).
- **P6: fails.** The source of the statistics matters. |S1 − S1′| reaches
  0.60 dev and 0.86 eval, and |E1 − E1′| reaches 0.41 on eval. The direction
  is consistent: **training-clip statistics are better**. See below.
- **P7: holds.** Mixed 2+2 ensembles average −0.02 dev and +0.17 eval
  against relu4. E1[mix8] reaches 96.02 dev and 96.41 eval.

### Decisions

- **D1: adopted.** E1[relu4] is this repo's reported *ensemble* system.
- **D2:** the claim allowed is "an ensemble of 4 OS-SCL models (4 × 1.38 M
  parameters) exceeds the paper's dev AUC, dev pAUC and eval score by point
  estimate". **[confirm-seeds]** On 4 fresh seeds E1 holds dev pAUC
  and eval but not dev AUC (95.64); E2′ over the same seeds holds all four. Not "clearly", and not on dev mAUC. `mix8` (dev 96.02 / 90.79 /
  91.29) and E2 score higher, but by the pre-registration they do not replace
  the headline.
- **D3: not adopted.** R0 stays the single-model rule. **[confirm-seeds]** S1′ was adopted
  there on 4 fresh seeds (4/4 on both sets).

### The unplanned result: held-out statistics hurt (post hoc)

I made the normalisation statistics come from the 5% held-out clips because
training clips are fit too well, and that was the principled-looking choice.
The exploration had used training-part clips. On this data the change was a
mistake:

```
mean Δ overall AUC vs R0      dev     eval    cells positive
S1   (held-out statistics)   +0.22   +0.14   9/16
S1′  (training statistics)   +0.42   +0.65   16/16
```

The same pattern shows in the ensembles. On eval, E1′[relu4] 96.42 > E0
(raw sum) 96.26 > E1 96.01. The pre-registered headline is the *weakest* of
the three normalisations on eval, so its claim against the paper is
conservative.

Likely mechanism (a hypothesis): each held-out ID has ~44 clips, so its mean
and especially its standard deviation are noisy. z-scoring then rescales IDs
inconsistently against each other. Per-ID ranking survives, but pooled
machine AUC suffers. The clearest case is eval valve under E1[relu4]: every
valve ID's AUC holds or rises (id_05 from ~0.969 to 0.976), yet pooled valve
AUC falls 0.96 below the mean single model, while E0 gains +0.47 and E1′
+0.24. Training-part statistics use ~840 clips per ID. Their optimism bias is
a near-constant shift per ID, which costs less than the variance.

S1′ was pre-specified, but only as a sensitivity check. Promoting it now,
after seeing eval, would be test-set selection. It is, however, exactly the
rule the dev exploration found (+0.44), now replicated on 4 new checkpoints
and on the unseen eval set in 16/16 cells. That makes it the natural primary
rule for a follow-up confirmation on new checkpoints.

## What next

1. **Confirm S1′ as the single-model rule** on checkpoints that don't exist
   yet. Any new 4-seed arm trained for the next ablation can serve: score it
   with S1′ as the pre-registered primary. No extra training needed.
2. **Turn the ensemble into one model.** D1 holds, so distillation (or a
   shared-trunk weight average) is the route to a single-model claim.
3. **Pump** (−1.29 vs the paper even ensembled) needs model-side work. No
   scoring rule here moves it.
