# Confirm seeds — fresh paper-model checkpoints for confirmation

> **Status: pre-registered (2026-10-15); training done; score stage committed. Nothing scored on the
> new checkpoints, R0 included.** All 4 seeds trained by 2026-10-15 15:00.
>
> **Resume:**
> - `run.sh score` waits for an idle GPU no other ablation has locked, then runs
>   eval.py (R0 CSVs) on N4, `../ensemble-backend/extract.py` on N4 + O4 and
>   `score.py`. Log: `results/run_score.log`. Resumable.
> - Then write Results against P0–P7 and D1–D4, `make_plots.py` and figures.
>   Nothing in *Pre-registration* may change after the first score.

## Why this exists

The open single-model rules have only been checked on checkpoints whose eval
scores we have already seen:

- S1′ from `../ensemble-backend`;
- F_A1 from `../memory-bank`, if adopted (it was not: the patch bank is null,
  so F_A1 is dropped here);
- an 8-member E1.

A clean confirmation needs new checkpoints. These are 4 more seeds of the
paper's model, trained with the exact recipe of `../tfgram-linear`'s `relu_max`
arm (`--num_workers 8`, 300 epochs, same snapshots). Only the seed and desc
differ.

| run | seed | GPU |
|-----|------|-----|
| `SCLTFSTgramMFN_cs_relu_max_s2028` | 2028 | 2 |
| `SCLTFSTgramMFN_cs_relu_max_s2029` | 2029 | 1 |
| `SCLTFSTgramMFN_cs_relu_max_s2030` | 2030 | 2 |
| `SCLTFSTgramMFN_cs_relu_max_s2031` | 2031 | 1 |

## Pre-registration

Written 2026-10-15, before any score of any kind on the four new checkpoints.

### What has already been seen

- **Old checkpoints (O4)**, `tfl_relu_max_s2024–2027`: every rule below except
  E2′ has been scored on them, on dev and eval (`../ensemble-backend`,
  `scores/summary.csv`). S1′ was found on dev, then held in 16/16 cells there,
  8 of them `relu_max`. All predictions below are calibrated on these numbers.
- **New checkpoints (N4)**, `cs_relu_max_s2028–2031`: only their training logs.
  No test clip of either set has been passed through them.
- **E2′** (defined below) has never been computed on any checkpoint.

Single models on O4, the calibration (overall, %):

```
                  dev AUC   dev pAUC   eval AUC   eval mean(AUC,pAUC)
S0 = R0  s2024    94.97     89.65      95.20      92.93
         s2025    94.59     90.01      95.53      93.20
         s2026    95.06     89.61      95.27      92.72
         s2027    94.66     89.70      95.86      93.31
S1'      s2024    95.43     90.09      96.05      93.71
         s2025    95.14     90.25      96.28      93.96
         s2026    95.63     90.24      96.09      93.57
         s2027    95.14     89.99      96.29      93.75
paper             95.71     90.23        —        93.22
```

S1′ − S0 on O4: dev AUC +0.46 / +0.55 / +0.57 / +0.48 (mean +0.52), eval
+0.85 / +0.75 / +0.82 / +0.43 (mean +0.71). Eval valve falls in all four
(−0.21 to −0.62); ToyConveyor (+1.7 to +3.7) and ToyCar carry the gain.
Ensembles on O4 (E1[relu4]): dev 95.79 AUC, eval mean 93.70; E1′[relu4] eval
mean 94.18.

### Checkpoints and sets

| name | checkpoints | role |
|------|-------------|------|
| **N4** | `cs_relu_max_s2028–2031` | **confirmatory**, dev and eval; eval is primary |
| O4 | `tfl_relu_max_s2024–2027` | calibration only, already seen |
| relu8 | N4 + O4 | reported, never decision-bearing (half seen) |

Both test sets count for N4: these are new models, though the dev clips shaped
the rule.

### Scores and rules

Exactly `../ensemble-backend`'s definitions and code (`extract.py`, the
`per_run_scores` of `score.py`). For checkpoint *m*, test clip *x* of machine
ID *i*:

- **R0(x)**: eval.py's score (margin ArcFace CE against *i*).
- **K(x)** = 1 − max cos to ID *i*'s training-part clips, on the L2-normalised
  pre-head embedding, k = 1.
- **Primed (′) statistics**: mean and sd over ID *i*'s **training-part** clips
  (K leave-one-out). Unprimed: over ID *i*'s held-out 5%.

| rule | definition | role |
|------|------------|------|
| S0 | R0 | single-model baseline |
| **S1′** | zR0′ + zK′ | **primary single-model rule** |
| S1 | zR0 + zK | reported, continuity |
| **E1** | Σ_m zR0 | **the adopted ensemble** (ensemble-backend D1) |
| E1′, E0 | Σ_m zR0′, Σ_m R0 | reported, sensitivity |
| **E2′** | Σ_m (zR0′ + zK′) = Σ_m S1′ | **challenger ensemble**, never scored anywhere |

Metrics and uncertainty are ensemble-backend's: eval.py's AUC, pAUC
(max FPR 0.1) and mAUC per machine, averaged over six machines, plus the eval
mean(AUC, pAUC). Stratified bootstrap over test clips (machine × ID ×
normal/anomalous, 2000 replicates), one resample shared by every checkpoint
and rule, so a paired difference and a mean over the 4 checkpoints get one CI.

### Predictions

**P0. Reproduction.** Batched R0 matches eval.py's per-ID AUC to ≤ 5×10⁻⁴
(ensemble-backend measured float32 reordering up to 3.6×10⁻⁴, so its 1×10⁻⁴
was too tight). The replayed held-out split reproduces each run's best
`Valid_loss` to ≤ 1×10⁻⁴.

**P1. N4 is the same population as O4.** Mean S0 dev AUC of N4 within ±0.4 of
O4's 94.82, eval within ±0.4 of 95.47. Seed sd of dev AUC between 0.1 and 0.5.
A failure here means the host or recipe drifted, and the rest is read with that
caveat.

**P2. S1′ beats R0 on every new checkpoint.** S1′ − S0 overall AUC > 0 for 4/4
N4 checkpoints on eval and 4/4 on dev. Mean +0.3 to +1.0 eval and +0.25 to
+0.75 dev: O4's +0.71 and +0.52 are the centre, and the dev figure is the one
the rule was found on, so it may shrink. Confidence: high for the sign on eval,
moderate for 4/4 on dev.

**P3. Where it acts.** ToyConveyor and ToyCar carry the gain (positive on ≥ 3/4
on both sets), and eval valve **falls** under S1′ on ≥ 3/4 checkpoints. Pump
moves less than ±0.5 on dev.

**P4. A single model beats the paper's eval number under S1′.** N4's S1′ eval
mean(AUC, pAUC) ≥ 93.22 for ≥ 3 of 4 checkpoints (O4: 4/4, 93.57–93.96).
Under S0 the same holds for ≤ 2 of 4 (O4: 1/4, s2027 at 93.31).
Dev AUC ≥ 95.71 under S1′ for at most 1 of 4 (O4: 0/4, max 95.63); dev pAUC
≥ 90.23 is a coin flip.

**P5. The ensemble result replicates on new seeds.** E1[N4] beats its best
member's S0 overall AUC on dev and eval, with a gain over the mean member of
+0.6 to +1.2 dev and +0.3 to +0.9 eval. Against the paper: eval mean ≥ 93.22
(high confidence), dev AUC ≥ 95.71 (coin flip: O4 cleared it by 0.08), dev
pAUC ≥ 90.23 (moderate), dev mAUC ≥ 91.23 (unlikely).

**P6. Summing S1′ beats summing zR0.** E2′[N4] − E1[N4] overall AUC is +0.0 to
+0.4 on dev and +0.2 to +0.8 on eval. Basis: on O4 eval both of its parts
helped, E1′ − E1 by +0.41 and E2 − E1 by +0.35; on dev each moved ≤ 0.2.
Confidence: moderate; the two gains may not add.

**P7. relu8, reported only.** E1[relu8] ≥ max(E1[N4], E1[O4]) on both sets, and
the mean E1 over all size-n subsets of relu8 is non-decreasing in n.

### Decision rules

Fixed now. N4 eval is the test set; nothing is tuned on it.

- **D1, adopt S1′ as the single-model rule.** Iff S1′ − S0 overall AUC > 0 on
  **4/4** N4 checkpoints on eval and **≥ 3/4** on dev, **and** the mean Δ is
  ≥ +0.2 on both sets, **and** the bootstrap 95% CI of the eval mean Δ excludes
  0. "Adopt" means single-model results are reported under S1′ next to R0,
  named "OS-SCL + kNN back-end (S1′)". eval.py's R0 is kept: the paper's
  numbers are R0 numbers.
- **D2, single-model claim against the paper.** Only if D1 holds. Per metric,
  allowed where N4's mean S1′ point estimate exceeds the paper; reported with
  the 4 per-seed values and the bootstrap CI of the mean. "Every seed" only if
  4/4 clear it. Wording: a scoring change on the paper's network, one model,
  no extra training.
- **D3, ensemble rule.** E1 stays the reported ensemble unless E2′[N4] beats
  E1[N4] in overall AUC on **both** dev and eval; then E2′ replaces it. The
  ensemble's claims against the paper are re-stated from N4 alone, by
  ensemble-backend's D2 wording ("4 × OS-SCL", point estimate, "clearly" only
  where the CI lower bound clears). Where N4 fails a metric that O4 passed,
  the O4 claim is annotated as not replicated.
- **D4.** relu8 numbers may be quoted as "the best available system", always
  marked as half-seen; they decide nothing.
- Single-model mAUC deltas under ~2 points are not read (`CLAUDE.md`).

### What this cannot answer

- n = 4 new seeds. A sign condition of 4/4 under a null of a fair coin has
  p = 1/16 per set, so D1 leans on the size and the CI as much as on the
  count.
- The bootstrap is over test clips; seed variance enters only through the
  per-checkpoint counts.
- Nothing here changes the network. Pump's dev gap to the paper (−1.3 even
  ensembled) is not expected to move.

### Planned files

```
run.sh score   1. eval.py --d/--e --csv scores/<run>/eval_{dev,eval}.csv for N4 (R0, for P0)
               2. ../ensemble-backend/extract.py on N4 + O4 -> results/emb (gitignored)
               3. score.py
score.py       imports per_run_scores / Evaluator from ../ensemble-backend/score.py;
               P0-P7 and D1-D4 printed to scores/scorecard.txt
scores/        rule_metrics.csv, summary.csv, ensemble_curve.csv, bootstrap.json,
               reproduction.json, scorecard.txt, and per-run eval CSVs + curve.csv
make_plots.py  reads scores/ only
```

The score code is committed before it is run. A bug fix after the first run is
allowed only for a crash or a P0 failure, and is logged in an addendum here.

**Addendum, 2026-10-16, before any score on N4.** `score.py` was dry-run on
already-seen checkpoints only: the cached ensemble-backend arrays, with
`linear_avg` s2024–2027 standing in for N4 and 20 bootstrap replicates. It
reproduced ensemble-backend exactly (P0 diffs, E1[lin4] 95.79 dev / 96.38
eval). That run computed E2′ for the first time, on old checkpoints:
E2′ − E1 [lin4] was +0.18 dev and +0.47 eval. These are inside P6's ranges
and were seen after P6 was committed; nothing above changed. No N4 checkpoint
was touched.
