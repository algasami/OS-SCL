# Confirm seeds — fresh paper-model checkpoints for confirmation

> **[2026-10-28, `../mixed-ensemble`] The best reported ensemble is now
> E2′[A8]:** the same E2′ rule over 8 `aug` models from `../lit-arms`. It beats
> E2′[relu8] on eval mean by +0.31 [+0.07, +0.55] and on all three dev numbers
> (dev 97.05 / 92.48 / 94.10, eval mean 94.78). The E2′ rule and every number
> below stand as written. E2′[N4] is no longer the top system.

> **Status: done (2026-10-16), 4 fresh seeds, dev and eval.** **D1: S1′ adopted**
> as the single-model rule (beats R0 on 4/4 fresh seeds on both sets; eval
> +0.65 [+0.53, +0.78]). One model under S1′ beats the paper's eval number by
> point estimate (93.63 vs 93.22, 3/4 seeds), not on dev. **D3: E2′ = Σ S1′
> replaces E1** as the ensemble; E2′ over 4 fresh seeds beats all four paper
> numbers by point estimate (dev 96.05 / 90.94 / 91.34, eval 94.34), clearly
> on dev pAUC and eval. E1's dev AUC claim from `../ensemble-backend` did not
> replicate (95.64).
>
> **Resume:** nothing running. Regenerate figures with `make_plots.py`;
> re-scoring is `run.sh score` (resumable, ~30 min on one GPU).

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

## Results

Scored 2026-10-16 20:48–21:19 on GPU 2 (eval.py on N4, one extraction pass
over 8 checkpoints, scoring and 2000 bootstrap replicates). Everything above
this heading was committed before the run (1cfd9f0, d2dc978) and is left as
written. Full scorecard: `scores/scorecard.txt`.

### Headline

```
                                      dev AUC   dev pAUC   dev mAUC   eval mean(AUC,pAUC)
paper (OS-SCL, one model)             95.71     90.23      91.23      93.22
one model, R0 (N4 mean)               94.90     89.62      88.92      93.09
one model, S1' (N4 mean)              95.36     90.06      90.26      93.63 [93.18, 94.11]
4 models, E1[N4]                      95.64     90.61      89.95      93.86 [93.12, 94.54]
4 models, E2'[N4]  (adopted, D3)      96.05     90.94      91.34      94.34 [93.86, 94.82]
                                      [95.71,   [90.30,    [90.21,
                                       96.36]    91.55]     92.28]
```

Brackets are bootstrap 95% intervals over test clips (seed variance is not in
them). N4 never touched a test clip before this run.

- **A single model scored with S1′ beats the paper's eval number**: 93.63,
  on 3 of 4 fresh seeds (93.76 / 93.57 / 93.16 / 94.04). The interval's lower
  bound (93.18) does not clear 93.22, so not "clearly". On dev a single model
  stays short of the paper on all three numbers.
- **The 4-seed ensemble scored with E2′ beats all four of the paper's numbers
  by point estimate**, on seeds no rule had seen. It clearly beats dev pAUC and
  the eval number. Dev AUC's lower bound sits exactly on the paper (95.710 vs
  95.71), and dev mAUC's interval spans it; neither is "clearly".

### Verification

- **P0: holds.** Per-ID AUC matches eval.py's own CSVs to ≤ 1.4×10⁻⁴ on N4
  (≤ 3.6×10⁻⁴ on O4, as in ensemble-backend). The replayed held-out split
  reproduces every run's best `Valid_loss` to ≤ 1.8×10⁻⁵.
- **P1: holds.** N4 under R0 is the same population as O4: dev 94.90 ± 0.18
  vs 94.82 ± 0.23, eval 95.28 ± 0.27 vs 95.47 ± 0.30.

### Against the predictions

- **P2: holds.** S1′ − R0 overall AUC is positive on 4/4 N4 checkpoints on
  both sets: dev +0.46 / +0.38 / +0.35 / +0.61 (mean +0.45 [+0.35, +0.56]),
  eval +0.59 / +0.80 / +0.56 / +0.66 (mean +0.65 [+0.53, +0.78]). Both means
  sit near O4's (+0.52, +0.71); the dev figure did not shrink much. Counting
  O4, S1′ beats R0 on 8 of 8 `relu_max` checkpoints on both sets.
- **P3: holds, except one pump cell.** ToyCar (+0.6 to +2.2) and ToyConveyor
  (+0.5 to +3.5) are positive on 4/4 on both sets. Eval valve falls on 4/4
  (−0.30 to −0.60), as predicted. Dev pump stays within ±0.5 on 3/4; s2030
  moves +0.77.
- **P4: holds.** Under S1′, 3/4 single models reach the paper's eval number;
  under R0, 2/4. Dev AUC ≥ 95.71: 0/4 under either.
- **P5: holds, eval gain a little above range.** E1[N4] beats its best member
  on dev (95.64 vs 95.15) and eval (96.21 vs 95.57). The gain over the mean
  member is +0.73 dev (predicted +0.6 to +1.2) and +0.93 eval (predicted +0.3
  to +0.9). Against the paper, E1[N4] holds eval (93.86) and dev pAUC (90.61).
  **Dev AUC misses: 95.64 vs 95.71**, the coin flip O4 had won by 0.08. Dev
  mAUC misses (89.95), as predicted.
- **P6: eval holds, dev misses high by 0.01.** E2′ − E1 [N4] is +0.41 dev
  [+0.27, +0.56] (predicted ≤ +0.40) and +0.34 eval [+0.15, +0.53]. On O4,
  first computed here, it is +0.19 dev and +0.72 eval. The two parts do not
  add: on N4 eval E1′ − E1 is +0.01, so the eval gain is the kNN term.
- **P7: holds.** E1[relu8] (96.00 dev, 96.30 eval) beats both halves, and the
  mean over subsets rises at every n for E1 and E2′ on both sets. E2′ at
  n = 1 (95.35 dev, 96.06 eval) scores between E1 at n = 2 and 3 on dev, and
  above E1 at n = 3 on eval.

### Decisions

- **D1: adopted.** S1′ wins on 4/4 eval and 4/4 dev, mean +0.65 / +0.45, eval
  interval [+0.53, +0.78]. Single-model results are now reported under S1′ next
  to R0, as "OS-SCL + kNN back-end (S1′)". eval.py is unchanged.
- **D2:** the single-model claim allowed is "one OS-SCL network, scored with
  S1′, exceeds the paper's eval number by point estimate (93.63 vs 93.22; 3 of
  4 seeds)". Not "clearly", not on dev.
- **D3: E2′ replaces E1** as the reported ensemble (E2′ − E1 positive on both
  sets). The claim allowed is "an ensemble of 4 OS-SCL networks (4 × 1.38 M
  parameters) scored with E2′ exceeds the paper's dev AUC, dev pAUC, dev mAUC
  and eval score by point estimate, on fresh seeds; clearly on dev pAUC and
  eval". `../ensemble-backend`'s E1[relu4] dev AUC claim did **not** replicate
  on N4 (95.64) and is annotated there.
- **D4 (half seen, decides nothing):** relu8 under E2′ reaches dev
  96.16 / 91.12 / 91.52 and eval mean 94.46.

### What this means

- **The best single-model system is the paper's network with a different
  score.** S1′ adds the distance to the nearest normal training clip in the
  embedding, both z-scored per machine ID on training clips. It costs one
  nearest-neighbour search per test clip and no training. It is now confirmed
  on fresh checkpoints, on both sets.
- **The score fixes the machines R0 is weak on** (ToyCar, ToyConveyor) and
  costs a little on eval valve. Pump, the largest dev gap to the paper, barely
  moves.
- **Ensembling and S1′ stack.** E2′ beats E1 on both sets for N4 and O4, and a
  single model under S1′ is worth two to three under R0.
- **What is still not beaten by one network** is the paper's dev numbers. That
  is the target for `../distill`, whose students are scored with R0 by their
  pre-registration; S1′ on them is a natural follow-up.

## Figures

Regenerate with `/home/f74134118/anaconda3/bin/python ablation/confirm-seeds/make_plots.py`
(matplotlib lives in `base`). Reads `scores/` only. E1 is blue, the
training-statistics rules S1′ and E2′ orange; R0 and the paper are achromatic.

| figure | shows |
|--------|-------|
| `plots/fig1_single_model.png` | each of the 8 checkpoints under R0 and S1′: S1′ wins on all 8 on both sets (N4 filled, O4 hollow) |
| `plots/fig2_eval_mean_vs_paper.png` | eval mean(AUC, pAUC) per N4 model under R0 and S1′, plus mean S1′, E1[N4] and E2′[N4] with 95% CIs: S1′ lifts 3 of 4 single models past the paper |
| `plots/fig3_ensemble_size.png` | score vs ensemble size over every subset of relu8: E2′ above E1 at every n, the gap narrowing from n = 1 to 8. The low E1 n = 1 dots are single models under held-out z-statistics |
| `plots/fig4_vs_paper.png` | E2′[N4], E1[N4] and mean S1′ with 95% CIs against the paper's four numbers: E2′ above on 4/4 by point estimate; its dev AUC lower bound touches the paper |
| `plots/fig5_per_machine.png` | S1′ − R0 per machine for the 4 N4 seeds: ToyConveyor and ToyCar carry the gain, eval valve falls on all 4 |
