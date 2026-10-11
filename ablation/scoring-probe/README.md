# Scoring probe — is eval.py's anomaly score the right function of the model?

Every number in this repo comes from one scoring rule, inherited from
STgram-MFN and never ablated here. For a test clip with known machine ID `y`,
`eval.py` takes the per-clip cross-entropy of the ArcFace logits:

```
score = CE(logits, y),   logit_y = 30·cos(θ_y + m)   <- the training margin, still applied at test time
                         logit_j = 30·cos θ_j          (j ≠ y, all 41 classes, every machine type)
```

(`m=0.4` from `--m`; the margin is applied because `eval.py` passes `labels`
into `ArcMarginProduct.forward`, `losses.py`.) Two choices are baked into that
rule without having been tested: the margin is kept at test time, and the
clip is compared against **all 41** class centres, including the other five
machine types. This probe changes only the scoring rule, on fixed trained
checkpoints, and asks whether either choice matters.

Prompted by a senior's suggestion to look at "machine-ID label hinting". This
is its zero-training version: the machine type is known at test time in DCASE
2020 Task 2, so the question is how much of that known label the score should
use. Conditioning the network on machine type *during training* is a separate,
costlier question that this probe can motivate but not answer.

> **Status: done at n=1 per arm (seed 2024); stage-2 seeds pending.** Answer:
> **keep eval.py's rule.** No alternative beats R0 consistently. Dropping the
> margin (R1) or dropping the margin *and* the other machine types (R3) loses
> to it in every cell. Restricting to the own machine type (R2) and own-centre
> cosine (R4) change sign between dev and eval. The design, predictions and
> decision rule below were written before any cosine was dumped and are left
> unedited; outcomes are under *Results*. Rerun `run.sh` after
> `../tfgram-linear` stage 2 lands to get n=4 per arm.

## Rules

All five are functions of the 41 cosines `cos θ_j` (normalised embedding ·
normalised ArcFace weight), dumped once per run.

| rule | score | asks |
|------|-------|------|
| **R0** | CE over 41 classes, margin on | `eval.py`; must reproduce the committed CSVs |
| R1 | CE over 41 classes, margin off | does keeping the training margin at test time help ranking? |
| R2 | CE over the **own machine type's** 6–7 IDs, margin on | is proximity to *other machine types'* centres signal or noise? |
| R3 | CE over own machine type, margin off | completes the 2×2 {margin} × {class set} |
| R4 | `1 − cos θ_y` | is distance to the own-ID centre alone enough? |

**Diagnostic** (`scores/cross_type.csv`): per machine, for normal vs.
anomalous clips, R0's softmax mass on other machine types, and how often the
nearest rival centre belongs to another type. If that mass is ~0 everywhere, R0
and R2 cannot differ.

## Setup

- **Runs:** every finished `tfgram-linear` checkpoint, `relu_max` (the paper's
  TFgram) and `linear_avg`. At launch that is **seed 2024 only** (n=1 per arm);
  seeds 2025–2027 are still training in `../tfgram-linear` stage 2 and
  `run.sh` picks them up once their eval CSVs are committed. The seed-sweep
  checkpoints are not on this host (`smartlab1`).
- **Forward pass:** exactly `eval.py`'s: EMA weights, batch 1, `train=False`,
  architecture flags from the run's committed `config.json`. GPU 0.
- **Scoring:** float64, offline. eval.py's CE runs in float32, where a
  confident clip's `log(1 + Σ e^{z_j − z_y})` with `Σ ~ e^{-17}` rounds to 0 or
  a few ulps. Ties between normal and anomalous clips at 0 count half in AUC, so
  float64 R0 can differ slightly from the committed CSV; `reproduction.json`
  records both the gap and the fraction of float32 scores that are exactly 0.
- **Metrics:** eval.py's (AUC and pAUC pooled per machine, mAUC = min per-ID
  AUC, each averaged over the six machines), plus **idAUC**, the mean per-ID
  AUC. idAUC only sees the ranking within an ID. Pooled machine AUC also depends
  on whether scores are comparable *across* IDs (calibration), and two rules can
  differ there while ranking every ID identically.

### Why the noise floor is lower here than elsewhere

Every comparison is **paired**: the same checkpoint and the same test clips,
scored two ways. The ~1-point seed-to-seed AUC noise from `../seed-sweep` is
variation *between trained models*, and it cancels in a within-checkpoint
difference. What remains is whether the rule effect generalises across seeds,
so the decision rule asks for a consistent sign across runs rather than a size
above 1 point.

## Predictions (written before running)

- **P0, R0 reproduces eval.py.** Per-ID AUC within 0.001 of the committed CSVs.
  Any larger gap should be fully explained by float32 ties
  (`frac_float32_score_zero` > 0 on the anomaly side).
- **P1, R2 ≈ R0.** At scale 30 a centre with cosine 0.4 below the clip's own
  contributes `e^{-12}` of the mass, so foreign machine types should be
  irrelevant for nearly every clip: |Δ overall AUC| ≤ 0.2 on every run and
  dataset, and median cross-type mass < 10⁻³ for normal clips on every machine.
  Most likely exception, if any: ToyCar/ToyConveyor (both toy sets recorded
  with the same setup).
- **P2, R1 vs R0: small, direction not predicted.** |Δ overall AUC| < 0.5. The
  margin re-weights own-centre distance against close within-type rivals; I
  have no basis for a sign.
- **P3, R4 ≈ R0 on idAUC, but not on pooled AUC.** |Δ idAUC| < 0.5 overall,
  with per-machine swings up to ~2 points where IDs of one machine sit close
  together (valve, slider). Pooled machine AUC can move more, because
  `1 − cos` and CE are calibrated differently across IDs.
- **P4, arm ordering survives.** For every rule, `linear_avg − relu_max` has the
  same sign as under R0 on each dataset. (At n=1 per arm that ordering is
  itself noise, see `../tfgram-linear`; the point is that the scoring choice
  should not be what decides it.)

## Decision rule

- A rule **changes the result** if its overall AUC delta vs. R0 has the **same
  sign on every (run, dataset) cell** and a mean |Δ| ≥ 0.3. Otherwise it is
  recorded as no effect.
- **Selection uses dev only; eval is confirmation.** The eval set is the test
  set. Picking a scoring rule because it scores best on eval would be
  test-set selection. A rule is adopted only if it wins on dev and the eval
  delta has the same sign.
- **What happens next:** if R2 or R3 wins, cross-type comparison is noise →
  try machine-type conditioning at training time. If R0 beats R2/R3,
  "sounds like another machine" is real anomaly evidence, and that is worth
  stating in any write-up. If R4 ≈ R0, the softmax adds nothing over the
  embedding geometry.

## Files

```
dump_cosines.py   GPU, once per run x dataset: 41 cosines + eval.py's float32 score per clip
score_rules.py    CPU, float64: R0-R4, metrics, cross-type diagnostic, reproduction check
run.sh            end to end; resumable; picks up finished stage-2 seeds
make_plots.py     figures from scores/ only
scores/           rule_metrics.csv, cross_type.csv, reproduction.json  (committed)
results/          per-clip cosines, gitignored (.gitignore here), regenerable
```

Run: `bash ablation/scoring-probe/run.sh` from the repo root. When run from
another worktree, set `PROBE_REPO` to the checkout holding `data/` and
`check_points/`.

## Results

Two checkpoints, `tfl_relu_max_s2024` and `tfl_linear_avg_s2024`, × dev and
eval = 4 paired cells. Wall clock 7 min on one A4000 (GPU 0), most of it Log-Mel
on CPU.

### Reproduction (P0): holds

R0 in float64 matches every committed per-ID AUC to within **6×10⁻⁵**
(`reproduction.json`). No float32 score is exactly 0 and >99.9% are distinct.
The tie mechanism I flagged didn't occur: with the margin on, the own-class
probability of a normal clip has a median of only ~0.79, nowhere near
saturation.

### Overall, rule − R0 (AUC points; `scores/rule_metrics.csv`)

```
                     pooled AUC (eval.py)          mean per-ID AUC
cell                 R1     R2     R3     R4       R1     R2     R3     R4
dev  relu_max      -0.14  -0.19  -0.64  -0.13    -0.26  -0.09  -0.58  -0.02
dev  linear_avg    -0.16  -0.23  -0.70  -0.14    -0.23  -0.19  -0.72  -0.06
eval relu_max      -0.60  +0.09  -0.46  +0.34    -0.61  +0.05  -0.53  +0.36
eval linear_avg    -0.51  -0.21  -1.03  +0.27    -0.45  -0.08  -0.74  +0.25
mean               -0.35  -0.14  -0.71  +0.09
```

R0 itself: dev 94.97 / 95.12, eval 95.20 / 95.45 pooled AUC (relu_max /
linear_avg).

**Decision rule** (same sign in all 4 cells and |mean| ≥ 0.3):

| rule | verdict |
|------|---------|
| R1 | **loses to R0**: negative in all 4 cells, mean −0.35. Keeping the training margin at test time helps. |
| R2 | no effect: sign flips (eval relu_max +0.09), mean −0.14 |
| R3 | **loses to R0**: negative in all 4 cells, mean −0.71; the worst rule, on every metric |
| R4 | no effect: −0.13 on dev, +0.3 on eval. Loses on dev, so not adopted under the dev-only selection rule, however good eval looks |

pAUC and mAUC point the same way (`fig1`; mAUC for R3 drops 1.0–2.1 points).
Per machine (`fig2`), most of the movement is in the toy sets: R1 costs
ToyConveyor 1.9–2.7 on eval, R3 costs ToyCar 1.4–2.6, and R4's eval gain is
mostly ToyConveyor (+1.0/+1.8). Per `../seed-sweep`, ToyConveyor is the noisiest
machine (>4 points seed to seed). fan, slider and valve barely move under any
rule.

### Cross-type diagnostic: P1's reasoning was wrong

I predicted foreign machine types would hold < 10⁻³ of R0's softmax mass.
They hold **~15–19% for normal clips on every machine** and **50–80% for
anomalous fan, pump, slider and valve** (`fig3`, `cross_type.csv`). The
"scale 30 makes rivals vanish" argument assumed tight clusters. In fact the
margin pulls the own logit down to `30·cos(θ+0.4)`, and 34 foreign centres at
modest cosine add up. The nearest rival is a foreign type for 33–65% of
normal fan/slider/valve clips, but almost never for ToyCar (0–4%), whose IDs
crowd each other.

So other machine types are a large part of eval.py's score, yet removing them
(R2) barely changes the ranking. The outcome half of P1 nearly held
(|Δ| ≤ 0.23, though two cells exceed the 0.2 I wrote).

### Prediction scorecard

| | prediction | outcome |
|---|---|---|
| P0 | R0 reproduces committed CSVs within 0.001 | **holds** (6×10⁻⁵) |
| P1 | R2 ≈ R0 (|Δ| ≤ 0.2); cross-type mass < 10⁻³ | **outcome nearly, mechanism wrong**: |Δ| ≤ 0.23, mass ~15–80% |
| P2 | R1: |Δ| < 0.5, sign unknown | **fails on size**: −0.60 / −0.51 on eval; sign is consistently negative |
| P3 | R4 ≈ R0 on idAUC (< 0.5); pooled moves more; swings in valve/slider | idAUC **holds** (≤ 0.36); pooled does **not** move more; swings are in ToyConveyor/ToyCar, not valve/slider |
| P4 | arm ordering survives every rule | **fails on eval for R2/R3**: `linear_avg − relu_max` is +0.25 under R0, −0.05 under R2, −0.32 under R3. That gap is below the n=1 noise floor either way, which is the point: a scoring choice is enough to flip it |

### Post-hoc mechanism (not pre-registered)

The loss ordering R3 < R1 < R2 < R0 is also the order of how close each rule
pushes a normal clip's own-class probability to 1:

```
            median p_own (normal clips)   share > 0.99    mean Δ AUC vs R0
R0                 0.79                     0%               —
R2                 0.95                     0%             −0.14
R1                 0.992                   83%             −0.35
R3                 0.998                   98%             −0.71
R4             (no softmax)                 —              +0.09
```

Identical across all 4 cells to ±0.01. One reading: once `p_own → 1`,
`CE ≈ Σ_j e^{z_j − z_y}`, which is dominated by the **single nearest rival**,
so the score turns into a max-type statistic of one gap and gets noisier.
The margin plus the 40-class denominator keep R0 out of that regime, so it
behaves like a smooth own-distance (R4, which has no softmax at all, ranks
about as well). This was found after looking at the results, from 2
checkpoints, and is a hypothesis. A direct test would be R0 with the
temperature varied at scoring time.

### What this means

- **Keep eval.py's rule.** All past ablation numbers stand. The margin at
  test time is doing real work, which nobody had checked before.
- **On "machine-ID label hinting":** the current rule already uses the true
  ID, and the stronger hint (restricting the comparison to the clip's own
  machine type) buys nothing. **This probe gives no reason to try
  machine-type conditioning at training time.** It can't rule it out either:
  conditioning changes the embedding, and this probe only re-reads a fixed one.
- **For `../tfgram-linear`:** the arm difference (+0.25 eval under R0) is
  smaller than the effect of some scoring choices. That's another reason to
  read it only at n=4.

## Figures

Regenerate with `/home/f74134118/anaconda3/bin/python ablation/scoring-probe/make_plots.py`
(matplotlib lives in `base`). Reads `scores/` only and picks up every run
present. Arm colours follow `../tfgram-linear` (`relu_max` achromatic).

| figure | shows |
|--------|-------|
| `plots/fig1_overall_delta.png` | overall pooled AUC and mean per-ID AUC, each rule minus R0 per checkpoint, dev and eval, against the ±0.3 decision band: R1 and R3 below it in every cell |
| `plots/fig2_machine_delta.png` | the same deltas per machine: movement concentrates in ToyCar/ToyConveyor; fan, slider and valve are flat |
| `plots/fig3_cross_type.png` | R0's median softmax mass on other machine types (~15% for normals, 50–80% for anomalous fan/pump/slider/valve), and how often the nearest rival centre is another type |
