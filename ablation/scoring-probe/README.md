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

> **Status: pre-registered.** Design, predictions and decision rule written
> before any cosine was dumped; outcomes go under *Results*.

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

*Pending.*
