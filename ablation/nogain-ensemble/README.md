# nogain-ensemble — does an ensemble of `ao_nogain` models beat E2′[A8]?

> **Status: PRE-REGISTERED 2026-11-01.** Nothing has been scored yet.
>
> CPU only. Every member was already trained and dumped.

## Why this exists

`../aug-ops` replaced full `aug` with `--wave_aug_ops noise,shift` (`ao_nogain`)
as the single-model recipe (C2). That replication left one question open: the
8 `ao_nogain` models have never been ensembled.

The reported best ensemble is E2′[A8] from `../mixed-ensemble`: the 8 `la_aug`
models, seeds 2040–2047.

- Dev: 97.05 / 92.48 / 94.10 (AUC / pAUC / mAUC).
- Eval mean(AUC, pAUC): 94.78.
- The paper reports 95.71 / 90.23 / 91.23 on dev and 93.22 on eval.

`ao_nogain` was trained on **the same 8 seeds**. So G8 and A8 are seed-paired
member for member.

Single models under S1′, 8-seed means (`../aug-ops/scores/s1p.csv`), already seen:

| S1′, single model | dev AUC | dev pAUC | dev mAUC | dev mean | eval AUC | eval mean |
|---|---|---|---|---|---|---|
| `ao_nogain` | 96.43 | 91.58 | 92.82 | 94.00 | 96.19 | 93.88 |
| `la_aug` | 96.24 | 91.22 | 92.66 | 93.73 | 96.12 | 93.82 |
| Δ | +0.19 | +0.36 | +0.16 | +0.27 | +0.07 | +0.06 |

Single models gain on dev and tie on eval. The question is whether averaging 8
of them changes that.

## What has already been seen

- **Every member individually,** under R0 and S1′, on dev and eval
  (`../aug-ops`, `../lit-arms`).
- **Every reference ensemble** (A8, F20, relu8) on dev and eval
  (`../mixed-ensemble`).
- **Not seen:** any ensemble that contains an `ao_nogain` model, on either set.

`ao_nogain` was itself picked post hoc. It was the best of three arms on dev
(`../aug-ops` stage 1), and its replication on seeds 2044–2047 held. No member
here was chosen by its eval score.

## Members (all 300 epochs, `fussion 1`, `m 0.4`, `ht basic`)

| type | runs | dump |
|---|---|---|
| `nogain` | `ao_nogain_s2040–2047` (8) | `OS-SCL-augops/ablation/aug-ops/results/emb` |
| `aug` | `la_aug_s2040–2047` (8) | `OS-SCL-litarms/ablation/lit-arms/results/emb` |
| `plain`, `gate`, `eca` | `fr_f1`, `la_gate`, `la_eca`, `s2040–2043` (4 each) | same |
| `old` | `cs_relu_max_s2028–2031` + `tfl_relu_max_s2024–2027` (8) | `OS-SCL-confirm-seeds/ablation/confirm-seeds/results/emb` |

The dumps come from `../ensemble-backend/extract.py` and are not regenerated.
At setup, all three `meta.npz` files list the same clips in the same order
below `data/`. `score.py` asserts this again before it scores anything.

## Rule and metrics

**Unchanged from `../ensemble-backend` / `../mixed-ensemble`, imported, not
copied:**

- Each member gets S1′.
- An ensemble is E2′, the plain sum of its members' S1′. No weights; nothing
  is fitted on test clips.
- Metrics are eval.py's: AUC, pAUC, mAUC, and mean(AUC, pAUC) as the eval number.

**Fixed systems:**

| system | members | n | role |
|---|---|---|---|
| `G8` | `nogain` | 8 | candidate (the main question) |
| `GA16` | `nogain` + `aug` | 16 | candidate |
| `F20g` | `plain` + `gate` + `eca` + `nogain` | 20 | candidate: `../mixed-ensemble`'s F20 with `aug` swapped for `nogain` |
| `A8` | `aug` | 8 | **the bar** |
| `F20` | `plain` + `gate` + `eca` + `aug` | 20 | reference |
| `relu8` | `old` | 8 | reference (the previous bar) |

**4-member families,** over all 70 four-seed subsets of 2040–2047:

- `hom4_nogain` and `hom4_aug`.
- `paired4`: for each subset, `hom4_nogain` minus `hom4_aug` on **the same
  seeds**. That removes seed luck from the comparison.

**Uncertainty:**

- Families report the 5–95% spread over subsets, which is model-choice noise.
- Each system is compared with A8 by a paired clip-bootstrap (2000 stratified
  resamples, seed 0, the same clips for both systems), which is test-sample
  noise.

## Workflow

1. **Stage A, dev only:** `bash ablation/nogain-ensemble/run.sh dev`. It
   builds an evaluator for dev only, so no eval number is computed. It writes
   `scores/*_dev.*`, which are committed.
2. **Dev pick:** the candidate (`G8`, `GA16`, `F20g`) with the highest dev
   mean(AUC, pAUC). A tie within 0.05 goes to the smaller n.
3. **Eval predictions:** an addendum with numeric stage-B predictions,
   committed before stage B runs.
4. **Stage B, eval:** `run.sh eval`, then the results, figures and decisions.

## Predictions, stage A (dev), written before any ensemble score

- **P0, reproduction.** A8, F20 and relu8 reproduce `../mixed-ensemble`'s dev
  numbers (A8 97.05 / 92.48 / 94.10, F20 96.64, relu8 96.16 AUC) to within
  0.01. `hom4_aug` reproduces its combo mean (AUC 96.93, mean 94.62). If any
  of them doesn't, stop: the dumps or the rule differ.
- **P1, G8.** Dev AUC in [97.05, 97.45], central 97.25. Dev mean in
  [94.85, 95.25], central 95.05. That is A8 plus most of the single-model
  +0.27.
- **P2.** G8 is at least A8 on all three dev numbers. mAUC is the least
  certain of the three. About 60% overall.
- **P3.** GA16's dev mean lies strictly between A8's and G8's. The two
  member types are near-copies, so a 50/50 mix should land near the
  midpoint.
- **P4.** F20g's dev mean is 0.3 to 0.7 below G8's. F20 was 0.53 below A8,
  because plain, gate and eca dilute the augmented models.
- **P5.** The dev pick is G8. About 75%; otherwise GA16.
- **P6.** `paired4` dev mean Δ is in [+0.10, +0.40], central +0.25, and
  positive in at least 85% of the 70 subsets.
- **P7.** On dev, G8's gain over A8 is mostly ToyConveyor: G8 ToyConveyor
  ≥ A8's + 0.5. Single `ao_nogain` models gained +1.86 there on seeds 2040–2043.

## Predictions, stage B (eval), first draft

The full set comes in the addendum after stage A. These are fixed now:

- **Q1.** G8 − A8 on eval mean is in [−0.20, +0.30], central +0.05. The
  paired 95% interval includes 0, so **D1 fails** if G8 is the pick. About 75%.
- **Q2.** G8's eval ToyCar AUC is within ±0.6 of A8's 96.00. Eval ToyCar id_05
  stays below 85 in both. Every member has the shift op, and `../aug-ops`
  showed shift carries half of the id_05 loss.
- **Q3.** F20g's eval mean is at least G8's and within ±0.15 of F20's 94.96.
  Mixing repaired eval ToyCar for F20, and it should do the same here.

## Decision rules

- **D1, a new reported ensemble.** The dev pick replaces E2′[A8] as the
  reported best ensemble if both hold:
  - its eval mean(AUC, pAUC) beats A8 with a paired-bootstrap 95% lower
    bound > 0;
  - its dev AUC, pAUC and mAUC are each ≥ A8's.

  This is the bar A8 cleared against relu8. Otherwise E2′[A8] stays.
- **D2, which member type ensembles better** (no adoption on its own), on
  eval `paired4`:
  - "`nogain` ensembles better" if the mean Δ on mean(AUC, pAUC) is ≥ +0.10
    and positive in ≥ 80% of subsets;
  - "`aug` ensembles better" if the mean Δ is ≤ −0.10 and positive in ≤ 20%;
  - otherwise "tie".
- **D3, secondary (no claims).** Every fixed system is reported on eval, per
  machine and per ToyCar ID. Choosing among them on eval would be test-set
  selection.

## What this cannot answer

- **Fresh checkpoints.** Every member's single-model eval score has been
  seen, and `ao_nogain` was a post-hoc pick that was then replicated. The
  ensembles are unseen, and none of their members was picked on eval.
- **ToyCar id_05.** Every G8 and A8 member uses the shift op. Neither can
  repair id_05; only mixing in non-augmented models can dilute it.

## Files

| file | what |
|---|---|
| `score.py` | the scorer, one split per call |
| `run.sh` | `dev`, `eval`, `plots` |
| `scores/` | committed record: `systems_*`, `perid_*`, `combos_*`, `bootstrap_*`, `scorecard_*` |
| `results/` | the scorer's console log (gitignored scratch) |
