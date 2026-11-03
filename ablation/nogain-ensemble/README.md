# nogain-ensemble — does an ensemble of `ao_nogain` models beat E2′[A8]?

> **Status: DONE 2026-11-01.**
> - **D1, new reported ensemble: fails.** G8 (the dev pick) leads A8 on dev,
>   +0.20 [+0.10, +0.31], but only +0.09 [−0.05, +0.23] on eval. **E2′[A8]
>   stays the reported best ensemble.**
> - **D2, member type: tie.** Seed-paired 4-member ensembles: `nogain` − `aug`
>   = +0.09 on eval mean, ahead in 73% of subsets (bar: +0.10 and 80%).
> - G8 still beats all four paper numbers: dev 97.19 / 92.76 / 94.12, eval
>   mean 94.87 (paper 95.71 / 90.23 / 91.23 / 93.22).
> - **The one gain that holds on both sets is ToyConveyor:** +0.62 dev,
>   +0.72 eval. Eval ToyCar id_05 does not move (86.83 vs 86.81).
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

## Stage A results (dev only, 2026-11-01)

`scores/scorecard_dev.txt`, `systems_dev.csv`, `perid_dev.csv`,
`combos_dev.csv` and `bootstrap_dev.json`. Eval was not computed.

```
dev, E2′     n    AUC    pAUC   mAUC   mean  |  pump  ToyCar  ToyConv   Δ mean vs A8 [paired 95%]
G8 (pick)    8   97.19  92.76  94.12  94.97  |  96.13  98.37   89.88    +0.20 [+0.10, +0.31]
GA16        16   97.17  92.68  94.18  94.93  |  96.08  98.48   89.79    +0.16 [+0.10, +0.21]
F20g        20   96.69  91.95  92.91  94.32  |  95.55  97.78   88.32    −0.45 [−0.57, −0.33]
A8 (bar)     8   97.05  92.48  94.10  94.77  |  96.00  98.54   89.26
F20         20   96.64  91.83  92.91  94.24  |  95.52  97.88   88.07    −0.53 [−0.64, −0.42]
relu8        8   96.16  91.12  91.52  93.64  |  95.45  96.83   86.40
paper, one model 95.71  90.23  91.23         |  96.25  96.55   83.56

4-member families over the 70 seed subsets: mean [5%, 95%]
hom4_nogain      AUC 97.07 [96.96, 97.18]   mean 94.83 [94.67, 94.99]
hom4_aug         AUC 96.93 [96.84, 97.02]   mean 94.62 [94.52, 94.72]
paired4 (G − A)  AUC +0.14 [+0.02, +0.28]   mean +0.20 [+0.03, +0.38]   nogain ahead in 97% of subsets
```

| | prediction | observed | verdict |
|---|---|---|---|
| P0 | A8, F20, relu8 and `hom4_aug` reproduce `../mixed-ensemble` to 0.01 | 97.05 / 92.48 / 94.10, 96.64, 96.16; 96.93 / 94.62 | holds |
| P1 | G8 dev AUC in [97.05, 97.45], mean in [94.85, 95.25] | 97.19, 94.97 | holds (both a little below central) |
| P2 | G8 ≥ A8 on AUC, pAUC and mAUC | +0.14, +0.28, +0.02 | holds; mAUC by 0.02 [−0.22, +0.33] |
| P3 | GA16 dev mean strictly between A8 and G8 | 94.93 (94.77 – 94.97) | holds, nearer G8 than the midpoint |
| P4 | F20g 0.3–0.7 below G8 | −0.65 | holds |
| P5 | the dev pick is G8 | G8 | holds. GA16 is within 0.05 (−0.04), so the tie rule picks G8 (smaller n) |
| P6 | `paired4` mean Δ in [+0.10, +0.40], ≥ 85% of subsets positive | +0.20, 97% | holds |
| P7 | G8 ToyConveyor ≥ A8 + 0.5; the gain is mostly ToyConveyor | +0.62; next largest fan +0.22 | holds |

What dev shows:

- **The single-model dev edge survives ensembling, shrunk.** Single models:
  +0.27 on dev mean. 8-member E2′: +0.20, with a paired-bootstrap interval
  clear of 0. Every 4-member subset but two of 70 favours `nogain`.
- **mAUC does not move.** +0.02 for G8 and +0.04 for `paired4`, both with
  intervals across 0. The gain is AUC and pAUC.
- **ToyConveyor carries it** (+0.62). ToyCar goes slightly the other way
  (−0.17). Dev ToyCar id_01 is 93.44 for G8 against A8's 94.35. That is
  the same direction as single `ao_nogain` models (−0.40 on id_01 in
  `../aug-ops`), a little of `aug`'s dev ToyCar gain given back.
- **Mixing behaves as for `aug`.** F20g sits 0.65 below G8, just as F20 sat
  0.53 below A8: plain, gate and eca dilute the augmented models on dev.

## Addendum: stage B (eval) predictions, written after stage A and before any eval score

References on eval (`../mixed-ensemble`):

- A8: mean **94.78**; ToyCar 96.00, ToyConveyor 88.97, pump 98.40.
- F20: mean 94.96.

Single-model S1′ eval means are `ao_nogain` 93.88 and `la_aug` 93.82 (+0.06).
On dev, the ensemble kept about 3/4 of the single-model edge (+0.20 of +0.27).

- **B1, G8 (the dev pick).** G8 − A8 on eval mean is in [−0.20, +0.30],
  central +0.05. The paired 95% interval includes 0, so **D1 fails**. About
  75%. (This sharpens Q1; the interval width on dev was ±0.10, and eval is
  noisier.)
- **B2, D2.** Eval `paired4` mean Δ is in [−0.15, +0.20], central +0.04,
  with 35–75% of subsets positive. So D2 reads **"tie"**. About 70%.
- **B3 (Q2).** G8's eval ToyCar AUC is within ±0.6 of A8's 96.00, and eval
  ToyCar id_05 is below 85 for both. G8 gave back 0.9 of dev id_01. If the
  dev id_01 / eval id_05 mirror from `../aug-ops` holds, G8's id_05 is above
  A8's, but by less than 2 points.
- **B4.** G8's eval ToyConveyor is within ±0.8 of A8's 88.97. The +0.62 on dev
  does not carry over: single `ao_nogain` models were −0.19 there on eval.
- **B5 (Q3).** F20g's eval mean is at least G8's and within ±0.15 of F20's
  94.96. Mixing repairs eval ToyCar here as it did for F20: F20g eval ToyCar
  ≥ 97.3.
- **B6.** GA16's eval mean is within 0.10 of the mean of G8's and A8's.

## Stage B results (eval, 2026-11-01)

`scores/scorecard_eval.txt`, `systems_eval.csv`, `perid_eval.csv`,
`combos_eval.csv` and `bootstrap_eval.json`. The run started after the
stage-B predictions were committed (`825e201`). A8, F20 and relu8 reproduce
`../mixed-ensemble`'s eval numbers exactly (94.78, 94.96, 94.46).

```
eval, E2′    n    AUC    pAUC   mAUC   mean  |  pump  ToyCar  ToyConv  | id_05   Δ mean vs A8 [paired 95%]
G8 (pick)    8   96.96  92.77  93.35  94.87  |  98.31  95.96   89.69   | 86.83   +0.09 [−0.05, +0.23]
GA16        16   96.97  92.78  93.39  94.87  |  98.37  96.03   89.49   | 86.96   +0.10 [+0.02, +0.17]
F20g        20   97.15  92.89  94.30  95.02  |  98.58  97.68   88.49   | 92.28   +0.24 [+0.07, +0.41]
A8 (bar)     8   96.88  92.67  93.26  94.78  |  98.40  96.00   88.97   | 86.81
F20         20   97.09  92.83  94.25  94.96  |  98.59  97.68   88.16   | 92.21   +0.19 [+0.03, +0.34]
relu8        8   96.75  92.18  93.62  94.46  |  98.54  97.55   86.63   | 91.59
paper, one model                        93.22

4-member families over the 70 seed subsets: mean [5%, 95%]
hom4_nogain      mean 94.71 [94.41, 95.00]
hom4_aug         mean 94.63 [94.32, 94.93]
paired4 (G − A)  mean +0.09 [−0.12, +0.31]   nogain ahead in 73% of subsets; ToyConveyor +0.74
```

### Decisions

- **D1, new reported ensemble: fails.** G8's dev AUC, pAUC and mAUC are each
  at least A8's, but its eval gain is +0.09 [−0.05, +0.23], an interval that
  includes 0. **E2′[A8] stays the reported best ensemble.** G8 is the
  better system on dev only. On eval it is level with A8, and no metric's
  interval clears 0 (AUC +0.08 [−0.01, +0.18]).
- **D2, which member type ensembles better: tie.** `paired4` eval mean Δ is
  +0.09, positive in 73% of subsets. The bar was +0.10 and 80%; both are
  missed narrowly. On dev the same comparison was +0.20 and 97%.
- **D3, secondary (no claims).**
  - **F20g is the highest eval system here:** 95.02, +0.24 [+0.07, +0.41] over
    A8. F20 is +0.19 [+0.03, +0.34]. Neither was the dev pick: both lose about
    0.5 to A8 on dev. Picking either now would be picking on eval.
  - **GA16's interval clears 0** (+0.10 [+0.02, +0.17]), but its point gain is
    the same as G8's. Its interval is narrower because half its members are
    A8's own, so the paired difference varies less. It was not the pick.
  - **Swapping `aug` for `nogain` inside the mix** (F20g − F20) gains +0.06
    on eval and +0.08 on dev.

### Predictions, checked as written

| | prediction | observed | verdict |
|---|---|---|---|
| B1 | G8 − A8 eval mean in [−0.20, +0.30], central +0.05; interval includes 0, D1 fails | +0.09 [−0.05, +0.23]; fails | holds |
| B2 | `paired4` eval Δ in [−0.15, +0.20], 35–75% positive; D2 "tie" | +0.09, 73%; tie | holds, at the edge of the 75% bound |
| B3 | G8 eval ToyCar within ±0.6 of A8's 96.00 | 95.96 | holds |
| | eval id_05 below 85 for both | 86.83 and 86.81 | **fails**: the bound came from single models (81–83). E2′ lifts id_05 about 5 points for both |
| | G8's id_05 above A8's, by less than 2 | +0.03 | holds, but only nominally: no mirror effect is visible |
| B4 | G8 eval ToyConveyor within ±0.8 of A8's | +0.72 | holds by the number |
| | the dev +0.62 does not carry over | it does: +0.72 eval, `paired4` +0.74 | **fails**: the reasoning was wrong |
| B5 | F20g eval ≥ G8 and within ±0.15 of F20; F20g ToyCar ≥ 97.3 | 95.02 vs 94.87; +0.06; 97.68 | holds |
| B6 | GA16 within 0.10 of mean(G8, A8) = 94.83 | 94.87 | holds |

### What the eval set shows

- **The dev gain shrinks by about half on eval** (+0.20 → +0.09). For single
  models it vanished (+0.27 → +0.06). Ensembling keeps more of it, but not
  enough to clear the bar.
- **ToyConveyor is the one real gain, and it holds on both sets.** G8 is
  +0.62 on dev and +0.72 on eval; `paired4` is +0.63 and +0.74. That
  matches `../aug-ops`: noise carries ToyConveyor, and without gain the
  noise-plus-shift models get a little more of it. Every other machine moves
  by 0.1 or less on eval, and four of them slightly down. So the eval mean
  gain is ToyConveyor's +0.72 spread over six machines, minus small losses
  elsewhere.
- **ToyCar is unchanged.** Eval id_05 is 86.83 for G8 and 86.81 for A8. G8's
  dev id_01 give-back (−0.91) did not buy anything on eval id_05. So no mirror
  is visible at the ensemble level. Mixing in non-augmented models is still
  the only repair: F20g reaches 92.28 on id_05.

## Figures

| figure | what it shows |
|---|---|
| `plots/fig1_ladder.png` | every system's mean(AUC, pAUC) against A8 on dev and eval, with paired-bootstrap 95% intervals. G8 leads on dev (+0.20) and ties on eval (+0.09, interval includes 0) |
| `plots/fig2_paired.png` | 4-member ensembles over all 70 seed subsets: `nogain` vs `aug`, and the seed-paired Δ per metric. 97% of subsets favour `nogain` on dev, 73% on eval |
| `plots/fig3_machines.png` | per-machine AUC minus A8 for G8, GA16, F20g and F20. ToyConveyor is G8's only gain on both sets; mixing (F20g, F20) repairs eval ToyCar |

## What next

1. **Nothing to adopt.** E2′[A8] stays. `ao_nogain` remains the single-model
   recipe (`../aug-ops` C2). As an ensemble member, it ties `aug`.
2. **A fresh-seed test of F20g (or F20)** is the only ensemble lead left. Both
   beat A8 on eval with intervals clear of 0, but both were chosen against dev.
   A claim would need fresh checkpoints and a pre-registration that names F20g
   before any score.
3. **ToyCar id_05** is untouched by anything that keeps the shift op. See
   `../aug-ops` "What next" (smaller or non-circular shift).

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
