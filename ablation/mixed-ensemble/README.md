# Mixed ensemble — do different kinds of model ensemble better than copies of one?

> **Status: DONE 2026-10-28.**
> - **D1, mixing helps: fails.** At equal size, a mix of model types ties an
>   all-`aug` ensemble (+0.06 at 4 members, +0.05 at 8).
> - **D2, new best ensemble: passes.** E2′ over the 8 `aug` models (`A8`, the
>   dev pick) beats E2′[relu8] on eval by +0.31 [+0.07, +0.55] and on all
>   three dev numbers. It is now the reported best ensemble: dev
>   97.05 / 92.48 / 94.10, eval mean 94.78 (paper 95.71 / 90.23 / 91.23 /
>   93.22).
> - **Weak categories:** mixing trades ToyConveyor (+`aug`) against eval
>   ToyCar (−`aug`); pump does not move on eval.
>
> CPU only: every member was already trained and dumped.

## Why this exists

The best system so far is `../confirm-seeds`' E2′ ensemble. It sums S1′ (R0
plus a kNN distance, both z-scored on training statistics) over 4 seeds of the
paper's model. It beats all four of the paper's numbers. Its members are copies
of one model trained with different seeds, and averaging them mostly cancels
seed luck.

`../lit-arms` produced models that differ in **where** they are good. Single
model, R0, mean Δ AUC vs `fr_f1`:

| | ToyCar | ToyConveyor | pump |
|---|---|---|---|
| `aug` (8 seeds), dev | +1.9 / +2.1 | +2.0 / +2.6 | +1.1 / +1.0 |
| `aug` (8 seeds), eval | **−2.3 / −1.9** | +2.6 / +2.3 | −0.1 / −0.2 |
| `gate`, eval | −0.3 | +2.9 | +0.1 |
| `eca`, eval | +0.3 | +1.7 | +0.1 |

The weak categories of E2′[N4] (AUC) are ToyConveyor (86.3 dev / 85.9 eval),
pump (95.3 dev, 0.9 below the paper) and ToyCar (96.7 / 97.4). An ensemble
gains most when its members err differently. Here they err differently in
exactly those categories. So the question is:

1. **Size-matched:** at 4 and 8 members, does a mix of model types beat an
   ensemble of one type?
2. **Overall:** does any ensemble of the available models beat E2′[relu8],
   the strongest existing one?
3. **Weak categories:** what happens to ToyConveyor, pump and ToyCar?

The one earlier test of mixing (`../ensemble-backend`, `mix8`) combined two
nearly identical TFgram variants. At equal size it gained −0.02 dev and +0.17
eval AUC (2+2 subsets vs `relu4`). The lit-arms types differ more than that.

## What has already been seen

- **Every member individually,** under R0 and S1′, on dev and eval.
- **Ensembles of the old-recipe paper model:** E1/E2/E2′ over
  `relu4`/`lin4`/`mix8` (`../ensemble-backend`) and over N4/O4/relu8
  (`../confirm-seeds`), on dev and eval.
- **Not seen:** any ensemble containing a fast-recipe model (`fr_f1`, `aug`,
  `gate`, `eca`), on either set.

## Members (32 checkpoints, all 300 epochs, `fussion 1`, `m 0.4`, `ht basic`)

| type | runs | recipe | dump |
|---|---|---|---|
| `plain` | `fr_f1_s2040–2043` (4) | fast | `../lit-arms/results/emb` |
| `aug` | `la_aug_s2040–2047` (8) | fast + waveform augmentation | same |
| `gate` | `la_gate_s2040–2043` (4) | fast + TriStat gate | same |
| `eca` | `la_eca_s2040–2043` (4) | fast + ECA | same |
| `old` | `cs_relu_max_s2028–2031` (= N4), `tfl_relu_max_s2024–2027` (= O4) (8) | old (fp32, non-lean) | `../confirm-seeds/results/emb` |
| `lin` | `tfl_linear_avg_s2024–2027` (4) | old, linear TFgram | `../ensemble-backend/results/emb` |

The dumps come from `../ensemble-backend/extract.py` and are not regenerated.
`score.py` asserts that all three dumps list the same clips in the same order
below `data/`. It checked out at setup (32 dumps, labels and order identical);
only scores were withheld.

## Rule and metrics

**Unchanged from `../ensemble-backend` / `../confirm-seeds`, imported, not
copied:**

- Each member gets S1′.
- An ensemble is E2′, the plain sum of its members' S1′. No weights; nothing
  is fitted on test clips.
- Metrics are eval.py's: AUC, pAUC, mAUC, and mean(AUC, pAUC) as the paper's
  eval number.

**Fixed systems** (one set of members each):

| system | members | n |
|---|---|---|
| `P4` | `plain` | 4 |
| `A8` | `aug` | 8 |
| `PA12` | `plain` + `aug` | 12 |
| `F20` | every fast-recipe model | 20 |
| `ALL32` | everything | 32 |
| `N4` | the adopted E2′[N4] (reference) | 4 |
| `relu8` | E2′[relu8], the bar (reference) | 8 |

**Size-matched families** (mean over every member combination, or 1000
sampled with seed 0 when there are more):

| family | members | combos |
|---|---|---|
| `hom4_plain` | the 4 `plain` | 1 |
| `hom4_aug` | 4 of 8 `aug` | 70 |
| `hom4_old` | 4 of 8 `old` | 70 |
| `mix4_PAGE` | 1 each of plain, aug, gate, eca | 512 |
| `mix4_PGE+P` | 2 plain, 1 gate, 1 eca: a mix without `aug` | 96 |
| `hom8_aug` | the 8 `aug` | 1 |
| `hom8_old` | the 8 `old` | 1 |
| `mix8_PAGE` | 2 each of plain, aug, gate, eca | 1000 of 6048 |

**Uncertainty.** Families report the 5–95% spread over combos, which is
model-choice noise. Each candidate is compared with `relu8` by a paired
clip-bootstrap (2000 stratified resamples, the same clips for both
systems), which measures test-sample noise.

## Workflow

1. **Stage A, dev only:** `python ablation/mixed-ensemble/score.py dev`.
   `score.py` builds an evaluator for one split only, so eval is never
   computed. It writes `scores/*_dev.*`, which are committed.
2. **Pick on dev:** the candidate (`P4`, `A8`, `PA12`, `F20`, `ALL32`) with the
   highest dev mean(AUC, pAUC). A tie within 0.05 goes to the smaller n.
3. **Eval predictions:** an addendum with numeric predictions for stage B,
   committed before stage B runs.
4. **Stage B, eval:** `score.py eval`, then the results, figures and
   decisions.

## Predictions, stage A (dev), written before any ensemble score

- **P0 (reproduction).** `N4` and `relu8` reproduce `../confirm-seeds`' dev
  E2′ (96.05 and 96.16 AUC) to within 0.01. If they don't, stop: the
  dumps or the rule differ.
- **P1.** `hom4_plain` dev AUC is in [95.6, 96.2]. That is the `fr_f1` single
  S1′ (95.27) plus E2′'s 1→4 gain on N4 (+0.69).
- **P2.** `hom4_aug` dev AUC is in [96.4, 97.0]. `aug`'s single S1′ is about
  96.24, and the ensemble gain is smaller on stronger members.
- **P3.** On dev, `mix4_PAGE` sits between the two: below `hom4_aug` by 0.2 to
  0.5 (its aug edge is diluted to 1/4), and above `hom4_plain` by 0.1 to 0.5.
- **P4.** On dev, ToyConveyor and pump are highest in the `aug`-heavy
  families (`hom4_aug`, `hom8_aug`). Their dev gain is `aug`'s, not
  mixing's.
- **P5.** The dev pick is `aug`-heavy: `A8` or `PA12`.

## Predictions, stage B (eval), first draft

The full set comes in the addendum after stage A. These are fixed now:

- **Q1.** On eval, `mix4_PAGE` beats `hom4_aug` on mean(AUC, pAUC). Mixing
  dilutes `aug`'s ToyCar loss and adds gate/eca's ToyConveyor gain.
- **Q2.** Every `aug`-containing system scores lower on eval ToyCar AUC than
  `P4`.

## Decision rules

- **D1, mixing helps** if both hold, on eval mean(AUC, pAUC), combo means:
  - `mix4_PAGE` ≥ max(`hom4_plain`, `hom4_aug`) + 0.20
  - `mix8_PAGE` ≥ `hom8_aug` + 0.20

  If only one size clears its bar, the verdict is "size-dependent", and the
  result is reported but not acted on.
- **D2, new reported ensemble** if the dev-picked candidate meets both
  conditions:
  - its eval mean(AUC, pAUC) beats E2′[`relu8`] with a paired-bootstrap 95%
    interval above 0
  - its dev AUC, pAUC and mAUC are each ≥ `relu8`'s

  It would then be reported next to E2′[N4] as the best system, with its
  member count stated. Otherwise E2′[N4] stays as it is.
- **D3.** Whatever D2 says, every fixed system is reported on eval, and per
  machine for ToyConveyor, pump and ToyCar. These are secondary and carry no
  claims, since picking among them on eval would be test-set selection.

## Stage A results (dev only, 2026-10-28)

`scores/scorecard_dev.txt`, `systems_dev.csv`, `combos_dev.csv` and
`bootstrap_dev.json`. Eval was not computed.

```
dev, E2′              n    AUC    pAUC   mAUC   mean  |  pump  ToyCar  ToyConv
P4 (plain)            4   95.94  90.83  91.35  93.39  |  94.89  96.74   85.77
A8 (aug)              8   97.05  92.48  94.10  94.77  |  96.00  98.54   89.26
PA12                 12   96.84  92.12  93.45  94.48  |  95.66  98.16   88.72
F20                  20   96.64  91.83  92.91  94.24  |  95.52  97.88   88.07
ALL32                32   96.54  91.67  92.57  94.10  |  95.55  97.61   87.70
N4 (adopted)          4   96.05  90.94  91.34  93.49  |  95.33  96.74   86.33
relu8 (bar)           8   96.16  91.12  91.52  93.64  |  95.45  96.83   86.40
paper, one model          95.71  90.23  91.23         |  96.25  96.55   83.56

family (combo mean)   n    AUC    mean   | pump  ToyCar ToyConv
hom4_plain            4   95.94  93.39   | 94.89  96.74  85.77
hom4_aug              4   96.93  94.62   | 95.88  98.48  88.82
hom4_old              4   96.03  93.50   | 95.36  96.78  85.87
mix4_PAGE             4   96.28  93.82   | 95.25  97.54  86.74
mix4_PGE+P            4   95.95  93.42   | 94.99  96.91  85.67
hom8_aug              8   97.05  94.77   | 96.00  98.54  89.26
mix8_PAGE             8   96.41  93.96   | 95.33  97.60  87.23
```

| | prediction | observed | verdict |
|---|---|---|---|
| P0 | `N4`, `relu8` reproduce 96.05 / 96.16 | 96.0505 / 96.1581, and pAUC and mAUC to 4 decimals | holds |
| P1 | `hom4_plain` dev AUC in [95.6, 96.2] | 95.94 | holds |
| P2 | `hom4_aug` in [96.4, 97.0] | 96.93 | holds |
| P3 | `mix4_PAGE` below `hom4_aug` by 0.2–0.5 | by 0.65 | **fails**: more dilution than expected |
| | and above `hom4_plain` by 0.1–0.5 | by 0.34 | holds |
| P4 | ToyConveyor and pump highest in the `aug`-heavy families | `hom8_aug`: 89.26 and 96.00 | holds |
| P5 | the dev pick is `A8` or `PA12` | **`A8`** (dev mean 94.77) | holds |

What dev shows:

- **On dev, the ensemble is `aug`, diluted.** Every system's dev score falls
  roughly in line with its `aug` share: A8 > PA12 > F20 > ALL32, and
  hom4_aug > mix4 > hom4_plain. Mixing adds little beyond that: `mix4_PAGE`
  sits +0.09 above the straight line from `hom4_plain` to `hom4_aug` at 1/4
  `aug`. Without `aug`, mixing gives nothing (`mix4_PGE+P` 95.95 vs
  `hom4_plain` 95.94).
- **`A8` is the strongest dev system yet:** 97.05 / 92.48 / 94.10. That is
  +0.90 AUC over `relu8` [+0.75, +1.04], ToyConveyor 89.26 (+2.9) and
  ToyCar 98.54 (+1.7). Pump reaches 96.00, still 0.25 short of the paper.
- **This is exactly the dev-selection trap `../lit-arms` found.** `aug`'s dev
  edge on ToyCar reverses on eval in all 8 seeds. Averaging cancels noise,
  not a systematic flip. Stage B tests whether `A8`'s lead survives.

## Addendum: stage B (eval) predictions, written after stage A and before any eval score

Reference, E2′[`relu8`] on eval (`../confirm-seeds`): AUC 96.75, pAUC 92.18,
mean **94.46**; ToyCar 97.55, ToyConveyor 86.63, pump 98.54. Single-model S1′
eval means: `plain` 93.41, `aug` 93.82 (8 seeds), `gate` 93.77, `eca` 94.06.
Going from 1 to 4 models under E2′ added about +0.7 to the eval mean on N4.

- **B1, `A8` (the dev pick).** Eval mean in [94.2, 95.0], central 94.6.
  Against `relu8` that is +0.15 central, with a paired 95% interval that
  includes 0. **D2 fails** (confidence moderate, about 65%). `A8`'s eval
  ToyCar is below `relu8`'s by 1 to 2.5. Its ToyConveyor is above by 1.5 to 3.
- **B2 (Q1, sharpened).** `mix4_PAGE` − `hom4_aug` on eval mean is in
  [−0.1, +0.4], central +0.15. **D1 fails at n=4** (it needs +0.20 over the
  better of the two homogeneous families; a coin flip, leaning fail).
  `mix8_PAGE` − `hom8_aug` is in [−0.1, +0.4], central +0.15. That gives D1 overall
  "fails" or "size-dependent".
- **B3 (Q2).** Every `aug`-containing system has lower eval ToyCar AUC than
  `P4`. The gap grows with the `aug` share: `A8` loses most, `mix4`/`mix8`
  about 1/4 as much.
- **B4.** On eval ToyCar, the mixed families beat `hom4_aug`/`hom8_aug` by
  +0.8 to +1.8. On eval ToyConveyor, they are within ±0.6 of the `aug`
  families.
- **B5 (secondary, D3).** On eval, the ranking reverses from dev. `F20` or
  `ALL32` has the highest eval mean among the fixed systems; `A8` does not.
- **B6.** The `old` family stays where `../confirm-seeds` put it: `hom4_old`
  eval mean within 0.15 of N4's 94.34.

## Stage B results (eval, 2026-10-28)

`scores/scorecard_eval.txt`, `systems_eval.csv`, `combos_eval.csv` and
`bootstrap_eval.json`. The run started after the stage-B predictions were
committed (`0325a10`).

```
eval, E2′             n    AUC    pAUC   mAUC   mean  |  pump  ToyCar  ToyConv   Δ mean vs relu8 [paired 95%]
P4 (plain)            4   96.48  91.76  93.24  94.12  |  98.48  97.57   85.93    −0.34 [−0.53, −0.15]
A8 (aug, dev pick)    8   96.88  92.67  93.26  94.78  |  98.40  96.00   88.97    +0.31 [+0.07, +0.55]
PA12                 12   96.96  92.68  93.82  94.82  |  98.46  97.30   88.27    +0.36 [+0.17, +0.54]
F20                  20   97.09  92.83  94.25  94.96  |  98.59  97.68   88.16    +0.50 [+0.35, +0.65]
ALL32                32   97.05  92.74  94.17  94.90  |  98.58  97.71   87.78    +0.43 [+0.32, +0.56]
N4 (adopted)          4   96.56  92.12  93.23  94.34  |  98.51  97.39   85.88
relu8 (bar)           8   96.75  92.18  93.62  94.46  |  98.54  97.55   86.63
paper, one model                                93.22

family (combo mean)   n   mean [5%, 95%]          | pump  ToyCar ToyConv
hom4_plain            4   94.12                   | 98.48  97.57  85.93
hom4_old              4   94.36 [94.22, 94.49]    | 98.51  97.45  86.26
hom4_aug              4   94.63 [94.32, 94.93]    | 98.38  95.87  88.61
mix4_PGE+P            4   94.44 [94.23, 94.66]    | 98.56  97.64  86.55
mix4_PAGE             4   94.68 [94.42, 94.94]    | 98.56  97.61  87.30
hom8_aug              8   94.78                   | 98.40  96.00  88.97
mix8_PAGE             8   94.82 [94.67, 94.98]    | 98.60  97.72  87.66
```

### Decisions

- **D1, mixing helps: fails at both sizes.**
  - 4 members: `mix4_PAGE` − `hom4_aug` = **+0.06** (bar +0.20).
  - 8 members: `mix8_PAGE` − `hom8_aug` = **+0.05**.

  A mix of types beats copies of the plain model (+0.56 at n=4), but no
  more than copies of `aug` do. Without `aug` in it, a mix scores below
  `hom4_aug` (`mix4_PGE+P` 94.44).
- **D2, new reported ensemble: passes.** The dev pick `A8` beats E2′[`relu8`]
  on eval mean by **+0.31 [+0.07, +0.55]** (paired clip-bootstrap). Its dev
  AUC, pAUC and mAUC are each above `relu8`'s: 97.05 / 92.48 / 94.10 vs
  96.16 / 91.12 / 91.52. **E2′[`A8`] is now the reported best ensemble**:
  8 members, one type (`aug`), E2′ rule unchanged. Against the paper's
  single model it scores:
  - dev AUC 97.05 vs 95.71
  - dev pAUC 92.48 vs 90.23
  - dev mAUC 94.10 vs 91.23
  - eval mean 94.78 vs 93.22

  **Where `A8`'s eval gain is and isn't:**
  - It is pAUC: +0.49 [+0.13, +0.87].
  - Eval AUC is +0.14 [−0.02, +0.28], and eval mAUC −0.36 [−0.81, +0.09].
    Both intervals include 0.
  - Its ToyCar is 1.55 below `relu8`'s on eval.
- **D3, secondary (no claims).** On eval, `F20` scores highest of the fixed
  systems: 94.96, +0.50 [+0.35, +0.65] over `relu8`, with no machine below
  `relu8` by more than 0.05. It was not the dev pick, so choosing it now
  would be choosing on eval. It is the obvious candidate for a fresh-seed
  confirmation; see "What next".

### Predictions, checked as written

| | prediction | observed | verdict |
|---|---|---|---|
| B1 | `A8` eval mean in [94.2, 95.0], central 94.6 | 94.78 | holds |
| | vs `relu8` +0.15, interval includes 0, **D2 fails** | +0.31 [+0.07, +0.55], **D2 passes** | **fails**: the gain was larger and clear of 0 |
| | ToyCar 1 to 2.5 below `relu8`, ToyConveyor 1.5 to 3 above | −1.55, +2.34 | holds |
| B2 | `mix4_PAGE` − `hom4_aug` in [−0.1, +0.4], central +0.15; D1 fails at n=4 | +0.06; fails | holds |
| | `mix8_PAGE` − `hom8_aug` in [−0.1, +0.4] | +0.05 | holds |
| B3 | every `aug`-containing system is below `P4` on eval ToyCar | `A8` 96.00, `PA12` 97.30 and `hom4_aug` 95.87 are below 97.57. `F20` 97.68, `ALL32` 97.71, `mix4` 97.61 and `mix8` 97.72 are not | **fails**: mixing fully repairs ToyCar |
| B4 | on eval ToyCar, mixed beats `aug` families by +0.8 to +1.8 | +1.74 (n=4), +1.72 (n=8) | holds |
| | on eval ToyConveyor, mixed within ±0.6 of `aug` | −1.31 (n=4), −1.31 (n=8) | **fails** |
| B5 | `F20` or `ALL32` has the best eval mean; `A8` does not | `F20` 94.96 | holds |
| B6 | `hom4_old` within 0.15 of N4's 94.34 | 94.36 | holds |

### The weak categories (the question that started this)

- **ToyConveyor** is where `aug` pays, on both sets. `A8` reaches 88.97 on
  eval (+2.34 over `relu8`) and 89.26 on dev. Diluting `aug` gives it back
  in proportion: `F20` 88.16, `ALL32` 87.78.
- **ToyCar** is where `aug` costs on eval: `A8` 96.00 vs 97.55. Mixing in
  other models repairs that fully (`F20` 97.68).
- **So mixing is a trade, not a bonus.** Relative to all-`aug`, a mix gains
  about +1.7 on ToyCar and loses about 1.3 on ToyConveyor. Over six
  machines, that nets to roughly zero, which is D1's +0.05. No one model
  type is best at both.
- **Pump does not move on eval:** 98.40 to 98.60 across every system. On dev,
  `A8` lifts pump to 96.00, the closest any system has come to the paper's
  96.25, and still 0.25 short.

## What next

1. **Per-machine routing.** The trade above suggests one: `aug` models for
   ToyConveyor and pump, a mixed ensemble for ToyCar. It would be chosen
   from evidence already seen on eval. So it would need its own
   pre-registration and fresh checkpoints, or an honest "post hoc" label.
2. **Confirm `F20` (or `A8`) on fresh seeds** if an ensemble claim is to be
   published. Every member's single-model eval score had been seen before
   this ablation, though no ensemble's had.
3. **Pump** still needs model-side work. No ensemble here moves eval pump.

## What this cannot answer

- **Why `aug` flips ToyCar.** An ensemble can hide that flip, not explain it.
- **Whether the gain survives at a fixed budget.** `F20` and `ALL32` cost 20
  and 32 trainings. The size-matched families are the fair comparison.
- **Fresh checkpoints.** Every member's single-model eval score has been
  seen. The ensembles have not, and no member was chosen by its eval score.
  The `aug` seeds 2044–2047 were trained for `../lit-arms`' replication, not
  for this.

## Figures

| figure | what it shows |
|---|---|
| `plots/fig1_ladder.png` | every fixed system's mean(AUC, pAUC) on dev and eval against E2′[relu8], with paired-bootstrap intervals on eval. The dev ranking follows the `aug` share; eval reorders it |
| `plots/fig2_families.png` | size-matched families at 4 and 8 members, mean over member combinations with their 5–95% spread. Mixed ties `aug` on eval (D1 fails) |
| `plots/fig3_weak.png` | ToyCar, ToyConveyor and pump AUC for P4, A8, F20, ALL32 and relu8. `aug` wins ToyConveyor, loses eval ToyCar; mixing repairs ToyCar; eval pump flat |
