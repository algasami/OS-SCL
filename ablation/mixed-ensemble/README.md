# Mixed ensemble — do different kinds of model ensemble better than copies of one?

> **Status: pre-registered 2026-10-28; nothing scored yet.** Stage A (dev)
> runs next; stage B (eval) only after the eval predictions are committed. CPU
> only: every member is already trained and dumped.

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

## What this cannot answer

- **Why `aug` flips ToyCar.** An ensemble can hide that flip, not explain it.
- **Whether the gain survives at a fixed budget.** `F20` and `ALL32` cost 20
  and 32 trainings. The size-matched families are the fair comparison.
- **Fresh checkpoints.** Every member's single-model eval score has been
  seen. The ensembles have not, and no member was chosen by its eval score.
  The `aug` seeds 2044–2047 were trained for `../lit-arms`' replication, not
  for this.
