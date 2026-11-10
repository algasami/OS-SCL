# synthesis/final — the best systems this repository has to offer

> **Status: DRAFT.** Everything below the fresh-seed section is final and read from committed
> scores. The fresh-seed confirmation (`../fresh-confirm`, seeds 2060–2063, rules and
> predictions F1–F6 frozen in d3472f2 before any fresh score) is still training; its rows
> and fig3 appear when `collect.py` finds its `scores/`.

## Answer

Amean = mean over the 6 machines of (AUC + pAUC) / 2, each pooled per machine as `eval.py`
does. The paper reports dev AUC / pAUC / mAUC 95.71 / 90.23 / 91.23 and an eval mean of 93.22.

| track | final system | dev AUC / pAUC / mAUC | dev Amean | eval Amean | vs paper eval |
|---|---|---|---|---|---|
| single network | `ao_nogain` scored by **C1ᵀ**, mean of 8 seeds | 96.61 / 91.91 / 93.35 | 94.26 | 94.10 | +0.88 |
| ensemble, from scratch | **E2′[A8]** (8 `la_aug`), reported best | 97.05 / 92.48 / 94.10 | 94.77 | 94.78 | +1.56 |
| ensemble, from scratch | E2′[GS12] (8 `ao_nogain` + 4 `sb_spec`, equal weights), dev pick, ties A8 | 97.38 / 93.03 / 94.71 | 95.21 | 94.93 | +1.71 |
| **with external pretraining** | **X1** = E2′[GS12] + 1.5·ΣzK′[BEATs ×3] | 97.50 / 93.24 / 95.12 | 95.37 | 95.42 | +2.20 |

- **One network beats every paper number.** Under C1ᵀ, an `ao_nogain` network beats the paper's
  dev AUC, pAUC and mAUC and its eval mean, averaged over 8 seeds. Under `eval.py`'s own score
  (R0), all three dev numbers pass (96.10 / 91.08 / 92.12) but eval only ties (93.14 vs 93.22).
- **From scratch, A8 stays the reported best.** GS12 was picked on dev only, before its eval score
  was seen. It is +0.16 over A8 on eval, but the CI [−0.02, +0.33] spans 0, so by the rule written
  in advance it ties.
- **With BEATs, X1 replaces F1k.** It is +0.17 [+0.02, +0.32] on eval, a thin pass that is mostly
  inherited from swapping A8 for GS12. This number uses external pretraining and never enters a
  from-scratch comparison.

All rows, including the secondary pools, are in `scores/summary.csv`; per-machine rows are in
`scores/machines.csv`.

## Full table

| track | system | members | dev AUC | dev pAUC | dev mAUC | dev Amean | eval AUC | eval pAUC | eval mAUC | eval Amean | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | OS-SCL (paper) | 1 | 95.71 | 90.23 | 91.23 | 92.97 | — | — | — | 93.22 | arXiv:2509.13853 |
| single | R0 = `eval.py` score, mean of 8 | 1 | 96.10 | 91.08 | 92.12 | 93.59 | 95.30 | 90.97 | 90.33 | 93.14 | `ablation/aug-ops` |
| single | C1, mean of 8 | 1 | 96.55 | 91.78 | 93.22 | 94.16 | 96.42 | 91.75 | 92.54 | 94.09 | `single-final` |
| single | **C1ᵀ**, mean of 8 | 1 | 96.61 | 91.91 | 93.35 | 94.26 | 96.45 | 91.76 | 92.61 | 94.10 | `single-final` |
| ensemble | **E2′[A8]** | 8 | 97.05 | 92.48 | 94.10 | 94.77 | 96.88 | 92.67 | 93.26 | 94.78 | `grand-ensemble` |
| ensemble | E2′[G8] | 8 | 97.19 | 92.76 | 94.12 | 94.97 | 96.96 | 92.77 | 93.35 | 94.87 | `grand-ensemble` |
| ensemble | E2′[GS12] (dev pick) | 12 | 97.38 | 93.03 | 94.71 | 95.21 | 97.00 | 92.86 | 93.31 | 94.93 | `grand-ensemble` |
| ensemble, secondary | E2′[NG18] | 18 | 97.38 | 93.04 | 94.67 | 95.21 | 97.02 | 93.10 | 93.35 | 95.06 | `grand-ensemble` |
| ensemble, secondary | E2′[FAST42] (dev rank 8) | 42 | 97.11 | 92.66 | 93.95 | 94.89 | 97.28 | 93.41 | 94.34 | 95.35 | `grand-ensemble` |
| ensemble, secondary | E2′[ALL50] (dev rank 10) | 50 | 97.02 | 92.48 | 93.67 | 94.75 | 97.25 | 93.30 | 94.35 | 95.28 | `grand-ensemble` |
| *external* | F1k = E2′[A8] + ΣzK′[B3] | 11 | 97.23 | 92.79 | 94.79 | 95.01 | 97.21 | 93.30 | 94.20 | 95.25 | `grand-ensemble` (`ablation/beats`) |
| *external* | **X1** = E2′[GS12] + 1.5·ΣzK′[B3] | 15 | 97.50 | 93.24 | 95.12 | 95.37 | 97.32 | 93.53 | 94.26 | 95.42 | `grand-ensemble` |

The paper's dev Amean (92.97) is the mean of its reported AUC and pAUC; the paper reports no eval
AUC, pAUC or mAUC separately.

## How each final system scores a clip

All systems use the EMA weights of networks trained with the fast recipe (`--amp fp16 --lean 1
--cache_dir data/cache`), 300 epochs, `--m 0.4 --fussion 1 --ht basic`.

- **R0** is `eval.py`'s score: the ArcFace cross-entropy of the clip against its own machine ID.
- **C1ᵀ** (single network; `../single-final/scores/final_rule.json`):
  - The score is zR0′ᵀ + zK′ᵀ + zGmean.
  - R0ᵀ is R0 averaged over circular shifts of 0, +1 % and −1 % of the clip.
  - K′ᵀ is the 1-NN cosine distance of the 128-d pre-head embedding to the ID's training clips,
    using the same shifts.
  - Gmean is the 1-NN cosine distance of the time-mean of `mobilefacenet.conv2` (512-d, unshifted).
  - Each term is z-scored per ID on the training clips the network trained on.
- **E2′[·]** (ensemble; `../grand-ensemble/score.py: ensemble_from_dumps`):
  - The score is the weighted sum of each member's S1′ = zR0′ + zK′, with no TTA and no Gmean.
  - In GS12 all 12 members have weight 1 (`../grand-ensemble/scores/dev_pick.json`). Only the
    size-matched fresh build (4 `ao_nogain` + 4 `sb_spec`) uses weights 2 : 1, which keeps GS12's
    8 : 4 type ratio.
- **X1** adds three BEATs members (`ablation/beats`, `bt_arc` s2040–2042) with weight 1.5 each.
  They contribute zK′ only, because their training CE sits at the fp32 floor.

## What this phase found

| study | question | answer |
|---|---|---|
| `single-final` | Does TTA add to C1 for one network? | Yes on dev (+0.10, 8/8 seeds). Flat on eval (+0.01, CI spans 0). **C1ᵀ adopted.** |
| `grand-ensemble` | Best ensemble from ~60 existing members, picked on dev only? | GS12: +0.44 dev, +0.16 eval vs A8. **Tie**, so A8 stays the reported best. Selection optimism +0.07. |
| `grand-ensemble` (external) | Does the GS12 swap carry over to the BEATs fusion? | X1 +0.17 [+0.02, +0.32] eval vs F1k. **Adopted** (thin). |
| `fresh-confirm` | Do the frozen rules hold on seeds nobody has scored (2060+)? | *pending* |

## Caveats

- **Valve, eval only.** Against R0, single-net C1ᵀ gains on 5 of 6 machines on eval but loses
  1.9 Amean points on valve (pAUC 91.70 vs 94.46). On dev, C1ᵀ is better on valve (99.42 vs
  99.23), so this is an eval-only effect and cannot be acted on without selecting on eval.
  - Part of it is the kNN terms: C1 already lost to S1′ on eval valve (`pooled-knn`, −0.35 AUC).
  - TTA adds a little (pooled valve pAUC −0.37). `single-final` found every per-ID valve AUC flat
    or up, so this part is a calibration effect of per-ID z-scoring across IDs, not a detection loss.
  - The ensembles do not lose on valve (GS12 +0.31 vs R0).
- **The diluted pools FAST42 and ALL50 have the highest from-scratch eval** (95.35, 95.28), but
  they rank 8th and 10th on dev. Picking them would be selecting on eval, so they are reported as
  secondary only.
- **Thin margins.** Every ensemble-vs-ensemble difference in this table is under 0.5 Amean. Only
  X1 vs F1k has a bootstrap CI above 0. The phase-1 noise floors still apply: ~1 point of eval
  AUC run to run, mAUC deltas under ~2 points unresolvable, ToyConveyor >4 points.
- **External pretraining** (BEATs, AudioSet) is a separate track throughout.

## Fresh-seed confirmation

*Pending.* Seeds `fc_nogain_s2060–2063` (round 1) and `fc_spec_s2060–2063` (round 2) are training.
Predictions F1–F6 are in `../fresh-confirm/README.md`.

## Reproduce

```bash
bash synthesis/final/run.sh            # collect.py -> scores/, make_plots.py -> plots/
```

`collect.py` reads only committed `scores/` from `ablation/aug-ops`, `synthesis/single-final`,
`synthesis/grand-ensemble` and (when present) `synthesis/fresh-confirm`. Re-scoring a system from
checkpoints is `../fresh-confirm/run.sh extract` + `score` (see that README).

## Figures

| figure | shows |
|---|---|
| ![](plots/fig1_progression.png) | Dev and eval Amean for each adopted system, from the paper to X1. C1ᵀ is +0.9 over the paper's eval mean, the ensembles +1.6 to +1.7, X1 +2.2. R0 only ties the paper. |
| ![](plots/fig2_machines_eval.png) | Eval Amean gain over R0 per machine for C1ᵀ, GS12 and X1. Most of the gain is ToyConveyor and ToyCar. BEATs adds most on ToyCar. The single-net rule loses 1.9 points on valve. |
