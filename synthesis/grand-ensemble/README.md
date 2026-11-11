# grand-ensemble — which ensemble of the whole member pool is best, picked on dev only?

> **Status: DONE 2026-11-09.** Pre-registered at `b5a7dec`; dev pick committed at `ee37d92`,
> before any eval score.
> - **D1, from scratch: tie, so E2′[A8] stays.**
>   - The dev pick is **GS12** = 8 `ao_nogain` + 4 `sb_spec`, all weights 1. On dev it scores
>     95.21, +0.23 over G8 (O1 optimism +0.07, so the honest margin is +0.16).
>   - On eval it scores **94.93**: **+0.16 [−0.02, +0.33]** vs A8 and +0.06 [−0.08, +0.21]
>     vs G8. The CI touches 0, so D1 fails.
> - **D2, with external pretraining: passes the pre-registered bar, thinly.**
>   - X1 = GS12 + 3 BEATs (zK′, w 1.5 each) scores eval **95.42**, **+0.17 [+0.02, +0.32]**
>     vs F1k, and dev 95.37 (+0.36).
>   - It replaces F1k as the best system *with external pretraining*, **pending the
>     fresh-confirm seeds.**
>   - **[review 2026-11-09]** Four reasons to read it as thin (details under *Decisions*):
>     - other bootstrap seeds put the lower bound at +0.018 / +0.008, with P(Δ ≤ 0) ≈ 2%;
>     - the CI covers clip resampling only, not seed or member variance;
>     - every member's eval had been seen;
>     - most of the gap is the GS12-vs-A8 swap that D1 calls a tie.
> - **[review 2026-11-09] F1k's eval Amean is 95.25** (95.2532). Where this README says 95.26,
>   read 95.25.
> - **The dev/eval reversal of phase 1 returns.** Dilute pools rank near the bottom on dev and
>   top on eval: FAST42 95.35 and ALL50 95.28, mostly from eval ToyCar +1.4. They were not dev
>   picks, so this is secondary (D3).
> - **For fresh-confirm:** 4 `ao_nogain` + 4 `sb_spec`, weights 2 : 1 (`scores/dev_pick.json`).
>   On dev, existing members give 95.15, which is +0.33 over 4-G subsets.
>
> CPU only. Every member was already trained and dumped.

## Why this exists

Phase 1 only ever ensembled **one member type at a time**, or the fixed mixes of
`../../ablation/mixed-ensemble`. Mixing tied there at equal size (+0.05). Since then the pool
gained member types that failed as single models but err differently:
- `sb_spec` (`../../ablation/spectrum-branch`): lifts dev pump id_02 by +2.9 (R0) / +4.1 (S1′).
- `tp` (`../../ablation/timepool-head`): better embedding kNN and pump; a different ToyCar profile.
- `pm` (PerMix) and the trunk-soup networks.

So the question is:
- **X0, from scratch:** which ensemble of the existing pool is best when picked on dev only?
  Does it beat E2′[A8] (the reported best, eval Amean 94.78)?
- **X1, with external pretraining:** the X0 pick plus the three BEATs nets. Does it beat F1k
  (eval Amean 95.26 **[95.25]**)?

## Incumbents (all already seen on dev and eval)

| system | members | dev Amean | eval Amean | source |
|---|---|---|---|---|
| E2′[A8] — reported best | 8 `la_aug` | 94.77 | 94.78 | `ablation/mixed-ensemble` |
| E2′[G8] — phase-1 dev pick, tied A8 | 8 `ao_nogain` | 94.97 | 94.87 | `ablation/nogain-ensemble` |
| GA16 | G8 + A8 | 94.93 | 94.87 | `ablation/nogain-ensemble` |
| F20g | G8 + plain/gate/eca | 94.32 | 95.02 | `ablation/nogain-ensemble` (not a dev pick) |
| F1k — with external pretraining | A8 + ΣzK′[B3] | 95.01 | 95.26 **[95.25]** | `ablation/beats` (secondary there) |

## P0: reproduction (done before anything new)

`score.py repro` recomputes the committed systems from the dumps: `scores/reproduction.{txt,csv}`.

- A8, G8, GA16, F20g, F20, relu8 and F1k reproduce on dev and eval to ≤ 4.8e-5 Amean.
- 59 of 60 single-member S1′ records (dev/eval AUC, pAUC) reproduce to ≤ 5.0e-5
  (`scores/members.csv`). Every dump file passes its zip CRC check.
- **One known discrepancy:** `tp_s2041`, eval. Only ToyConveyor S1′ differs: 88.63 here,
  88.14 in `ablation/timepool-head/scores/metrics.csv`, giving eval AUC +0.08 overall.
  - R0 matches, and the dump CRCs pass.
  - `timepool-head`'s own `run_scores` on the same dump gives this study's value, 4 times.
  - So the committed number is most likely a faulty-RAM event during `timepool-head`'s scoring.
    It is eval-only and does not touch selection here.

## Member pool (50 from scratch + 3 external)

Dumps are `../../ablation/ensemble-backend/extract.py` npz files. `tp` uses the
train/dev/eval split dumps of `../../ablation/timepool-head` and is reassembled into meta order
by clip index. Every dump dir's `meta.npz` is asserted identical to lit-arms' below `data/`.

| type | runs | n | single-member S1′ dev Amean (seen) |
|---|---|---|---|
| `G` | `ao_nogain_s2040–2047` | 8 | 94.00 |
| `A` | `la_aug_s2040–2047` | 8 | 93.73 |
| `P` | `fr_f1_s2040–2043` | 4 | 92.64 |
| `Ga` | `la_gate_s2040–2043` | 4 | 92.64 |
| `E` | `la_eca_s2040–2043` | 4 | 92.70 |
| `S` | `sb_spec_s2040–2043` | 4 | 94.06 |
| `T` | `tp_s2040–2043` | 4 | 93.82 |
| `PM` | `pm_s2040–2043` | 4 | 92.62 |
| `TS` | `ts_soup4_s2050`, `ts_soup4_s2051` | 2 | 94.21 |
| `O` | `tfl_relu_max_s2024–2027`, `cs_relu_max_s2028–2031` | 8 | 92.72 |
| `B` (external) | `BEATsASD_bt_arc_s2040–2042` | 3 | 90.80 (zK′) |

- Trunk soup: forks of one trunk are near-copies. Each trunk counts as **one member**, and its
  soup of 4 forks stands in for it. That soup is the 4-fork score ensemble in one network.
- `tfl_linear_avg` (old recipe, linear TFgram) is left out. It was not in the phase-2 pool, and
  `O` already covers the old recipe.

## Rule

- **Member score:** S1′ = zR0′ + zK′ (`ablation/ensemble-backend/score.py`, imported, not copied).
  z is per machine ID, on the member's own training-part clips.
- **BEATs members:** zK′ only. Their training CE sits at the fp32 floor, so z(R0′) explodes.
- **Ensemble:** Σ_member w · score(member). w = 1 unless a candidate says otherwise.
- Metrics are `eval.py`'s. Amean = mean(AUC, pAUC), pooled per machine, averaged over the 6.
- Nothing is fitted on test clips. Dev clips choose among a fixed, finite candidate list.

## X0 candidates (fixed now; per-member weight 1 unless stated)

| name | members | n | seen before? |
|---|---|---|---|
| `G8` | G | 8 | yes (dev and eval) |
| `GA16` | G + A | 16 | yes (dev and eval) |
| `GS12` | G + S | 12 | no |
| `GT12` | G + T | 12 | no |
| `GST16` | G + S + T | 16 | no |
| `GAST24` | G + A + S + T | 24 | no |
| `NG18` | every `ao_nogain`-based member: G + S + T + TS | 18 | no |
| `FAST42` | every fast-recipe from-scratch member: G A P Ga E S T PM TS | 42 | no |
| `ALL50` | FAST42 + O | 50 | no |
| `GREEDY` | type-block forward selection, below | ≤ 50 | no |

`A8` is the bar, not a candidate.

**`GREEDY`** is Caruana ensemble selection with replacement, over member **types** as blocks:
- Blocks are the 10 from-scratch types. A block's score is its type's **mean** member S1′.
  Each step therefore adds one type at equal block weight, whatever the type's member count.
- Start empty. At each step, add the block that maximises dev Amean of the running sum.
  Blocks may repeat, which upweights a type. Ties go to the earlier type in the table above.
- **Stop** at the first step whose best addition gains less than **+0.02** dev Amean. That step
  is not taken. There are at most 10 steps.
- Output: block counts c_t, so per-member weight = c_t / n_t. Its n is the number of distinct
  members used.

## Selection procedure (dev only)

1. `score.py dev` computes every candidate on **dev only**. Eval is never computed by that stage.
2. **X0 pick** = the candidate with the highest dev Amean. A tie within 0.05 goes to the
   smaller n (training cost), then to the earlier row.
3. **X1** (with external pretraining) = X0 pick + the 3 BEATs nets (zK′). The BEATs block's
   total weight is a ratio r of the X0 pick's total weight W. Per-BEATs-member weight =
   r·W/3, with r ∈ {**3/8** (F1k's ratio: 3 zK′ against 8 S1′), **3/4**, **3/2**}.
   - For an 8-member pick this gives w_B = 1, 2, 4.
   - X1 pick = the r with the highest dev Amean. A tie within 0.05 goes to the smaller r.
4. **Selection optimism**, both computed on dev only:
   - **O1, clip split-half.** 20 random 50/50 splits of the dev clips, stratified by machine ×
     ID × label (rng seeds 0–19). On each split, run step 2 (with GREEDY re-run) on half A, and
     score the half-A pick on half B. Optimism = mean over splits of
     [Δ_A(pick − G8) − Δ_B(pick − G8)]. Also report how often each candidate is picked.
     Honest dev margin = full-dev Δ(pick − G8) − optimism.
   - **O2, seed halves.** Split each type's runs in two by seed order: first half H1, second
     half H2 (G: 2040–43 / 2044–47; 4-member types 2+2; TS 1+1; A and O likewise). Run step 2 on
     H1's members only, with candidates rebuilt from H1. Score that pick rebuilt from H2's
     members against G-H2. Then swap the halves. Report Δ on the selection half against the
     other half.
5. Paired clip-bootstrap on dev (2000 stratified resamples, rng 0) of every candidate − G8 and
   − A8. These are descriptive.
6. Write `scores/selection_dev.json` and `scores/dev_pick.json`, then commit them. Eval
   predictions for the specific pick go in as an addendum in that same commit.
7. `score.py eval` refuses to run unless `selection_dev.json` is committed and unmodified. It
   scores eval **once**: every candidate, X1 options and bars, plus the paired bootstraps.

## Size-matched version for fresh-confirm (rule fixed now)

`fresh-confirm` trains 4 fresh `ao_nogain` (2060–2063) and later 4 fresh members of one more
type. `dev_pick.json` states how to build the pick from them:
- Let W_t be the pick's total weight on type t: n_t for fixed candidates, c_t for GREEDY.
- **If the pick contains no type except G** (or G-recipe TS), build 8 G.
- **Otherwise** X is the non-G type with the largest W_t. A tie goes to the type whose removal
  costs the most dev Amean. Build 4 G + 4 X with per-member weights W_G/4 and W_X/4, and drop
  every other type.
- A dev-only check on existing members goes in `selection_dev.json`, descriptive:
  - the reduced 4G+4X form over every 4-of-8 G subset × 4-of-n_X X subset;
  - against 4-G subsets and against G8.

## Predictions (written before any new combination is scored)

Single-member dev S1′ (seen): S ≈ G (94.06 vs 94.05, same seeds), T −0.2, TS +0.2, and A −0.3.
P, Ga, E, PM and O are −1.3 to −1.4. On dev, mixing in the weak types dilutes: F20g 94.32 vs
G8 94.97.

- **Q1, dev Amean of the new fixed candidates** (central [range]):
  - GS12 95.05 [94.90, 95.20]
  - GT12 94.98 [94.85, 95.12]
  - GST16 95.05 [94.90, 95.20]
  - GAST24 94.98 [94.85, 95.12]
  - NG18 95.07 [94.92, 95.22]
  - FAST42 94.55 [94.35, 94.75]
  - ALL50 94.40 [94.20, 94.60]
- **Q2, GREEDY:**
  - It starts with G, takes 2–4 steps and includes S or T. It never takes P, Ga, E, PM or O.
  - Dev 95.12 [95.00, 95.30].
- **Q3, the X0 pick** is GREEDY (60%), otherwise one of GS12 / GST16 / NG18. Its dev margin over
  G8 is +0.10 to +0.25.
- **Q4, optimism:**
  - O1 removes about half of that margin: optimism +0.05 to +0.10. Honest margin over G8 is
    +0.03 to +0.12.
  - The full-dev pick is also the half-A pick in fewer than half of the splits.
- **Q5, X0 pick on eval:**
  - Amean 94.90 [94.65, 95.15]. Δ vs A8 is +0.12, with a paired CI that includes 0.
  - **D1 → tie, A8 stays** (65%).
  - Δ vs G8 is within ±0.15.
- **Q6, X1:**
  - The pick is r = 3/8 or 3/4. Dev is +0.15 to +0.30 over the X0 pick.
  - Eval Amean 95.30 [95.05, 95.55]. Δ vs F1k is +0.05, with a CI that includes 0.
  - **D2 → tie, F1k stays** (70%).
- **Q7, weak spots:**
  - The X0 pick raises dev pump by +0.1 to +0.4 over G8, from S/T.
  - Eval ToyCar id_05 moves within ±1.5 of G8's 86.8.
  - Eval ToyConveyor moves within ±0.8 of G8.

## Decision rules (fixed now)

- **D1, X0 replaces E2′[A8]** as the reported best from-scratch ensemble iff the pick is new,
  its eval Δ Amean vs A8 > 0, and the paired clip-bootstrap 95% CI lower bound is > 0.
  - Otherwise: tie, and A8 stays.
  - If the pick is `G8` or `GA16` (already eval-scored), there is no new eval information.
    D1 then reads the existing record: G8 − A8 = +0.09 [−0.05, +0.23], a tie, and A8 stays.
  - Δ vs G8 is reported next to it and carries no decision.
- **D2, X1 replaces F1k** as the best system with external pretraining iff eval Δ Amean
  (X1 − F1k) > 0 with a paired 95% CI lower bound > 0. Otherwise: tie, and F1k stays.
- **D3, secondary, no claims:**
  - every candidate, the GREEDY trace prefixes and the X1 options are reported on eval,
    overall and per machine;
  - picking among them on eval would be test-set selection.
- **D4, a flag on the pick, not a decision:** if O1's honest dev margin over G8 is ≤ 0, the
  README calls the pick's dev lead selection noise. D1 is unchanged.

Every X1 number is labelled **with external pretraining** and stays out of from-scratch tables.

## Stage A results (dev only, 2026-11-09)

Source: `scores/scorecard_dev.txt`, `systems_dev.csv`, `x1_dev.csv`, `selection_dev.json`,
`bootstrap_dev.json` and `type_corr_dev.csv`. Eval was not computed.

```
dev, E2′      n   AUC   pAUC  mAUC  Amean | fan   pump  slider ToyCar ToyConv valve  Δ vs G8: bootstrap mean [paired 95%]
G8            8  97.19 92.76 94.12 94.97 | 99.04 96.13 99.73 98.37 89.88 99.96
GA16         16  97.17 92.68 94.18 94.93 | 98.98 96.08 99.74 98.48 89.79 99.96   −0.05 [−0.10, +0.01]
GS12  PICK   12  97.38 93.03 94.71 95.21 | 99.21 96.59 99.80 98.41 90.33 99.97   +0.24 [+0.13, +0.36]
GT12         12  97.20 92.76 94.21 94.98 | 98.99 96.38 99.74 98.43 89.68 99.98   +0.00 [−0.05, +0.06]
GST16        16  97.36 93.02 94.68 95.19 | 99.15 96.69 99.79 98.46 90.11 99.99   +0.22 [+0.11, +0.33]
GAST24       24  97.32 92.95 94.58 95.14 | 99.12 96.52 99.79 98.51 90.00 99.98   +0.17 [+0.07, +0.27]
NG18         18  97.38 93.04 94.67 95.21 | 99.15 96.64 99.80 98.44 90.25 99.99   +0.24 [+0.13, +0.35]
FAST42       42  97.11 92.66 93.95 94.89 | 98.94 96.03 99.76 98.37 89.60 99.98   −0.09 [−0.21, +0.04]
ALL50        50  97.02 92.48 93.67 94.75 | 98.90 95.93 99.76 98.21 89.31 99.98   −0.23 [−0.35, −0.10]
GREEDY       14  97.42 93.06 94.75 95.24 | 99.19 96.58 99.83 98.37 90.56 99.98   +0.27 [+0.14, +0.41]
A8 (bar)      8  97.05 92.48 94.10 94.77 | 98.82 96.00 99.74 98.54 89.26 99.95   −0.20
paper, one model 95.71 90.23 91.23       |       96.25             96.55 83.56          (per machine: AUC)
```

**[review 2026-11-09]** The Δ column is the **bootstrap mean** over 2000 resamples, not the point
estimate. GS12 − G8 has a point estimate of **+0.23** (95.2061 − 94.9717 = 0.2343), which is the
figure the text uses. The column shows +0.24 (bootstrap mean 0.236). The other rows differ from
their point estimates by ≤ 0.01.

- **GREEDY** took G, then S (+0.23), then TS (+0.04), and stopped (+G would gain −0.01). That is
  G + S + TS at equal block weight: per-member weights G 1/8, S 1/4, TS 1/2.
- **The pick is GS12.** GREEDY (95.24) leads GS12 (95.21) by 0.03, which is inside the 0.05 tie
  band, so the smaller n (12 < 14) wins.
- **The gain is `sb_spec`'s.** Every candidate with S gains +0.17 to +0.27 over G8, and none
  without it does: GT12 +0.00, GA16 −0.05. Adding the weak types dilutes: FAST42 −0.09,
  ALL50 −0.23.
- **Where the gain is:**
  - dev pump 96.13 → 96.59, now above the paper's 96.25 for the first time. Pump id_02 goes
    92.77 → 95.35.
  - ToyConveyor +0.44, and mAUC +0.59.
- **Selection optimism:**
  - **O1** (20 clip split-halves): the half-A pick leads G8 by +0.28 on its own half and by
    +0.21 on the other. Optimism is +0.07, so the honest dev margin is **+0.16**. The half-A
    pick was GS12 in 12 of 20 splits and GREEDY in 8. GREEDY on half A shrinks most (+0.33 →
    +0.18 on average); GS12 barely shrinks (+0.25 → +0.22 **[+0.23]**).
    - **[review 2026-11-09]** Two corrections, both from `selection_dev.json` O1 splits:
      - GREEDY's held-out value read "+0.17" when committed at `ee37d92`. Commit `9b842f6`
        changed it to +0.18 after eval was scored, without a note. **+0.18 is correct** (mean
        0.182 over its 8 splits). Only the wording changed; no number was recomputed.
      - GS12's held-out mean is 0.226, so **+0.23**, as `fig3_selection.png` shows, not +0.22.
  - **O2** (seed halves): picking on H1 gives GS12, +0.24 on H1 and +0.31 rebuilt on H2.
    Picking on H2 gives GREEDY = G + S + TS + **PM**, +0.42 on H2 and +0.21 rebuilt on H1. The
    S effect replicates across seed halves; the greedy extras do not.
- **The size-matched form holds on dev.** 4 G + 4 S at weights 2 : 1, over all 70 G subsets:
  95.15 [95.07, 95.23]. That is +0.33 over 4-G subsets (94.83) and +0.18 over G8.
- **X1** (with external pretraining): r = 3/8 gives 95.37, r = 3/4 95.30, r = 3/2 95.09.
  F1k is 95.01. The pick is r = 3/8, i.e. w = 1.5 per BEATs member against 12 S1′ members.
  On dev it leads F1k by +0.36 [+0.25, +0.48] and GS12 by +0.16 [+0.07, +0.26].

### Stage A predictions, checked

| | prediction | observed | verdict |
|---|---|---|---|
| Q1 | GS12 95.05 [94.90, 95.20] | 95.21 | **fails** (0.01 above) |
| | GT12 94.98 [94.85, 95.12] | 94.98 | holds |
| | GST16 95.05 [94.90, 95.20] | 95.19 | holds |
| | GAST24 94.98 [94.85, 95.12] | 95.14 | **fails** (0.02 above) |
| | NG18 95.07 [94.92, 95.22] | 95.21 | holds |
| | FAST42 94.55 [94.35, 94.75] | 94.89 | **fails**: less dilution than predicted |
| | ALL50 94.40 [94.20, 94.60] | 94.75 | **fails**: less dilution than predicted |
| Q2 | GREEDY starts with G, takes 2–4 steps, includes S or T, never a weak type | G, S, TS (3 steps) | holds |
| | GREEDY dev 95.12 [95.00, 95.30] | 95.24 | holds |
| Q3 | pick GREEDY (60%), otherwise GS12 / GST16 / NG18 | GS12 | the alternative holds |
| | margin over G8 +0.10 to +0.25 | +0.23 | holds |
| Q4 | O1 optimism +0.05 to +0.10 | +0.07 | holds |
| | honest margin +0.03 to +0.12 | +0.16 | **fails**: the gain is sturdier than predicted |
| | full-dev pick is the half-A pick in fewer than half of the splits | 12 / 20 | **fails** |
| Q7 | dev pump +0.1 to +0.4 over G8 | +0.46 | **fails** (slightly above) |

T added nothing on dev (GT12 +0.00), against the expectation that `tp`'s better kNN and pump
would help. The diversity that pays is `sb_spec`'s.

## Addendum: stage B (eval) predictions, written after stage A and before any eval score

Seen single-member S1′ on eval, 4 paired seeds (`scores/members.csv`):
- `sb_spec` − `ao_nogain` = −0.31 eval Amean;
- by machine: fan +0.2, pump +0.3, slider −0.7, ToyCar +0.15, ToyConveyor 0.0, valve −0.6.

Eval references: G8 94.87, with pump 98.31, ToyCar 95.96 (id_05 86.83) and ToyConveyor 89.69.
A8 is 94.78, and F1k is 95.26 **[95.25]**.

- **B1, GS12 on eval:**
  - Amean 94.93 [94.70, 95.15].
  - vs A8: +0.15, with a paired CI of about [−0.05, +0.35]. **D1 → tie, A8 stays** (60%).
  - vs G8: +0.06, with a CI that includes 0.
- **B2, per machine vs G8 (AUC):**
  - pump +0.1 to +0.4;
  - slider and valve −0.3 to 0;
  - ToyConveyor within ±0.6;
  - ToyCar within ±0.4, and id_05 within ±1.5 of 86.83.
- **B3, the dev lead shrinks.** Δ(GS12 − G8) on eval is less than half the dev +0.23. S's
  single-model eval deficit and the dev/eval pattern of phase 1 both point that way.
- **B4, X1 = GS12 + B3 (r = 3/8), with external pretraining:**
  - eval Amean 95.38 [95.10, 95.60];
  - vs F1k: +0.12, with a CI that includes 0. **D2 → tie, F1k stays** (60%).
  - BEATs adds +0.3 to +0.6 to GS12 on eval, mostly ToyCar.
- **B5, secondary.** GREEDY and NG18 score within ±0.10 of GS12 on eval. FAST42 and ALL50 rank
  higher on eval than on dev (phase 1's F20g pattern).

## Stage B results (eval, scored once, 2026-11-09)

Source: `scores/scorecard_eval.txt`, `systems_eval.csv`, `x1_eval.csv`, `greedy_trace.csv`,
`bootstrap_eval.json`, `decisions.json` and `type_corr_eval.csv`. The run started after
`ee37d92`. `score.py eval` checks that `selection_dev.json` and `dev_pick.json` are committed
and unmodified.

```
eval, E2′     n   AUC   pAUC  mAUC  Amean | fan   pump  slider ToyCar ToyConv valve  Δ vs A8: point [paired 95%]   dev
G8            8  96.96 92.77 93.35 94.87 | 99.64 98.31 99.77 95.96 89.69 98.40   +0.09 [−0.05, +0.23]   94.97
GA16         16  96.97 92.78 93.39 94.87 | 99.68 98.37 99.75 96.03 89.49 98.46   +0.10 [+0.02, +0.17]   94.93
GS12  PICK   12  97.00 92.86 93.31 94.93 | 99.80 98.47 99.76 95.98 89.64 98.36   +0.16 [−0.02, +0.33]   95.21
GT12         12  96.89 92.82 93.15 94.85 | 99.65 98.37 99.79 95.66 89.40 98.46   +0.08 [−0.05, +0.21]   94.98
GST16        16  96.95 92.95 93.22 94.95 | 99.79 98.47 99.78 95.76 89.44 98.47   +0.18 [+0.01, +0.33]   95.19
GAST24       24  96.97 92.97 93.31 94.97 | 99.78 98.45 99.77 95.88 89.37 98.54   +0.19 [+0.08, +0.30]   95.14
NG18         18  97.02 93.10 93.35 95.06 | 99.79 98.45 99.79 95.86 89.61 98.64   +0.28 [+0.12, +0.43]   95.21
FAST42       42  97.28 93.41 94.34 95.35 | 99.78 98.51 99.79 97.32 89.44 98.86   +0.57 [+0.43, +0.70]   94.89
ALL50        50  97.25 93.30 94.35 95.28 | 99.76 98.54 99.74 97.46 89.10 98.90   +0.50 [+0.35, +0.64]   94.75
GREEDY       14  97.14 93.23 93.54 95.19 | 99.81 98.45 99.78 96.13 89.87 98.81   +0.41 [+0.21, +0.60]   95.24
A8 (bar)      8  96.88 92.67 93.26 94.78 | 99.71 98.40 99.72 96.00 88.97 98.49                          94.77
paper, one model                   93.22                                                    (per machine: AUC)

WITH EXTERNAL PRETRAINING           AUC   pAUC  mAUC  Amean | ToyCar ToyConv   Δ vs F1k [paired 95%]   dev
X1[r=3/8]  PICK  GS12 + B3 w 1.5   97.32 93.53 94.26 95.42 | 97.67  89.60     +0.17 [+0.02, +0.32]    95.37
X1[r=3/4]        GS12 + B3 w 3     97.35 93.57 94.40 95.46 | 98.13  89.36     +0.21 [+0.07, +0.36]    95.30
X1[r=3/2]        GS12 + B3 w 6     97.29 93.52 94.38 95.40 | 98.51  88.76     +0.15 [−0.01, +0.32]    95.09
F1k (bar)        A8 + B3 w 1       97.21 93.30 94.20 95.25 | 97.72  88.99                             95.01
```

### Decisions

- **D1, X0 replaces E2′[A8]: fails, so it is a tie.**
  - GS12 is new, and its eval Δ vs A8 is +0.16, but the paired 95% CI is [−0.02, +0.33]:
    the lower bound is not above 0.
  - **E2′[A8] stays the reported best from-scratch ensemble.**
  - Against G8, GS12 is +0.06 [−0.08, +0.21]. The +0.23 dev lead (honest +0.16) shrank to a
    quarter on eval.
  - GS12 still beats all four paper numbers: dev 97.38 / 93.03 / 94.71, eval 94.93 (paper
    95.71 / 90.23 / 91.23 / 93.22).
  - It is the first system above the paper's dev pump (96.59 vs 96.25).
- **D2, X1 replaces F1k: passes.** X1 = GS12 + 3 BEATs nets at r = 3/8 (w 1.5 per BEATs
  member, zK′).
  - Eval Amean 95.42, +0.17 [+0.02, +0.32] vs F1k. Dev 95.37, +0.36 vs F1k.
  - **It is the new best system with external pretraining.** It needs fresh-seed confirmation
    like every phase-2 system. Its margin is small, and the lower CI bound sits at +0.02.
  - BEATs adds +0.49 [+0.36, +0.64] to GS12 on eval, mostly ToyCar (95.98 → 97.67).
  - **[review 2026-11-09] Read D2 as "passes the pre-registered bar, thinly; pending
    fresh-confirm".**
    - *Thin.* An independent rerun of the bootstrap with rng 1 and rng 2 gives lower bounds
      +0.018 and +0.008. P(Δ ≤ 0) is about 2%.
    - *Clip noise only.* The CI resamples test clips. It does not cover seed or member
      variance, which phase 1 puts at about 1 point of single-model eval AUC.
    - *Seen members.* Every member's single-model eval score had been seen in phase 1. Only
      the combination was new.
    - *Mostly inherited from D1.* X1 − F1k (+0.17) is largely the GS12-vs-A8 swap in the
      from-scratch block (+0.16), which D1 calls a tie. The BEATs block is the same in both.
      Per machine on eval (Amean), X1 − F1k is fan +0.25, pump +0.19, slider +0.15,
      **ToyCar +0.06**, ToyConveyor +0.52 and valve −0.14. So the X1 gain over F1k is fan,
      pump and ToyConveyor, not BEATs' ToyCar.
- **D4: no flag.** The honest dev margin over G8 is +0.16 > 0.
- **D3, secondary (no claims; choosing among these on eval would be test-set selection):**
  - FAST42 has the highest eval Amean of all from-scratch candidates: 95.35, +0.57 vs A8. It
    is 8th of 10 on dev.
  - ALL50 is 95.28, and GREEDY 95.19.
  - Their gain is eval ToyCar (97.3–97.5 vs 96.0) and valve. The weak types (plain, gate, eca,
    PerMix, old) repair eval ToyCar exactly as `mixed-ensemble`'s F20 did, and dev never sees it.
  - GREEDY's eval jump comes at its soup step: @2 (G + S) 94.83 → @3 (+ soup) 95.19. Both
    trunks drew lucky eval seeds (`ablation/trunk-soup`), and they carry half-weight per
    member here.

### Stage B predictions, checked

| | prediction | observed | verdict |
|---|---|---|---|
| B1 | GS12 eval 94.93 [94.70, 95.15] | 94.93 | holds |
| | vs A8 +0.15, CI ≈ [−0.05, +0.35] → D1 tie | +0.16 [−0.02, +0.33], tie | holds |
| | vs G8 +0.06, CI includes 0 | +0.06 [−0.08, +0.21] | holds |
| B2 | pump +0.1 to +0.4 (AUC vs G8) | +0.16 | holds |
| | slider and valve −0.3 to 0 | −0.01, −0.04 | holds |
| | ToyConveyor within ±0.6 | −0.05 | holds |
| | ToyCar within ±0.4, and id_05 within ±1.5 of 86.83 | +0.02; 86.91 | holds |
| B3 | eval Δ vs G8 less than half the dev +0.23 | +0.06 | holds |
| B4 | X1 eval 95.38 [95.10, 95.60] | 95.42 | holds |
| | vs F1k +0.12, CI includes 0 → D2 tie | +0.17 [+0.02, +0.32] → **passes** | **fails** |
| | BEATs adds +0.3 to +0.6 to GS12, mostly ToyCar | +0.49, ToyCar +1.69 AUC | holds |
| B5 | GREEDY and NG18 within ±0.10 of GS12 | +0.26, +0.13 | **fails** |
| | FAST42 and ALL50 rank higher on eval than on dev | 1st and 2nd (dev 8th and 10th) | holds |

## What this means

- **`sb_spec` is the only member type that adds on dev, and on eval it adds only to fan and
  pump.**
  - It is the least correlated strong type in the pool (fig2): about 0.90 with `ao_nogain`.
    `tp`, the soups and `la_aug` sit at 0.98–0.99, and are near-copies as ensemble members.
  - On dev, GS12 gains fan, pump (id_02 +2.6) and ToyConveyor. On eval, fan and pump hold
    (+0.4 / +0.3 Amean). ToyConveyor reverses (−0.2), and so do slider and valve.
- **Dev cannot see the eval ToyCar effect.** It decides every eval ranking here:
  - Members that are weak on dev but different (plain, gate, eca, pm, old) lift eval ToyCar
    by 1.4 AUC when pooled, and lose 0.1–0.3 on dev.
  - Phase 1 saw this as F20 / F20g; here it is FAST42 / ALL50.
  - A dev-only rule will keep missing it. Only a ToyCar-specific mechanism, or BEATs (+1.7 on
    eval ToyCar), reaches it.
- **Selection optimism was small and measurable.** O1 says +0.07 of the +0.23 dev lead is
  selection. The rest, +0.16, is real on dev but mostly does not transfer to eval (+0.06).
  The dev→eval gap is a domain shift, not selection noise. Split-half on dev cannot measure it.

## What does not replicate / limits

- **Fresh seeds.** Every member's single-model eval score had been seen in phase 1. Only the
  combinations were new. The fresh-confirm study tests the size-matched GS12 (4 G + 4 S, 2 : 1)
  on seeds 2060+.
- **The X1 margin over F1k is thin**, with a CI lower bound of +0.02. The r = 3/4 option did
  better on eval (+0.21), but it was not the dev pick, and reporting it would be eval
  selection.
- **`tp_s2041`'s committed timepool-head eval record is off by +0.08 AUC** in ToyConveyor
  (see P0). This study uses the CRC-verified dump.
- One soup per trunk stood in for 8 forks. The 4 forks of a trunk are near-copies, so a fork
  ensemble would be about the same member.

## Figures

| figure | what it shows |
|---|---|
| `plots/fig1_candidates.png` | every candidate's Amean on dev and eval vs A8 (dashed) and G8 (dotted), with paired 95% CIs on eval. GS12 is the dev pick, a tie on eval; FAST42/ALL50 reverse from dev to eval |
| `plots/fig2_diversity.png` | between-type score correlation (type-mean S1′ within machine ID, averaged), dev and eval. `sb_spec` is the least correlated strong type; BEATs is far from everything |
| `plots/fig3_selection.png` | GREEDY's dev trace with its eval prefixes (descriptive), and O1's 20 split-halves: selection-half vs held-out Δ over G8, by half-A pick |
| `plots/fig4_machines.png` | per-machine Δ Amean, dev vs eval: GS12 − G8 (from scratch) and X1 − F1k (with external pretraining) |

## Files

| file | role |
|---|---|
| `score.py` | loaders, the adopted ensemble rule (`ensemble_from_dumps`, importable), stages `repro`, `dev`, `eval` |
| `run.sh` | `repro`, `dev`, `eval` and `plots`; CPU only, 4 BLAS threads (the GPUs' dataloaders share the CPUs) |
| `make_plots.py` | figures from `scores/` only |
| `scores/reproduction.{txt,csv}`, `scores/members.csv` | P0, plus single-member S1′ dev/eval for the whole pool (all seen in phase 1) |
| `scores/selection_dev.json`, `scorecard_dev.txt`, `systems_dev.csv`, `x1_dev.csv`, `bootstrap_dev.json` | stage A: candidates, GREEDY trace, picks, O1/O2, size-matched dev check |
| `scores/dev_pick.json` | the X0/X1 picks and the size-matched build for fresh-confirm (weights, recipe, how to score) |
| `scores/scorecard_eval.txt`, `systems_eval.csv`, `x1_eval.csv`, `greedy_trace.csv`, `bootstrap_eval.json`, `decisions.json` | stage B, scored once |
| `scores/type_corr_{dev,eval}.csv` | between-type correlation (diversity diagnostic) |
