# grand-ensemble — which ensemble of the whole member pool is best, picked on dev only?

> **Status: PRE-REGISTERED 2026-11-09.** Reproduction check done (P0 passes). No new combination
> has been scored on dev or eval yet. Candidates, selection procedure, predictions and decision
> rules below are fixed before any of that.
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
  (eval Amean 95.26)?

## Incumbents (all already seen on dev and eval)

| system | members | dev Amean | eval Amean | source |
|---|---|---|---|---|
| E2′[A8] — reported best | 8 `la_aug` | 94.77 | 94.78 | `ablation/mixed-ensemble` |
| E2′[G8] — phase-1 dev pick, tied A8 | 8 `ao_nogain` | 94.97 | 94.87 | `ablation/nogain-ensemble` |
| GA16 | G8 + A8 | 94.93 | 94.87 | `ablation/nogain-ensemble` |
| F20g | G8 + plain/gate/eca | 94.32 | 95.02 | `ablation/nogain-ensemble` (not a dev pick) |
| F1k — with external pretraining | A8 + ΣzK′[B3] | 95.01 | 95.26 | `ablation/beats` (secondary there) |

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

## Files

| file | role |
|---|---|
| `score.py` | loaders, the adopted ensemble rule (`ensemble_from_dumps`, importable), stages `repro`, `dev`, `eval` |
| `run.sh` | `repro`, `dev`, `eval` and `plots`; CPU only, 4 BLAS threads (the GPUs' dataloaders share the CPUs) |
| `make_plots.py` | figures from `scores/` only |
| `scores/reproduction.{txt,csv}`, `scores/members.csv` | P0, plus single-member S1′ dev/eval for the whole pool (all seen in phase 1) |
