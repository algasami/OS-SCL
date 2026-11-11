# ensemble-g8: which fresh ensemble do we report?

**Status: PRE-REGISTERED 2026-11-11. Training is running; nothing below the line "Results" exists yet.**
The predictions and decision rules in this file were committed before any of the four new networks
was scored, and before any of their probe lines was read.

## Question

`../final` reports one network (C1ᵀ) and says of the ensemble only that it sits at about 94.9 eval
Amean, and that the `sb_spec` members "are not shown to help on eval". The ensemble will be reported
beside the single network, because N networks cost N times the training but stay cheap at inference.
That needs one named build, and `../fresh-confirm` left the choice open:

- fresh GS8 (4 `ao_nogain` at weight 2 + 4 `sb_spec` at weight 1) beat fresh G4 (4 `ao_nogain`) by
  +0.40 dev but only +0.09 eval, CI [−0.08, +0.24];
- that comparison gives GS8 twice the networks. Count-matched (2 G + 2 S against 4 G) the mix was
  +0.26 on dev (36 of 36 subsets) and −0.09 on eval (8 of 36);
- so the eval gain of GS8 may be the gain of *any* four extra networks. The direct test, eight fresh
  `ao_nogain` networks against the mix, was never run. This study runs it.

## What is trained

Four more `ao_nogain` networks, seeds **2064–2067**, with `../fresh-confirm/run_one.sh`'s `nogain`
recipe flag for flag (`run.sh train_one`): 300 epochs, fast recipe, `--lean 1`,
`--wave_aug 1 --wave_aug_ops noise,shift`, `--num_workers 8`. They are named `fc_nogain_s2064…67` and
live beside the first four. Nothing else is trained; the four fresh `sb_spec` networks and the three
phase-1 BEATs networks are reused as they are.

No seed is dropped, replaced or rerun after scoring. If a run cannot be finished (three crash-resume
attempts fail) the study stops and reports that; G8f is not rebuilt from seven networks.

## Systems

All are Σ S1′ over members (`../grand-ensemble/score.py: ensemble_from_dumps`), the rule frozen in
phase 1. No rule, weight or member list is fitted here.

| name | members (weight each) | n | status |
|---|---|---|---|
| G4 | `fc_nogain` 2060–63 (1) | 4 | scored in `../fresh-confirm`; recomputed here and asserted equal |
| G4b | `fc_nogain` 2064–67 (1) | 4 | new; a second draw of G4 |
| **G8f** | `fc_nogain` 2060–67 (1) | 8 | new; phase 1's G8 on fresh seeds |
| **GS8** | `fc_nogain` 2060–63 (2) + `fc_spec` 2060–63 (1) | 8 | scored in `../fresh-confirm`; recomputed, asserted equal |
| **GS12f** | `fc_nogain` 2060–67 (1) + `fc_spec` 2060–63 (1) | 12 | new; GS12's exact recipe (the phase-2 dev pick) |
| X1 | GS8 + BEATs ×3 (1.5) | 11 | external; scored in `../fresh-confirm`; recomputed, asserted equal |
| X1f | GS12f + BEATs ×3 (1.5) | 15 | external; X1's exact recipe |
| XG8 | G8f + BEATs ×3 (1.0) | 11 | external; the same BEATs share (0.375 of the from-scratch weight) |

The three bold systems are the candidates for "the from-scratch ensemble we report".

## Decision rules (fixed before scoring)

Amean = mean over machine types of (pooled AUC + pooled pAUC)/2, as everywhere in `synthesis/`. CIs are
the paired clip bootstrap of `../grand-ensemble` (2000 resamples, seed 0), 2.5–97.5 %.

| id | rule |
|---|---|
| E1 | **Level.** Dev and eval of G4b, G8f, GS12f are reported once each, pooled and per-ID. |
| E2 | **The reported from-scratch ensemble** is the candidate (G8f, GS8, GS12f) with the highest *dev* Amean (exact tie: the earlier in that order). Its eval number is the reported one. Eval never enters the choice. |
| E3 | **`sb_spec` on eval.** "Confirmed" only if the GS12f − G8f eval CI excludes 0 from above. Otherwise the standing wording stays: not shown to help on eval. The count-matched families (8 networks, equal weights, 2 or 4 of them `sb_spec`, every subset, against G8f) are reported with one fixed reading: a *type effect* on a set needs mean Δ > 0 **and** ≥ 75 % of subsets above G8f. |
| E4 | **Simplest build not shown worse.** If E2 picks G8f, it is also this. If E2 picks a mix, the mix is "the recommended recipe" only if its eval CI against G8f excludes 0; otherwise both are printed and G8f (one recipe, no spectrum branch) is named as the recipe for a new build. |
| E5 | **4 → 8 networks.** G8f − G4 and G8f − G4b, dev and eval, with eval CIs. Reported. G4b − G4 is printed as one draw of the seed noise of a four-network ensemble. |
| E6 | **Single network, second replication.** The four new networks under C1ᵀ must clear `../fresh-confirm`'s F1 bars again: dev ≥ 93.88, eval ≥ 93.34 (phase-1 mean − 2·se, 4 against 8 seeds). The eight fresh seeds together replace the n = 4 row in `../final` by annotation, with paper counts per seed for C1ᵀ and R0. |
| E7 | **External.** The system reported "with external pretraining" is the E2 pick + BEATs at the fixed share (G8f → XG8, GS8 → X1, GS12f → X1f). Both X1f and XG8 are printed. It is never mixed into the from-scratch rows. |

What this study cannot say: anything about more than four `sb_spec` networks, about a best *N*, or
about a weighting other than the two frozen ones. The four `fc_spec` networks are the ones
`../fresh-confirm` found weaker than phase 1's (S1′ 93.36 / 92.74 against 94.06 / 93.42), so a null
here is a null for these four networks.

## Predictions (80 % intervals)

Anchors. Seen seeds: G8 94.97 / 94.87, GS12 95.21 / 94.93 (GS12 − G8: +0.24 / +0.06), G4 → G8
+0.13 dev / +0.25 eval. Fresh: G4 94.52 / 94.80, GS8 94.92 / 94.90, X1 95.13 / 95.42; single C1ᵀ
94.02 / 94.23 against 94.26 / 94.10 seen, so fresh networks have run lower on dev and level on eval.

| quantity | dev | eval |
|---|---|---|
| new four, C1ᵀ mean | 94.10 (93.85–94.35) | 94.15 (93.70–94.60) |
| G4b | 94.60 (94.35–94.85) | 94.70 (94.40–95.00) |
| G8f | 94.65 (94.45–94.85) | 94.95 (94.75–95.15) |
| GS12f | 94.95 (94.75–95.15) | 94.97 (94.80–95.15) |
| GS12f − G8f | +0.28 (+0.15 to +0.40) | +0.02 (−0.12 to +0.15) |
| G8f − GS8 | −0.27 (−0.45 to −0.10) | +0.05 (−0.12 to +0.22) |
| G8f − G4 | +0.13 (0.00 to +0.25) | +0.15 (0.00 to +0.30) |
| X1f | 95.15 (95.00–95.30) | 95.45 (95.30–95.60) |
| XG8 | 94.95 (94.75–95.15) | 95.45 (95.25–95.65) |

Predicted outcomes: E2 picks GS12f (0.70; GS8 0.25, G8f 0.05). E3 not confirmed (0.80); count-matched
type effect on dev yes, on eval no. E4 names G8f (0.75). E6 passes (0.90). So the expected headline is
"report GS12f at about 95.0 eval, and say that eight `ao_nogain` networks reach the same eval level".
The outcome that would change the write-up most is a G8f eval clearly *below* GS8: that would mean the
mix does carry eval information after all, and `../final`'s "not shown to help" would have been too
cautious in the other direction.

## Scoring (`score.py`, `run.sh`)

`score.py` adds no rule. It imports `../fresh-confirm/score.py` and uses its machinery unchanged:
single networks through `../single-final`'s `score_member`, ensembles through `ensemble_from_dumps`,
every vector computed twice from separate CRC-checked reads. It refuses to write anything unless G4,
GS8 and X1 come out within 0.0005 Amean of `../fresh-confirm/scores/ensembles.csv`.

`run.sh score` runs it in fresh processes until two outputs are byte-identical (smartlab1's RAM flips
bits) and is run **once**. `run.sh dryrun` runs the same code on seen seeds (`ao_nogain_s2040–43` as
"old", `2044–47` as "new", `sb_spec_s2040–43`) and must reproduce the committed G8, GS12 and
X1[r=0.375] of `../grand-ensemble` and the committed C1 / C1ᵀ of `../single-final`.

`run.sh post` (detached) waits for training, copies `config.json` / `wall.csv` into `scores/<run>/`
and extracts the dumps of the four new runs with `../fresh-confirm/extract.py`; the eight existing
dumps are linked, not re-extracted. `run.sh evalpy` (after `score`) writes `eval.py`'s own CSVs, the
probe trace, the distilled curves and the R0 cross-check.

## Results

*Not scored yet.*
