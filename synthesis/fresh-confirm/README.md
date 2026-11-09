# fresh-confirm — do the final systems hold on seeds nobody has seen?

> **Status: TRAINING.** Round 1 `fc_nogain_s2060–2063` launched 2026-11-09 16:31; round 2
> `fc_spec_s2060–2063` queued 2026-11-09 behind it (set by `../grand-ensemble`'s dev pick GS12,
> `scores/dev_pick.json` at ee37d92, committed before that pick's eval score).
> No fresh checkpoint has been scored under any rule. Scoring waits for
> `../single-final` and `../grand-ensemble` to commit their final rules.

## Why this exists

Every phase-1 member has been scored on eval under R0, S1′ and mostly C1, and each
phase-1 decision looked at eval once it passed dev. Phase 2 then picks combinations of
those same checkpoints. Even with dev-only selection, the reported eval numbers of the
final systems are the end of a long chain of looks at one test set.

The cleanest check available is new checkpoints: train the final recipe on seeds no
rule was ever tuned on, freeze every rule first, then score them once.

## Design

- **Round 1 (needed in every outcome):** 4 × `ao_nogain` — the adopted single-network
  recipe (`ablation/aug-ops` C2) — seeds 2060–2063, as `fc_nogain_s<seed>`. Recipe
  identical to `ao_nogain` (`run_one.sh`); `--num_workers 8` as in phase 1 (CLAUDE.md:
  `nw` changes the run).
- **Round 2 (set by `../grand-ensemble`'s dev pick, before any fresh score):** 4 more
  members of whichever second type the from-scratch pick needs (expected `la_aug`,
  `fc_aug_s2060–2063`), or 4 more `ao_nogain` (2064–2067) if the pick is single-type.
  Optionally 2–3 fresh BEATs seeds if the external-track pick contains BEATs
  (~19 h each; decided with the user).
- ~11.3 h per run on GPUs 0/2, ~14.2 h on GPU 1 (x1 PCIe; CLAUDE.md).

- **Round 2, decided:** `../grand-ensemble` picked GS12 (8 `ao_nogain` + 4 `sb_spec`) on dev, so
  round 2 is 4 × `sb_spec` = the `ao_nogain` flags plus `--spec_branch 1` (`fc_spec_s2060–2063`).
  The size-matched fresh ensemble is 4 G (weight 2) + 4 S (weight 1).

## Scoring — frozen 2026-11-09, before any fresh checkpoint was scored

Rules, verbatim:
- **Single network:** C1ᵀ = zR0′ᵀ + zK′ᵀ + zGmean (`../single-final`, `scores/final_rule.json`,
  e24a022), via `single-final/score.py: score_member`. Also reported: C1, and R0 (eval.py's score,
  the paper-comparable one).
- **Ensemble (from scratch):** the size-matched GS12 build from `../grand-ensemble`'s
  `scores/dev_pick.json` (ee37d92): Σ S1′ over 4 `fc_nogain` (weight 2 each) + 4 `fc_spec` (weight 1
  each), via `grand-ensemble/score.py: ensemble_from_dumps`. Its total weight is 12, as GS12's.
- **Ensemble (with external pretraining):** the same + 1.5 × Σ zK′ over the three phase-1 BEATs nets
  (`bt_arc_s2040–2042`), the X1 pick's weighting. The BEATs nets are not fresh; the combination is.

`score.py` here converts this study's dumps (`extract.py`) to the inputs those two functions take; it
adds no rule of its own. Amean = mean over machines of (pooled AUC + pooled pAUC)/2.

Phase-1 references (8 seeds, mean / sd across seeds): C1ᵀ dev Amean 94.26 / 0.31, eval 94.10 / 0.62;
C1 dev 94.16, eval 94.09. Size-matched 4 G (w 2) + 4 S (w 1) over the 70 subsets of the existing
members: dev 95.15 (p05–p95 95.07–95.23), eval 94.86 (94.67–95.05), i.e. +0.33 dev / +0.15 eval over
4-G subsets (min +0.22 / +0.02). That eval reference was computed after the pick was committed.

| id | claim | prediction (80 % interval) | pass bar |
|---|---|---|---|
| F1 | one fresh network under C1ᵀ replicates phase 1 | 4-seed mean dev Amean 94.26 (93.9–94.6), eval 94.10 (93.5–94.7) | mean ≥ phase-1 mean − 2·se(4 vs 8 seeds): dev ≥ 93.88, eval ≥ 93.34 |
| F2 | one network beats the paper | C1ᵀ: all 3 dev numbers and eval mean above the paper on ≥ 3/4 seeds; R0: dev yes, eval ~50/50 | C1ᵀ: each of dev AUC > 95.71, pAUC > 90.23, mAUC > 91.23, eval Amean > 93.22 on ≥ 3/4 seeds |
| F3 | C1ᵀ ≥ C1 (TTA adds to pooled kNN) | dev +0.10, eval +0.01 | reported only; dev 4-seed Δ > 0 counts as replication |
| F4 | adding `sb_spec` helps the ensemble | fresh GS8 − fresh G4: dev +0.30 (0.15–0.45), eval +0.15 (−0.05–0.35) | dev Δ > 0 and eval Δ > 0; "confirmed" if the eval paired-bootstrap CI excludes 0 |
| F5 | fresh GS8 level | dev Amean 95.1 (94.8–95.4), eval 94.85 (94.4–95.3) | reported against paper 93.22 and A8 94.78 |
| F6 | BEATs fusion helps (external) | X1 − GS8: dev +0.15, eval +0.45 | eval Δ > 0 |

`fc_spec` single-network numbers (S1′, C1) are reported for completeness; nothing is decided on them.
If a run dies or a dump fails its CRC after three rewrites, that seed is dropped and reported, and the
bars use the remaining seeds (n ≥ 3); no seed is replaced or rerun after it has been scored.

## Extraction (`extract.py`)

One pass per split dumps, per clip, R0 and the pre-head embedding under `small` TTA shifts
(0, +1 %, −1 %) plus the shift-0 time-mean / time-max of `mobilefacenet.conv2`'s map, i.e. the
union of what `ablation/tta` and `ablation/pooled-knn` dumped. Reproduction test (2026-11-09,
`ao_nogain_s2040`, first 640 dev clips, GPU 0 shared with a training run): against the phase-1
dumps, max |ΔR0| 0.015 (0.14 % relative), max |Δpre| 0.013, max |Δgmean| 5e-4, rank correlation
1.000. The phase-1 dumps agree with each other bit-exactly, so the gap is this pass, most likely
cuDNN choosing other conv algorithms on a busy GPU (unverified). Fresh checkpoints are only ever
scored from this extractor, so the gap never enters a comparison.
