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

## Scoring (to be filled in before any fresh score exists)

The rules come verbatim from the committed decisions of `../single-final` (single
network) and `../grand-ensemble` (ensemble, matched to the fresh set's size). The
per-member bank, z statistics and TTA are exactly phase 1's (replayed 5 % split held out
of the bank). The predictions and pass bars go here, committed before scoring.

## Extraction (`extract.py`)

One pass per split dumps, per clip, R0 and the pre-head embedding under `small` TTA shifts
(0, +1 %, −1 %) plus the shift-0 time-mean / time-max of `mobilefacenet.conv2`'s map, i.e. the
union of what `ablation/tta` and `ablation/pooled-knn` dumped. Reproduction test (2026-11-09,
`ao_nogain_s2040`, first 640 dev clips, GPU 0 shared with a training run): against the phase-1
dumps, max |ΔR0| 0.015 (0.14 % relative), max |Δpre| 0.013, max |Δgmean| 5e-4, rank correlation
1.000. The phase-1 dumps agree with each other bit-exactly, so the gap is this pass, most likely
cuDNN choosing other conv algorithms on a busy GPU (unverified). Fresh checkpoints are only ever
scored from this extractor, so the gap never enters a comparison.
