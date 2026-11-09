# fresh-confirm — do the final systems hold on seeds nobody has seen?

> **Status: TRAINING (round 1 launched 2026-11-09).** `fc_nogain_s2060–2063` on GPUs 0/2/1.
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

## Scoring (to be filled in before any fresh score exists)

The rules come verbatim from the committed decisions of `../single-final` (single
network) and `../grand-ensemble` (ensemble, matched to the fresh set's size). The
per-member bank, z statistics and TTA are exactly phase 1's (replayed 5 % split held out
of the bank). The predictions and pass bars go here, committed before scoring.
