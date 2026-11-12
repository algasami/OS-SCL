# fresh-confirm — do the final systems hold on seeds nobody has seen?

> **Status: SCORED ONCE, 2026-11-11.** Rules and F1–F6 were frozen on 2026-11-09 (d3472f2); the
> *Addendum of 2026-11-10* was committed before any fresh score (32740ba, 8d11e7d).
> **The single-network rule C1ᵀ replicates** (94.02 dev / 94.23 eval Amean; every fresh seed beats
> every paper number). **The ensemble level replicates** (GS8 94.92 / 94.90), **but `sb_spec` is not
> shown to matter on eval** (GS8 − G4 = +0.09, CI includes 0; at equal size −0.09). BEATs fusion
> replicates (+0.52 eval). See *Results* and *What does not replicate* below.
>
> **Extended 2026-11-12 by `../ensemble-g8`** (four more `ao_nogain` seeds, 2064–2067; nothing here
> was rescored). **[n=8]** single network under C1ᵀ: 94.11 / 94.37 (was 94.02 / 94.23 at n=4).
> The untrained option named below, an 8-network fresh `ao_nogain` ensemble, scores 94.76 / 95.00;
> the 12-network mix GS12f scores 95.01 / 95.03. The `sb_spec` verdict is unchanged.

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

## Scorer self-test (`score.py selftest`, before any fresh score)

`score.py` was run on "pseudo-fresh" dumps rebuilt in this study's format from phase-1 data
(`ao_nogain_s2040–2043` from the tta + pooled-knn dumps, `sb_spec_s2040–2043` from spectrum-branch).
It reproduces single-final's committed per-seed C1ᵀ and C1 Amean on dev and eval to ≤ 4.9e-5 (the
CSVs' 4-decimal rounding), and the 4 G (w 2) + 4 S (w 1) ensemble exactly (dev 95.1790, eval 94.8030)
against grand-ensemble's own loader on the phase-1 flat dumps (`scores/selftest.json`).

Faulty RAM, seen during the self-test: one segfault (GP fault in `dmesg`); two npz files that this
study had written CRC-clean then failed their CRC when read back, and were clean again once evicted
from the page cache, i.e. the bits flipped in cached pages, not on disk; and five score vectors that
needed a third computation to get two bit-identical ones. `score.py` therefore evicts every `.npz`
before reading it, reads it whole and retries on a CRC failure; every vector and metric is still
computed until two runs agree.

## Addendum of 2026-11-10, before any fresh score

A new session took the study over on 2026-11-10 and audited it before scoring (one independent
read-only audit of `extract.py`, `score.py` and `run.sh`, plus a dry run). F1–F6, their bars and the
three scoring rules above are unchanged. What changed is how safely they are computed, and one
reported contrast was added.

### The scorer had never run end to end

`selftest` covered the conversion and the two scoring functions, but `score()` itself (CSV writers,
bootstrap, the F1–F6 table) had not been executed once. `run.sh dryrun` now runs all of `score()` on
the self-test's pseudo-fresh dumps (`ao_nogain` / `sb_spec` s2040–2043, all seen before). Its output
is kept in `scores/dryrun_p1/` as the phase-1 reference for the figures; it holds no fresh number.

- **It found a bug that would have stopped the real scoring half way.** `score.py` loaded the two
  scorers without registering them in `sys.modules`, so the bootstrap's worker function could not be
  pickled. The crash came after `single.csv` and `ensembles.csv` were written and before
  `bootstrap.json` and `decisions.json`. Fixed.
- **No seed is dropped silently.** `score()` used to skip any run with a missing dump. It now stops,
  unless the run is named in `FC_DROP` (for the two reasons given under *Scoring*), and records it.
- **The flat ensemble inputs are rebuilt in every process** from two reads that agree, instead of
  being reused from an earlier run.
- **Every recomputation starts from its own read of the dumps**, so a bit flipped in a loaded array
  is caught like one flipped during the arithmetic. The `sb_spec` single-network vector is now
  recomputed too.

### One process cannot vouch for itself on this host

During the dry runs of 2026-11-10 the host's RAM was flipping bits much more often than during the
self-test of 2026-11-09:

- One score vector was computed six times and no two results agreed.
- Single elements of 17,971 differed between two computations (by 4e-5 and by 1.6e-3).
- A bootstrap index came back as 2⁵² + 7924, which is bit 52 flipped, in the byte lane that
  `~/.cache/hwcheck` reports.

`run.sh score` (and `dryrun`) therefore run `score.py` as independent processes, each writing to its
own directory under `results/`, until two of them produce byte-identical files. Only that output is
copied to `scores/`. The computation is deterministic (the bootstrap is seeded), so this repeats one
scoring; it is not a second look. "Scored once" means that no rule, bar or seed list changes after
the first fresh number is seen.

Not covered: the extraction is a single pass. A flip while a dump sits in RAM before it is written
passes the CRC (which is computed from the flipped data) and, if small, the range check. The damage
is bounded to single clips out of 10,868 per set. As a check on the R0 path, `evalpy` recomputes R0
with `eval.py` after scoring and `crosscheck.py` compares the two (`scores/r0_crosscheck.csv`).

- **Reference.** On the dry run's 8 seen networks the largest gap in AUC, pAUC or mAUC is 0.004 points
  (`scores/dryrun_p1/r0_crosscheck.csv`).
- **Rule, set before scoring.** If a fresh run differs from `eval.py` by more than 0.05 points on any of
  the three, its dumps are deleted and extracted again and everything is rescored. Both scorings are
  then reported. A smaller gap changes nothing.

### F4b — is it the `sb_spec` type, or just more members?

F4 compares 8 networks with 4. In phase 1 the 4 G + 4 S build scored 94.86 on eval and the 8-G
ensemble 94.87 (`../grand-ensemble`), so the +0.15 that F4 predicts on eval is about what four more
networks of *any* type give. F4 can pass without `sb_spec` contributing anything.

F4b holds the count at four: each of the 36 ensembles of 2 `fc_nogain` (weight 2) + 2 `fc_spec`
(weight 1) against the 4-`fc_nogain` ensemble. It is reported, with no pass bar. The subsets share
networks with each other and with the baseline, so they are not 36 independent draws.

The dry run gives the phase-1 reference on seeds 2040–2043 (all seen; `scores/dryrun_p1/`):

| contrast, s2040–2043 | dev Δ Amean | eval Δ Amean |
|---|---|---|
| F4: GS8 − G4 (8 nets against 4) | +0.34 | +0.18, bootstrap CI [+0.03, +0.32] |
| F4b: mean of 2 G + 2 S subsets − G4 (4 against 4) | +0.18, 34 of 36 subsets above G4 | +0.00, 19 of 36 above G4 (range −0.53 to +0.45) |

So on seen seeds F4 is "confirmed" by its own bar while the type adds nothing on eval at equal size.

- **Prediction for the fresh seeds:** F4b dev +0.15 (80 % interval 0.00 to +0.30); eval 0.00 (−0.25
  to +0.25).
- **How the two are read together.** "`sb_spec` helps at equal size" is claimed for a set only if the
  F4b mean is above 0 and at least 27 of the 36 subsets are above G4. If F4 passes and F4b does not,
  the result is written as "more members help; the member type is not shown to matter".

### Other additions to the output, none of them a decision

- **Per-ID metrics.** `eval.py`, and so every Amean in this repository, pools the machine IDs of a
  type into one AUC. The papers that state a convention compute AUC per machine ID and average
  (`../final/PUBLISHED.md`). The CSVs now carry `AUC_perID`, `pAUC_perID` and `Amean_perID` beside
  the pooled columns. F1–F6 use the pooled numbers, as frozen. On the seen seeds of the dry run the
  two differ by less than 0.2 (GS8: dev 95.18 pooled, 95.25 per ID; eval 94.80 and 94.64).
- **F2, joint count.** `decisions.json` also reports how many seeds beat all four paper numbers at
  once (`seeds_above_all4`). The pass rule stays the per-number count frozen in `score.py` at
  5c41150: each number is beaten on at least n − 1 seeds.
- **No training-health exclusion.** Every run with a `done` marker is scored, whatever its curve.

### What the pass bars do not show

- The bootstrap intervals in F4 and F6 resample test clips. They are conditional on these 8
  networks and ignore seed-to-seed variance, which is about 1 point of eval Amean per network.
- F1 is a lenient non-inferiority bar: on eval it sits 0.76 below the phase-1 mean.
- F3: C1ᵀ was adopted on the same 8 seeds it is compared with, so its margin is expected to shrink.
- F5 sets an 8-network ensemble beside the paper's single network.
- F6 is only half fresh. The BEATs networks and their weight were chosen in phase 1, on data that
  includes this eval set.
- mAUC is a minimum over IDs; in F2 it is near a coin flip at small margins.
- Nothing is corrected for the number of comparisons.

### Operations

- **Order:** wait for `results/post.status` = `READY_TO_SCORE` → `bash synthesis/fresh-confirm/run.sh
  score` → `FC_GPU=0 bash synthesis/fresh-confirm/run.sh evalpy` → `crosscheck.py` → results, `make_plots.py`.
- **Figures:** `make_plots.py` reads `scores/` only and skips what is missing. Its layout was checked on
  the dry-run record; no figure is committed before the fresh scores exist.
- **`FAIL extract`:** rerun `FC_GPU=0 bash synthesis/fresh-confirm/run.sh extract` (resumable).
- **A run shows `FAIL train ... after 3 attempts`:** do not rerun `train_one.sh`, which deletes
  `state_last.pth`. Resume by hand from the worktree root with the run's own flags (`run_one.sh`)
  plus `--crash_resume check_points/<run>/state_last.pth`, appending to its log. Then write
  `check_points/<run>/wall.csv` (`start,end,seconds,epochs,gpu`), `touch check_points/<run>/done`,
  and start `run.sh post` again.
- **Editing `run.sh` while a stage runs:** write a new file and rename it over the old one. Bash reads
  a running script by byte offset, and `post` re-enters the file with `bash $0`.

## Results — scored once, 2026-11-11

All 8 runs finished 300 epochs with no crash, and no seed was dropped. `run.sh score` needed two
processes and they agreed byte for byte; inside the second, one vector took a third computation to
get two identical results (the host's RAM again). Six runs were extracted on 2026-11-10 on the idle GPU 1 as
they finished and the last two by `post`; the extractor prints shapes only. After scoring, the only
change to `make_plots.py` was the spacing of fig1's groups. Everything
below is from `scores/` (`decisions.json`, `single.csv`, `ensembles.csv`, `bootstrap.json`,
`f4b_subsets.csv`). Amean pools the IDs of a machine type, as everywhere in this repository; the
per-ID figure is given beside it where a number is compared with the literature.

### Against the frozen predictions

| id | predicted (80 % interval) | fresh seeds 2060–2063 | bar | verdict |
|---|---|---|---|---|
| F1 | C1ᵀ mean dev 94.26 (93.9–94.6), eval 94.10 (93.5–94.7) | dev **94.02**, eval **94.23** (sd over seeds 0.15 / 0.19) | dev ≥ 93.88, eval ≥ 93.34 | **passes**; both inside the interval |
| F2, C1ᵀ | all four paper numbers beaten on ≥ 3 of 4 seeds | AUC 4/4, pAUC 4/4, mAUC 4/4, eval Amean 4/4; all four at once on 4/4 | ≥ 3/4 on each | **passes** |
| F2, R0 | dev yes, eval about 50/50 | dev AUC 2/4, pAUC 3/4, mAUC 2/4; eval Amean 2/4; all four at once on 1/4 | same | **fails**; the dev half of the prediction was wrong |
| F3 | C1ᵀ − C1: dev +0.10, eval +0.01 | dev +0.12 (4/4 seeds positive), eval +0.03 (3/4) | dev Δ > 0 | replicates on dev |
| F4 | GS8 − G4: dev +0.30 (0.15–0.45), eval +0.15 (−0.05–0.35) | dev +0.40, eval +0.09, eval CI [−0.08, +0.24] | both Δ > 0; "confirmed" if the CI excludes 0 | **passes, not confirmed** |
| F4b | 2 G + 2 S − G4: dev +0.15 (0.00–0.30), eval 0.00 (−0.25–0.25) | dev +0.26, 36/36 subsets above G4; eval **−0.09**, 8/36 above | mean > 0 and ≥ 27/36 | type helps at equal size on dev, **not on eval** |
| F5 | GS8 dev 95.1 (94.8–95.4), eval 94.85 (94.4–95.3) | dev 94.92, eval 94.90 | reported | inside both intervals |
| F6 | X1 − GS8: dev +0.15, eval +0.45 | dev +0.21, eval +0.52, eval CI [+0.40, +0.67] | eval Δ > 0 | **passes** (external pretraining) |

All eight interval predictions landed inside their 80 % intervals. With eight for eight the
intervals were probably wider than 80 %, so this says little about calibration.

### One network (`fc_nogain`, 4 seeds, mean ± sd over seeds)

| rule | dev AUC / pAUC / mAUC | dev Amean (per ID) | eval AUC / pAUC / mAUC | eval Amean (per ID) |
|---|---|---|---|---|
| R0 (`eval.py`) | 95.83 / 90.66 / 91.46 | 93.24 ± 0.31 (93.42) | 95.34 / 91.34 / 89.53 | 93.34 ± 0.35 (93.34) |
| C1 | 96.41 / 91.40 / 92.93 | 93.91 ± 0.14 (94.05) | 96.44 / 91.96 / 92.50 | 94.20 ± 0.23 (94.23) |
| **C1ᵀ** (final rule) | **96.48 / 91.57 / 93.09** | **94.02 ± 0.15** (94.16) | **96.47 / 91.99 / 92.59** | **94.23 ± 0.19** (94.29) |
| paper, one run | 95.71 / 90.23 / 91.23 | 92.97 | not given | 93.22 |
| phase 1, C1ᵀ, 8 seen seeds | | 94.26 ± 0.31 | | 94.10 ± 0.62 |

- **The final single-network rule holds on unseen seeds.** Each of the four fresh networks beats
  every paper number under C1ᵀ. The margin over the paper is +1.05 dev and +1.01 eval Amean. The
  worst fresh seed is 93.91 on dev and 93.96 on eval.
- **The rule is what carries it.** Under the paper's own score (R0) the 4-seed mean is above the
  paper on all four numbers, by 0.1 to 0.4, but a single seed beats all four only one time in four.
  C1ᵀ adds 0.78 dev and 0.89 eval Amean to R0 on the same checkpoints, and 3.1 points of eval mAUC.
- **Dev came out 0.24 below phase 1 and eval 0.13 above.** That is 1.3 and 0.3 standard errors of
  the difference between a 4-seed and an 8-seed mean (0.19 dev, 0.38 eval). The eval number did not
  shrink, which gives no sign that phase 1's single-network eval figure was inflated by selection.
- TTA (F3) is again worth about 0.1 on dev and nothing measurable on eval.

### Ensembles (one build each; brackets are the same build on seen seeds 2040–2043)

| system | members | dev AUC / pAUC / mAUC | dev Amean (per ID) | eval AUC / pAUC / mAUC | eval Amean (per ID) |
|---|---|---|---|---|---|
| G4 | 4 `fc_nogain` | 96.98 / 92.06 / 93.86 | 94.52 (94.66) [94.84] | 96.76 / 92.85 / 92.88 | 94.80 (94.67) [94.62] |
| GS8 | G4 + 4 `fc_spec` | 97.25 / 92.60 / 94.39 | 94.92 (95.07) [95.18] | 96.90 / 92.90 / 93.15 | 94.90 (94.69) [94.80] |
| X1 | GS8 + 3 phase-1 BEATs | 97.37 / 92.89 / 94.77 | 95.13 (95.26) [95.34] | 97.24 / 93.60 / 94.19 | 95.42 (95.27) [95.32] |

- **The from-scratch ensemble level replicates:** fresh GS8 is 94.92 dev / 94.90 eval against the
  paper's single network at 92.97 / 93.22 and phase 1's A8 at 94.78 eval.
- **`sb_spec` is not shown to be the reason.** On eval, GS8 is +0.09 over G4 and the interval
  includes 0. At equal size (F4b) the mixed ensembles are 0.09 *below* four `fc_nogain` networks,
  and only 8 of 36 are above. By the reading rule set before scoring: more members may help; the
  member type is not shown to matter on eval. On dev the type effect is clear (+0.26, 36 of 36),
  which is where `sb_spec` was picked. Under the per-ID convention GS8 − G4 on eval is +0.02.
- **Four plain networks already give most of it.** Fresh G4 scores 94.80 on eval; the four extra
  training runs of GS8 buy 0.09.
- **BEATs fusion replicates** at +0.52 eval Amean, clear of 0. Almost all of it is ToyCar
  (93.78 → 96.08) and slider (99.22 → 99.82). The BEATs networks are the phase-1 ones, so only the
  from-scratch half of X1 is fresh.

Per machine, fresh GS8 against fresh G4 (Amean): dev fan +1.09, ToyConveyor +0.57, valve +0.29,
slider +0.26, pump +0.15, ToyCar +0.02; eval pump +0.53, fan +0.36, ToyCar +0.18, slider −0.09,
ToyConveyor −0.12, valve −0.30.

### What does not replicate

- **The fresh `sb_spec` networks are weaker than the phase-1 ones.** Under S1′ the four `fc_spec`
  seeds average 93.36 dev / 92.74 eval (sd 0.61 / 0.77); phase 1's four `sb_spec` seeds averaged
  94.06 / 93.42 (sd 0.19 / 0.38). The loss is on valve (dev AUC 96.3–99.4 against 99.2–99.8) and on
  eval slider (97.5–99.0 against 98.7–99.2) and valve. The recipe is the same: the two `config.json`
  differ only in seed, GPU and name, and the training-code diff since the `sb_spec` runs
  (permix, timepool-head, BEATs) is behind flags that are off. Four seeds against four cannot
  separate bad luck now from good luck then, but `sb_spec` was chosen on those four phase-1 seeds,
  so some regression was to be expected. Nothing was decided on these single-network numbers.
- **F4 is not confirmed and F4b fails on eval** (above). The claim "adding `sb_spec` helps the
  ensemble" stands on dev only.
- **R0 does not reliably beat the paper on dev.** On seeds 2040–2043 `ao_nogain` under R0 was above
  each of the paper's three dev numbers on 4 seeds of 4 (`scores/dryrun_p1/decisions.json`). On the
  fresh seeds it is 2, 3 and 2 of 4 for AUC, pAUC and mAUC, although the 4-seed mean is still above.
- **Dev ensembles are 0.2 to 0.3 lower than the same builds on seeds 2040–2043**, mostly
  ToyConveyor (−1.1) and fan (G4 −0.5). Eval is 0.1 to 0.2 higher. Neither shift is outside seed
  noise, and they point in opposite directions.

### R0 cross-check against `eval.py`

`crosscheck.py` compared the R0 column of `single.csv` with `eval.py` on all 8 checkpoints and both
sets (`scores/r0_crosscheck.csv`). The largest gap in AUC, pAUC or mAUC is 0.0033 points, against
0.004 on the dry run and a re-extraction threshold of 0.05. No dump was extracted again and nothing
was rescored.

### What this changes for `../final`

- Single network: **C1ᵀ on `ao_nogain` is confirmed** — fresh 94.02 dev / 94.23 eval (per ID
  94.16 / 94.29), every seed above every paper number.
- From-scratch ensemble: report GS8 at its fresh level (94.92 / 94.90), and say that on eval a
  same-size all-`nogain` ensemble is not shown to be worse. The `sb_spec` members are not a
  confirmed ingredient.
- External track: X1 at 95.13 / 95.42, with the BEATs half not fresh.

## Figures

`make_plots.py` (run with `/home/f74134118/anaconda3/bin/python`) reads `scores/` and phase-1
`scores/` directories only.

| figure | shows |
|---|---|
| `plots/fig1_single_seeds.png` | Amean of each network under R0, C1 and C1ᵀ, fresh seeds beside the 8 phase-1 seeds, with the F1 bar and the paper's number |
| `plots/fig2_ensembles.png` | G4, GS8 and X1 on fresh seeds beside the same builds on seeds 2040–2043 |
| `plots/fig3_f4b_type_vs_count.png` | the 36 count-matched 2 G + 2 S ensembles minus G4, fresh and phase 1, with GS8 marked |
| `plots/fig4_machines.png` | per machine type, fresh C1ᵀ Amean minus the phase-1 mean |
