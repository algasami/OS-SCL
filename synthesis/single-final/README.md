# single-final — does C1 + small TTA (C1ᵀ) beat C1 for one `ao_nogain` network?

> **Status: PRE-REGISTERED 2026-11-09.** Design, predictions and decision rule below are
> committed before any C1ᵀ number exists. The reproduction check (C1, S1′, S1′ᵀ, all already
> seen) has been run and passes; see *Reproduction check*.

## Why this exists

Phase 1 adopted two single-model improvements for the `ao_nogain` network, each tested only
against plain S1′ = zR0′ + zK′:

- **C1 = S1′ + zGmean** (`ablation/pooled-knn`, 2026-11-05): adds the z-scored 1-NN cosine
  distance of `gmean`, the 512-d time-and-frequency mean of the pre-`linear7` map. 8-seed dev
  96.55 / 91.78 / 93.22 (Amean 94.16), eval Amean 94.09.
- **S1′ᵀ** (`ablation/tta`, 2026-11-03): S1′ with R0 and the pre-head embedding averaged over
  circular shifts of 0 and ±1 % (`small`, K = 3), the training bank treated the same way.
  +0.11 dev / +0.05 eval Amean over S1′; dev 96.52 / 91.72 / 93.01 (94.12), eval 93.93.

Both READMEs flag the combination as untested (`pooled-knn`: "C1 + TTA was not tested ... That
is the next confirmation, not something to assume"). This study is that confirmation, and it
fixes the single-model rule that `fresh-confirm` will apply to new seeds.

## What has already been seen

- Every number of C1, S1′ and S1′ᵀ for the 8 `ao_nogain` seeds (2040–2047), dev and eval,
  per seed and per machine (`ablation/pooled-knn/scores/confirm_metrics.csv`,
  `ablation/tta/scores/{dev,eval}_runs.csv`).
- zGmean standalone (dev 4-seed AUC 93.40), and its ToyCar roll diagnostic.
- **Not seen:** any score that combines a shifted R0 or K with zGmean, on any clip.

## Design

### Rules

Per checkpoint and machine ID. Bank = the ID's training-part clips (`valid == False`, the
replayed 95 % the checkpoint trained on). Distances are 1-NN cosine distances on L2-normalised,
uncentred float64 vectors. Every z-statistic is the mean and population sd over the bank; K and
G bank clips are scored leave-one-out. This is exactly `ablation/ensemble-backend`'s
`per_run_scores` and `ablation/pooled-knn`'s `Bank`.

| rule | definition | source |
|---|---|---|
| C1 (incumbent) | zR0′ + zK′ + zGmean | `ablation/pooled-knn` |
| S1′ᵀ (reference) | zR0′ᵀ + zK′ᵀ | `ablation/tta` |
| **C1ᵀ (candidate)** | **zR0′ᵀ + zK′ᵀ + zGmean** | this study |

- **ᵀ is `ablation/tta`'s `small` TTA, replicated exactly** (`tta_reduce` in `score.py` mirrors
  `ablation/tta/score.py: versions()`):
  - R0ᵀ = mean over the 3 copies (shifts 0, +1 %, −1 % of the 160,000-sample clip) of the
    per-copy CE.
  - preᵀ = mean over the copies of the L2-normalised pre-head embedding; the kNN normalises it
    again.
  - **The training clips get the same treatment.** They are the bank and the source of the z
    statistics, so query and reference stay on one scale.
- **zGmean stays unshifted. This is a design choice, not an omission.**
  - `pooled-knn`'s diagnostic showed the time mean of the pre-`linear7` map is roll-invariant.
    A half-clip roll of a ToyCar clip moves zGmean by 0.13 sd (dev) and 0.20 sd (eval), against
    27 / 44 sd for S1′.
  - A ±1 % shift is ~40× smaller than those rolls, so averaging gmean over it would change almost
    nothing.
  - No shifted gmean dump exists, and making one needs a GPU.
- **Inputs, all existing dumps, read-only:**
  - R0 / pre per shift: `OS-SCL-tta/ablation/tta/results/emb/{train,dev,eval}` (train and dev
    hold 9 shifts; `small`'s 3 columns are picked by value; eval holds exactly those 3).
  - gmean (and the unshifted R0 / pre that define C1): `OS-SCL-pknn/ablation/pooled-knn/results/emb`.
  - `score.py` aborts unless both `meta.npz` files list the same clips (paths compared below
    `/data/`) and both dump families list every split's clips in the same order.
  - It also asserts that the unshifted R0 and embedding agree across the two families: bit for
    bit on the training bank, ≤ 1e-3 on test clips. (They turn out bit-identical on test clips
    too.)
- **Faulty host RAM** (`CLAUDE.md`). Every loaded array is range-checked, and `np.load` checks
  each member's zip CRC.
  - The first reproduction run lost 0.17 AUC on one cell, ToyConveyor of s2047 on eval. The
    inputs there were bit-identical to a sibling path that reproduced the committed value, so a
    bit flipped during the computation.
  - So every score vector is now computed until two runs agree bit for bit (`twice`).
  - The input arrays are hashed before and after scoring, and a member is reloaded if they
    changed while in RAM.

### Stages (dev-only selection, eval once)

There is one candidate and no free parameter, so stage A is a gate, not a selection.

1. **Reproduction check** (`score.py check`; done, committed with this README). It recomputes C1,
   S1′ and S1′ᵀ for all 8 seeds on dev and eval, and compares them to the committed
   `pooled-knn` / `tta` per-seed, per-machine cells. Bar: ≤ 1e-3 on every cell. These numbers
   were all seen before, and no C1ᵀ number is computed.
2. **Stage A, gate** (`score.py stageA`; refuses unless the check is committed): seeds
   2040–2043, dev only, C1ᵀ vs C1.
3. **Stage B, confirm** (`score.py stageB`; refuses unless `stageA.json` is committed):
   seeds 2044–2047 dev, plus the 8-seed dev summary.
4. **Stage C, eval** (`score.py eval`; refuses unless `stageB.json` is committed, and refuses
   again once `eval.json` is committed): all 8 seeds, scored once. This stage writes the
   decision and `scores/final_rule.json`.

**Metrics.**
- Amean = mean over the 6 machines of (pooled AUC + pooled pAUC) / 2, as `pooled-knn` reports
  it (`eval.py`'s definitions).
- Also dev AUC / pAUC / mAUC against the paper (95.71 / 90.23 / 91.23), and eval Amean against
  93.22.
- Paired per-seed deltas. A paired stratified clip-bootstrap of the seed-mean Δ: 2000
  resamples, seed 0, `pooled-knn`'s `boot_delta` (imported), the same resamples for both rules.
- Per machine and per ID (especially ToyCar, ToyConveyor, valve, pump; eval ToyCar id_05 and
  valve id_05).

## Predictions (written 2026-11-09, before any C1ᵀ number)

Δ = C1ᵀ − C1, seed-mean, in points; intervals are 80 %.

Reasoning. TTA lifted S1′ by +0.11 on dev and +0.05 on eval. Most of the dev gain was
ToyConveyor (+0.28 AUC), which reads as smoothing frame-grid jitter. C1 should keep less of it,
for three reasons:
- C1 already gained on ToyConveyor (+0.31 dev) from a time-mean view, which is itself
  insensitive to jitter.
- In a three-term sum, the two terms TTA de-noises carry less weight.
- Every variance-reduction step in phase 1 shrank when composed with another one (TTA on G8:
  −0.05).

So, about half the S1′ gain:

- **P0, reproduction.** Every committed C1, S1′ and S1′ᵀ cell is reproduced to ≤ 1e-3.
  *(Seen before this commit: passes once the RAM guard is in; see below.)*
- **P1, gate A (seeds 2040–2043, dev).** Δ Amean +0.06 [−0.03, +0.15]. The gate passes (70 %).
- **P2, dev 8 seeds.**
  - Δ AUC +0.03 [−0.03, +0.10], Δ pAUC +0.08 [−0.05, +0.20], Δ Amean +0.06 [−0.02, +0.14].
  - ≥ 6/8 seeds positive (60 %).
  - The bootstrap 95 % CI of Δ Amean includes 0 (55 %).
- **P3, confirm seeds (2044–2047, dev).** Δ Amean > 0 (70 %). Smaller than stage A's (55 %).
- **P4, eval 8 seeds.** Δ Amean +0.03 [−0.05, +0.10]. Δ ≥ 0 (65 %). The 95 % CI includes 0
  (65 %).
- **P5, where it moves.**
  - Dev: ToyConveyor has the largest Δ AUC of the six machines (50 %), central +0.15.
  - Fan, pump and slider each |Δ AUC| ≤ 0.05 on dev. ToyCar Δ AUC in [−0.10, +0.15] on both
    sets. Valve |Δ AUC| ≤ 0.10 on both sets.
  - Eval ToyCar id_05 |Δ| < 0.5.
  - Eval valve id_05 (C1's one consistent loss, −0.51 vs S1′) is not repaired: |Δ| ≤ 0.2.
- **P6, C1ᵀ vs S1′ᵀ.** C1ᵀ is above S1′ᵀ on both dev and eval 8-seed Amean (85 %).
- **P7, decision.** D1 passes (45 %). The central call is a narrow fail: the most likely way to
  fail is fewer than 6/8 dev seeds positive, or an eval Δ just below 0.

## Decision rule (fixed now)

- **Gate A.** The stage-A 4-seed dev Δ Amean must be > 0. If it isn't, D1 cannot pass. Stages B
  and C are still run once as descriptives, so the record is complete.
- **D1: adopt C1ᵀ as the single-model rule in place of C1** only if all of these hold:
  1. gate A passed;
  2. 8-seed dev Δ Amean > 0, with ≥ 6 of 8 seeds positive;
  3. confirm-seed (2044–2047) dev Δ Amean > 0;
  4. eval 8-seed Δ Amean ≥ 0 (point estimate).
- **Otherwise C1 stays.** It is simpler and costs one forward pass per clip instead of three,
  for the bank too.
- The bootstrap CIs are reported but do not gate. The gain sought is small by construction, and
  the seed count plus the eval sign are the agreed bar. mAUC is reported, not read (single-model
  deltas under ~2 points are unresolvable).
- The adopted rule is written to `scores/final_rule.json`. `score.py: score_member()` reads it as
  its default, and `fresh-confirm` imports that function.

## Reproduction check (run before this commit; `scores/repro_check.json`, `scorecard_check.txt`)

**P0 passes, exactly.** The recomputed values match every committed cell to 0.0 at 4 dp. That
covers C1 and S1′ against `pooled-knn` (224 cells per set) and S1′ᵀ against `tta` (176 cells
per set), on dev and eval, per seed and per machine.

- C1 rebuilt from the TTA dump's shift-0 copy is identical to C1 from the pooled dump, on both
  sets. The two dump families agree bit for bit on the unshifted R0 and embedding, test clips
  included: both passes batched the clips the same way.
- Both `meta.npz` files list the same clips in the same order, and so does every dump's `idx`.
- **How the check got there.**
  - The first run failed one cell: s2047 eval ToyConveyor, 0.17 AUC off. Its sibling path from
    bit-identical inputs (`C1_0`) reproduced the committed value, so a bit flipped during
    computation.
  - The second run, with the two-agreeing-runs guard, aborted. It saw input arrays change in RAM
    5 times in 6 members.
  - The guard now re-reads a changed digest before reloading, since a misread looks the same as a
    change.
  - The third run passed with no guard events.
  - The faulty host RAM is the explanation in `CLAUDE.md`. Every later stage runs with these
    guards.

8-seed means, AUC / pAUC / mAUC / Amean (paper dev 95.71 / 90.23 / 91.23, eval Amean 93.22):

```
          dev                            eval
S1′    96.43  91.58  92.82  94.00    96.19  91.57  91.96  93.88
S1′ᵀ   96.52  91.72  93.01  94.12    96.24  91.61  92.12  93.93
C1     96.55  91.78  93.22  94.16    96.42  91.75  92.54  94.09
```

## Files

| file | what |
|---|---|
| `score.py` | CPU scorer. Stages `check`, `stageA`, `stageB`, `eval`, each gated on the previous one's committed output. Importable: `score_member(tr, te, tr_label, te_label, valid, rule=FINAL_RULE)`, `tta_reduce`, `member_terms` |
| `run.sh` | stages `check`, `stageA`, `stageB`, `eval`, `plots`; resumable |
| `make_plots.py` | figures from `scores/` only |
| `scores/` | committed record: `repro_check.json`, `metrics_*.csv`, `perid_*.csv`, `stage*.json`, `eval.json`, `final_rule.json`, `scorecard_*.txt` |
| `results/` | logs only: gitignored scratch |
