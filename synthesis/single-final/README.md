# single-final — does C1 + small TTA (C1ᵀ) beat C1 for one `ao_nogain` network?

> **Status: DONE 2026-11-09. D1 passes by the letter, so C1ᵀ = zR0′ᵀ + zK′ᵀ + zGmean replaces C1
> as the single-model rule. The eval margin is +0.01 and is not resolved.**
> - **Dev, 8 seeds:** C1ᵀ − C1 = **+0.10 Amean** [+0.01, +0.21], 8/8 seeds positive. The gate
>   seeds (2040–2043) give +0.08 and the confirm seeds (2044–2047) +0.11.
> - **Eval, 8 seeds, scored once:** **+0.014 Amean** [−0.024, +0.051], 5/8 seeds positive. The
>   rule's bar was ≥ 0, so it passes; the CI includes 0.
>   - That is TTA's usual dev-to-eval shrink: S1′ᵀ went from +0.11 to +0.05.
> - **One `ao_nogain` net under C1ᵀ** (8-seed means): dev 96.61 / 91.91 / 93.35 (Amean 94.26);
>   eval AUC 96.45, pAUC 91.76, Amean **94.10**. Paper: 95.71 / 90.23 / 91.23, eval 93.22. C1:
>   dev 94.16, eval 94.09.
> - **Where it moves:** ToyConveyor (+0.17 dev / +0.09 eval AUC) and ToyCar (+0.09 / +0.10;
>   eval id_05 +0.25, 7/8 seeds). The one loss is eval valve (−0.09 AUC, −0.37 pAUC). It comes
>   from pooling scores across valve IDs: every per-ID valve AUC is unchanged or up.
> - **Reproduction:** C1, S1′ and S1′ᵀ reproduce every committed cell exactly (0.0 at 4 dp).
> - **Faulty host RAM** corrupted scores mid-computation. Every score vector here was computed
>   until two runs agreed bit for bit (see *Design*).
> - Pre-registered at e4ebd98. Stage A was committed at e7d62a5 and stage B at 61432c5, each
>   before the next stage was scored.
> - **Cost:** 3 forward passes per clip, for the bank too. `fresh-confirm` (new seeds) is the
>   real test of a +0.01 eval margin.

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

## Stage A — gate (2026-11-09, before any confirm-seed or eval C1ᵀ number)

`scores/scorecard_stageA.txt`, `stageA.json`, `metrics_stageA.csv`, `perid_stageA.csv`.

- **Gate A passes.** C1ᵀ − C1 on seeds 2040–2043 dev: Δ Amean **+0.083**, 95 % CI
  [−0.001, +0.188], 4/4 seeds positive (+0.09, +0.06, +0.06, +0.13).
  - Δ AUC +0.05, Δ pAUC +0.12. mAUC moves +0.16 and is not read.
- Δ AUC by machine: ToyConveyor +0.15, ToyCar +0.10, slider +0.06, pump +0.03, valve +0.02,
  fan −0.04.
- S1′ᵀ alone is 0.04 below C1 here, so TTA's gain adds to the pooled term rather than replacing it.
- P1 (+0.06 [−0.03, +0.15], gate passes): holds.

## Stage B — confirm dev (2026-11-09, before any eval C1ᵀ number)

`scores/scorecard_stageB.txt`, `stageB.json`, `metrics_stageB.csv`, `perid_stageB.csv`. Seeds
2040–2043 were rescored here and match stage A exactly.

- **Confirm seeds (2044–2047):** Δ Amean **+0.112** [+0.007, +0.248], 4/4 positive (+0.14,
  +0.08, +0.06, +0.16). Δ AUC +0.06, Δ pAUC +0.16.
- **Dev, 8 seeds:** Δ AUC +0.057, Δ pAUC +0.138, Δ Amean **+0.098** [+0.006, +0.214], 8/8
  seeds positive. C1ᵀ scores 96.61 / 91.91 / 93.35, Amean 94.26 (C1 94.16, S1′ᵀ 94.12).
- Δ AUC by machine (8 seeds): ToyConveyor +0.17, ToyCar +0.09, slider +0.06, pump +0.03,
  valve +0.01, fan −0.02.
- **D1's dev conditions all hold** (gate A, 8-seed Δ > 0 with 8/8 positive, confirm Δ > 0).
  Eval decides.
- Guard events: in this run, 7 score vectors needed a third or fourth computation, and 2 members
  were reloaded. The first stage-B attempt aborted on s2047 (the inputs kept changing). It was
  rerun with verified per-member caching and more retries; no score from the aborted attempt was
  read.

## Results — eval (2026-11-09, scored once after 61432c5)

`scores/scorecard_eval.txt`, `eval.json`, `metrics_eval.csv`, `perid_eval.csv`; 8-seed summary
over both sets in `summary.csv` and `perid_delta.csv`; the adopted rule in `final_rule.json`.

```
8-seed means       dev AUC  pAUC   mAUC   Amean  |  eval AUC  pAUC   mAUC   Amean
S1′                 96.43  91.58  92.82  94.00   |     96.19  91.57  91.96  93.88
S1′ᵀ                96.52  91.72  93.01  94.12   |     96.24  91.61  92.12  93.93
C1                  96.55  91.78  93.22  94.16   |     96.42  91.75  92.54  94.09
C1ᵀ (adopted)       96.61  91.91  93.35  94.26   |     96.45  91.76  92.61  94.10
paper               95.71  90.23  91.23  92.97   |                          93.22
```

- **Eval C1ᵀ − C1:** Δ AUC +0.022, Δ pAUC +0.006, Δ Amean **+0.014** [−0.024, +0.051].
  - 5/8 seeds positive (+0.05, +0.08, −0.06, −0.03, +0.01, +0.01, +0.08, −0.03).
  - mAUC +0.07 (not read).
- **Per machine, Δ AUC dev / eval (8 seeds):**

  | machine | fan | pump | slider | ToyCar | ToyConveyor | valve |
  |---|---|---|---|---|---|---|
  | dev | −0.02 | +0.03 | +0.06 | +0.09 | **+0.17** | +0.01 |
  | eval | +0.01 | +0.01 | +0.01 | **+0.10** | +0.09 | **−0.09** |

- **Per ID** (`perid_delta.csv`, 8-seed mean, seeds positive):
  - ToyConveyor gains on every ID on both sets: +0.09 to +0.23, 5–7/8 seeds each.
  - ToyCar: dev id_01 +0.31 (8/8) and eval id_05 +0.25 (87.59 → 87.83, 7/8) gain. These are
    the weakest IDs, the same ones C1 helped.
  - Pump: dev id_00 +0.14 and id_02 +0.12; eval id_03 −0.04.
  - Valve: every per-ID AUC is unchanged or up on eval (id_05 +0.06, 5/8). Yet pooled valve
    AUC falls 0.09 and pAUC 0.37.
    - So TTA shifts the per-ID score scales relative to each other on valve, and the pooled
      ranking across IDs gets worse.
    - That is a calibration effect of the z-scoring, not a detection loss inside any ID.
- **S1′ᵀ against C1** is −0.05 on dev and −0.16 on eval. It loses 0.97 Amean on eval
  ToyConveyor and 0.56 on ToyCar, and wins 0.43 on valve.
  - So the pooled term and TTA are complementary: C1ᵀ keeps C1's ToyCar / ToyConveyor gains and
    adds TTA's on top.
  - On eval valve, C1 already lost to S1′ (`pooled-knn`: −0.35 AUC), and TTA does not recover it.

## Decision

| D1 condition | value | pass |
|---|---|---|
| gate A: stage-A dev Δ Amean > 0 | +0.083 (4/4) | yes |
| dev 8-seed Δ Amean > 0 with ≥ 6/8 seeds positive | +0.098, 8/8 | yes |
| confirm-seed (2044–2047) dev Δ Amean > 0 | +0.112 (4/4) | yes |
| eval 8-seed Δ Amean ≥ 0 | +0.014 | yes (narrowly) |

**C1ᵀ is the single-model rule** (`scores/final_rule.json`; `score.py: score_member`, default
rule `C1T`).

- Read it as "TTA's dev gain carries to C1, and eval does not contradict it". Eval does not show
  a gain: the eval CI includes 0, and 3 of 8 seeds are negative.
- The rule was fixed before eval, and it passes. `fresh-confirm` scores C1ᵀ and C1 on seeds no
  rule has seen. If C1ᵀ falls below C1 there, that is the evidence to revert. A small eval
  shortfall alone is not: C1 is the fallback, simpler and 3× cheaper.

## Prediction check

| prediction | outcome | held? |
|---|---|---|
| P0 reproduction ≤ 1e-3 | 0.0 at 4 dp on every cell | **yes** |
| P1 gate A Δ +0.06 [−0.03, +0.15]; passes (70 %) | +0.083; passes | **yes** |
| P2 dev8 Δ AUC +0.03 [−0.03, +0.10] | +0.057 | **yes** |
| P2 dev8 Δ pAUC +0.08 [−0.05, +0.20] | +0.138 | **yes** |
| P2 dev8 Δ Amean +0.06 [−0.02, +0.14] | +0.098 | **yes** (above central) |
| P2 ≥ 6/8 seeds positive (60 %) | 8/8 | **yes** |
| P2 95 % CI includes 0 (55 %) | [+0.006, +0.214] | **no** |
| P3 confirm Δ > 0 (70 %); smaller than stage A (55 %) | +0.112 > +0.083 | yes; **no** (no winner's curse, as in `pooled-knn`) |
| P4 eval Δ +0.03 [−0.05, +0.10]; ≥ 0 (65 %); CI includes 0 (65 %) | +0.014; ≥ 0; CI includes 0 | **yes** |
| P5 ToyConveyor largest dev Δ AUC (50 %), central +0.15 | +0.17, largest | **yes** |
| P5 fan, pump, slider \|Δ\| ≤ 0.05 dev | −0.02, +0.03, **+0.06** | **no** (slider, just) |
| P5 ToyCar Δ AUC in [−0.10, +0.15] both sets | +0.09 / +0.10 | **yes** |
| P5 valve \|Δ AUC\| ≤ 0.10 both sets | +0.01 / −0.09 | **yes** (just) |
| P5 eval ToyCar id_05 \|Δ\| < 0.5 | +0.25 | **yes** |
| P5 eval valve id_05 not repaired, \|Δ\| ≤ 0.2 | +0.06 (C1 94.62, S1′ᵀ 95.25) | **yes** |
| P6 C1ᵀ above S1′ᵀ on dev and eval (85 %) | 94.26 vs 94.12; 94.10 vs 93.93 | **yes** |
| P7 D1 passes (45 %); central call a narrow fail | passes, narrowly on eval | central call **wrong** |

The dev effect came in near the top of the predicted range, not at "half of S1′'s gain". The
three-term sum did not dilute TTA. On dev, C1ᵀ's gain over C1 (+0.10) equals TTA's gain over S1′
(+0.11). On eval, the shrink was as large as predicted.

## What does NOT replicate / what this cannot answer

- **The dev gain does not carry to eval at strength.** +0.10 on dev against +0.01 on eval, 8/8
  against 5/8 seeds. The same happened to S1′ᵀ (+0.11 → +0.05) and to most phase-1 dev gains.
  Selection was dev-only, so this is not overfitting to eval. It may be dev-to-eval transfer
  of a jitter-smoothing effect, or simply noise.
- **Eval valve goes the other way** (−0.09 AUC pooled, with no per-ID loss). It is not
  understood beyond "cross-ID calibration".
- **zGmean was not shifted.** The roll-invariance argument makes that a small omission, but it
  is untested. C1ᵀ with a shifted gmean would need a GPU dump.
- **One recipe** (`ao_nogain`), 8 seeds that every earlier rule has also seen. `fresh-confirm` is
  the out-of-sample check.
- **Ensembles are not addressed.** `tta` found TTA useless or harmful on E2′ (G8 −0.05 eval).

## Figures

`/home/f74134118/anaconda3/bin/python synthesis/single-final/make_plots.py` reads `scores/` only
(or `bash synthesis/single-final/run.sh plots`).

| figure | shows |
|---|---|
| `plots/fig1_delta.png` | C1ᵀ − C1 Amean by stage (gate, confirm, dev 8, eval 8): per-seed dots, paired 95 % bootstrap, seeds positive; orange ticks are S1′ᵀ − C1. Dev +0.10 (8/8), eval +0.01 (5/8) |
| `plots/fig2_per_machine.png` | C1ᵀ − C1 AUC per machine, dev and eval, with seed dots. ToyCar and ToyConveyor gain on both sets; eval valve −0.09 |
| `plots/fig3_absolute.png` | 8-seed Amean of S1′, S1′ᵀ, C1, C1ᵀ on dev and eval, with seed dots and the paper line |

## Using the adopted rule (for `fresh-confirm`)

```python
import importlib.util
spec = importlib.util.spec_from_file_location('sf', 'synthesis/single-final/score.py')
sf = importlib.util.module_from_spec(spec); spec.loader.exec_module(sf)
# tr / te: dicts with 'ce' (n, K), 'pre' (n, K, 128), 'shifts' (K,) incl. 0, +0.01, -0.01
#          (ablation/tta/extract_tta.py --shifts 0,0.01,-0.01), and 'gmean' (n, 512), unshifted
#          (ablation/pooled-knn/extract.py); tr = all training clips in meta order.
# tr_label / te_label: machine-ID class (meta.npz tr_label / te_label); valid: the run's replayed
# 5 % held-out mask (pooled-knn train dump 'valid').
score = sf.score_member(tr, te, tr_label, te_label, valid)            # rule = 'C1T' (final_rule.json)
c1 = sf.score_member(tr, te, tr_label, te_label, valid, rule='C1')    # the fallback, uses shift 0 only
```

On smartlab1, compute each score twice and compare (`sf.twice(lambda: ...)`), as this study did.

## Files

| file | what |
|---|---|
| `score.py` | CPU scorer. Stages `check`, `stageA`, `stageB`, `eval`, each gated on the previous one's committed output. Importable: `score_member(tr, te, tr_label, te_label, valid, rule=FINAL_RULE)`, `tta_reduce`, `member_terms` |
| `run.sh` | stages `check`, `stageA`, `stageB`, `eval`, `summary`, `plots`; resumable (verified per-member score cache in `results/cache`, gitignored) |
| `make_plots.py` | figures from `scores/` only |
| `scores/` | committed record: `repro_check.json`, `metrics_{check,stageA,stageB,eval}.csv` (per seed × rule × machine), `perid_*.csv`, `stageA.json`, `stageB.json`, `eval.json` (Δs, CIs, D1), `summary.csv` (8-seed means and Δ vs C1), `perid_delta.csv`, `final_rule.json`, `scorecard_*.txt` |
| `results/` | logs only: gitignored scratch |
