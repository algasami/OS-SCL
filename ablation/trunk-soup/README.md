# Trunk soup: one network with an ensemble's stability

> **Status: TRAINING since 2026-11-03.** The base is `ao_nogain`
> (`../aug-ops` C2), fixed in the addendum below before any trunk was
> trained, and smoke-tested on that base. Trunks T1 and T2 are on GPUs 0 and
> 2; 8 forks follow from epoch 150.

## Why this exists

A 4-seed ensemble beats the paper on every number. Its best version, E2′,
scores 96.05 dev AUC, while one network under S1′ reaches 95.36
(`../confirm-seeds`). Most of the ensemble's gain is ToyConveyor's seed
variance averaging out: +4.5 dev AUC on ToyConveyor against the mean member
(`../ensemble-backend`). Distilling the ensemble into one network was null
(`../distill`).

A different way to get one network: train a **trunk**, **fork** it halfway
into K copies that finish training under different data orders and mixup
draws, then **average the forks' weights** (a uniform "model soup", Wortsman
et al. 2022). Forks of a shared trunk usually stay in one linearly connected
basin (Frankle et al. 2020), so their average is a working network. The
result has the same architecture and the same inference cost as one network.
It is a single model under any reading of the paper's comparison. Independent
seeds can't be averaged this way, because they sit in different basins.

The open question: is late-training variance (data order, mixup draws) a
large enough share of ToyConveyor's seed variance for fork diversity to
matter? `../fast-recipe`'s probe logs dev AUC every 5 epochs, so it shows how
much ToyConveyor moves late in training. Those curves get read before the
addendum fixes the fork point.

## Code (on this branch, from `../lit-arms`)

- `--state_snapshots 150` saves the full training state at epoch 150. That is
  weights, EMA, optimiser, scheduler, scaler, best validation loss and RNG
  streams.
- `--resume_state <trunk>/state_e150.pth --fork_seed N`, run with `--seed`
  set to the **trunk's** seed. That seed redraws the trunk's exact train/valid
  split; the loaded state then replaces the weights, and every RNG is re-seeded
  with N. A fork picks its own best epoch from the fork point on.
- After loading, a fork recomputes the snapshot epoch's validation loss and
  prints `resume check ... ok`. It must match the trunk's logged value.
- `soup.py` averages the members' `model.pth` (EMA weights, BatchNorm running
  statistics included). It refuses members that aren't forks of one trunk.

**Smoke test (2026-10-17):** a 2-epoch trunk was snapshotted at epoch 1 and
forked twice for one more epoch. Both forks print `resume check | valid loss
18.13256 vs trunk 18.13256 | ok`. They diverge (max |Δw| 0.31). The soup is
the exact element-wise mean and `eval.py` scores it. `soup.py` rejects a
non-fork member.

## Design (base recipe: see addendum)

- **Trunks:** T1 and T2 (seeds 2050 and 2051), 300 epochs, snapshot at epoch
  150. The trunk's own epochs 150–300 count as one more fork, `k0`, with no
  re-seed.
- **Forks:** 4 per trunk (fork seeds 1–4), each finishing epochs 150 → 300.
- **Soups:** `soup(T)` = the average of T's 4 forks. Sensitivity: the average
  of 5 (with `k0`).
- **Comparators,** all on the same base recipe:
  - each fork alone (single networks);
  - the forks as a score ensemble, E2′ over the 4 forks. This is the ceiling
    of fork diversity; not a single model.
  - the independent-seed baseline: the base recipe's 4 seeds, their mean and
    their E2′ ensemble.
- **Cost:** 2 trunks plus 8 half-runs = 6 run-equivalents. At the f4 speed
  that is about 48 GPU-hours, or 16–20 h on three GPUs.

## Predictions (written 2026-10-17, before any trunk exists)

Δ in points of dev R0 AUC unless stated.

- **P1, soup beats its own members.** soup(T) − mean fork of T is +0.1 to
  +0.4, on both trunks. EMA already averages along one trajectory; forks add
  data-order diversity.
- **P2, forks are less diverse than seeds.** The fork ensemble's gain over its
  mean member is ≤ 60% of the independent-seed ensemble's gain over its mean
  member. For reference, E1 independent gains +0.73 dev (`../confirm-seeds`).
- **P3, the soup tracks the fork ensemble.** |soup(T) − E-ensemble of T's
  forks| ≤ 0.25 under R0. That is linear connectivity from epoch 150.
- **P4, ToyConveyor carries the gain.** In soup − mean fork, ToyConveyor
  contributes the most of any machine type.
- **P5, versus independent single models.** The mean soup over 2 trunks minus
  the independent baseline's mean is +0.1 to +0.4.

## Decision rules

- **D1, adopt the soup recipe** if, under R0, (mean soup − independent mean) ≥
  +0.20 on dev *and* ≥ 0 on eval, and P3 holds on both trunks.
- **D2.** If adopted, add trunks 3–4 (seeds 2052 and 2053) to reach n=4 soups.
  Report the soup under R0 and S1′ against the paper's dev numbers as the
  single-network claim.
- **D3.** If P3 fails (the soup is far below its fork ensemble), the forks left
  the basin. Re-run one trunk with the fork point at 225 before giving up.

## Addendum: base recipe (to be fixed before the first trunk)

*Written 2026-11-03, before any trunk was trained. The design, P1–P5 and
D1–D3 above are unchanged; this fixes the base recipe and turns P1–P5 into
numbers for the new base.*

**Base recipe: `ao_nogain`**, the single-model recipe since `../aug-ops` C2.

- Flags: `TS_FUSSION=1`, `TS_FLAGS="--wave_aug 1 --wave_aug_ops noise,shift"`,
  on top of `run_one.sh`'s fast-recipe base (`--amp fp16 --cudnn_bench 1
  --lean 1 --cache_dir data/cache --probe_every 5 --num_workers 8`, 300
  epochs).
- This is the exact recipe of `ao_nogain_s2040–2047`, so those 8 runs are
  the **independent baseline**. No new independent seeds are trained.
- Launch: `TS_FUSSION=1 TS_FLAGS="--wave_aug 1 --wave_aug_ops noise,shift"
  GPUS="0 2 1" bash ablation/trunk-soup/run.sh train`. The GPU order puts
  the two 300-epoch trunks on the fast cards (GPU 1 is PCIe x1, CLAUDE.md).

**Fork point stays at epoch 150.** The probe logs of the 8 `ao_nogain`
runs show the EMA net still learning there:

| epoch | 99 | 124 | 149 | 174 | 199 | 224 | 249 | 299 |
|---|---|---|---|---|---|---|---|---|
| dev AUC, probe mean over 8 | 69.3 | 84.6 | 91.1 | 93.5 | 95.3 | 96.0 | 96.1 | 96.1 |
| dev ToyConveyor, sd across seeds | 3.00 | 3.57 | 1.80 | 0.84 | 0.69 | 0.59 | 0.62 | 0.67 |

So forks have half of training, including the whole climb from 91 to 96,
in which to diverge. Within one run, ToyConveyor's probe moves by an sd of
0.27–0.77 over epochs 200–299, so late training still varies. D3's fallback
(a fork point at 225) stays as written.

**Smoke test on this base (2026-11-03, GPU 1).** A 2-epoch trunk
(`--state_snapshots 1`) was forked twice for 1 epoch:

- Both forks print `resume check | valid loss 19.15265 vs trunk 19.15265 | ok`.
- Their `config.json` carries `wave_aug_ops: noise,shift`.
- The forks differ (max |Δw| 0.155), and the soup is their exact element-wise
  mean.
- The smoke checkpoints were deleted, so `run.sh score` cannot pick them up.

**Reference numbers.** Independent `ao_nogain`, 8 seeds, R0, mean (sd across
seeds):

| | dev AUC | eval AUC | eval mean(AUC, pAUC) | eval ToyConveyor |
|---|---|---|---|---|
| independent `ao_nogain` | 96.10 (0.23) | 95.30 (0.48) | 93.14 (0.60) | 83.62 (2.07) |
| paper | 95.71 | — | 93.22 | — |

Independent ensembles, E2′ = Σ S1′ (`../nogain-ensemble`):

- A single model under S1′ scores dev AUC 96.43.
- `hom4_nogain` scores 97.07, a gain of **+0.64** from 4 independent members.
- G8 scores 97.19.

**Scoring.**

- Primary: R0 via `run.sh score`.
- Secondary: S1′, E1 and E2′. These use `../ensemble-backend/extract.py`
  dumps and `../aug-ops/score_s1p.py`'s rule, both unchanged.
- E1 (sum of per-ID z-scored R0) over a trunk's forks is the "fork ensemble"
  in P3. E2′ over the forks is the one in P2.

**Power.** D1 compares the mean of 2 soups with the mean of 8 independent
runs. If soups vary like single seeds, that difference has se ≈ 0.18 dev and
≈ 0.38 eval. So D1's +0.20 dev bar is about 1.1 se: only a large gain passes
it on 2 trunks. That is why D2 adds trunks 3–4. P1 is paired within a trunk,
so it is the better-powered test of the mechanism.

**Predictions, made numeric** (R0 dev AUC unless stated):

- **P1.** soup4(T) − mean fork of T lies in [+0.10, +0.40] for both trunks,
  central +0.20. On eval, soup4 − mean fork ≥ 0 for both trunks.
- **P2.** E2′ over T's 4 forks, minus the mean fork under S1′, is ≤ +0.38
  (60% of +0.64) on both trunks. Central +0.25.
- **P3.** |soup4(T) under R0 − E1 over T's forks| ≤ 0.25 on dev, for both
  trunks.
- **P4.** In soup4 − mean fork, ToyConveyor has the largest per-machine Δ on
  eval. Dev ToyConveyor is already tight (sd 0.48), so P4 is judged on eval.
- **P5.** Mean soup4 over the 2 trunks, minus the independent mean, lies in
  [+0.10, +0.40], central +0.20, i.e. dev AUC ≈ 96.30.
- **P6 (new).** Mean soup4 eval mean(AUC, pAUC) under R0 is ≥ the paper's
  93.22. Central 93.40, about 55%. The independent runs sit at 93.14, with 4
  of 8 seeds above.

## Results

*(empty)*

## Figures

*(empty)*
