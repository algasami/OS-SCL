# Trunk soup: one network with an ensemble's stability

> **Status: designed 2026-10-17; code and smoke test done; nothing trained.**
> Phase 2 of the plan. It runs after `../lit-arms`, on the best single-model
> recipe at that point. That base is fixed in a dated addendum below **before**
> any trunk is trained.

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

*(empty: waiting for `../lit-arms`)*

## Results

*(empty)*

## Figures

*(empty)*
