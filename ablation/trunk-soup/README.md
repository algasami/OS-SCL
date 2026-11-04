# Trunk soup: one network with an ensemble's stability

> **Status: DONE 2026-11-04. D1 fails; the soup recipe is not adopted.**
> - **Weight averaging works.** The soup of 4 forks is the 4-fork score
>   ensemble in one network: under S1′ it is −0.01 and +0.00 dev AUC from it.
>   The forks stay in one basin.
> - **There is little to average.** The 4-fork ensemble gains only +0.11
>   over its mean fork, against +0.64 for 4 independent seeds (P2: 17–18%).
>   Most seed variance is fixed before epoch 150. So the soup is +0.06 / +0.11
>   dev over its own forks (P1 fails) and +0.05 over the independent seeds
>   (D1 needs +0.20).
> - **Both trunks drew lucky eval seeds.** Every network from them scores
>   +0.5 to +0.9 eval AUC over the independent mean. The code and recipe are
>   identical, so this is seed luck, not the soup. Within-trunk eval gains
>   are +0.08 and 0.00.
> - Base `ao_nogain` (`../aug-ops` C2), fixed in the addendum before any
>   trunk was trained. 10 runs, no crashes.

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

## Results (2026-11-04)

All 10 runs finished without a crash or retry. Trunks took 11.2–11.4 h on
GPUs 0 and 2. Forks took 5.7 h on GPUs 0 and 2, and 6.9 h on GPU 1. Every
fork printed `resume check ... ok` against its trunk's epoch-150 validation
loss. `run.sh soup` built soup4 (the 4 forks) and soup5 (the 4 forks plus the
trunk's own second half) per trunk.

**Scoring:**

- `run.sh score` produced the R0 CSVs with `eval.py`.
- `score_ts.py` scores the `../ensemble-backend/extract.py` dumps
  (`results/emb`, scratch) with ensemble-backend's rules, imported unchanged.
- Its S0 matches `eval.py`'s per-ID AUCs to 0.019 points.
- Full output: `scores/summary.txt`, `systems.csv`, `machines.csv` and
  `check.json`.

R0, AUC (dev | eval):

| | trunk s2050 | trunk s2051 |
|---|---|---|
| forks 1–4 | 96.20 95.90 95.90 95.91 \| 96.24 95.86 96.06 95.89 | 96.04 96.19 96.12 96.25 \| 95.66 96.08 95.72 95.92 |
| mean fork | 95.98 \| 96.01 | 96.15 \| 95.84 |
| **soup4 (one network)** | **96.04 \| 96.10** | **96.26 \| 95.84** |
| soup5 | 96.03 \| 96.13 | 96.30 \| 95.87 |
| trunk, own second half | 96.02 \| 96.25 | 96.32 \| 95.99 |
| E1′ over 4 forks (4 networks) | 96.13 \| 96.05 | 96.33 \| 95.97 |
| independent `ao_nogain`, 8 seeds | 96.10 \| 95.30 | |

S1′ (dev AUC | eval mean(AUC, pAUC)):

| | s2050 | s2051 |
|---|---|---|
| mean fork | 96.38 \| 94.50 | 96.53 \| 94.54 |
| soup4 | 96.47 \| 94.65 | 96.65 \| 94.61 |
| E2′ over 4 forks | 96.49 \| 94.63 | 96.65 \| 94.63 |
| independent: one model 96.43 \| 93.88; 4 seeds (`hom4_nogain`) 97.07 \| 94.71 | | |

### Predictions, checked as written

| | prediction | result | verdict |
|---|---|---|---|
| P1 | soup4 − mean fork in [+0.10, +0.40] dev, ≥ 0 eval, both trunks | dev +0.06 / +0.11; eval +0.08 / −0.00 | **fails** (s2050 dev below the range) |
| P2 | E2′(4 forks) − mean fork under S1′ ≤ +0.38 (60% of the independent +0.64) | +0.11 / +0.11 (17% / 18%) | **holds**, far inside the bound |
| P3 | \|soup4 R0 − E1(4 forks)\| ≤ 0.25 dev | +0.27 / +0.09 | **fails by the letter**; see below |
| P4 | ToyConveyor has the largest soup4 − mean fork gain on eval | s2050: ToyCar +0.23 > ToyConveyor +0.19; s2051: ToyConveyor −0.22 | **fails** |
| P5 | mean soup4 − independent mean dev AUC in [+0.10, +0.40] | 96.15 − 96.10 = +0.05 | **fails** |
| P6 | mean soup4 eval mean(AUC, pAUC) ≥ 93.22 | 93.98 | holds, but only through trunk luck (below) |

**P3 is a letter-only failure, and it goes the other way from what D3
guards against.**

- `score_ts.py` took P3's "E1" to be `../ensemble-backend`'s E1, with
  held-out z-statistics. `../confirm-seeds` replaced that rule because
  held-out statistics mis-scale IDs.
- Here it scores *below* the mean fork on s2050 (95.77 vs 95.98). So the
  +0.27 is the soup sitting **above** a weak ensemble.
- With training-part statistics (E1′) the gaps are −0.09 / −0.07. Under S1′
  the soup equals the E2′ fork ensemble: −0.01 / +0.00.
- Linear connectivity from epoch 150 holds by every reasonable reading.

### Decisions

- **D1 fails.** The dev gain is +0.05 against the ≥ +0.20 bar, and P3
  fails by the letter. **The soup recipe is not adopted.** `ao_nogain`
  stays the single-model recipe.
  - The eval AUC gain over the independent seeds, +0.67, meets D1's eval
    half. It is not a soup effect, though: the trunks' own second halves
    score the same, and within-trunk gains are +0.08 and 0.00.
- **D2 is not run.** It only applies after D1.
- **D3 is not triggered.** D3 targets the soup "far below its fork
  ensemble", i.e. forks that left the basin. The soup is at or above every
  fork ensemble here. A later fork point (225) would also make forks *more*
  alike, which is the opposite of what is missing. (`score_ts.py` first
  printed "D3 trigger: yes" from P3's letter. It was corrected to test the
  pre-registered direction before this write-up.)

### What this shows

- **Seed variance is set early.** Two forks of one trunk differ from each
  other far less than two independent seeds:
  - the fork ensemble gain is +0.11, against +0.64 for independent seeds;
  - forks track their trunk on dev AUC after epoch 150 (fig3), and spread by
    ~1.5 points only on dev ToyConveyor.

  So the ensemble's stability comes from the init and the first half of
  training, not from late data order or mixup draws.
- **The trunk seed dominates.** All 10 networks from s2050 and s2051 sit
  +0.5 to +0.9 eval AUC above the independent mean.
  - The recipe and code are identical (the configs differ only in seed and
    `--state_snapshots`, and saving a snapshot draws no random numbers).
  - s2051's trunk (95.99) sits at the top of the independent seeds' range,
    and s2050's (96.25) is above its maximum of 96.01.
  - Comparisons against the independent baseline on eval are therefore
    confounded at n=2. The within-trunk numbers (P1, P2) are the clean ones.
- **A single soup under S1′ beats every paper number:** dev 96.47–96.65 /
  91.67–92.04 / 92.80–93.36, eval mean 94.61–94.65. So do its forks, so this
  is not a claim for the soup.

### What next

- An **earlier fork point** (e.g. epoch 50–100) is the only version of this
  idea left. Forks would have more room to differ, at the risk of leaving
  the basin (the EMA probe sits at 69 dev AUC at epoch 99). Not
  recommended: the independent-seed ensemble already gets the diversity, and
  the single-network goal is met by `ao_nogain`.
- Seed luck at the level seen here (two trunks both near the top on eval) is
  a reminder that n=2 comparisons against a different seed pool are
  unreliable, even for eval.

## Figures

| file | shows |
|---|---|
| `plots/fig1_soup_vs_forks.png` | R0 dev and eval AUC per trunk: forks, mean fork, soup4/soup5, the trunk's own second half and the E1′ fork ensemble, beside the 8 independent seeds. The soup matches the fork ensemble, both sit barely above the mean fork, and both trunks sit high on eval |
| `plots/fig2_machines.png` | soup4 − mean fork per machine, dev and eval. Gains are ≤ 0.5 everywhere; ToyConveyor leads on dev but not on eval (P4 fails) |
| `plots/fig3_probe_curves.png` | Probe dev AUC and dev ToyConveyor per epoch for each trunk and its forks. Forks track the trunk after epoch 150 and spread only on ToyConveyor |
