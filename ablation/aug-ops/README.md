# aug-ops: which waveform-augmentation op costs ToyCar id_05?

> **Status: stage 1 DONE 2026-10-31 (D0). The `ao_nogain` replication on seeds 2044–2047 is
> pre-registered and training** (see "Replication of `ao_nogain`").
> **Stage 1: no arm passes D1, so stage 2 does not run.**
> E2′[A8] stays the best ensemble and `aug` stays as it is.
> - No single op causes the id_05 loss. Dropping the shift recovers the most:
>   id_05 85.11 against `aug`'s 81.33 and the 86.0 bar.
> - Noise carries `aug`'s ToyConveyor gain.
> - Unplanned: `ao_nogain` beats the paper's three dev numbers on 4/4 seeds
>   under R0 (dev AUC 95.96–96.49). It is post hoc, see "Not planned".
>
> Predictions and decision rules below were written before any 300-epoch run.

## Why this exists

`aug` (`../lit-arms`, `--wave_aug 1`) drives the best ensemble, E2′[A8]
(`../mixed-ensemble`: dev AUC 97.05, eval mean(AUC, pAUC) 94.78). Its one known
hole is eval ToyCar: A8 scores 96.00 there, against 97.5–97.7 for ensembles
without `aug`.

The hole is **one machine ID**. Eval ToyCar per ID, R0, 4-seed means
(seeds 2040–2043):

| eval ToyCar | `fr_f1` (plain) | `aug` | Δ |
|---|---|---|---|
| id_05 | 88.66 | **81.33** | **−7.33** |
| id_06 | 98.99 | 98.89 | −0.10 |
| id_07 | 100.00 | 99.72 | −0.28 |

- It is systematic. `aug`'s id_05 is 78.8–84.0 on all 8 seeds (2040–2047);
  plain's is 87.9–90.4 on its 4.
- It has a mirror on dev. `aug` *gains* +7.15 on dev ToyCar id_01 (84.94 →
  92.09), the hardest dev ToyCar ID. So dev alone cannot see the eval loss.
- **Two explanations are ruled out** (checked 2026-10-28, before this
  pre-registration):
  - *The noise drowns a quiet ID.* id_05 is the loudest ToyCar ID: median
    training RMS 0.028, against 0.021–0.027 for the others. `aug`'s noise
    (std 0.001–0.005) is at an SNR of 15–29 dB there.
  - *The anomalies are a loudness change that ±10% gain teaches the model to
    ignore.* Clip RMS alone separates eval id_05's anomalies at AUC 54.1, no
    better than id_06 (54.2) or id_07 (55.2).

`aug` is three ops, each applied per clip with p=0.3: Gaussian noise (std ~
U[0.001, 0.005]), a circular shift of ±U[3%, 8%] of the clip, and gain ~
U[0.9, 1.1]. This ablation drops one op at a time.

## Design

- **Arms (leave-one-out):**
  - `ao_nonoise`: shift + gain
  - `ao_noshift`: noise + gain
  - `ao_nogain`: noise + shift

  Leave-one-out rather than single-op: the arm that repairs id_05 is then
  itself the candidate recipe, so no untrained combination has to be
  extrapolated.
- **Recipe:** the fast recipe at fussion 1 (`--amp fp16 --cudnn_bench 1
  --lean 1 --cache_dir data/cache --probe_every 5 --num_workers 8`, 300 epochs),
  plus `--wave_aug 1 --wave_aug_ops <ops>`. Identical to `la_aug` except for
  the dropped op.
- **Seeds:** 2040–2043, the seeds of `fr_f1` (no ops) and `la_aug` (all
  three). smartlab1 is not bit-deterministic over a full run (CLAUDE.md), so the
  pairing is by seed and split, not bit-exact.
- **Cost:** 12 runs. At 11.5 h on GPUs 0/2 and ~14 h on GPU 1, that is
  about 50 h on three GPUs.
- **Scoring for stage 1:** R0 (`eval.py`), per ID, as in `../lit-arms` D1.
  No extraction is needed.

### Code

- `--wave_aug_ops` (`train.py`, `trainer.py`) takes a comma-separated subset
  of `noise,shift,gain`; the default is all three. A disabled op still draws
  its random mask, so the remaining ops keep their place in the random stream.
- Checked before launch:
  - With all three ops, `augment` reproduces the old code exactly at a fixed
    seed (`torch.equal` on waves and mels).
  - At p=1, gain-only output is exactly g·w per clip, with g in [0.903, 1.099].
  - Shift-only output is an exact roll of each clip.
  - Noise+shift never changes the gain.
- A 2-epoch smoke run (`ao_nonoise_s1_e2`) trains and probes, and its
  `config.json` records `wave_aug_ops: shift,gain`.

## Predictions (written 2026-10-28, before any run)

R0, 4-seed means. Reference values: `la_aug` and `fr_f1`, seeds 2040–2043.

- **P1, a single culprit.** Exactly one arm brings eval ToyCar id_05 to
  ≥ 86.0. That is about two thirds of the 7.3-point loss back, and over 5 se
  above `aug`'s 81.33. Prior on which op it is:
  - noise 50%
  - shift 30%
  - gain 5%
  - no single op (an interaction, or none repairs it) 15%

  So the central call is `ao_nonoise`. Reasoning: noise is the only op that
  changes the spectrum's low-level fine structure, and loudness has been ruled
  out above.
- **P2, gain does nothing.** `ao_nogain` is within ±0.3 of `la_aug` on dev
  AUC and within ±2.5 on eval id_05. A ±10% gain is ±0.8 dB on the log-mel.
- **P3, the mirror.** The arm that repairs eval id_05 also gives back at least
  half of `aug`'s dev ToyCar id_01 gain: dev id_01 ≤ 88.5 (`fr_f1` 84.94,
  `la_aug` 92.09). One op, one invariance, two IDs.
- **P4, ToyConveyor.** Dropping noise costs at least half of `aug`'s eval
  ToyConveyor gain: `ao_nonoise` eval ToyConveyor ≤ 82.1 (`fr_f1` 80.81,
  `la_aug` 83.38). Coin-flip confidence.
- **P5, net gain.** The repairing arm's eval overall AUC is ≥ `la_aug`'s
  95.45 + 0.20.

## Decision rules

- **D1, a candidate recipe.** An arm is a candidate if all three hold:
  - (a) eval ToyCar id_05 ≥ 86.0;
  - (b) eval overall AUC ≥ `la_aug`'s 95.45, so the repair isn't paid for
    elsewhere;
  - (c) dev overall AUC ≥ `fr_f1` + 0.40 = 95.30, keeping at least half of
    `aug`'s +0.83 dev gain.

  If several qualify, the highest dev mean(AUC, pAUC) wins. That is the same
  dev-picks rule that chose A8.

  **Note:** criterion (a) is by construction an eval-set selection, because dev
  cannot see id_05. That is why D2 exists.
- **D0, no candidate.** Stop. Report which op does what. E2′[A8] stays the
  best ensemble, and `aug` stays as it is.
- **D2, stage 2 (only if D1 picks an arm).**
  - Train the candidate on 8 fresh seeds, 2060–2067. That is about 33 h.
  - Dump embeddings with `../ensemble-backend/extract.py` and build E2′[A8′]
    exactly as `../mixed-ensemble` built A8.
  - **Adopt A8′ as the best ensemble** if its eval mean(AUC, pAUC) minus
    E2′[A8]'s has a paired clip-bootstrap 95% lower bound > 0 (2000
    resamples), **and** its dev mean(AUC, pAUC) is no more than 0.30 below
    A8's 94.77.
  - Projection: eval 94.95–95.05 (A8 94.78); dev ≈ 94.5–94.7.
  - Report single models under R0 and S1′ against the paper, as before.

## Results (2026-10-31)

All 12 runs finished. One segfault (`ao_nogain_s2040`, rc 139) resumed from
`state_last.pth`. No cached item needed recomputing and no step was skipped.
R0, 4-seed means, seeds 2040–2043. Δ is seed-paired against `la_aug`, with se in
brackets. Full table: `scores/summary.txt`.

| | `fr_f1` | `la_aug` | `ao_nonoise` | `ao_noshift` | `ao_nogain` |
|---|---|---|---|---|---|
| dev AUC | 94.90 | 95.73 | 95.12 −0.61 (0.07) | 95.70 −0.04 (0.25) | **96.17 +0.44 (0.15)** |
| dev pAUC | 89.58 | 90.85 | 89.91 | 90.53 | 91.16 |
| eval AUC | 95.25 | 95.45 | 95.06 −0.38 (0.27) | 95.50 +0.05 (0.30) | 95.17 −0.27 (0.25) |
| eval pAUC | 90.24 | 91.20 | 91.00 | 90.80 | 90.90 |
| **eval ToyCar id_05** | 88.66 | 81.33 | 82.32 +1.00 (2.30) | **85.11 +3.79 (2.49)** | 81.60 +0.27 (0.76) |
| eval ToyCar id_06 | 98.99 | 98.89 | 99.52 | 97.79 −1.10 (0.44) | 98.60 |
| dev ToyCar id_01 | 84.94 | 92.09 | 89.01 −3.08 | 89.02 −3.07 | 91.69 −0.40 |
| dev ToyConveyor | 82.14 | 84.10 | 81.67 | 84.75 | 85.96 |
| eval ToyConveyor | 80.81 | 83.38 | **80.42** | 83.50 | 83.18 |

Per-seed eval id_05:

- `ao_noshift`: 85.07, 81.35, 86.39, 87.64
- `ao_nonoise`: 80.51, 80.13, 84.04, 84.62
- `ao_nogain`: 82.20, 82.98, 79.97, 81.25

### D1: no candidate

| arm | id_05 ≥ 86.0 | eval AUC ≥ 95.45 | dev AUC ≥ 95.30 | |
|---|---|---|---|---|
| `ao_nonoise` | 82.32 no | 95.06 no | 95.12 no | out |
| `ao_noshift` | **85.11 no** | 95.50 ok | 95.70 ok | out |
| `ao_nogain` | 81.60 no | 95.17 no | 96.17 ok | out |

**D0 applies:** stop. E2′[A8] stays the best ensemble, and stage 2 (fresh seeds
2060–2067) is not run.

`ao_noshift` misses only (a), by 0.9. Two of its four seeds clear 86. Even so, it
would not be worth an ensemble:

- Its eval AUC ties `aug` (+0.05 ± 0.30), and its eval pAUC is 0.39 lower.
- It pays for part of the id_05 repair on id_06 (−1.10 ± 0.44).

So the id_05 repair does not carry through to the overall score that A8′ would
need to win on.

### What each op does

- **Shift is the largest share of the id_05 loss, but not all of it.**
  Dropping it recovers 3.8 of 7.3 points, and also gives back 3.1 of `aug`'s
  7.2-point dev id_01 gain. That supports the mirror: one invariance, two IDs.
- **Noise carries the ToyConveyor gain.** Without noise, eval ToyConveyor is
  80.42, at `fr_f1`'s level (80.81), against `aug`'s 83.38. Noise also costs id_05
  about 1 point.
- **Gain does nothing to id_05** (+0.27 ± 0.76). Dropping it *raises* dev
  (+0.44 ± 0.15) and lowers eval (−0.27 ± 0.25).
- The three leave-one-out recoveries (+1.0, +3.8, +0.3) sum to 5.1 of 7.3. The
  rest is either an interaction or noise: these deltas carry se ≈ 2.4.

### Predictions, checked as written

| | prediction | result | verdict |
|---|---|---|---|
| P1 | exactly one arm reaches id_05 ≥ 86; central call noise | none does; shift recovers most (85.11), noise least of the two (82.32) | **fails**, and the central call is wrong |
| P2 | `ao_nogain` within ±0.3 of `aug` on dev AUC and ±2.5 on id_05 | id_05 +0.27 holds; dev +0.44 (se 0.15) | **fails** on dev |
| P3 | the repairing arm gives back ≥ half of the dev id_01 gain (id_01 ≤ 88.5) | no arm repairs; best repairer `ao_noshift` at 89.02 gives back 43% | **fails narrowly** (direction holds) |
| P4 | `ao_nonoise` eval ToyConveyor ≤ 82.1 | 80.42 | **holds** |
| P5 | the repairing arm's eval AUC ≥ 95.65 | `ao_noshift` 95.50 | **fails** |

### Not planned: `ao_nogain` against the paper (post hoc)

Under R0, the paper's own scoring rule, `ao_nogain` clears all three of the
paper's dev numbers on every seed:

| seed | dev AUC | dev pAUC | dev mAUC |
|---|---|---|---|
| 2040 | 95.99 | 90.32 | 92.12 |
| 2041 | 96.25 | 91.43 | 92.91 |
| 2042 | 95.96 | 91.35 | 91.57 |
| 2043 | 96.49 | 91.54 | 93.79 |
| paper | 95.71 | 90.23 | 91.23 |

`aug` cleared dev AUC on 1 of these 4 seeds and on 2 of its 4 fresh ones
(2044–2047), with a 4-seed mean of 95.73, level with the paper. No earlier recipe
cleared all three dev numbers on every seed under R0.

**This is not a claim.**

- P2 expected `ao_nogain` to be a no-op. It is the best of three arms on dev,
  so some winner's curse is expected.
- Its eval mean(AUC, pAUC) under R0 is 93.04, below the paper's 93.22.

It needs fresh seeds before it means anything.

## Replication of `ao_nogain` (pre-registered 2026-10-31, before any run)

The user asked for this follow-up. It is post hoc: `ao_nogain` was the best of
three arms on dev, and P2 predicted it would do nothing. So these predictions
shrink its stage-1 gain.

- **Runs:** `ao_nogain` on seeds **2044–2047**, the same recipe as stage 1.
  These seeds are new to `ao_nogain`. `la_aug` already has 300-epoch runs on
  them from `../lit-arms` D2, so every new run has a seed-paired `aug` partner.
  Seeds 2060–2067, reserved for stage 2, stay unused because D0 applied.
- **Command:** `AO_ARMS=nogain AO_SEEDS="2044 2045 2046 2047" bash ablation/aug-ops/run.sh train`
- **Cost:** 4 runs on 3 GPUs, about 1 day.
- **`la_aug` on 2044–2047, R0** (reference):
  - dev AUC 95.80, pAUC 90.60, mAUC 91.80
  - eval AUC 95.22, pAUC 90.71, mean 92.97

Predictions (R0 unless stated; 4-seed means over 2044–2047):

- **R1, the paper on dev.** All three dev means exceed the paper's 95.71 /
  90.23 / 91.23. Projection: dev AUC ≈ 96.05 (`aug`'s 95.80 plus a shrunk
  +0.25), pAUC ≈ 90.9, mAUC ≈ 92.2. About 70% likely.
- **R2, versus `aug`.** The seed-paired dev AUC Δ against `la_aug` is ≥ +0.20.
  Central +0.25, against stage 1's +0.44. About 55% likely.
- **R3, eval.** Eval mean(AUC, pAUC) stays below the paper's 93.22 (central
  93.0). The paired eval AUC Δ against `la_aug` is ≤ 0 (central −0.2).
- **R4, under S1′** (secondary, reported, no decision attached). The
  4-seed means beat all three paper dev numbers and the eval 93.22 (central
  eval ≈ 93.6).
- **R5.** Eval ToyCar id_05 stays ≤ 84; gain was not the culprit in stage 1.

Decision rules:

- **C1, the single-network claim.** One network beats the paper's dev numbers
  under R0 if both hold:
  - R1 holds;
  - over all 8 seeds (2040–2047), the dev AUC mean − 1.96·se > 95.71.

  Report per-seed counts alongside.
- **C2.** If C1 and R2 both hold, `--wave_aug_ops noise,shift` replaces full
  `aug` as the single-model recipe. An ensemble of these models against A8
  would be its own pre-registered ablation, not part of this one.
- **Otherwise,** record the stage-1 dev gain as selection luck and leave `aug`
  unchanged.

## What next

1. **Replicate `ao_nogain` on 4 fresh seeds** (about 1 day). This is the open
   single-network route to the paper's dev numbers. It needs its own
   pre-registration with a shrunk central estimate, since it was picked post
   hoc.
2. **ToyCar id_05** stays open. Shift explains about half. A smaller shift
   range, or a linear rather than circular shift (no splice), are the obvious
   variants. Any of them can only be judged on eval, so they would need fresh
   seeds again.
3. A8′ from `ao_noshift` is not worth training (D1 discussion above).

## Figures

| file | shows |
|---|---|
| `plots/fig1_toycar.png` | Eval ToyCar id_05 and dev ToyCar id_01 per arm, seeds and means, with the D1 bar. No shift repairs most, short of 86 |
| `plots/fig2_machines.png` | Per-machine Δ AUC vs plain, dev and eval. Noise carries ToyConveyor; every arm still loses eval ToyCar |
| `plots/fig3_d1.png` | Dev vs eval AUC per seed and arm, with D1's bars. No shift is in the region but misses id_05 |
