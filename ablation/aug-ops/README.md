# aug-ops: which waveform-augmentation op costs ToyCar id_05?

> **Status: pre-registered 2026-10-28; stage 1 training.** Predictions and
> decision rules below were written before any 300-epoch run of this ablation.

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

## Results

*(empty)*

## Figures

*(empty)*
