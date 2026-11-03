# TTA: does averaging over time-shifted copies of each clip help?

> **Status: pre-registered 2026-11-03; stage A (dev) running.** The pilot (held-out
> training clips only) is done and fixed the scheme. Nothing below *Results* has
> seen a test clip.

## Why this exists

Every model here scores a clip once, at the position it was recorded in. MobileFaceNet
ends in `linear7`, a depthwise 8×20 convolution (`model/net.py`). That is a *learned,
position-weighted* average over the final 8×20 grid, so the score depends on where in the
10 s clip things happen. Tgram's `LayerNorm(313)` has a per-frame affine, and every conv
zero-pads at the clip edges; both are position-dependent too.

Test-time augmentation (TTA) is the standard way to average such variance out: score K
shifted copies of each clip and average. It costs no training. The best single-model recipe,
`ao_nogain` (`../aug-ops`: `--wave_aug_ops noise,shift`), was trained with a circular shift
of ±3–8% on 30% of clips, so it should tolerate shifts in that range.

**Question.** Does averaging R0 and the pre-head embedding over circularly shifted copies raise:
- single-model R0 and S1′ for `ao_nogain`;
- E2′[G8], and through it E2′[A8] (the reported best ensemble)?

## What has already been seen

- **Every member, unshifted,** under R0 and S1′ on dev and eval (`../aug-ops`, `../lit-arms`).
- **G8 and A8 under E2′** on dev and eval (`../nogain-ensemble`).
  - G8: dev 97.19 / 92.76 / 94.12 (mean 94.97), eval mean 94.87.
  - A8: eval mean 94.78.
- **`ao_nogain`, 8 seeds, references:**
  - S1′: dev 96.43 / 91.58 / 92.82 (mean 94.00), eval mean 93.88.
  - R0: dev 96.10 / 91.08 / 92.12, eval mean 93.14.
- **Not seen:** any score of a shifted test clip, on either set.

## Pilot (training clips only, done 2026-11-03)

`extract_tta.py --split valid` scored each run's own replayed 5% held-out split under 16
circular shifts. These are 1,815 normal clips the run never trained on. `pilot.py`
(`scores/pilot.csv`, `scores/pilot_shift.csv`) measures, per ID and then averaged over IDs:
- **penalty:** how far the mean R0 rises over a scheme's copies, in sd of the unshifted R0;
- **rho:** across-shift variance divided by across-clip variance;
- **shrink:** the sd of the averaged R0 divided by the unshifted sd.

The selection rule was written into `pilot.py` before its output existed: among `match`
(0, ±3, ±5.5, ±8%) and `wide` (eighths of the clip), take the larger rho for `ao_nogain`
whose penalty is ≤ 0.25 sd.

| scheme | shifts | `ao_nogain` penalty | rho | shrink | `fr_f1` penalty |
|---|---|---|---|---|---|
| `small` | 0, ±1% | +0.00 | 0.06 | 0.97 | +0.04 |
| `match` | 0, ±3, ±5.5, ±8% | **+0.08** | 0.38 | 1.05 | +0.77 |
| `wide` | k/8, k=0..7 | **+5.98** | 225 | 2.2 | +4.4 |

**Pick: `match`.** `wide` fails the penalty bar by a factor of 24.

What the pilot shows (per shift and machine: `scores/pilot_shift.csv`, figure 1):

1. **Shifting beyond the training range makes normal clips look anomalous, and that is
   ToyCar.** At a 50% roll, normal ToyCar clips rise 56–69 sd for the shift-trained models
   and 27–41 sd for `fr_f1`. ToyConveyor and valve rise about 1 sd. Fan, pump and slider
   don't move at any shift (|penalty| < 0.15).
   - So the ToyCar models key on *absolute time position*: ToyCar clips evidently share a
     temporal structure.
   - The ±3–8% shift op buys tolerance only inside its range. At ±8% the shift-trained
     models' ToyCar penalty is ~1 sd, against 4.7–14 sd for `fr_f1`.
2. **The splice itself costs almost nothing.** A ±1% roll has the same end-to-start splice
   as any other roll, but its ToyCar penalty is 0.0–0.2 sd, even for `fr_f1`. The penalty
   grows smoothly with shift size.
   - So what the model reacts to is displacement, not the splice click.
   - This bears on `../aug-ops`' open item, eval ToyCar id_05: a click-free shift is
     unlikely to repair it, if this mechanism is the one at work there. That inference is
     untested.
3. **Inside the training range, averaging widens the normal-score spread** (shrink 1.04–1.09
   for `match`). The extra copies add ToyCar distortion faster than they cancel position
   noise. This is the main reason the predictions below lean null or negative.
4. **`fr_f1` (never trained with shift) is penalised even inside `match`** (+0.6 to +0.9
   sd). TTA is unusable for unshifted models. This answers the `fr_f1` contrast, so `fr_f1`
   is not extracted on test clips.

## Design

- **Versions** of every checkpoint, built from one 9-shift dump (0, ±1, ±3, ±5.5, ±8%):
  - `T0`: shift 0 only, i.e. no TTA.
  - `Tm` (`match`, primary, K=7).
  - `Ts` (`small`, secondary, K=3).
- **TTA rule:**
  - R0ᵀ = mean over copies of the per-copy R0.
  - Embeddingᵀ = mean over copies of the L2-normalised pre-head embedding.
  - **The training clips get the same treatment.** They are the S1′ kNN bank and the
    source of the z-statistics, so test and reference scores stay on one scale.
- **Rules unchanged,** imported from `../ensemble-backend/score.py`:
  - S0 = R0;
  - S1′ = z(R0) + z(1-NN cosine distance), with per-ID training-clip statistics;
  - E2′ = Σ members' S1′.
- **Extraction:** `extract_tta.py`, the forward pass of `../ensemble-backend/extract.py`
  (imported). Shift 0 uses the CPU mel, so `T0` reproduces the existing dumps; other
  shifts roll the wave on the GPU and recompute the mel there, as `trainer.augment` does.
- **Stage A (dev):** 8 `ao_nogain` models, train bank + dev clips. Eval clips are not
  extracted in stage A.
- **Stage B (eval), only if gate A passes:** extract eval for `ao_nogain`, and train + dev
  + eval for `la_aug` s2040–2047, to rebuild A8. Eval predictions go in a dated addendum
  first.
- **Uncertainty:** a paired clip-bootstrap (2000 stratified resamples, seed 0, the
  resampler from `../ensemble-backend`).
  - Single models: of the 8-seed mean Δ, the same resample for both versions.
  - Ensembles: of the E2′ Δ.

## Predictions (written 2026-11-03, after the pilot, before any test clip is scored)

Δ = version − T0, on mean(AUC, pAUC) unless stated; single models are 8-seed means.

- **P0, reproduction.**
  - T0's R0 reproduces the original dumps to ≤ 1e-4 (train and dev).
  - T0's S0 per-ID AUC matches eval.py to ≤ 2.5e-4.
  - T0's S1′ overall matches `../aug-ops/scores/s1p.csv` to ≤ 0.01.
  - If any of these fails, stop.
- **P1, `match`, S1′, single model, dev.** Δ in [−0.30, +0.15], central −0.05. Positive on
  ≤ 4 of 8 seeds. About 65%.
- **P2, `match`, R0, dev.** Δ in [−0.30, +0.15], central −0.05.
- **P3, where it moves.** Under `match`, ToyCar has the most negative Δ AUC of the six
  machines. Fan, pump and slider are each within ±0.10.
- **P4, `small`.** |Δ| ≤ 0.05 on dev, under both S0 and S1′. With rho 0.06, three copies
  remove at most ~3% of the spread.
- **P5, E2′[G8], `match`, dev.** Δ in [−0.25, +0.10], central −0.05. Ensembling already
  averages away most of the variance TTA could remove.

## Decision rules

- **Gate A (stage A, dev).** A scheme goes to stage B only if both hold:
  - the 8-seed mean single-model S1′ dev Δ is > 0, with ≥ 6 of 8 seeds positive;
  - E2′[G8]'s dev Δ is ≥ 0.

  `match` is the primary scheme. `small` goes to stage B only if it passes and `match`
  does not.
- **D0.** If no scheme passes gate A, stop: TTA is not adopted. Eval clips are never
  extracted, and nothing is claimed about eval.
- **D1, adopt TTA in the single-model rule (stage B).** Gate A passed, and the paired
  clip-bootstrap 95% lower bound of the 8-seed mean S1′ eval Δ is > 0.
- **D2, adopt TTA for the reported ensemble (stage B).** E2′ᵀ[A8] − E2′[A8] on eval has a
  paired lower bound > 0, and A8's dev Δ is ≥ 0. The rule must improve the reported system
  itself, not only G8.

## Files

| file | what |
|---|---|
| `extract_tta.py` | per-clip R0 and pre-head embeddings under K circular shifts; `--split train/dev/eval/valid` |
| `pilot.py` | the pilot statistics and scheme pick (held-out training clips only) |
| `score.py` | T0 / Tm / Ts under S0, S1′ and E2′, bootstrap, scorecard; one split per call |
| `run.sh` | stages: `pilot`, `extractA`, `dev`, `extractB`, `eval`, `plots` |
| `scores/` | committed record: `pilot*.csv`, then `<ds>_runs.csv`, `<ds>_systems.csv`, `<ds>_perid.csv`, `<ds>_bootstrap.json`, `<ds>_check.json`, `scorecard_<ds>.txt` |
| `results/` | dumps (`emb/`, ~2 GB) and logs: gitignored scratch |

## Results

*(stage A pending)*

## Figures

| file | shows |
|---|---|
| `plots/fig1_pilot_shift.png` | Pilot: R0 rise on normal held-out clips per circular shift, per machine and model type. Past the training range (grey), ToyCar rises ~60 sd at half a clip; fan, pump and slider stay flat. Inside ±1%, the splice alone costs nothing |
