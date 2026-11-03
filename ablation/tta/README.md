# TTA: does averaging over time-shifted copies of each clip help?

> **Status: stage A (dev) done 2026-11-03; stage B (eval) running.**
> - **Gate A:** `match` (±3–8%, the primary scheme) fails. It is +0.07 on single-model
>   S1′, but E2′[G8] goes −0.02. `small` (±1%, the secondary) passes: +0.11 S1′ on 8/8
>   seeds, +0.09 R0 on 8/8, E2′[G8] +0.06. By the pre-registered rule, `small` goes to
>   stage B.
> - **Pilot (training clips only):** shifting a clip past the training range makes normal
>   ToyCar look anomalous (~60 sd at half a clip). The splice itself costs nothing, so the
>   ToyCar models key on absolute time position.
> - **P0 fails by the letter** on dev-clip R0 (2.4e-4 against 1e-4). Every difference is in
>   the last, differently sized batch; see Stage A.
> - The eval predictions (addendum) were committed before any eval clip was extracted.

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

### Stage A (dev, 2026-11-03)

8 `ao_nogain` models; `scores/scorecard_dev.txt`, `dev_runs.csv`, `dev_systems.csv`,
`dev_perid.csv`, `dev_bootstrap.json` and `dev_check.json`. No eval clip had been extracted.

**Reproduction (P0).**
- T0 training-bank R0 and embeddings equal the original dumps bit for bit (max |Δ| 0).
- T0 S0 per-ID AUC matches eval.py to ≤ 1.4e-4 (bar 2.5e-4).
- T0 S1′ overall matches `../aug-ops/scores/s1p.csv` to ≤ 5e-5 (bar 0.01).
- **Dev-clip R0 differs from the original dump by up to 2.4e-4, against a bar of 1e-4.**
  - All of it sits in the 20 clips of the final batch, on every seed. Every other clip is
    bit-identical.
  - There that batch held 20 clips; the original pass's final dev batch held the same 20
    plus 12 eval clips. So it is GPU float noise from a different batch size (relative
    ≤ 7e-4).
  - T0, Tm and Ts are computed from the same batch, so it cannot favour one version.
  - P0 fails by the letter. It was meant to catch a broken pipeline, and nothing else
    points to one, so the run was not stopped.

```
dev, ao_nogain ×8      AUC    pAUC   mAUC   mean  | Δ mean   seeds>0  paired 95% (mean)
S0  (R0)   T0         96.10  91.08  92.12  93.59
           Tm match   96.05  91.01  91.97  93.53  | −0.06    0/8     [−0.11, −0.01]
           Ts small   96.18  91.19  92.26  93.68  | +0.09    8/8     [+0.06, +0.12]
S1′        T0         96.43  91.58  92.82  94.00
           Tm match   96.47  91.67  92.95  94.07  | +0.07    8/8     [−0.03, +0.19]
           Ts small   96.52  91.72  93.01  94.12  | +0.11    8/8     [+0.02, +0.24]
E2′[G8]    T0         97.19  92.76  94.12  94.97
           Tm match   97.13  92.77  94.02  94.95  | −0.02            [−0.13, +0.12]
           Ts small   97.21  92.86  94.17  95.04  | +0.06            [−0.04, +0.20]
paper                 95.71  90.23  91.23
```

Δ AUC by machine, single-model 8-seed means (fan / pump / slider / ToyCar / ToyConveyor / valve):

| | fan | pump | slider | ToyCar | ToyConveyor | valve |
|---|---|---|---|---|---|---|
| R0, `match` | +0.08 | +0.10 | +0.00 | **−0.49** | −0.02 | +0.05 |
| R0, `small` | +0.05 | +0.07 | −0.00 | +0.05 | **+0.30** | +0.04 |
| S1′, `match` | +0.01 | +0.05 | +0.08 | +0.00 | +0.11 | +0.03 |
| S1′, `small` | −0.00 | +0.05 | +0.05 | +0.12 | **+0.28** | +0.02 |

**Gate A.**

| scheme | single S1′ dev Δ > 0, ≥ 6/8 seeds | E2′[G8] dev Δ ≥ 0 | |
|---|---|---|---|
| `match` (primary) | +0.07, 8/8: ok | −0.02: **no** | fails |
| `small` (secondary) | +0.11, 8/8: ok | +0.06: ok | **passes**; goes to stage B because `match` failed |

**Predictions, checked as written.**

| | prediction | observed | verdict |
|---|---|---|---|
| P0 | reproduction bars | all hold except dev-clip R0 2.4e-4 > 1e-4 (last batch only) | fails by the letter; explained above |
| P1 | `match` S1′ Δ in [−0.30, +0.15], central −0.05; ≤ 4/8 seeds positive | +0.07; 8/8 | interval holds; **seed count fails** |
| P2 | `match` R0 Δ in [−0.30, +0.15], central −0.05 | −0.06 | holds |
| P3 | `match`: ToyCar most negative; fan, pump, slider within ±0.10 | R0 ToyCar −0.49 (S1′ +0.00, lowest); pump +0.098 | holds |
| P4 | `small` \|Δ\| ≤ 0.05 under S0 and S1′ | +0.09, +0.11 | **fails**: `small` does twice the bound |
| P5 | E2′[G8] `match` Δ in [−0.25, +0.10], central −0.05 | −0.02 | holds |

**What dev shows.**
- **`match` behaves as the pilot said.** Under R0, the ±3–8% copies cost ToyCar 0.49 AUC,
  the machine whose normal clips the pilot saw rise ~1 sd at ±8%. S1′ hides it, probably
  because its per-ID z-scoring absorbs a rise that is uniform within an ID. On G8 nothing is left.
- **`small` is a consistent, small gain:** positive on 8/8 seeds under both rules, with
  the R0 interval clear of 0. It is mostly ToyConveyor (+0.28 to +0.30), the most
  seed-noisy machine. P4's reasoning was wrong.
  - The pilot's rho of 0.06 measured *how much* of the normal-score spread is positional
    noise.
  - It did not measure whether that noise lands on the clips that matter: the normal clips
    in the upper tail, which set AUC and pAUC.
  - An unverified reading: a ±1% roll moves the clip by 1,600 samples. That is about 3 mel
    frames, plus a 1/8-hop change in where the STFT frames fall. Averaging such copies
    smooths frame-grid jitter rather than event position.
- Ensembling keeps about half of it (+0.06 on G8, interval across 0). The pattern matches
  every earlier variance-reduction step here.

## Addendum: stage B (eval) predictions, written 2026-11-03 after stage A, before any eval clip is extracted

Stage B runs `small` only (gate A). Extraction is restricted to its 3 shifts (0, ±1%):
- eval clips for the 8 `ao_nogain` models;
- train, dev and eval for `la_aug` s2040–2047, so A8 can be scored with and without TTA.

`dev` is then rerun to add A8's dev Δ. The existing dev numbers are unchanged, because
the bootstrap draws the same resamples.

Eval references, unshifted:
- `ao_nogain` 8-seed S1′ eval mean 93.88 (R0 93.14);
- E2′[G8] 94.87;
- E2′[A8] 94.78.

Every earlier dev gain here shrank on eval, usually by about half or more (`aug`, `nogain`,
G8). ToyConveyor gains were the ones that carried over (`../nogain-ensemble`).

- **B1, single S1′ (D1).** Eval Δ mean(AUC, pAUC) in [−0.10, +0.25], central +0.06.
  Positive on ≥ 5 of 8 seeds. The paired lower bound is > 0 with about 40% probability, so
  the central call is that **D1 fails**.
- **B2, single R0.** Eval Δ in [−0.05, +0.20], central +0.06, positive on ≥ 6 of 8 seeds.
  R0's dev interval was narrow (±0.03), so the lower bound clears 0 with about 55%
  probability. No decision is attached.
- **B3, where.** ToyConveyor has the largest eval Δ AUC of the six machines under S1′. Eval
  ToyCar id_05 moves by < 1.0 AUC.
- **B4, A8 (D2).**
  - E2′[A8] dev Δ ≥ 0, central +0.05.
  - Eval Δ in [−0.10, +0.20], central +0.04, interval including 0, so **D2 fails**. About
    65%.
- **B5, G8.** E2′[G8] eval Δ in [−0.10, +0.20], central +0.04.

### Stage B results

*(pending)*

## Figures

| file | shows |
|---|---|
| `plots/fig1_pilot_shift.png` | Pilot: R0 rise on normal held-out clips per circular shift, per machine and model type. Past the training range (grey), ToyCar rises ~60 sd at half a clip; fan, pump and slider stay flat. Inside ±1%, the splice alone costs nothing |
| `plots/fig2_dev_delta.png` | Stage A: per-seed and bootstrap Δ dev mean(AUC, pAUC), single models (R0, S1′) and E2′[G8], for `match` and `small`. `small` gains on every seed; `match` loses under R0 and on G8 |
| `plots/fig3_dev_machines.png` | Stage A: Δ dev AUC per machine. `small`'s gain is mostly ToyConveyor; `match` costs ToyCar 0.49 under R0 |
