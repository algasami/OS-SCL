# Spectrum branch: does a whole-clip |FFT| embedding close the pump gap?

> **Status: DONE 2026-11-05 — D1 FAILS → D0, `--spec_branch` not adopted; stage 2
> not run.** Stage 1 (sb_spec s2040–2043) trained 11-04/05 with no crashes. Under R0
> dev pump moved −0.65 (pre-registered gate ≥ +0.40), and (b) and (c) failed too.
> The branch *is* used (P7 0.22) and lifts dev pump id_02 by +2.9 on 4/4 seeds, but it
> costs pump id_00/id_06, valve and slider. Under S1′ (secondary) pump turns +0.36,
> still short of the gate. See Results.

## Why this exists

`ao_nogain` (`../aug-ops`) is the single-model recipe: the fast recipe at
fussion 1 plus `--wave_aug 1 --wave_aug_ops noise,shift`. Under R0 it beats
the paper's three dev numbers on 8/8 seeds. Per machine (8 seeds, dev, R0) it
is level with or above the paper everywhere except **pump**:

| dev AUC, R0 | fan | pump | slider | ToyCar | ToyConveyor | valve |
|---|---|---|---|---|---|---|
| `ao_nogain`, 8 seeds (sd) | 98.34 (0.52) | **95.53 (0.48)** | 99.62 | 97.71 | 85.77 | 99.60 |
| paper, one network | 98.41 | **96.25** | 99.63 | 96.55 | 83.56 | 99.85 |

- The pump gap (−0.72) is about 4 standard errors, so it isn't seed noise.
- It sits in one dev ID, **pump id_02 (91.08 ± 1.34)**. The paper's worst dev
  pump ID is 94.17 (its Table IV mAUC).
- On eval, pump id_03 (90.82 ± 1.74) is the weak pump ID.
- No ensemble or scoring rule has moved pump. G8 is still at 96.13 dev pump
  (`../nogain-ensemble`). Patch-level kNN was −0.20 on pump
  (`../memory-bank`). `../ensemble-backend` concluded that pump "needs
  model-side work".

**The idea.** The network sees the clip only through 128-band representations:
the Log-Mel (n_fft 2048, 128 mel bands) and two learned 1-D conv front ends with
128 output channels. Pump faults (leakage, contamination, clogging) are
largely stationary changes in the spectrum. Near 4 kHz the 128 Slaney mel
filters are ~100 Hz apart and ~200 Hz wide, so a narrow tonal shift inside
one is averaged away. The **whole-clip
magnitude spectrum** keeps all 80,000 bins of 0.1 Hz each, and it gives up
time resolution, which stationary faults don't need.

## Literature (checked 2026-11-03)

- **What the branch is.** Wilkinghoff's DCASE 2023 Task 2 system
  ([code](https://github.com/wilkinghoff/DCASE2023_task2), `main.py`,
  `model_emb_cnn`) has two sub-networks on the mixed-up waveform:
  - **"FFT":** |FFT| of the whole clip, keeping the first half of the bins
    (0–8 kHz). No log and no input normalisation.
  - Then Conv1D(128, k 256, s 64) → ReLU → Conv1D(128, k 64, s 32) → ReLU →
    Conv1D(128, k 16, s 4) → ReLU → flatten.
  - Then four × [Dense(128) → BN → ReLU], and a final Dense(128) as the
    embedding. No biases anywhere; L2 weight decay 1e-5.
  - The other sub-network is a ResNet on a temporally mean-normalised
    magnitude spectrogram.
  - The two 128-d embeddings are **concatenated while training**, and
    sub-cluster AdaCos (16 sub-clusters) is applied to the 256-d result.
- **Reported effect: DCASE 2022 only, not DCASE 2020.** Wilkinghoff, ICASSP
  2023, "Design choices for learning embeddings from auxiliary tasks for domain
  generalization in ASD", Table 4 (reproduced in his thesis, Bonn 2024). Five
  trials, harmonic means, domains mixed:

  | input | dev AUC | dev pAUC | eval AUC | eval pAUC |
  |---|---|---|---|---|
  | magnitude spectrum alone | 76.81 ± 0.94 | 63.09 | 66.00 ± 1.11 | 57.11 |
  | log-mel spectrogram (128 bins) alone | 68.93 ± 1.52 | 57.79 | 65.04 ± 0.68 | 57.00 |
  | magnitude spectrogram alone | 77.60 ± 1.02 | 62.31 | 72.10 ± 0.81 | 58.87 |
  | spectrum + spectrogram, concatenated while training | **81.36 ± 0.66** | 66.55 | **73.43 ± 0.54** | 59.78 |
  | the same, embeddings added while training | 80.21 ± 0.73 | 64.40 | 72.48 ± 0.53 | 59.22 |
  | the same, concatenated after separate training | 81.17 ± 0.66 | 65.23 | 72.18 ± 0.67 | 59.45 |

  The text gives the reason: temporal normalisation strips stationary
  frequency information from the spectrogram, "but [it is] clearly contained
  in the spectra".
- **What does not transfer.** That system's spectrogram branch is far weaker
  than ours on its own data, and DCASE 2022 has no pump. Our Log-Mel is not
  mean-normalised, so it still carries stationary information, only at mel
  resolution. **No DCASE 2020 or per-pump number exists** for this branch that
  I could find. The +3.8 dev AUC above is a different dataset, a different
  backbone and a weaker partner branch, so the prior here is much smaller.

## Implementation (`--spec_branch 1`, off by default)

- `model/net.py: SpecNet` is a port of the FFT sub-network.
  - Keras `same` padding is reproduced exactly for 160,000 samples:
    80,000 → 1,250 → 40 → 10 frames.
  - Differences: PyTorch default init instead of Glorot, and AdamW's decay
    instead of per-layer L2.
  - It has **1.57 M parameters**, which takes the lean model from 1.38 M to
    2.96 M. This is Wilkinghoff's size; see *Confounds*.
- **Fusion follows the paper's best variant: concatenate while training.**
  - `SCLTFSTgramMFN.embed()` returns [MobileFaceNet 128-d, SpecNet 128-d].
  - The FPH head's first layer widens from 128→64 to 256→64. Everything after
    it is unchanged: 128-d output, L2 norm, ArcFace, SupCon.
  - The pre-head embedding that S1′'s kNN uses is the 256-d concatenation.
- **The spectrum is computed inside the network** from whatever wave the net is
  given. In training that is the augmented, then mixed-up wave, as in
  Wilkinghoff's system. At test time it is the raw clip.
- **fp32 under `--amp`.** The FFT and the whole branch run with autocast off.
  Raw |FFT| values reach ~1e3, and the first conv sums 256 of them.
- **Initialisation.** The branch is built **last** in `__init__`, and the head
  only widens when the flag is on. With the flag off, the init draw order and
  every op are unchanged.
- **Other code that uses the embedding:**
  - `eval.py` reads `spec_branch` from the checkpoint's `config.json`.
  - `trainer.teacher_cos` and `../ensemble-backend/extract.py` call
    `net.embed()`. Checked: `extract.forward`'s logits equal `net.forward`'s,
    with the flag both off and on.
  - `../memory-bank/bank.py` still reads MobileFaceNet internals. It is not
    used here.

### Flag-off equality (2026-11-03)

`eqtest.py` seeds as `train.py` does (seed 2040). It builds the net and the
EMA, then runs `Trainer.train` for 3 real batches from `data/cache` (wave_aug
noise+shift, mixup, SupCon, EMA) plus one validation. It saves both
state_dicts. The same script then runs against main's checkout (f57b58a) and
against this branch with the flag off:

- CPU, batch 8: **444/444 tensors `torch.equal`** (net + EMA), and the valid
  loss is identical (18.789768).
- GPU 1, fp16 autocast, batch 16, deterministic cudnn: **444/444 tensors
  `torch.equal`**, and the valid loss is identical (18.322233).

With the flag on, the same 3 steps give a finite loss (CPU: train 14.81,
valid 19.78).

### Smoke and bench (GPU 1, 2026-11-03)

- **Smoke** (`sb_smoke_s1_e2`): seed 1, the full `sb_spec` recipe, 2 epochs,
  probe every epoch.
  - Training finished (rc 0) with no skipped fp16 steps. The probe logged both
    epochs.
  - `eval.py --d` on the saved checkpoint (batch-1 `train=False` path, config
    read from `config.json`) gives dev AUC / pAUC / mAUC 0.50967 / 0.50805 /
    0.43800. The batched probe's row for the same epoch is 0.50967 / 0.50806 /
    0.43800.
  - Chance-level AUC is expected: the EMA net is still near init after 2
    epochs.
- **Bench** (`scores/bench.txt`; random inputs, batch 64, fp16, lean):
  4.92 it/s with the branch against 5.10 without, i.e. **+3.7% step time**.
  Peak memory is the same (5.7 GB).
- **`head_share.py`** runs on the smoke checkpoint.
  - The 2-epoch EMA net gives 1.000. Its MobileFaceNet output is still almost
    constant across clips (std 2e-6), so the share is degenerate there.
  - The 2-epoch student gives 0.177.
  - So P7 is meaningful only on trained nets; the smoke row is not kept.

## Design

- **Arm `sb_spec`:** the `ao_nogain` recipe plus `--spec_branch 1`. That is
  `--m 0.4 --fussion 1 --ht basic --num_workers 8 --amp fp16 --cudnn_bench 1
  --lean 1 --cache_dir data/cache --probe_every 5 --wave_aug 1 --wave_aug_ops
  noise,shift`, 300 epochs. `run_one.sh` has the exact command.
- **Seeds 2040–2043** (stage 1). Each is paired with the existing
  `ao_nogain_s2040–2043`, trained on this host with this recipe. Their scores
  are in `../aug-ops/scores/`, and they are not retrained.
  - The pairing is by **seed and split only**. The train/valid split is drawn
    before the net is built, so it is identical.
  - The extra branch shifts every later random draw: the EMA net's init, batch
    order and mixup. As with `--lean` (CLAUDE.md), a run here is not a paired
    copy of its baseline seed.
  - smartlab1 is not bit-deterministic over a full run anyway.
- **Stage 2** (only if D1 passes): seeds **2044–2047**, paired with
  `ao_nogain_s2044–2047`.
- **Scoring:** R0 (`eval.py`) on dev and eval, per machine and per ID, written
  to `scores/<run>/eval_{dev,eval}.csv`.
  - S1′ (`../aug-ops/score_s1p.py` on `extract.py` dumps) is secondary: it is
    reported and drives no decision.
- **Diagnostic:** `head_share.py` measures how much of the head's first-layer
  variance comes from the spectrum half (P7). It runs on 512 dev-test clips
  and writes `scores/head_share.csv`.
- **Cost:** `ao_nogain` took 11.2–12.0 h per run on GPUs 0/2 and 13.6–14.2 h
  on GPU 1. At +3.7% step time, `sb_spec` should take about 11.6–12.4 h and
  14.1–14.7 h.
  - Stage 1 (4 runs on 3 GPUs; the fourth queues behind the first to finish)
    is about **26 h wall, ~50 GPU-hours**.
  - Stage 2 costs the same.

## Noise reference

From `ao_nogain`'s 8 seeds: dev pump AUC has sd 0.48 per seed. A 4-pair
mean delta between unpaired-init runs therefore has se ≈ √2 · 0.48 / 2 ≈ 0.34.
Dev overall AUC has sd ≈ 0.2 per seed, so its se ≈ 0.14.

## Predictions (written 2026-11-03, before any training run)

Δ = `sb_spec` − `ao_nogain`, mean over the 4 seed pairs, R0, AUC points.
Reference values (`ao_nogain`, 2040–2043):

- dev: pump 95.57, pump id_02 91.3, ToyConveyor 85.96, AUC 96.17, pAUC 91.16
- eval: pump 97.63, pump id_03 90.6, mean(AUC, pAUC) 93.04

- **P1, dev pump.** Δ central **+0.35**, 80% interval [−0.30, +1.00].
  P(Δ ≥ +0.40, i.e. D1(a)) ≈ 40%. Shrunk hard from the DCASE 2022 +3.8: our
  Log-Mel already carries stationary information, and our backbone is much
  stronger.
- **P2, the gain concentrates in dev pump id_02.** Δ id_02 central +0.8,
  interval [−1.0, +2.5]. It is larger than the pump-wide Δ, because id_04 and
  id_06 are at 99–100.
- **P3, eval pump id_03** (secondary). Δ central +0.5, interval [−1.5, +2.5].
- **P4, overall dev AUC.** Δ central +0.05, interval [−0.30, +0.30]. Four
  machines are within 0.4 of 100. Valve and fan may gain a little (both are
  partly stationary). The time-local ToyCar and ToyConveyor gain nothing.
- **P5, dev ToyConveyor.** Δ central 0.0, interval [−1.0, +1.0]. The
  concatenation could dilute the embedding that ToyConveyor's events need.
  That is why D1(d) exists.
- **P6, eval mean(AUC, pAUC).** Δ central 0.0, interval [−0.40, +0.40].
- **P7, the branch is used.** Over the head's first-layer units, var(z_spec) /
  (var(z_mel) + var(z_spec)) has a 4-seed mean in [0.15, 0.50], central 0.30. If
  it is below 0.05, the head has learned to ignore the spectrum, and a null P1
  would say nothing about the spectrum's information content.

## Decision rules

- **D1, stage 1 → stage 2.** All of these must hold (4 seed pairs, R0):
  - (a) dev pump Δ ≥ **+0.40**, and Δ > 0 on ≥ 3 of 4 seeds;
  - (b) dev AUC Δ ≥ −0.10, and dev mean(AUC, pAUC) Δ ≥ −0.10;
  - (c) eval mean(AUC, pAUC) Δ ≥ −0.20;
  - (d) dev ToyConveyor Δ ≥ −0.80.

  `score.py` prints the checklist. Under no effect, (a) passes about 10% of
  the time; stage 2 guards against that.
- **D0.** If D1 fails, stop. `--spec_branch` is not adopted. Report which
  criterion failed, P7, and per-ID pump on both sets.
- **D2, adoption** (stage 2, seeds 2044–2047, fixed now). `--spec_branch 1`
  joins the single-model recipe if all hold:
  - the fresh-seed dev pump Δ ≥ +0.25;
  - over all 8 pairs, the dev pump Δ has mean − 1.96·se > 0;
  - (b), (c) and (d) hold on the fresh seeds.

  Report the 8-seed `sb_spec` dev pump against the paper's 96.25. That is a
  number, not a claim criterion.
- **D3, no claims.** Eval per ID, S1′, and whether `sb_spec` models help an
  E2′ ensemble are reported only. An ensemble with them would need its own
  pre-registration, as in `../nogain-ensemble`.

## Confounds

- **Capacity.** The branch adds 1.57 M parameters, more than doubling the
  model. A D1 pass shows the *branch* helps, not that the *spectrum* does. P7
  shows whether the head uses the spectrum half. No capacity-matched control is
  in stage 1, to keep it to 4 runs. A deeper MobileFaceNet of equal size would
  be that control.
- **Linear magnitude.** The branch follows Wilkinghoff in using raw |FFT|
  with no log. Strong tonal peaks dominate the first conv. A log-magnitude
  variant is the obvious second arm if the branch is unused (P7 < 0.05).

### Addendum 2026-11-04: capacity control (user decision, before any run)

- **Stage 1 runs without a control,** as designed: 4 runs of `sb_spec`.
- **If D1 passes,** stage 2 trains a capacity control **alongside** `sb_spec`
  on the same fresh seeds, 2044–2047. That is 8 runs.
- **The control is `sb_melres`.** It is the identical SpecNet branch, the
  same 1.57 M parameters and the same fusion. Its input is the |FFT| reduced
  to mel resolution: the 128-band mel filterbank applied to |FFT|, then
  spread back over the same 0–8 kHz bins. So it gets the extra parameters and
  only information the Log-Mel channel already carries. Only the frequency
  resolution differs from `sb_spec`.
  - This replaces the "deeper MobileFaceNet" suggested under Confounds,
    because it changes one thing instead of two.
  - It is implemented only if D1 passes.
- **C-spec, the "spectrum, not capacity" claim.** On the fresh seeds, the
  seed-paired dev pump Δ of `sb_spec` − `sb_melres` must be ≥ +0.25. Without
  that, a D2 pass is reported as "the branch helps", not "the spectrum
  helps". D2's adoption rule is unchanged.

## Files

| file | what |
|---|---|
| `run.sh` | `train` / `score` / `plots`; all resumable |
| `run_one.sh` | one `sb_spec` run on `../train_one.sh` (locks under `/tmp/os-scl-spectrum-branch-locks`) |
| `score.py` | 4-seed means, paired Δ with se, per-seed pump Δ, and the D1 checklist → `scores/summary.txt` |
| `make_plots.py` | figures from `scores/` only; skips any figure whose inputs are missing |
| `head_share.py` | P7 diagnostic → `scores/head_share.csv` |
| `score_s1p.py` | secondary S1′ on `extract.py` dumps, paired with `../aug-ops` dumps → `scores/s1p.csv`, `scores/s1p_summary.txt` (`run.sh s1p`) |
| `eqtest.py` | the flag-off equality test |
| `bench.py` | step rate with and without the branch → `scores/bench.txt` |

## Figures

| file | shows |
|---|---|
| `plots/fig1_machines.png` | paired Δ AUC per machine, dev and eval, seeds and mean |
| `plots/fig2_pump.png` | pump per ID for both arms, dev and eval, with the paper's dev pump AUC |
| `plots/fig3_probe.png` | dev-test probe over training (pump AUC, overall AUC), mean over seeds: faster early, lower plateau |

## Results

Δ = `sb_spec` − `ao_nogain`, 4 seed pairs (2040–2043), AUC points, se over pairs.
Full output: `scores/summary.txt` (R0) and `scores/s1p_summary.txt` (S1′). S0
recomputed from the dumps matches eval.py to 1.2e-4 AUC per ID.

| | sb_spec | ao_nogain | Δ R0 (se) | Δ S1′ (se) |
|---|---|---|---|---|
| dev AUC | 95.95 | 96.17 | −0.22 (0.21) | +0.20 (0.13) |
| dev mean(AUC, pAUC) | 93.41 | 93.67 | −0.26 (0.22) | +0.01 (0.18) |
| **dev pump** | 94.92 | 95.57 | **−0.65 (0.49)** | +0.36 (0.09) |
| dev pump id_02 | 94.22 | 91.28 | **+2.93 (0.29)** | +4.06 (0.73) |
| dev pump id_00 / id_06 | 92.65 / 97.21 | 94.23 / 99.30 | −1.57 / −2.09 | |
| dev ToyConveyor | 85.80 | 85.96 | −0.16 (0.94) | +0.64 (0.58) |
| eval mean(AUC, pAUC) | 92.72 | 93.04 | −0.32 (0.54) | −0.31 (0.53) |
| eval pump | 98.01 | 97.63 | +0.38 (0.11) | +0.33 (0.18) |
| eval pump id_03 | 91.84 | 90.58 | +1.26 (0.97) | |
| eval ToyCar | 91.81 | 93.48 | −1.67 (0.14) | +0.15 (0.23) |
| eval ToyCar id_05 | 78.79 | 81.60 | −2.81 (0.10) | |

(sb_spec / ao_nogain columns are R0.) Per-seed dev pump Δ under R0: −0.20, +0.45,
−1.83, −1.01.

Per machine, Δ mean(AUC, pAUC), dev / eval:

| rule | fan | pump | slider | ToyCar | ToyConveyor | valve |
|---|---|---|---|---|---|---|
| R0 | +0.84 / +0.49 | −0.78 / +0.23 | −0.21 / −1.26 | +0.02 / −0.95 | −0.22 / +0.81 | −1.19 / −1.21 |
| S1′ | +1.46 / +0.53 | −0.21 / +0.32 | −0.30 / −2.02 | +0.07 / +0.16 | +0.18 / −0.26 | −1.11 / −0.58 |

**P7 (branch used):** spectrum share of the head's first-layer variance 0.227,
0.207, 0.225, 0.219 (mean 0.22). The head uses the branch, so the null is
informative.

### D1 checklist (R0)

| criterion | value | |
|---|---|---|
| (a) dev pump Δ ≥ +0.40 | −0.65 | FAIL |
| (a) Δ > 0 on ≥ 3/4 seeds | 1/4 | FAIL |
| (b) dev AUC Δ ≥ −0.10 | −0.22 | FAIL |
| (b) dev mean Δ ≥ −0.10 | −0.26 | FAIL |
| (c) eval mean Δ ≥ −0.20 | −0.32 | FAIL |
| (d) dev ToyConveyor Δ ≥ −0.80 | −0.16 | ok |

**D1 fails → D0: stop.** Stage 2 and the `sb_melres` control are not run.
Read under S1′ instead, D1 would still fail: (a) +0.36 < +0.40 and (c) −0.31 < −0.20.

### Predictions

| | predicted (80% interval) | observed | |
|---|---|---|---|
| P1 dev pump | +0.35 [−0.30, +1.00] | −0.65 | **fails** (below interval) |
| P2 dev pump id_02 | +0.8 [−1.0, +2.5] | +2.93 | direction right, **above** interval |
| P3 eval pump id_03 | +0.5 [−1.5, +2.5] | +1.26 | holds |
| P4 dev AUC | +0.05 [−0.30, +0.30] | −0.22 | holds (sign wrong) |
| P5 dev ToyConveyor | 0.0 [−1.0, +1.0] | −0.16 | holds |
| P6 eval mean | 0.0 [−0.40, +0.40] | −0.32 | holds (sign wrong) |
| P7 share | 0.30 [0.15, 0.50] | 0.22 | holds |

### What this shows

- **The spectrum carries the information pump id_02 needs.** +2.9 under R0 and
  +4.1 under S1′, on 4/4 seeds, with the branch taking ~22% of the head's
  variance. id_02 was the ID the idea was built for (P2), and the effect is
  larger than predicted. Eval pump id_03 moves the same way (+1.3, noisier).
- **It hurts transient machines.** Valve and slider lose on both sets and
  both rules. The other pump IDs (id_00, id_06) lose 1.6–2.1. A whole-clip
  spectrum has no time axis, so it adds weight to stationary content at the
  expense of events. That is the premise of the branch, and also its cost.
- **R0 understates the branch; S1′ recovers most of it.** Under R0 the
  branch's CE score loses on eval ToyCar (−1.7, id_05 −2.8, 4/4 seeds). Under
  S1′ that loss disappears (+0.15) and dev pump turns positive. The 256-d
  embedding holds the information; the ArcFace logit calibrates it badly.
- **Faster early, lower plateau** (fig3): the branch lifts the probe by
  several points before epoch 150, then both arms flatten and the branch ends
  0.2 (all) and 0.7 (pump) lower.
- Not tested: whether the spectrum or the extra capacity is responsible (no
  `sb_melres`), and linear vs log magnitude.

### What next (not pre-registered)

- **Not as a recipe change.** One network has to serve every machine, and
  the transient-machine cost cancels the pump gain.
- **As an ensemble member, it may be useful.** Its errors differ from
  `ao_nogain`'s by machine (fan/pump up, valve/slider down), which is the
  diversity the trunk-soup forks lacked. A mixed E2′ with sb_spec members would
  need its own pre-registration (D3).
- **A time-aware version** (spectrum of short windows, or a gate that lets the
  head down-weight the branch per clip) would address the transient cost.
  This sits close to the per-band, pooled kNN scoring proposed after the
  2026-11-05 literature survey; that cheaper route comes first.
