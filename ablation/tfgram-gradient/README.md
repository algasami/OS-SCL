# TFgram gradient probe — is the branch starved, or constrained?

Step 0 of the follow-up to [`../frozen-tfgram`](../frozen-tfgram). That ablation
showed frozen TFgram ≈ trained TFgram ≈ no TFgram. Two hypotheses survived:

1. **starved** — Log-Mel is already an input, so improving TFgram buys ~0 loss
   reduction; the branch receives no usable learning signal and never moves.
2. **constrained** — the branch trains normally but its architecture can only
   compute an energy envelope, i.e. a spectrogram, so training returns it to
   where it started.

They imply opposite fixes, so they have to be separated before spending a
300-epoch run. This probe is cheap (~20 min, no training) and decides it.

Run: `python ablation/tfgram-gradient/probe.py --gpu 2 --steps 400 --seed 2024`

## Measurement 1 — drift of the trained checkpoint from its own init

`train.py` seeds before constructing the model, so the exact init is
reproducible by replaying its order (`random_seed` → `train_dataset` →
`dataset_split` → `model` → `model_ema`; `random_split` consumes the torch RNG,
and `WeightEMA.__init__` copies `ema_net`'s weights *into* the student, so
`model_ema`'s draw is the init the system actually starts from).

Checkpoint: `check_points/SCLTFSTgramMFN_sweep_f1_s2024/model.pth`.

```
group               params       cosine    |Wt|/|Wi|    rel_drift
TFgram_live        198,720       0.9501       0.7283       0.3828
TFgram_dead     79,797,225       1.0000       0.7508       0.2492   <- control
Tgram              280,406       0.9193       0.7531       0.4272
MobileFaceNet      876,201       0.9711       0.6998       0.3613
head                16,960       0.6168       0.7450       0.7975
arcface              5,248       0.3845       0.8115       1.0171
```

**`TFgram_dead` is a free control.** Those 79.8M parameters are allocated in
`TFgram.__init__` but never reached by `TFgram.forward` (`conv_block1..6`,
`fc1`, `fc_audioset`, `bn0`, `pre_block4` — 48 tensors, confirmed by the probe
as never receiving a gradient). They isolate what `WeightEMA`'s weight decay
does alone: cosine exactly 1.0000 (no rotation) and a shrink to 0.7508. That
independently reproduces the 0.72x decay figure from the frozen-tfgram commit,
and it shows `rel_drift` is contaminated by decay while **cosine is not** —
cosine is scale-invariant, so it measures pure rotation.

Against that baseline `TFgram_live` moved: cosine 0.9501, the same order as
Tgram's 0.9193, and *more* rotation than MobileFaceNet's 0.9711.

## Measurement 2 — gradient health during real training steps

400 real training steps (same mixup + `os_scl` loss as `trainer.py`), at random
init and at the converged checkpoint. AdamW divides by `sqrt(v)`, so raw
gradient norm does **not** predict learning speed — `|dW|/|W|`, the step
actually taken, does.

```
grad_trained (converged)      |g|      |g|/|W|          SNR     |dW|/|W|
TFgram_live             4.375e-01    1.663e-02       0.0406    3.810e-04
Tgram                   4.959e-01    1.977e-02       0.0489    4.908e-04
MobileFaceNet           8.519e+00    1.000e-01       0.0282    2.582e-04
head                    2.469e+00    2.071e-01       0.0280    2.631e-04
arcface                 1.141e+00    1.781e-01       0.0343    2.780e-04
```

TFgram relative to Tgram — the branch that demonstrably carries signal:

```
              rel_grad    SNR    effective step
at init          1.48x   1.05x           0.72x
converged        0.84x   0.83x           0.78x
```

## Conclusion: hypothesis 1 is refuted

TFgram is **not** starved. At convergence it receives 0.84x Tgram's relative
gradient and takes 0.78x its effective step — and a *larger* step than the
MobileFaceNet backbone (3.81e-4 vs 2.58e-4). AdamW is updating the branch at
full size on every one of the 161,700 steps, right to the end of training, and
the weights really do rotate away from init.

So the branch trains normally and converges to a spectrogram anyway. By
elimination, and corroborated by CKA 0.89 with Log-Mel, the cause is
**architectural**: `ReLU` + `max_pool` + 511x downsampling (160000 -> 32000 ->
8000 -> 626 -> 313) is a rectified peak-energy envelope detector on a
filterbank, which is what a magnitude spectrogram is. Training rotates the
filterbank; it cannot change what kind of object the stack computes.

Contrast Tgram at CKA 0.06: `LeakyReLU(0.2)` preserves sign, `LayerNorm` not BN,
and no pooling at all — phase and temporal fine structure survive to its output.

## What this changes about the fix

- **Architecture change is now first choice**, not last: make TFgram capable of
  representing something Log-Mel cannot (sign-preserving nonlinearity, strided
  convs instead of max-pools).
- **Log-Mel channel dropout is demoted.** It was justified by "add gradient
  pressure", and there is no pressure deficit. It changes *what the optimum is*
  rather than how fast it is reached — but the architecture still bounds what is
  reachable, so it would yield a better spectrogram, not a different feature.
- **A decorrelation / CKA penalty should not be run alone.** Forcing
  decorrelation from Log-Mel on a stack that can only compute energy envelopes
  pushes it toward a *degraded* envelope, not a novel feature. Only meaningful
  paired with the architecture change.
- **`fussion 5: ('tgram','tfgram')` is still worth it, as a diagnostic not a
  fix** — it answers whether TFgram *can* carry the spectral load alone.

## Limits

n=1 checkpoint for the drift measurement. The 400-step window puts the
pure-noise SNR floor at 0.05, and every branch lands at 0.03-0.05, so **SNR did
not discriminate** — it shows only that minibatch gradients are noise-dominated
equally for all branches, with no TFgram-specific deficit. The discriminating
evidence is `|dW|/|W|` and the drift cosine against the dead-weight control.

## Figures

Regenerate with `/home/f74134118/anaconda3/bin/python ablation/tfgram-gradient/make_plots.py`
(matplotlib lives in the `base` conda env, not `os-scl`). All four read
`results/probe.json`; nothing is hardcoded.

| figure | shows |
|--------|-------|
| `plots/fig1_rotation_vs_control.png` | rotation from init per encoder module, against the zero-gradient dead-weight control |
| `plots/fig2_adam_normalizes.png` | raw \|g\| vs the effective AdamW step -- why a small gradient is not a starved branch |
| `plots/fig3_parity_with_tgram.png` | TFgram / Tgram ratio for all three gradient metrics, both regimes |
| `plots/fig4_snr_null.png` | the SNR null, with arcface as the metric's positive control |

Blue and orange are bound to TFgram and Tgram in every figure. fig3/fig4 vary
the training regime instead, which is ordinal, so they use one hue in two steps
rather than borrowing the identity pair.

## Incidental findings

- The checkpoint is 325MB, of which 79.8M params (58x the live model's 1.38M)
  are the never-executed TFgram submodules. This is what `model_prune.py`
  strips; see the `--no-strict` trap in CLAUDE.md.
- `MobileFaceNet.fc_out` (`model/net.py:136`) is never trained — 2 tensors,
  never a gradient. `MobileFaceNet.forward` computes it, but `SCLTFSTgramMFN`
  discards that output and uses `arcface` on the embedding instead.
