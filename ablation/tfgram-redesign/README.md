# TFgram redesign — can a waveform branch learn what Log-Mel can't?

Follow-up to [`../tfgram-gradient`](../tfgram-gradient), which established that
TFgram is **constrained, not starved**: it trains at full step size, rotates
away from init, and still converges to a spectrogram (clip CKA 0.89 with
Log-Mel). Its explanation is architectural. `ReLU` followed by `max_pool`, with
511× downsampling, amounts to a rectified peak-energy envelope detector on a
learned filterbank, which is what a magnitude spectrogram is. Tgram, the branch
that *does* carry new information (CKA 0.06), has a sign-preserving
`LeakyReLU(0.2)` and no pooling.

That explanation is a claim about two specific operations, so this ablation
switches them off one at a time.

> **Status: closed.** Continued in [`../tfgram-linear`](../tfgram-linear),
> which trains the one variant this ablation found that escapes the envelope.
>
> What actually ran, in order (the plan below had a pilot and a sweep):
>
> 1. **Stage 1, 30-epoch pilot: void.** The gate didn't test anything: the
>    saved EMA checkpoints were ~55% random init (see *Results*).
> 2. **Init probes: the result.** `init_probe.py`, `tgram_probe.py` and
>    `band_probe.py`, with no training. **None of the four 2×2 cells stops
>    TFgram computing an energy envelope at random init**; only a fully linear
>    stack with linear pooling does, and that variant is not a second Tgram.
>    This is a statement about init, not training (see *What an init probe can
>    and can't show*).
> 3. **Stage 2, 300-epoch sweep: cancelled, never run.** At init the four cells
>    are the same spectrogram detector (see *What this means for the sweep*).
>
> The design, predictions and gate below were written before any data and are
> left unedited; outcomes are recorded under *Results*.

## Design — 2×2 factorial

Two switches on the TFgram branch, both new in `model/waveq.py`:

| flag | `relu`/`max` (paper) | alternative |
|------|----------------------|-------------|
| `--tf_act` | `ReLU` after every conv+BN (4 sites) | `leaky`: `LeakyReLU(0.2)`, as in Tgram |
| `--tf_pool` | `max_pool1d(4)`, `adaptive_max_pool1d(626)`, `(313)` | `avg`: the same three as average pools |

| cell | act | pool | role |
|------|-----|------|------|
| `relu_max` | ReLU | max | **the paper's TFgram = f1**, retrained on this host |
| `leaky_max` | Leaky | max | sign kept, peak-picking kept |
| `relu_avg` | ReLU | avg | rectification kept, pooling made linear |
| `leaky_avg` | Leaky | avg | both removed: the README's proposed fix |

Everything else is f1: `--fussion 1 --m 0.4 --ht basic`, batch 64,
`--num_workers 8`, 300 epochs. BatchNorm is left in place (Tgram uses
LayerNorm, the third difference `../tfgram-gradient` listed); holding it fixed
keeps this to two variables.

**Why avg pool and not a strided conv.** Average pooling is linear, so it
preserves sign, adds **zero parameters**, and handles the non-integer
32000→626 ratio exactly. A strided conv would also change capacity and mix a
third variable into the cell.

**Why the pairing is unusually clean.** Neither switch adds a parameter, so at
a given seed all four cells draw the **same weight init and the same
train/valid split**, and the same RNG stream follows. Verified bit-for-bit:
`relu_max` reproduces the original `TFgram` init and outputs exactly, and the
other three cells share its init (`check_equiv` output under *Verification*).

**Why f1 is retrained instead of reused.** The seed-sweep ran on RTX PRO 6000s
under torch 2.8 / cu128. This host (smartlab1: RTX A4000, CentOS 7, glibc 2.17)
can't install that stack and runs `environment-cu124.yml` (torch 2.6). Bit-
determinism is only verified within one machine and stack, so the committed
seed-sweep scores are not paired baselines for runs made here. They remain
useful as an unpaired cross-host check on `relu_max`.

## Stage 1 — pilot and gate

4 cells × 30 epochs × seed 2024, then `cka_probe.py` on each cell at its exact
init and at its checkpoint. It is cheap, and it also measures A4000 throughput,
which sets the sweep's real wall-clock budget.

**Gate (fixed in advance).** Proceed to the 300-epoch sweep if any redesigned
cell's trained clip-level CKA(Log-Mel, TFgram) is **≥ 0.15 below `relu_max`'s**
at the pilot. If none is, the architectural explanation is in trouble: report
the pilot as the result and don't spend the sweep.

The 0.15 margin is a judgement call, not derived. It is about a fifth of the gap
between TFgram–Log-Mel (0.89) and Tgram–Log-Mel (0.06), large enough that a
30-epoch, n=1 difference is unlikely to be noise in a deterministic CKA over
656 fixed clips.

## Stage 2 — sweep (cancelled, not run)

4 cells × seeds 2024–2027 × 300 epochs = 16 runs on GPUs 1 and 2, then the CKA
probe on all 16.

## Predictions

Written before any run.

**P1 — the mechanism (CKA).** If `../tfgram-gradient`'s explanation is right,
removing *either* operation alone is not enough. The mean of rectified values
is still an energy envelope (`relu_avg`), and the max of a leaky signal still
picks peaks (`leaky_max`). So the drop in CKA(Log-Mel, TFgram) should be
**concentrated in `leaky_avg`**, an interaction rather than two additive
effects:

    expected order of trained CKA(Log-Mel, TFgram):
    relu_max ≈ leaky_max  >  relu_avg  >>  leaky_avg

If instead one switch alone produces most of the drop, the explanation was
right about the cause but wrong to need both. If no cell drops, it was wrong.

**P2 — the risk (redundant with Tgram instead).** `leaky_avg` is
architecturally close to Tgram: sign-preserving, linear downsampling. Expect
CKA(Tgram, TFgram) to **rise** in that cell from ~0.06. If it rises toward
Log-Mel's level, the branch has swapped which neighbour it duplicates, and the
next step is the sub-frame-folding design (option C in the brainstorm).

**P3 — performance (AUC).** Expected: **null**. f4 (no TFgram at all) already
matches f1, so a redesigned branch only helps if the new feature carries
anomaly information Log-Mel + Tgram miss, and nothing so far shows such
information exists. The minimum detectable paired effect at n=4 is about
**0.5 AUC** (f1 sd 0.14–0.23 in the seed sweep). A cell counts as a gain only if
its paired delta vs `relu_max` is positive on dev *and* eval, survives
excluding ToyConveyor, and passes p < 0.0083 (Bonferroni over 3 cells × 2
sets). mAUC deltas under ~2 points are not read at all.

A **mechanism-only** outcome (P1 holds, P3 null) is still a result: it would
show the branch *can* be made non-redundant and that the non-redundant part
doesn't help detection, which goes further than "TFgram is redundant".

## Running it

```bash
conda env create -f environment-cu124.yml          # smartlab1; environment.yml elsewhere
bash ablation/tfgram-redesign/run.sh pilot         # 4 cells x 30 ep, then CKA (void gate, see Results)
bash ablation/tfgram-redesign/run.sh probes        # init/Tgram/band probes -> scores/*.json (no training)
/home/f74134118/anaconda3/bin/python ablation/tfgram-redesign/make_plots.py
```

The `sweep` stage was removed from `run.sh` when it was cancelled. The probes
need no checkpoints and take minutes.

`TFR_GPUS` overrides the card list (default `"1 2"`; GPU 0 is usually taken).
Each cell writes `scores/<model_name>/{eval_dev,eval_eval,config}.*`, and
`run.sh` distils `curve.csv` at the end. `eval.py` now reads `tf_act`/`tf_pool`
from the checkpoint's `config.json` and refuses to run on a mismatch, because
these switches add no parameters and a wrong-architecture load would otherwise
succeed silently.

## Verification

Before any training, on smartlab1 (torch 2.6.0+cu124, RTX A4000), random
input `(4, 1, 160000)`, `torch.manual_seed(0)` before each construction:

```
default == original: init and output bit-identical, train+eval mode
relu  max  shape (4, 128, 313)  frac<0 0.000  std 0.669  same init: yes
relu  avg  shape (4, 128, 313)  frac<0 0.000  std 0.424  same init: yes
leaky max  shape (4, 128, 313)  frac<0 0.260  std 0.697  same init: yes
leaky avg  shape (4, 128, 313)  frac<0 0.323  std 0.470  same init: yes
```

"Original" is `model/waveq.py` at `b663cb4`, loaded side by side. So
`relu_max` *is* the paper's TFgram, not an approximation of it, and all four
cells start from the same weights. The leaky cells' outputs keep a negative
part (26–32% of values), which ReLU cells can't have. Avg pooling shrinks the
output scale (std 0.42–0.47 vs 0.67–0.70). That's a scale change the
learned conv1 weights can compensate for, but it isn't normalised away per
channel: the three branches are concatenated *before* MobileFaceNet's first
conv+BN.

The eval guard was checked both ways: a `leaky/avg` `config.json` evaluated
without flags exits with `tf_act mismatch: checkpoint trained with 'leaky',
eval asked for 'relu'`, and with matching flags it goes on to load the weights.

## Results

### Stage 1 pilot: the gate is void, not failed

Four cells × 30 epochs × seed 2024 on two A4000s (`scores/pilot_timing.csv`:
2.12–2.35 h per cell, **0.075 h/epoch, so a 300-epoch cell is ~22 h** here vs
~4.6 h on the PRO 6000 host).

```
                   clip CKA(Log-Mel, TFgram)        eval AUC   valid_acc
cell               init    "trained" (saved ckpt)
relu_max           0.810   0.777                    52.39      0.000
leaky_max          0.812   0.800                    53.03      0.000
relu_avg           0.806   0.758                    52.00      0.000
leaky_avg          0.805   0.796                    52.52      0.000
```

By the letter of the gate (a cell ≥ 0.15 below `relu_max`), **no cell passes**.
But the measurement behind it is broken, and it would be wrong to read this as
evidence:

- **Every model is at chance** (dev AUC 47–48, eval 52–53) with validation
  accuracy exactly 0.000 throughout.
- The cause is `WeightEMA` (alpha 0.9999, `losses.py`). `trainer.py`
  validates and saves the **EMA net**, which starts as the init and absorbs the
  student at `1 − alpha` per step. After *e* epochs the init still makes up
  `0.9999^(539·e)` of its weights: **55% at epoch 11**, where validation loss
  last improved and the checkpoints were saved, and 20% at epoch 30 (fig2).
  BatchNorm running stats are blended the same way.
- So the "trained" column mostly measures the init a second time. The gate
  needed either the student's weights or a run long enough for the EMA to
  forget its start (≥ ~90 epochs for < 1% init). This is my design error;
  the 30-epoch figure was picked for cost, without checking the EMA horizon.

### Init probe: what the gate should have asked

The pilot did produce one valid measurement: **at random init, all four cells
are already at clip CKA 0.81 with Log-Mel** (fig1), within about 0.01 of each
other. The old host's *trained* f1 sits at 0.89, so most of TFgram's
spectrogram-likeness is architectural and present before any learning, which
is the claim `../tfgram-gradient` inferred by elimination.

That raised the question of *why* `leaky_avg` doesn't change it.
`init_probe.py` (no training, 3 seeds, the same 656 clips) sweeps the
activation's negative slope from 0 (ReLU) to 1 (identity) under both pools.
Output in `scores/init_probe.json`:

```
clip CKA(Log-Mel, TFgram), mean of 3 seeds
slope      0.0    0.2    0.5    0.8    0.95   1.0
max pool   0.82   0.82   0.82   0.82   0.82   0.82
avg pool   0.81   0.81   0.81   0.81   0.77   0.08
```

Two mechanisms, each sufficient on its own:

1. **Any rectification before a large average gives an envelope.**
   `LeakyReLU(a) = (1+a)/2 · x + (1−a)/2 · |x|`. Averaging over ~500 samples
   cancels the zero-mean linear term and keeps the `|x|` term. Even a = 0.95,
   only 2.5% rectifying, keeps CKA at 0.77.
2. **Max pooling is itself a rectifier.** With a fully linear stack (a = 1),
   max pooling still gives 0.82.

Only **linear activations and linear pooling together** decorrelate, reaching
0.08, which is exactly Tgram's level against Log-Mel (0.07). That fits Tgram's
design: it does all of its downsampling *first*, linearly (a stride-512 conv),
and applies its nonlinearity only afterwards, at the frame rate. TFgram applies
its nonlinearities at the sample rate and pools after them, and that **order**,
not the choice of ReLU or max, is what makes it an envelope detector.

### What an init probe can and can't show

It doesn't measure a trained model. It supports two narrower kinds of claim:

1. **Structural, weight-independent.** Rectify-then-average gives a band energy
   for *any* filter (`mean|h * x|` over a window is the energy of `x` in `h`'s
   band), so training can choose the bands but not make the output stop being
   an energy. Conversely, a fully linear stack with avg pooling is a linear map
   of the waveform for *any* weights (BN is affine), so it can never become an
   envelope. The probe illustrates these two ends; neither depends on the
   weights being random.
2. **Init is this branch's operating point.** `../frozen-tfgram` showed a
   random-init TFgram scores as well as a trained one.

What it can't rule out: training the rectifying cells to compute *nonlinear*
functions of band energies (modulation, envelope differences) with lower
*linear* CKA to Log-Mel. The only evidence against that is indirect and from
`relu_max` alone (trained rotation without escaping the spectrogram in
`../tfgram-gradient`, trained f1 at 0.89 vs 0.81 at init, frozen ≈ trained).
One relu_max run of ≥ 90 epochs that saves the student's weights would test it
directly.

### Is a decorrelated TFgram just a second Tgram? (P2, at init)

The worry from the brainstorm: a linear stack of linear decimations is a
learned filterbank on the waveform, which is also what Tgram's front end
(`conv_extrctor`, 1024 taps, stride 512) is. `tgram_probe.py` (random init, 3
seeds, the same 656 clips) compares each TFgram variant with Tgram's full
output and with its front end alone, using frame-level linear R² in both
directions. A control calibrates what "same information, different random
filterbank" looks like: two independently seeded Tgram front ends. Output in
`scores/tgram_probe.json`:

```
frame R², share of TFgram predictable from ...    Log-Mel   Tgram   Tgram front
relu_max (paper)                                     0.84     0.02     0.01
leaky_avg                                            0.85     0.02     0.01
linear_max                                           0.81     0.02     0.01
linear_avg                                           0.00     0.01     0.01
linear_first (linear_avg, then LeakyReLU)            0.10     0.01     0.01
control: Tgram front from another Tgram front                          0.86
```

> **Reproducibility note (2026-10-09).** Rerunning `tgram_probe.py` via
> `run.sh probes` reproduces every value in `scores/tgram_probe.json` to 1e-6
> except three `r2_b_to_a` entries for seed 2026 `relu_max`, the direction
> where TFgram is the *predictor*: Tgram 0.006 → −0.123, Tgram front −0.002 →
> −0.020, Log-Mel 0.455 → 0.455. The cause is a dead (near-constant) channel
> in that init (smallest/largest singular value ≈ 2e-18). `lstsq` then fits
> last-bit noise, and a negative in-sample R² is the symptom. Recomputed
> stably (rank-truncated SVD), the committed 0.006 is correct. No figure or
> conclusion above uses that direction, and the table reports the opposite one.
> `../tfgram-linear/branch_probe.py` uses the stable computation.

**Not a second Tgram, at init.** Two random Tgram front ends predict each
other at R² 0.86, while linear_avg and Tgram's front end predict each other at
0.01 both ways. The same holds for Tgram's full output.

**And not empty.** My follow-up guess was that linear_avg decorrelates only
because ~511× averaging low-passes it to near-DC (cutoff ≈ 16000/511 ≈ 31 Hz).
`band_probe.py` refutes that: high-passing the input at 31 Hz leaves **95%** of
its output variance, and at 100 Hz still 73%. It responds to the same 31–300
Hz band as the paper's stack (which keeps 26% after a 300 Hz high-pass, against
11% for linear_avg). Avg pooling is a poor anti-aliasing filter, so that band
arrives at the 31 Hz frame rate as **aliased, phase-sensitive values** rather
than energies. That is why it matches neither branch. Whether such a feature
carries anomaly information is a training question. Output in
`scores/band_probe.json`.

**Where the nonlinearity goes matters.** Adding a LeakyReLU *after* the linear
stack (`linear_first`) brings back some Log-Mel similarity (R² 0.10 Log-Mel→TF,
0.38 TF→Log-Mel, frame CKA 0.27). Rectifying a low-band signal at the frame
rate is a crude low-band energy. Tgram avoids this despite the same order,
presumably because its LayerNorm + 1024-tap front end feeds the rectifier a
broadband projection instead.

### A correction to the metric: clip-level CKA is biased toward 1

Clip-level linear CKA (656 clips × 40,064 dims) is **not reliable** for the
noise-like features a linear branch produces. Two completely independent random
signals of that shape score clip CKA **0.98**, because with N ≪ D both Gram
matrices are close to a scaled identity, so their CKA approaches 1 whatever the
content (the known small-N bias of linear CKA). This is why linear_avg vs
Tgram's front end shows clip CKA 0.955 alongside frame CKA 0.004 and R² 0.01.

Frame-level CKA (205k rows × 128 dims) and R² don't have the problem. Every
conclusion in this README holds on them: the rectifying cells sit at frame CKA
0.73–0.75 with Log-Mel against 0.000 for linear_avg (fig1, fig3). The older
ablations' headline "clip CKA 0.89" (`../gram-branch-redundancy`) is
upward-biased by the same mechanism, though its frame-level 0.81 and R² 0.63
support the same conclusion. That README has not been annotated yet.

### Against the predictions

- **P1 — refuted at init, for the cell it was about.** P1 expected the drop to
  be concentrated in `leaky_avg`. At init `leaky_avg` doesn't drop at all (0.81),
  because slope 0.2 still rectifies. The "both operations are needed"
  intuition was right in form (only the act+pool combination works), but the
  2×2 sampled the act axis at 0.2, far from the only working value, 1.0.
  Caveat: this is init-only. Training could in principle move a cell away,
  though the one trained data point (f1, 0.81 → 0.89 on the old host) moved
  *toward* Log-Mel.
- **P2 and P3 — untested.** No cell decorrelated, so there was nothing to
  become Tgram-like and nothing to score.

### What this means for the sweep

**Running the 2×2 sweep as designed is not recommended.** It would cost 16 ×
~22 h ≈ 5–7 days on 2–3 A4000s to train four cells that, at init, are the
same spectrogram detector within 0.01 CKA. The probe instead points at designs
the 2×2 didn't include:

- **linear-first TFgram**: identity activations through the downsampling
  stack, with the nonlinearity moved after the 313-frame output (the Tgram
  order). The risk: a linear stack of linear decimations is a learned
  filterbank on the raw waveform, i.e. close to Tgram's `conv_extrctor`, so
  P2 (redundancy with Tgram) becomes the main question.
- **lossless downsampling** (option C): fold sub-frame samples into channels
  instead of pooling, so nothing is averaged and the nonlinearity can stay.

Either one needs a gate that actually measures training: probe the student
(save `net` alongside `ema_net`) or train ≥ 90 epochs.

**Taken up in [`../tfgram-linear`](../tfgram-linear)**, which trains
`linear_avg` (identity activations, avg pooling) against a retrained
`relu_max` at 300 epochs and saves student snapshots.

## Figures

Regenerate with `/home/f74134118/anaconda3/bin/python ablation/tfgram-redesign/make_plots.py`
(matplotlib lives in `base`, not `os-scl-cu124`). Reads `scores/` only.

`relu_max` is the paper's branch and the baseline, so it is drawn achromatic.
The three redesigned cells take the three categorical slots that validate under
`--pairs all`. In fig3, avg pool is blue and direct-labelled.

| figure | shows |
|--------|-------|
| `plots/fig1_init_cka.png` | clip and frame CKA with Log-Mel at random init for all four cells: they're within about 0.01 of each other and close to trained f1; hollow markers are the void pilot EMA checkpoints |
| `plots/fig2_ema_init_share.png` | share of the EMA weights still at init vs epoch: why the 30-epoch gate measured mostly init |
| `plots/fig3_slope_sweep.png` | init CKA vs activation slope under max and avg pooling: only slope 1 with avg pooling decorrelates |
| `plots/fig4_not_tgram.png` | frame R² of each TFgram variant from Log-Mel / Tgram / Tgram's front end, against the two-random-front-ends control; and which input band drives the output |
