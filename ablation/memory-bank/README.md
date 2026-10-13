# Memory bank — does patch-level nearest-neighbour scoring beat clip-level?

> **Status: running (2026-10-13).** Pre-registered at 332ac22 and the code
> committed at 4e59b4d before any test-clip patch score. The primary arms (A1–A4)
> have been running on GPU 0 since 17:39, about 11 h.
>
> **Resume:**
> - Progress: `ablation/memory-bank/results/run_bank_primary.log`. Per-ID outputs are
>   in `results/full/<arm>/<run>/`; the run is resumable, so a rerun skips finished
>   IDs.
> - When it finishes:
>   1. `bash ablation/memory-bank/run.sh score` writes `scores/`.
>   2. Run the sensitivity arms with `MB_ARMS="mfn_l2 mfn_l3 mfn_cs10" MB_REPO=/home/f74134118/OS-SCL bash ablation/memory-bank/run.sh bank`, then score again.
>   3. Write `make_plots.py` and the Results section against the predictions below. Do not edit them.

## Why this exists

`../ensemble-backend` gave each clip a second score next to eval.py's R0: the
1-nearest-neighbour cosine distance K from the clip's 128-d pre-head embedding
to the embeddings of its ID's ~886 training clips. That is a **memory bank
with one vector per clip**. Fused with R0, using per-ID statistics from the
training clips, it was positive in 16/16 cells (S1′: +0.42 dev, +0.65 eval AUC
over R0). It was pre-registered only as a sensitivity check, so it is not
adopted yet.

PatchCore (Roth et al., CVPR 2022), the standard memory-bank method for visual
defect detection, differs in three ways:

1. **Patches, not whole inputs.** It keeps one vector per spatial location of a
   mid-level feature map. A test input is scored by its *worst* patch: the
   largest distance from any of its patches to that patch's nearest normal
   patch.
2. **Mid-level features from a generic ImageNet network** (WideResNet-50,
   layers 2 and 3). The target data is never trained on.
3. **A coreset**: a subsample of the bank chosen to cover it, for speed.

**Why patches might matter here.** MobileFaceNet turns the 3×128×313 input
into an 8×20 grid of 512-d vectors. `linear7` (`model/net.py:131`) is a
depthwise 8×20 convolution, i.e. a learned weighted average over that grid,
and its output is the one vector both R0 and K see. A fault lasting one of
the 20 time steps reaches that vector diluted by the other 19. A patch-level
score is a maximum over locations, so it is not diluted. Pump is the one
machine no scoring change has moved: −1.29 dev AUC against the paper even
ensembled (`../ensemble-backend`). Whether pump's faults are local in time or
frequency is unknown, and that is part of what this tests.

**Prior work.** ImageNet features on mel spectrograms, with GMM or OC-SVM on
top, beat autoencoders on MIMII (Müller et al., ICAART 2021, arXiv:2006.03429).
Vision anomaly detectors, PatchCore among them, have been run on DCASE 2020
audio with mixed, representation-dependent results (a Stuttgart thesis; and
arXiv:2502.18328 on other data). AnoPatch (arXiv:2406.11364) reports
state-of-the-art DCASE 2020 results with patch-level modelling on an
AudioSet-pretrained ViT, but its numbers were not checked here. None of these
tested PatchCore on the features of an ID-supervised ASD model, which is the
main arm here.

## What has already been seen

- **From `../ensemble-backend`:** every R0, K, S1 and S1′ score on dev and eval
  for the 8 checkpoints below. The clip-level reference arm A0 is therefore
  fully known, and the new arms will be compared against it.
- **Smoke test, 2026-10-13:** a forward pass of the 853 training-part clips of
  one pump ID through `tfl_relu_max_s2024` and WideResNet-50, to record
  feature-map shapes and time the nearest-neighbour search. No test clip was
  touched and no anomaly score was computed. Shapes for a 3×128×313 input:

  ```
  MobileFaceNet  blocks[3] (end of stage 2)  128 × 16 × 40
                 blocks[5] (end of stage 3)  128 ×  8 × 20
  TFgram         pre_block2                  128 × 626   (channels × time)
                 pre_block3 (branch output)  128 × 313
  WideResNet-50  layer2                      512 × 16 × 40
                 layer3                     1024 ×  8 × 20
  input stack    Log-Mel mean −24.1 sd 8.7 · Tgram 0.04 / 0.89 · TFgram 0.15 / 0.24
  ```

  One ID's 16×40 bank is 545,920 patches. Exact search on one A4000 costs
  42 ms per test clip at 128-d and 56 ms at 1536-d, so the full bank, without a
  coreset, is affordable.

- **Environment change:** `torchvision 0.21.0+cu124` and `pillow` were added to
  `os-scl-cu124` with `--no-deps` for WideResNet-50. torch stays 2.6.0.

**Addendum, 2026-10-13, after writing the code and before any test-clip patch
score.** Implementation choices the text above left open:

- **imagenet_mel's bank uses all of each ID's training clips.** It has no
  checkpoint, so there is no held-out split to replay. That is about 5% more
  bank than the model arms get, a slight edge in P4's A3-vs-A4 comparison.
- **The nearest-neighbour search runs in fp16** on features centred on the bank
  mean. Centring leaves L2 distances unchanged. Rounding is about 0.1–1%
  relative per patch.
- **Leave-one-out was verified against physically deleting the clip from the
  bank** (fp32): the max relative difference is ≤ 1.4%, which is fp16 rounding.
- **P0 is checked per ID inside `bank.py`**: its R0 must match the cached
  ensemble-backend R0. S1′ recomputed from the cache matches the committed
  `rule_metrics.csv` to 4.5e-6.
- **P4's "F_A4 does not beat S1′"** is scored as mean dev Δ ≤ 0.
- **Runtime:** the primary arms (A1–A4) take about 11 GPU-h and run first. The
  sensitivity arms (single-layer banks, the 10% coreset) take about 6 GPU-h and
  run later on whichever GPU is free.
- **Two out-of-memory crashes, neither changing a score.** The first was in
  imagenet_mel, from allocator fragmentation; `run.sh` now sets
  `expandable_segments` (d904ccc). The second, at id28 (ToyConveyor id_01, the
  largest test set), came from centring every test patch at once (2.2 GB on top
  of 11.9 GB). Test clips are now searched in chunks of about 200. Each
  patch's nearest-neighbour search is row-independent, so this is the same
  computation. The run resumes at id28. IDs 0–27 are unchanged.

## Design

### Checkpoints

The same 8 as `../ensemble-backend`: `tfl_relu_max` (the paper's model) and
`tfl_linear_avg`, seeds 2024–2027, EMA weights.

Their eval scores have been seen under R0, K, S1 and S1′, but never under any
patch-level score. Eval therefore still confirms the *new* rules. A pass over
new checkpoints follows if anything here is adopted (see *Decision rules*).

### Arms

Every arm builds one memory bank **per machine ID per checkpoint**, from that
ID's training-part clips (the 95% the checkpoint trained on, replayed from its
seed as in `../ensemble-backend`; median 886 clips per ID, minimum 408).

| arm | features | vector per patch | patches per clip |
|-----|----------|------------------|------------------|
| **A0 clip** | pre-head embedding, cosine distance (= K, already computed) | 128-d | 1 |
| **A1 mfn** | trained MobileFaceNet: `blocks[3]` ⊕ `blocks[5]` upsampled to 16×40 | 256-d | 640 |
| A2 tfgram | trained TFgram: `pre_block3` ⊕ `pre_block2` avg-pooled ×2 to 313 | 256-d | 313 |
| A3 imagenet-stack | WideResNet-50 `layer2` ⊕ `layer3` upsampled, on the model's 3-channel stack | 1536-d | 640 |
| A4 imagenet-mel | as A3, on Log-Mel copied to all 3 channels; uses no OS-SCL weights | 1536-d | 640 |

- **A1 is the primary arm, fixed now.** It is the direct PatchCore analogue on
  our own trained network: stage-2 and stage-3 outputs, which correspond to
  ResNet layers 2 and 3.
- **A2** is reading (b) of the brief, TFgram's own layers as a fixed extractor.
  `../frozen-tfgram` found an untrained TFgram matches a trained one inside
  the model.
- **A3** is reading (a), feeding our input to an ImageNet network.
- **A4** is the model-free control. It answers whether anything from the
  ID-supervised training helps at all. Being seed-free, it has one value per
  test clip, not eight.
- **Shared PatchCore details:**
  - Each patch vector is first averaged with its 3×3 neighbourhood (3 frames
    for A2).
  - Distance is L2.
  - The clip score P = max over the clip's patches of the distance to the
    nearest patch in the bank.
  - Each input channel for A3/A4 is standardised with mean and sd over that
    checkpoint's training-part clips.
  - The bank is used in full, with no coreset (affordable, see above).

### Statistics and fused rules

Raw distances are not comparable across IDs, so every score is z-scored per ID
with **training-part statistics**, the choice `../ensemble-backend` found
better than the ~44 held-out clips:

- **zP:** for 200 training-part clips per ID (fixed random subset, numpy seed
  0; all clips if fewer), P is computed **leave-one-clip-out**, with all of the
  clip's own patches removed from the bank. zP = (P − mean) / sd over those 200
  values.
- **zR0′ and zK′:** exactly as in `../ensemble-backend`.

| rule | definition | role |
|------|------------|------|
| R0 | eval.py | baseline |
| S1′ | zR0′ + zK′ | reference: clip-level fusion (A0), already measured |
| **F_a** | zR0′ + zP_a | **patch-level fusion, per arm a** |
| P_a | zP_a alone | standalone memory bank |
| E_a | Σ over relu4 of F_a | ensemble, reported only |

### Metrics

As `../ensemble-backend`: pooled AUC, pAUC and mAUC averaged over the 6
machines, plus per machine. Paired comparisons are per checkpoint. The
uncertainty on a mean Δ is a stratified bootstrap over test clips, 2000 draws.

## Predictions

Written before any score.

**P0. Pipeline.** Recomputed A0 reproduces `../ensemble-backend`'s committed
S1′ overall AUC for every checkpoint and set within 1e-3, so the new
extraction sees the same clips, splits and weights.

**P1. Patch beats clip as a standalone score:** P_A1 > zK′ in overall dev AUC
on ≥ 6 of 8 checkpoints. Confidence: low (40%). ArcFace training puts its
discriminative signal into the pooled vector, and the mid-level grid was
never trained to be compared location by location.

**P2. Patch beats clip inside the fusion, the decision-bearing one:** F_A1 −
S1′ mean overall AUC ≥ +0.2 on dev, with F_A1 > S1′ on ≥ 6/8 checkpoints on
eval. Confidence: low (30%).

**P3. Pump.** Mean dev pump AUC of F_A1 − S1′ ≥ +0.5. Confidence: low (30%).
If the max-over-patches score helps anywhere, I expect transient faults:
valve (no room left: 99.9) and ToyCar.

**P4. Generic features lose to ours.** P_A4 is below R0 by ≥ 2 overall dev AUC
(confidence: 80%). Fused, F_A4 does not beat S1′ (75%). P_A3 and P_A4 differ
by < 0.5 dev AUC (60%), because the Tgram and TFgram channels largely repeat
Log-Mel (`../gram-branch-redundancy`, CKA 0.89).

**P5. TFgram alone is weak:** P_A2 < P_A1 in overall dev AUC on ≥ 6/8
checkpoints (75%). It is a shallow 1-D stack whose training adds nothing.

**P6. Ensembles absorb most of it:** E_A1 − Σ_relu4 S1′ lies in [−0.1, +0.2]
AUC, i.e. smaller than any single-model gain.

## Decision rules

Fixed now. Dev selects, eval confirms. A configuration is never picked for its
eval score.

- **D1, adopt the patch bank in place of the clip bank.** F_A1 replaces S1′
  as the fusion rule carried into the next confirmation iff P2 holds. Both
  conditions are needed: dev mean ≥ +0.2, and eval positive on ≥ 6/8.
  "Adopted" here means it becomes the pre-registered primary for the next new
  checkpoints. It is not a claim against the paper.
- **D2, other arms are exploratory.** An arm other than A1 whose F beats S1′ by
  ≥ +0.2 dev AUC is reported and becomes a pre-registered candidate for the
  next new checkpoints. It is not adopted here, since it was one of several.
- **D3, sensitivity checks never replace the primary configuration.** These are:
  - a coreset of 10% (greedy k-centre on a 128-d random projection, as in
    PatchCore);
  - a top-10-patch mean in place of the max;
  - single-layer banks.
- Single-model mAUC deltas under ~2 points are not read (`CLAUDE.md`).

## What this cannot answer

- **S1′ itself is still unconfirmed.** It is the reference here because it is
  the best known single-model rule, not because it is adopted.
- **Location within the clip is not validated.** Patch scores give a
  time-frequency map of where a clip is anomalous. DCASE 2020 has no
  localisation labels, so any map shown is illustrative only.
- **One run per arm per checkpoint.** All nearest-neighbour searches are
  exact. Variation between checkpoints is the only seed variance measured.

## Verification (planned, before scoring)

1. P0: A0 is recomputed from a fresh extraction and must match the committed S1′.
2. Leave-one-out: for each statistics clip, assert that the bank searched
   excludes exactly that clip's patches.
3. Clip alignment: every per-clip array is checked by path against
   `../ensemble-backend`'s `meta.npz` order.

## Planned files

```
bank.py         GPU: per checkpoint, per ID, extract patch features, search the bank, write per-clip P
score.py        CPU: zP, F, E, bootstrap, scorecard -> scores/
run.sh          end to end, resumable; MB_GPU, MB_REPO as in ../ensemble-backend
make_plots.py   reads scores/ only
scores/         rule_metrics.csv, summary.csv, bootstrap.json, scorecard.txt
results/        per-clip P arrays, gitignored
```

Estimated cost: about 10 GPU-hours, mostly the leave-one-out statistics and
the 1536-d ImageNet banks. It splits across GPUs by checkpoint.
