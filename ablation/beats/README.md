# beats — a fine-tuned BEATs model, alone and fused with our ensemble

> **Every number in this ablation is WITH EXTERNAL PRETRAINING.** The model starts from
> BEATs_iter3, which was pre-trained self-supervised on AudioSet-2M. Report it as a separate
> track: "with external pretraining". Never put it in a from-scratch table without that label.
> The from-scratch bests stay where they are: C1 single model (`ablation/pooled-knn`) and
> E2′[A8] ensemble (`ablation/mixed-ensemble`).

**Status (2026-11-08): TRAINED, stage A (dev) committed; stage B (eval) next.**
- 3 runs done (17.6–19.5 h each, 0 skipped steps, final valid accuracy 0.9989).
- Stage A picks zKp and F1 (see *Stage A note*: the BEATs CE degeneracy, and a declared
  descriptive secondary F1k).

## Question

Our best from-scratch single network (`ao_nogain`, scored with C1, 8 seeds) reaches dev
Amean 94.16 and eval Amean 94.09. Amean is the mean of AUC and pAUC. Our best ensemble,
E2′[A8], reaches dev 94.77 and eval 94.78. Most of the remaining eval error is in
ToyConveyor (~60%), ToyCar (22%) and pump.

Published fine-tuned BEATs systems show the opposite profile. They are weaker than us on dev,
especially fan and ToyConveyor. On eval they are as good or better, especially ToyCar
(AnoPatch eval ToyCar 96.0 vs our C1 93.8).

1. Does one fine-tuned BEATs model beat our C1 single model on eval?
2. Its errors should differ from ours. Does adding it to E2′[A8] help the ensemble?

## Literature (read 2026-11-06; numbers are Amean = (AUC+pAUC)/2 unless stated)

| system | dev | eval | notes |
|---|---|---|---|
| AnoPatch (arXiv 2406.11364), BEATs_iter3 + ASP + ArcFace, 1-NN kNN | 90.93 | 94.28 | per machine dev fan 86.46 pump 93.10 slider 99.20 ToyCar 96.10 ToyConveyor 73.20 valve 97.53; eval 95.56 / 94.34 / 99.77 / 96.00 / 83.74 / 96.26. The text says "92.58 on both sets"; 92.58 is the *combined* mean of the table. Seeds not stated |
| PerMix (arXiv 2610.31434) Table 3, BEATs + ArcFace baseline, kNN, 3 seeds | 93.03/87.97 AUC/pAUC (90.50) | 95.96/91.39 (93.68) | lr 1e-5, batch 32, 100 epochs, AdamW, s 30 m 0.4; "architecture follows AnoPatch" |
| PerMix, BEATs + OS-SCL loss (mixup + SupCon), 3 seeds | 94.63/88.92 (91.78) | 95.90/90.68 (93.29) | our loss on BEATs: +1.3 dev, −0.4 eval vs plain ArcFace |
| PerMix, BEATs + PerMix loss, 3 seeds | 96.56/92.41 (94.49) | 97.99/93.17 (95.58) | a different loss (ablation/permix tests it on the MFN) |
| BEATs+LoRA (2508.12230), BEAM/AdaBEAM (2603.13749) | — | — | not read in detail; abstracts only. Figures quoted in the task (91.1/95.0; 95.6/96.2) are unverified |

**AnoPatch setup as published (Sec. 3, 4.3, Table 4):**
- Model: ViT, 12 layers, 90 M parameters, from BEATs iteration 3, all parameters trained.
- Input: 128 mel bins, 25 ms window, 10 ms hop, 16×16 patches.
- Head: ECAPA attentive statistics pooling, then a linear layer to a 128-d embedding.
- Loss: ArcFace over 41 classes (machine type × ID).
- Optimiser: AdamW, lr 1e-4, 960 warm-up steps, 10k steps, batch 32, gradient accumulation 8.
- Augmentation: SpecAugment, frequency mask 40, time mask 80, no time warping.
- Scoring: 1-NN cosine distance, computed per machine ID.
- Hardware: one A6000.

**Not specified, so these are our choices:**

| item | our choice | why |
|---|---|---|
| ArcFace s, m | **s 30, m 0.4** | PerMix's values, also our MFN's |
| "10k steps" | **10k optimizer steps** (×8 micro-batches = 2.56 M clips ≈ 74 epochs) | Read as micro-batches it would be only 9 epochs. PerMix's 100 epochs (3.4 M clips) supports the optimizer-step reading. **Main ambiguity** |
| LR schedule after warm-up | **cosine to 0** | not stated |
| weight decay | **0.01** (AdamW default) | not stated |
| SpecAugment | **one mask of each kind, widths uniform up to 40 / 80, mask value 0** (= the fbank mean after BEATs normalisation) | number of masks not stated |
| dropout / layerdrop | **the checkpoint's own cfg**: dropout 0.1, attention 0.1, input 0.1, layerdrop 0.05 | not stated |
| fbank | BEATs' own `preprocess`: Kaldi fbank, 128 bins, 25/10 ms, mean 15.41663, 2 × std 6.55582 | matches "25 ms window, 10 ms hop, 128 Mel" |
| checkpoint selection | **final weights, no EMA** | AnoPatch mentions neither EMA nor early stopping. Cosine-to-0 already averages the last steps. Validation loss is logged, not used |
| grad clipping | none (non-finite steps skipped and logged) | not stated |
| seeds | not stated | 3 here |

## Recipe (the one arm: `bt_arc`)

`ablation/beats/train_beats.py`, with flags fixed in `run.sh` (`RECIPE`):

- **Data.** Same protocol as `train.py`:
  - Split: `random_seed(seed)` → `train_dataset` → `dataset_split` (95/5, the first torch RNG
    draw). Seed s therefore holds out exactly the clips `ao_nogain_s<s>` / `la_aug_s<s>` held
    out, and the scorers' `replay_valid` reproduces it. `extract.py` asserts this against the
    run's config.
  - Labels: the same 41 machine-ID classes (`dataloader.machine_ids`).
  - Waves are read from `data/cache/train_wav.npy` and range-checked (finite, |x| ≤ 1); a bad
    item is re-decoded.
- **Model** (`model/beats_asd.py`):
  - BEATs_iter3, fully fine-tuned.
  - SpecAugment on the normalised fbank (training only).
  - Pooling: ASP over the 496 tokens (62 time × 8 frequency patches) → 1536 → Linear → 128-d
    `pre` → ArcMarginProduct (s 30, m 0.4).
- **Loss and optimiser.** Plain ArcFace CE: no mixup, no SupCon. That is AnoPatch, and PerMix's
  better-on-eval baseline. AdamW 1e-4, wd 0.01, 960 warm-up steps, cosine decay, 10k steps of
  32 × 8 clips.
- **Precision and speed.**
  - bf16 autocast; fbank and ArcFace stay in fp32.
  - Fused attention (`--sdpa 1`, a `model/beats` edit marked `OS-SCL:`). It equals upstream
    in fp32 (logits max |Δ| 5e-6, gradient relative Δ 7e-7).
  - No activation checkpointing.
  - Micro-batch **16 × 16** accumulated = 256 clips per step. That is AnoPatch's effective
    batch; the only difference from 32 × 8 is the batch the ASP BatchNorm sees. In the bench,
    32 × 8 is no faster and peaks at 12.1 of 14.7 GiB; 16 × 16 peaks at 6.9 GiB.
- **Data order.** A pure function of (seed, step): epoch e is
  `randperm(generator seed*1000+e)`. It does not depend on `num_workers`, and `--crash_resume`
  continues it exactly.
- **Crash safety.** `state_last.pth` every 500 steps; `run_one.sh` retries up to 3 times.
- **Seeds.** 2040, 2041, 2042. They are paired by split with `ao_nogain` and `la_aug` s2040–42.

We fix one recipe and do not tune it. Tuning on dev would use up the dev set before it is
needed for scoring and fusion selection.

### Checkpoint

- **Choice: BEATs_iter3** (SSL pre-training on AudioSet-2M), not iter3+ (AS2M).
  - AnoPatch and PerMix both name iter3, so the published numbers are our reference.
  - iter3+ uses a tokenizer distilled from a model fine-tuned on AudioSet labels, which adds
    supervised label information.
  - We use neither of the AS2M-*fine-tuned* (cpt1/cpt2) variants.
- **Source: <https://huggingface.co/lpepino/beats_ckpts>, `BEATs_iter3.pt`, 361,499,833 bytes,
  sha256 `8d1b234032a9ccff353612dc6c20982346dc2968b205b79d97303eb5e77bfb34`.** This is the
  HF LFS oid; the downloaded file matches it.
  - The official source is a OneDrive link in microsoft/unilm/beats. It returns 403 /
    "unauthenticated" to scripted downloads, and Microsoft publishes no hashes.
  - Corroboration that the mirror is faithful: its `BEATs_iter3_plus_AS2M.pt` (sha256
    `d43cbfad…8d34`) is byte-identical to three independent HF mirrors (alibabasglab/beats,
    mooneyko/BEATs, SongsongBaobao8694/beats-iter3).
  - The iter3 file has the expected cfg (deep-norm post-LN, 12 × 768, gated relative position
    bias) and 250 tensors / 90.35 M parameters.
  - Loaded with `weights_only=True`. `model.beats_asd.load_pretrained` re-checks the hash.
- Stored at `data/pretrained/hf/BEATs_iter3.pt` (gitignored, under `data/`).
  `run.sh fetch` re-downloads and verifies it.
- **Code:** `model/beats/{BEATs,backbone,modules}.py` are vendored from microsoft/unilm
  commit `732d834` (MIT, LICENSE kept). Edits: package-relative imports, and optional
  per-layer `torch.utils.checkpoint`. The wrapper's token output equals upstream
  `extract_features` exactly (max |Δ| = 0.0 on CPU, eval mode).

## Scoring (`extract.py`, `score.py`)

**Dumps.** `extract.py` writes `ensemble-backend/extract.py`'s format: `meta.npz` (identical
clip order, checked), plus per run `tr_ce tr_pre valid valid_order te_ce te_pre seed`, plus
`tr_gmean te_gmean`.
- `pre` is the 128-d embedding before ArcFace, the counterpart of the MFN's pre-head embedding.
- `gmean` is the mean of the 496 final BEATs tokens (768-d), the counterpart of pooled-knn's
  Gmean.
- Writing uses pooled-knn's `save_verified`: fsync, page-cache drop, CRC re-read.
- The unchanged `ensemble-backend/score.py:per_run_scores` reads these dumps.

**Single-model rules**, per machine ID. The bank is the ID's training-part clips; z-statistics
come from the bank, scored leave-one-out:

| rule | definition |
|---|---|
| S0 | R0, eval.py's score; written as `scores/<run>/eval_{dev,eval}.csv` in eval.py's layout |
| K | raw 1-NN cosine distance on `pre` — **AnoPatch's score** |
| zKp | z(K) |
| S1p | zR0′ + zK′ (S1′) |
| C1b | S1′ + z(1-NN on gmean) — our C1 with BEATs' token mean as the pooled map |

All five are also written per run to `scores/<run>/rules_{dev,eval}.csv`.

**Fusion.** Every member is scored by S1′, as in E2′:
- E2′[A8] = Σ over the 8 la_aug S1′ (the bar). P0: reproduces the committed 94.7692 dev /
  94.7764 eval.
- F1 = E2′[A8] + Σ_B S1′: plain E2′ over all 11 members.
- F2 = E2′[A8] + (8/3) Σ_B S1′: block-balanced, so the BEATs block weighs as much as A8.

## Pre-registration (written before any full run)

**Two stages:**
- **Stage A (dev only).**
  - `score.py dev` picks the BEATs single rule from {K, zKp, S1p, C1b}, by the 3-seed mean dev
    Amean.
  - It picks the fusion from {F1, F2}, by dev Amean.
  - Both picks go to `scores/selection.json`, which is committed **before** `score.py eval`
    runs.
- **Stage B (eval).** Scored once, with the stage-A picks.

**Primary metric.** Amean = mean of AUC and pAUC (%), averaged over the 6 machine types.

**Decision rules (stage B):**

- **D1 (single model):** the selected rule's 3-seed mean eval Amean beats our C1 8-seed eval
  Amean (94.09) by **≥ +0.50**, AND **≥ 2/3 seeds** are above 94.09.
  - If it passes: report "BEATs single model = best single model *with external pretraining*".
  - Either way, also report dev and the seed-paired C1 values (s2040–42).
  - D1 does not require dev to match: published BEATs systems are 3–4 points weaker on dev.
- **D2 (ensemble):** all four must hold for the chosen fusion vs E2′[A8]:
  - eval Δ Amean ≥ **+0.20**;
  - the paired clip-bootstrap 95% CI (2000 draws, stratified per machine × ID × label) is
    above 0;
  - dev Δ ≥ **−0.10**;
  - adding a single BEATs seed (same weighting) is positive on eval for **≥ 2/3** seeds.
  - If it passes: E2′[A8 + BEATs] becomes the best system *with external pretraining*. E2′[A8]
    stays the best from-scratch system.
- Whatever the outcome, nothing from this ablation changes the from-scratch rules (C1, E2′[A8]).

**Numeric predictions** (3-seed mean, selected rule; 80% intervals):

| | dev Amean | eval Amean |
|---|---|---|
| BEATs single, overall | **91.3** [89.5, 93.0] | **94.2** [93.0, 95.3] |
| fan | 87.5 [83, 92] | 95.5 [93, 97.5] |
| pump | 93.0 [90.5, 95.5] | 94.5 [92, 96.5] |
| slider | 99.2 [98, 99.8] | 99.6 [99, 99.9] |
| ToyCar | 96.3 [94, 98] | 95.5 [93, 97.5] |
| ToyConveyor | 75 [70, 82] | 84 [80, 87.5] |
| valve | 97.5 [94, 99.5] | 96 [93, 98] |
| fused (chosen F) − E2′[A8] | −0.2 [−0.8, +0.3] | **+0.4** [−0.2, +1.0] |
| fused, eval ToyCar Δ | | +1.0 [0, +2.5] |

The reasoning behind the predictions:
- The recipe is AnoPatch's. Its published 90.9 / 94.3 and PerMix's ArcFace baseline
  (90.5 / 93.7) bracket the expectation.
- Our S1′/C1b rules have added +0.4 to +0.7 over a plain score on the MFN. On a model whose
  raw kNN is already good, that should shrink, hence +0.4 dev and ±0 eval over AnoPatch.
- Fusion gains come from eval ToyCar, where BEATs is ~2 points better than us. Fan and pump
  should lose a little, because BEATs is far worse there on dev.
- P(D1 passes) ≈ 0.35; P(D2 passes) ≈ 0.35.

**Mechanism predictions** (each true/false is recorded in stage B):

| ID | prediction |
|---|---|
| P1 | BEATs dev fan Amean is ≥ 5 points below our C1 (96.5): pretrained features miss fan's anomalies |
| P2 | BEATs eval ToyCar Amean (selected rule) is above our C1's 93.8 |
| P3 | R0 is below K on dev Amean for BEATs: the classifier score is the weak one, as AnoPatch argues |
| P4 | validation accuracy is ≥ 0.99 by step 2000 |
| P5 | seed spread (max − min) of eval Amean is ≤ 1.5 |
| P6 | dev picks F1, not F2: the BEATs block is weaker on dev |

## Stage A note (2026-11-08, written after `score.py dev` and BEFORE `score.py eval`)

**The stage-A picks are unchanged: single rule `zKp`, fusion `F1`** (`scores/selection.json`).
The decisions D1 and D2 use exactly these picks.

Dev, 3-seed mean Amean: K 90.30, **zKp 90.80**, S1p 89.38, C1b 90.53, S0 (R0) 77.69.
Fusion vs E2′[A8] (94.77): F1 +0.16, F2 −0.02.

**A degeneracy the pre-registration did not anticipate.** The fine-tuned BEATs classifier
saturates on its training clips:
- In 26–30 of the 41 IDs per seed, the training-part CE sits at the fp32 floor (std ~1e-7).
  10–17% of training clips have CE exactly 0.
- z(R0′) divides by that std. So BEATs S1′ (and C1b, which contains it) is finite but has
  values up to 1e9 (`score.py` asserts finiteness, and that passed).
- The `RuntimeWarning: divide by zero` lines come from the hold-out z (`zR0`). It is computed
  but not used by any rule here.
- K and zKp do not use CE, so the selected single rule and D1 are clean.
- F1 adds BEATs S1′ to A8. On dev, the BEATs block is small for the median clip (1.7–59)
  but reaches 1e8–1e9 at its 99th percentile. So F1 is effectively A8 with the clips the
  BEATs classifier flags pushed to the top.
- The MFN members show the same effect, much milder: the A8 sum's median |score| is 11–900
  per machine. This is a property S1′ always had, made extreme here.

**Secondary, declared now (descriptive, never a decision): `F1k` = E2′[A8] + Σ_B zKp.** It
is the CE-free counterpart of F1, built from the rule stage A selected. Stage B reports it
with the D2 checks, under `secondary_F1k` in `decisions.json`. Disclosure: its dev number
is already seen, **95.01 (+0.24 vs A8)**, with one-seed deltas +0.22 / +0.17 / +0.16. It
was not chosen by looking at eval, and eval has not been scored.

## Smoke test and bench

**Bench** (`run.sh bench`, `scores/bench.csv`): GPU 2, idle, bf16 unless stated, 4 timed
optimizer steps of 256 clips each.

| config (micro × accum, checkpointing, attention) | clips/s | peak GiB | 10k steps |
|---|---|---|---|
| 32 × 8, ckpt, upstream attention | 23.5 | 4.4 | 30.2 h |
| 32 × 8, ckpt, upstream attention, fp16 | 23.0 | 4.4 | 30.9 h |
| 16 × 16, no ckpt, upstream attention | 29.8 | 9.4 | 23.8 h |
| 32 × 8, no ckpt, upstream attention | OOM | > 14.4 | — |
| 32 × 8, ckpt, fused attention | 33.0 | 4.1 | 21.6 h |
| 32 × 8, no ckpt, fused attention | 38.9 | 12.1 | 18.3 h |
| **16 × 16, no ckpt, fused attention (chosen)** | **38.8** | **6.9** | **18.3 h** |

- Fused attention gains 1.3× and is exact in fp32. bf16 vs fp16 makes no difference.
- The GPU fbank costs 28 ms per 32 clips (~2% of a step).
- GPU 1 could not be benched: it was busy for the whole preparation window. The smoke
  ran there at ~40 clips/s, measured over 200 steps including two validation passes, so its
  x1 PCIe link barely matters here. Each micro-batch moves only 10 MB.
- **Projected cost:**
  - ~18.3 h per run on GPUs 0/2, and 18–23 h on GPU 1 (the upper end allows for the 20–40%
    slowdown seen on MFN runs).
  - 3 runs in parallel ≈ **19–23 h wall, ~56–62 GPU-hours**.
  - Extraction: fp32, one run per pass measured at 17 min (train 11.5 + test 5.7 min).
    Three runs in one pass share the decoding: ~45 min.
  - Scoring is CPU only: dev ~1 min, eval ~3 min with the bootstrap.

**Smoke** (`run.sh smoke`; summary in `scores/smoke.txt`): GPU 1 while idle, seed 2040, 200
steps × 16 clips (0.09 epoch), warm-up 50.
- Training: valid loss 13.17 → 9.59, valid accuracy 0 → 0.078, 0 skipped steps, peak 6.55 GiB.
- Crash resume from `state_last.pth` at step 100 gives the same trajectory: valid loss
  9.58560 vs 9.58535, accuracy identical. It is not bit-exact; smartlab1 never is.
- Extraction: 17 min; CRC-verified dump; meta.npz equals lit-arms'.
- Both scoring stages ran end to end:
  - P0: E2′[A8] reproduced to 3e-5 on dev and eval.
  - Even this 0.09-epoch model gets dev kNN Amean 80.6, where R0 gets 63.7.
  - eval.py-layout CSVs: verified against a committed `eval.py --csv` file on an MFN dump
    (identical per-ID AUC/pAUC).
- Figures render from the smoke scores: checked by eye, x-labels and legends fixed.
- Nothing from the smoke is a result.

## Launch

```bash
cd /home/f74134118/OS-SCL-beats
setsid nohup env GPUS="0 2 1" bash ablation/beats/run.sh train > ablation/beats/results/launch.log 2>&1 &
# after all three finish (check_points/BEATsASD_bt_arc_s204{0,1,2}/done):
GPU=0 bash ablation/beats/run.sh extract
bash ablation/beats/run.sh dev        # commit scores/ (selection.json) BEFORE the next line
bash ablation/beats/run.sh eval
bash ablation/beats/run.sh curves     # config.json + curve.csv into scores/, plots
```

## Figures

Rendered by `make_plots.py` from `scores/` after stage B. None exist yet: there are no results.

| figure | shows |
|---|---|
| `plots/fig1_single_per_machine.png` | per-machine Amean, dev and eval: BEATs (selected rule, seeds as dots) vs our C1 single model (8 seeds) vs AnoPatch's published numbers |
| `plots/fig2_rules.png` | overall Amean of every BEATs scoring rule (R0, AnoPatch kNN, z kNN, S1′, C1b) per seed, dev and eval, with our C1 as a reference line |
| `plots/fig3_fusion.png` | per-machine Δ Amean of F1 and F2 vs E2′[A8], dev and eval |
| `plots/fig4_curves.png` | training and validation ArcFace loss and validation accuracy per optimizer step |

## Files

| file | role |
|---|---|
| `train_beats.py` | fine-tuning (also `--bench N`) |
| `run_one.sh` | one run with `../train_one.sh`'s GPU lock / idle check / crash retry |
| `extract.py` | dumps for the scorers |
| `score.py` | single rules, fusion, decisions |
| `make_plots.py` | figures |
| `run.sh` | everything, by stage |
| `../../model/beats/` | vendored BEATs (MIT) |
| `../../model/beats_asd.py` | wrapper |
