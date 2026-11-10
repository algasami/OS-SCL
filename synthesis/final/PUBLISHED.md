# Published DCASE 2020 Task 2 numbers, checked against the papers

Checked 2026-11-10 by a web-research agent that opened each source; this file records what it saw.
AUC / pAUC in %, pAUC at p = 0.1. **Amean = (AUC + pAUC) / 2**, averaged over the six machine types.

Status: **V** seen in the source · **S** quoted from another paper (named) · **C** arithmetic from V numbers.

## Read this before comparing

- **Our numbers pool machine IDs inside a type; the literature does not.** `eval.py` (inherited from
  the STgram-MFN / Noisy-ArcMix / OS-SCL code) computes one AUC per machine type over all its IDs.
  Every paper below that states its convention computes AUC per machine ID first and then averages
  (the official DCASE rule). Pooling also penalises score offsets between IDs, so the two are not
  the same number. The comparison with the OS-SCL paper is like for like only if the paper used its
  own released `eval.py`, which it does not say.
- **One run against a seed mean.** Only PerMix (3 seeds), AdaBEAM (5 trials) and ASDKit (4) report
  several runs. Every other row, the OS-SCL paper included, is one run with no stated checkpoint or
  seed selection rule. Our single-network rows are means over seeds.
- **The scoring rule differs by paper**: OS-SCL uses the ArcFace CE, PerMix a 1-NN cosine distance to
  the training embeddings, AdaBEAM a sub-band kNN.
- **Ensembles and externally pretrained models are separate classes.**
- None of these papers states how a checkpoint was chosen.

## From scratch (no external data)

| System | Source | Dev AUC / pAUC (Amean) | Eval AUC / pAUC (Amean) | Runs | Status |
|---|---|---|---|---|---|
| PerMix, TFSTgram-MFN, kNN score | Choi & Choi, arXiv 2610.31434 (ICASSP 2027 submission, unrefereed) | 95.89 / 90.98 (93.44) | 96.45 / 91.49 (93.97) | 3 seeds | V |
| ESTM (dual-branch Mamba) | arXiv 2509.02471, preprint | 95.76 / 90.49 (93.13) | — | not stated | V |
| **OS-SCL, TFSTgram-MFN** (the paper reproduced here) | arXiv 2509.13853 | 95.71 / 90.23 (92.97; paper prints 92.96), mAUC 91.23 | Amean 93.22 only; no eval AUC or pAUC is given | not stated | V |
| OS-SCL as rerun by PerMix (kNN score) | arXiv 2610.31434 Tab. 3 | 94.28 / 89.22 (91.75) | 95.58 / 90.73 (93.16) | 3 seeds | V |
| OS-SCL, Log-Mel only | arXiv 2509.13853 | 94.64 / 88.42 (91.53), mAUC 89.24 | Amean 93.27 | not stated | V |
| Noisy-ArcMix, TASTgram-MFN | ICASSP 2024, arXiv 2310.06364 | 94.65 / 89.31 (91.98), mAUC 89.78 | — | not stated | V |
| Noisy-ArcMix as rerun by PerMix | arXiv 2610.31434 | 94.10 / 88.49 (91.30) | 94.59 / 89.67 (92.13) | 3 seeds | V |
| MSN (multi-scale scanning network) | arXiv 2508.17194 | Amean 92.27 | Amean 92.21 | — | V |
| CLP-SCF (STgram-MFN, contrastive pretraining on the task data) | ICASSP 2023, arXiv 2304.03588 | 93.75 / 88.48 (91.12), mAUC 87.62 | — | — | V |
| ASD-AFPA | Interspeech 2023, arXiv 2308.14063 | 93.91 / 87.95 (90.93) | — | — | V |
| STgram-MFN (ArcFace) | ICASSP 2022, arXiv 2201.05510 | 92.36 / 86.34 (89.35), mAUC 84.86 | — | — | V |
| GeCo (trained on the dev training set only) | ICASSP 2023, arXiv 2305.12111 | 92.47 / 86.34 (89.41) | 96.67 / 90.61 (variant not specified) | — | V |
| SSL4ASD + 10-model ensemble (Wilkinghoff) | quoted in BEAM Tab. III | Amean 94.2 | Amean 93.3 | — | S |
| DCASE 2020 rank 1, Giri_Amazon_task2_2 (ensemble) | official results page | Amean 86.27 (S, AnoPatch) | 94.58 / 84.95 (89.77) | — | C |

The paper's per-machine dev AUC / pAUC: fan 98.41 / 94.50, pump 96.25 / 88.74, slider 99.63 / 98.04,
valve 99.85 / 99.23, ToyCar 96.55 / 91.17, ToyConveyor 83.56 / 69.87. Per-machine mAUC: fan 93.39,
pump 94.17, slider 98.61, valve 99.57, ToyCar 85.56, ToyConveyor 76.09. Its baseline rows are lower
than those baselines' own papers (STgram-MFN 91.73 / 85.86 against 92.36 / 86.34 published).

## With external pretraining

| System | Source | Dev AUC / pAUC (Amean) | Eval AUC / pAUC (Amean) | Runs | Status |
|---|---|---|---|---|---|
| AdaBEAM (BEATs-iter3 full fine-tune + sub-band kNN) | arXiv 2603.13749 v2, accepted at TASLP | Amean 95.6 ± 0.1 | Amean 96.2 ± 0.2 | 5 trials | V |
| … same paper: DMM / BEAM / global kNN (fine-tuned BEATs + LDN) | | 94.6 / 93.0 / 91.6 ± 0.2 | 95.2 / 94.0 / 93.0 ± 0.6 | 5 trials | V |
| PerMix, BEATs_iter3 fine-tuned | arXiv 2610.31434 | 96.56 / 92.41 (94.49) | 97.99 / 93.17 (95.58) | 3 seeds | V |
| BEATs + multi-branch LoRA + group adapter | Han et al., TASLP, arXiv 2508.12230 | Amean 91.08 | Amean 95.01 | not stated | V |
| AnoPatch (BEATs full fine-tune) | Interspeech 2024, arXiv 2406.11364 | Amean 90.93 | Amean 94.28 | not stated | V |
| OS-SCL on BEATs, rerun by PerMix | arXiv 2610.31434 | 94.63 / 88.92 (91.78) | 95.90 / 90.68 (93.29) | 3 seeds | V |
| ASDKit BEATs-LoRA | quoted in BEAM | Amean 90.6 ± 0.2 | Amean 93.5 ± 0.3 | 4 trials | S |

## What follows for this repository

- The paper's dev 95.71 / 90.23 was not reproduced by an independent group: PerMix's three-seed rerun
  of OS-SCL gives 94.28 / 89.22 (with its kNN score). Phase 1 here found the same direction: the
  plain reproduction sat below the paper on dev (`ablation/seed-sweep`).
- "Beats the paper" is a statement about one reported run. The seeded from-scratch reference is
  PerMix: Amean 93.44 dev / 93.97 eval (per-ID convention, 3 seeds).
- Not verified here: whether any paper pools IDs as `eval.py` does. Treat every cross-paper
  comparison of our pooled numbers as approximate until the per-ID numbers are reported beside them.
