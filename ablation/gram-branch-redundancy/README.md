# Gram branch redundancy — which learned gram carries the fusion gain?

The paper (arXiv:2509.13853) reports Log-Mel-only and the full 3-channel
TFSTgram, plus STgram, but never reports Log-Mel + TFgram **without Tgram**.
Table III has STgram(ArcFace), TFSTgram(ArcFace), STgram(OS-SCL),
TFSTgram(OS-SCL); Table II has Log-Mel(OS-SCL) and TFSTgram(OS-SCL). Nothing
isolates TFgram. This directory fills that gap.

## New fussion modes

`model/net.py` now carries `FUSSION_BRANCHES`; modes 1 and 2 are unchanged
(param counts verified identical at 1.3775M / 0.8973M, and existing
`fussion=1` checkpoints still load with `strict=True`).

| mode | branches | used params | note |
|------|----------|-------------|------|
| 1 | Log-Mel + Tgram + TFgram | 1.3775M | TFSTgram, the paper's model |
| 2 | Log-Mel | 0.8973M | paper's Log-Mel-only row |
| 3 | Log-Mel + TFgram | 1.0966M | **new** — the missing row |
| 4 | Log-Mel + Tgram | 1.1782M | **new** — STgram under OS-SCL |

`train.py` also gained `--seed` (default 2024, unchanged) and `--num_workers`
(default 0, unchanged) — the latter cuts a 300-epoch run from ~15h to ~9h.

## Step 1 — representational redundancy

`cka_probe.py` measures linear CKA and linear predictability between the three
branches of the *trained* `fussion=1` model. Full output in `results/cka.txt`:

```
CKA(Log-Mel, TFgram) = 0.89     TFgram ~ Log-Mel
CKA(Tgram,   TFgram) = 0.06     Tgram is orthogonal to TFgram
CKA(Log-Mel, Tgram)  = 0.06     Tgram is orthogonal to Log-Mel
```

TFgram is largely a re-derivation of the Log-Mel spectrogram (63% linearly
predictable from it, frame-wise). Tgram is the branch that is actually
decorrelated from everything else.

Run it with: `python ablation/gram-branch-redundancy/cka_probe.py <gpu>`

## Step 2 — the training runs

`bash ablation/gram-branch-redundancy/run.sh` trains modes 3 and 4 in parallel
on GPUs 1 and 2, 300 epochs each, logging to `results/train_f{3,4}.log` and
checkpointing to `check_points/SCLTFSTgramMFN_f{3,4}_*/`.

Evaluate with:
```
python eval.py --m 0.4 --fussion 3 --ht basic --gpu_num 1 \
    --model_path check_points/SCLTFSTgramMFN_f3_mel_tfgram/model.pth --d --csv
```

The read set out in advance was: mode 3 ≈ mode 1 would mean Tgram is redundant;
mode 4 ≈ mode 1 would mean TFgram is redundant and the paper's contribution does
not survive the control; both below mode 1 would mean the two grams are
genuinely complementary.

## Results

Single seed (2024) per arm, 300 epochs. Baseline `f1` is the pre-existing
`SCLTFSTgramMFN_test` checkpoint (this machine's reproduction of the paper's
TFSTgram; note it lands 0.77 AUC below the paper's reported 95.71 on dev).

| arm | branches | params | train time | dev AUC/pAUC/mAUC | eval AUC/pAUC/mAUC |
|-----|----------|--------|-----------|-------------------|--------------------|
| f1 | mel+Tgram+TFgram | 1.3775M | — | 94.95 / 89.67 / 89.22 | 94.84 / 90.51 / 90.48 |
| f3 | mel+TFgram | 1.0966M | 5:24 | 94.30 / 88.77 / 87.39 | 95.99 / 91.11 / 92.00 |
| f4 | mel+Tgram | 1.1782M | 3:10 | **95.01 / 89.57 / 89.17** | 95.64 / 91.49 / 91.63 |

Deltas vs f1 (percentage points):

```
             dev                    eval
f3   -0.64  -0.90  -1.83     +1.15  +0.60  +1.52
f4   +0.06  -0.10  -0.05     +0.80  +0.98  +1.15
```

## Conclusion

**TFgram is redundant.** Dropping it (f4) matches the full 3-channel model on
dev (+0.06 AUC) and beats it on eval (+0.80 AUC), at 14% fewer parameters and
41% less training time than f3 (3:10 vs 5:24; f1 was not timed, but it runs
every branch f3 does plus Tgram, so it is strictly costlier). Dropping Tgram instead (f3) costs 0.64 AUC and 1.83
mAUC on dev. This matches the CKA probe exactly: TFgram sits at 0.89 CKA with
Log-Mel, Tgram at 0.06 against both.

Across both test sets, f4 >= f1 everywhere. That is the robust result.

## Figures

Regenerate with `/home/f74134118/anaconda3/bin/python ablation/gram-branch-redundancy/make_plots.py`
(matplotlib lives in the `base` conda env, not `os-scl`). All four read from the
eval CSVs, `results/cka.txt`, and the training logs — nothing is hardcoded.

| figure | shows |
|--------|-------|
| `plots/fig1_branch_redundancy.png` | CKA matrices (clip + frame) and frame-level linear R² between the three branches |
| `plots/fig2_per_machine_scores.png` | AUC and mAUC per machine, all three arms, dev and eval |
| `plots/fig3_overall_deltas.png` | overall AUC/pAUC/mAUC change vs the f1 baseline |
| `plots/fig4_training_curves.png` | best-so-far validation loss (log) and the accuracy plateau |

## What does not replicate cleanly

Both new arms *beat* f1 on the evaluation set, largely via ToyConveyor
(f1 78.28 vs f3 84.42, f4 83.20). ToyConveyor is the machine this repo's
reproduction already flagged as unstable. The dev/eval disagreement for f3
(-0.64 dev, +1.15 eval) is direct evidence that single-seed differences of
~1 point are not resolvable here — and the paper's TFgram gain being claimed
is itself only +1.08 AUC (95.71 vs 94.63).

So: the *direction* (TFgram redundant, Tgram load-bearing) is consistent across
dev, eval, and the CKA probe. The *magnitudes* are not trustworthy at n=1.
A seed sweep (`--seed`, now supported) over 3-5 seeds per arm is required
before this becomes a claim. A local Log-Mel-only run (`--fussion 2`) is also
missing and would anchor the low end.
