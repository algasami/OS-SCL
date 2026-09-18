# Seed sweep — how much of the f1 / f4 / frozen spread is seed noise?

Both earlier ablations end on the same caveat: every arm is n=1, and
[`../frozen-tfgram`](../frozen-tfgram) shows a 4.4–6.9 point ToyConveyor spread
across five runs, which is direct evidence that ~1-point differences are not
resolvable at a single seed. This directory runs the sweep those READMEs asked
for, and the answer is that the caveat was right: **at n=4, no TFgram variant is
distinguishable from the full model, and once ToyConveyor is excluded every
difference collapses to within 0.09 AUC.**

## Design

4 arms x 4 seeds (2024–2027) = 16 runs, 300 epochs each. 37 h wall clock on two
RTX PRO 6000 cards.

| arm | flags | what it tests |
|-----|-------|---------------|
| `f1` | `--fussion 1` | TFSTgram, the paper's model — the baseline |
| `f4` | `--fussion 4` | TFgram **removed** |
| `ft_strict` | `--fussion 1 --freeze_tfgram strict` | TFgram frozen, BN pinned to `eval()` |
| `ft_bnadapt` | `--fussion 1 --freeze_tfgram bn_adapt` | TFgram frozen, BN tracks data |

Everything else at repo defaults (`--m 0.4 --ht basic`, batch 64,
`--num_workers 8`).

## Results

### Overall, mean +/- sd over 4 seeds

| arm | dev AUC / pAUC / mAUC | eval AUC / pAUC / mAUC |
|-----|------------------------|------------------------|
| `f1` | 94.90 +/- 0.14 / 89.76 +/- 0.25 / 88.66 +/- 0.46 | 95.53 +/- 0.23 / 91.11 +/- 0.54 / 91.51 +/- 0.56 |
| `f4` | 95.10 +/- 0.28 / 89.67 +/- 0.07 / 89.95 +/- 1.09 | 95.91 +/- 0.29 / 91.39 +/- 0.24 / 91.89 +/- 0.32 |
| `ft_strict` | 95.08 +/- 0.27 / 89.65 +/- 0.14 / 89.58 +/- 0.73 | 96.01 +/- 0.23 / 91.32 +/- 0.57 / 92.00 +/- 0.19 |
| `ft_bnadapt` | 94.85 +/- 0.10 / 89.63 +/- 0.23 / 88.84 +/- 0.33 | 95.69 +/- 0.12 / 91.06 +/- 0.11 / 92.12 +/- 0.36 |

Per-seed overall AUC, which shows how little of the ordering survives:

```
DEV   arm              2024     2025     2026     2027     mean     sd
      f1              94.70    95.02    94.93    94.95    94.90   0.14
      f4              95.01    95.50    95.03    94.87    95.10   0.27
      ft_strict       95.36    94.72    95.05    95.18    95.08   0.27
      ft_bnadapt      94.92    94.91    94.88    94.70    94.85   0.10

EVAL  arm              2024     2025     2026     2027     mean     sd
      f1              95.85    95.36    95.37    95.53    95.53   0.23
      f4              95.64    96.02    96.26    95.73    95.91   0.29
      ft_strict       95.98    96.02    95.73    96.29    96.01   0.23
      ft_bnadapt      95.59    95.59    95.83    95.75    95.69   0.12
```

On dev, `ft_strict` is best at 2024 and worst at 2025. On eval, `f1` is best at
2024 and worst at 2025. Rank is not stable across seeds.

### Nothing separates from the baseline

Paired t-tests on overall AUC (arms share each seed's train/valid split):

```
             dev                          eval
f4           +0.20  p=0.189               +0.38  p=0.216
ft_strict    +0.18  p=0.436               +0.48  p=0.045 *
ft_bnadapt   -0.05  p=0.661               +0.16  p=0.370
```

`*` nominal only. With 6 tests the Bonferroni threshold is p<0.0083, so **no
comparison is significant.**

### Excluding ToyConveyor, everything collapses

Paired vs `f1`, overall AUC recomputed over the other 5 machines:

```
             dev                          eval
f4           +0.02  p=0.774               -0.07  p=0.599
ft_strict    -0.09  p=0.714               +0.02  p=0.881
ft_bnadapt   -0.07  p=0.663               -0.08  p=0.593
```

Every delta is within **0.09 AUC** and every p-value above 0.59. This is the
central result, and it confirms at n=4 what `../frozen-tfgram` argued from a
single seed: outside ToyConveyor, trained / frozen / deleted TFgram are
indistinguishable.

### Per-machine AUC, mean over 4 seeds

```
DEV   arm             fan    pump  slider  ToyCar  ToyConv   valve
      f1            98.23   94.55   99.62   95.84    81.51   99.65
      f4            97.65   94.94   99.62   95.92    82.65   99.84
      ft_strict     97.85   94.28   99.60   95.89    83.01   99.84
      ft_bnadapt    98.01   94.82   99.62   95.23    81.59   99.84
      (f1 sd)        0.28    0.36    0.04    0.20     0.58    0.47

EVAL  arm             fan    pump  slider  ToyCar  ToyConv   valve
      f1            99.57   98.19   99.15   95.52    81.66   99.08
      f4            99.13   98.23   99.44   95.36    84.33   98.98
      ft_strict     99.37   98.23   99.39   95.49    84.40   99.15
      ft_bnadapt    99.16   97.95   99.22   95.52    83.00   99.28
      (f1 sd)        0.19    0.21    0.24    0.64     0.92    0.19
```

## How big is run-to-run noise?

Large enough to explain every effect either earlier README reported. Five draws
of the *identical* `f1` configuration (4 seeds plus the pre-existing off-stream
run) span **1.01 points of eval AUC** (94.84 to 95.85) and 1.73 of eval mAUC.
For comparison: `f4`'s previously reported +0.80 eval AUC, `ft_strict`'s +1.14,
and the paper's own claimed TFgram gain of +1.08 (95.71 vs 94.63) all fit inside
that interval.

`mAUC` is the noisiest metric, which is structural: it is the mean over machines
of the *minimum* per-ID AUC, so it is an order statistic. `f4`'s dev mAUC sd is
1.09 and one seed pair moved it 2.29 points. Treat mAUC deltas under ~2 points
as unresolvable.

### The old f1 baseline was off-stream

`check_points/SCLTFSTgramMFN_test` — the `f1` row in both earlier tables —
predates `--num_workers` and ran with 0, while every ablation arm ran with 8. At
a fixed seed that is not the same run:

```
split identical : True     (drawn before the DataLoader exists)
init  identical : True     (models built after the loader, but
                            DataLoader.__init__ draws nothing from the RNG)
epoch 0 order   : same
epoch 1+ order  : diverges
torch stream    : diverges  -> mixup pairings differ
numpy stream    : same      -> mixup lambda sequence is shared
```

`persistent_workers=nw>0` makes the multiprocessing iterator draw its base seed
once instead of once per epoch. That single draw desynchronises the global torch
stream from the first epoch boundary. Rerunning seed 2024 here put all four arms
on one stream; the rerun `f1` scored 95.85 eval AUC against the old run's 94.84,
with ToyConveyor at 82.29 vs 78.28. **The clean sweep of eval-side "gains" in
both earlier READMEs was that one unlucky baseline draw.**

### Determinism was verified

Three of the four seed-2024 cells duplicate pre-existing checkpoints, and all
three reproduced them to 5 decimals on every metric:

| cell | dev AUC/pAUC/mAUC | matches |
|------|-------------------|---------|
| `f4_s2024` | 0.95006 / 0.89573 / 0.89173 | `f4_mel_tgram` exactly |
| `ft_strict_s2024` | 0.95356 / 0.89622 / 0.90610 | `ft_strict` exactly |
| `ft_bnadapt_s2024` | 0.94925 / 0.89534 / 0.89172 | `ft_bnadapt` exactly |

So training is bit-deterministic at a fixed seed here (`cudnn.deterministic`),
the `--freeze_tfgram` plumbing does not perturb the default path over 300 epochs
of training (not just at checkpoint load), and this harness reproduces the
original ablation scripts.

## What this changes in the earlier READMEs

- `../gram-branch-redundancy`: "**TFgram is redundant**" **stands** — but the
  supporting numbers do not. "beats it on eval (+0.80 AUC)" should read *matches
  it*; `f4`'s eval advantage over a correctly-streamed baseline is +0.38,
  p=0.216, and −0.07 excluding ToyConveyor. "Across both test sets, f4 >= f1
  everywhere" is an artifact of the off-stream baseline.
- `../frozen-tfgram`: "**Training TFgram buys nothing**" **stands**, and its
  chosen phrasing — "freezing TFgram costs nothing," not "freezing helps" — was
  exactly right. Its ToyConveyor caveat is confirmed: it correctly predicted the
  collapse this sweep measured. `ft_strict`'s bolded dev 95.36 was its best of
  four draws (mean 95.08 +/- 0.27); the +1.39 dev mAUC is +0.92 +/- 0.73 at n=4.
- Neither directional claim is affected. `f3` (removing *Tgram*) remains the only
  arm that moved outside the noise band, at −0.74 dev AUC excluding ToyConveyor.

## Open: a ToyConveyor-specific effect

`f1` ranks **last of the four arms on eval ToyConveyor in all four seeds**
(81.66 mean vs 84.33 `f4`, 84.40 `ft_strict`, 83.00 `ft_bnadapt`), and paired
`ft_strict` vs `f1` there is +2.75, p=0.009.

This is *not* claimed as a result. ToyConveyor was selected because it looked
anomalous, so the p-value is inflated by that choice, and with 36 machine-level
comparisons the Bonferroni threshold is p<0.0014. What makes it worth recording
is the rank consistency (4/4, independent of the effect size) and the direction:
it points the opposite way from the paper, i.e. toward a trained TFgram *hurting*
the one machine it affects. Testing it properly needs a pre-registered
ToyConveyor-only comparison on fresh seeds.

## Figures

Regenerate with `/home/f74134118/anaconda3/bin/python ablation/seed-sweep/make_plots.py`
(matplotlib lives in the `base` conda env, not `os-scl`). All four read from the
16 sweep eval CSVs plus the four pre-existing single-seed checkpoints — nothing
is hardcoded, including the n=1 deltas shown for reference in fig2, which are
recomputed from the old checkpoints rather than copied from the READMEs.

`f1` is drawn achromatic because it is the *baseline*, not a fourth category.
That keeps the categorical set at the three slots that validate under
`--pairs all` (CVD dE 9.2, normal-vision dE 24.0); a fourth slot would put
yellow beside orange and fail the floors. Aqua (`f4`) sits at 2.74:1 on the
light surface, so the relief rule applies and every aqua mark is direct-labelled.

| figure | shows |
|--------|-------|
| `plots/fig1_seed_spread.png` | every seed as a dot, per arm — the band f1 alone covers is wider than the gaps between arms |
| `plots/fig2_effect_vs_noise.png` | paired delta vs f1 with 95% intervals, against the noise band, with the published n=1 values as hollow marks |
| `plots/fig3_toyconveyor_collapse.png` | the decisive one: dropping ToyConveyor pulls every arm onto zero |
| `plots/fig4_per_machine_variance.png` | where the variance lives — ToyConveyor swings >4 points between seeds |

## Running it

```bash
bash ablation/seed-sweep/run.sh                  # 2 slots, GPUs 0 and 1
SWEEP_SLOTS=3 SWEEP_GPUS="0 1 2" bash ablation/seed-sweep/run.sh
python ablation/seed-sweep/collect.py            # -> results/summary.csv, per_machine.csv
```

Resumable: a cell whose two eval CSVs exist is skipped, so the runner can be
relaunched after an interruption.

Results land in `scores/<model_name>/` (`eval_{dev,eval}.csv`, `config.json`,
`curve.csv`), which is the committed record. `check_points/` holds only the
weights and is gitignored — see the Ablations section of `CLAUDE.md`. After a
run, distil the curves: `python ablation/distill_curves.py seed-sweep`.

### One run per GPU, measured

These runs are **GPU-bound**, which is not obvious — `train_dataset.__getitem__`
decodes the wav and computes the mel per item, so the natural guess is
dataloader-bound. Measured on 2 free cards:

| concurrency | per-run | aggregate | GPU util |
|-------------|---------|-----------|----------|
| 1 per GPU | 13.2 it/s | 26.3 it/s | 92–99% |
| 3 per GPU | 3.1 it/s | 18.8 it/s | 99–100% |

Stacking runs on a card *lowers* aggregate throughput. CPU sat at 72% idle with
0% iowait throughout (the 15 GB dataset is fully page-cached) and
`OMP_NUM_THREADS=1` changed nothing. So `run_one.sh` claims a whole card via a
lock directory rather than packing jobs.

## What the seed controls

1. **The train/valid split** (`utils.py:17`, `random_split` with no generator).
   `trainer.py` saves at each new minimum validation loss, so the seed decides
   which 5% defines "best epoch" — it changes *model selection*, not just
   training dynamics.
2. **Weight init.** For the frozen arms this *is* the experiment: they never
   train TFgram, so at n=1 the claim rested on a single random projection. Four
   draws now agree.
3. **Batch composition.** `SupConLoss` builds positives and negatives within the
   batch, so shuffle order changes what the contrastive term compares.
4. **Mixup**: `np.random.beta` per step plus `torch.randperm` for the pairing.

It does not sample GPU nondeterminism — `random_seed()` also sets
`cudnn.deterministic=True, benchmark=False`, as the reproductions above confirm.

## Still missing

A local `--fussion 2` (Log-Mel-only) sweep to anchor the low end — the only item
from the earlier READMEs' priority list still outstanding.
