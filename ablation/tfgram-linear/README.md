# TFgram linear — does a waveform branch that isn't a spectrogram help?

Follow-up to [`../tfgram-redesign`](../tfgram-redesign). That ablation found
that every TFgram with a rectifier anywhere in its downsampling stack, including
the paper's `ReLU → max_pool` and the 2×2's `LeakyReLU(0.2)`/avg-pool cells, computes
an energy envelope, i.e. a spectrogram (frame CKA 0.73–0.75 with Log-Mel at
random init). Exactly one variant escapes: **identity activations through the
stack plus average pooling** (`linear_avg`, frame CKA 0.000 at init). At init it
is also not a second Tgram (frame R² 0.01 against Tgram's front end, where two
random Tgram front ends score 0.86), and its output is not near-DC (95% of its
variance survives a 31 Hz high-pass).

Everything in `../tfgram-redesign` was measured at random init. This ablation
trains `linear_avg` and asks what that ablation couldn't.

> **Status: stage 1 done, gate passed; stage 2 running** (launched
> 2026-10-10, ~3 days). Predictions and gate below were written before any
> training and are left unedited; outcomes are under *Results*.

## Questions

1. **Does training make `linear_avg` a copy of Tgram?** Both branches start
   with a linear filterbank on the raw waveform (`linear_avg`'s stack, Tgram's
   `conv_extrctor`), and nothing structural stops training from converging the
   two. This is the open mechanism question.
2. **Does it help detection?** The paper's TFgram doesn't: dropping it
   entirely (f4) matches the full model at n=4 (`../seed-sweep`). A decorrelated
   branch only helps if it carries anomaly information that Log-Mel and Tgram
   miss.

**What is *not* a question: whether training pulls `linear_avg` back toward
Log-Mel.** It can't. With identity activations and average pooling, every
operation in the stack (conv, BatchNorm in eval mode, avg pool) is affine, so
the branch output is `A·x + b` for *any* weights: an odd function of the
waveform up to a constant. Log-Mel is even (`mel(−x) = mel(x)`). On a
sign-symmetric input distribution their cross-covariance is zero, whatever the
weights. Frame CKA with Log-Mel is measured anyway, as a check on that
argument (a non-zero value would mean the data isn't sign-symmetric, e.g. DC
offsets, not that training did anything).

## Design

Two arms, seed 2024, otherwise f1: `--fussion 1 --m 0.4 --ht basic`, batch 64,
`--num_workers 8`, 300 epochs, smartlab1 (RTX A4000, `environment-cu124.yml`,
torch 2.6), one arm per GPU on GPUs 1 and 2.

| arm | `--tf_act` | `--tf_pool` | role |
|-----|------------|-------------|------|
| `relu_max` | `relu` | `max` | the paper's TFgram = f1, retrained on this host as the paired baseline |
| `linear_avg` | `linear` | `avg` | identity activations at all 4 sites, all 3 pools average |

`--tf_act linear` is new (`model/waveq.py`). Like the 2×2's switches it adds no
parameters, so both arms draw the **same weight init, the same train/valid split
and the same RNG stream** (verified, see *Verification*). The only difference
between the runs is TFgram's activation and pooling.

**Why `relu_max` is retrained.** The seed-sweep's f1 runs were made on another
host under torch 2.8. Bit-determinism is only verified within one machine and
stack, so they are not paired baselines here. They stay useful as an unpaired
cross-host check.

**Student snapshots.** `../tfgram-redesign`'s pilot failed because
`trainer.py` saves only the EMA net (alpha 0.9999), which is still ~55% init at
epoch 11. Both arms now also save the **student** (`net`) at epochs 10, 30, 90
and 300 (`--student_snapshots`, new in `trainer.py`). The EMA at its best
validation loss (`model.pth`) remains the deliverable and is what `eval.py`
scores.

## Measurements

`branch_probe.py`, the same 656 training clips (16 per machine ID) as
`../tfgram-redesign`, **frame level only** (205k frames × 128 dims). Clip-level
CKA is biased toward 1 at N ≪ D (`../tfgram-redesign`, *A correction to the
metric*) and is not reported. For each arm, at the replayed init, each student
snapshot and the final EMA:

- frame CKA of TFgram with Log-Mel, Tgram and Tgram's front end
  (`conv_extrctor` alone);
- frame linear R² in both directions between TFgram and Tgram's front end,
  and Log-Mel → TFgram.

The reference for "same information, different filterbank" is the
`../tfgram-redesign` control: two independently seeded Tgram front ends
predict each other at frame R² **0.86**.

Plus `eval.py` dev and eval AUC/pAUC/mAUC for each arm's EMA checkpoint.

## Predictions

Written before any run.

**P1 — structural check.** `linear_avg`'s frame CKA with Log-Mel is ≤ 0.02 at
init, at every student snapshot and at the EMA. `relu_max` stays ≥ 0.70
throughout; old-host trained f1 is 0.81.

**P2 — not a Tgram copy (the mechanism question).** `linear_avg`'s frame R²
from Tgram's front end stays **≤ 0.20** in the trained EMA, far from the 0.86
control. Moderate confidence. Each frame of `linear_avg` is a 511-sample
box-car average of a learned filterbank output, so it aliases its band into the
31 Hz frame rate; Tgram's front end is a 1024-tap windowed projection at the
same rate. Training has no loss term that rewards the two agreeing, and Tgram
already supplies its own view, so there is no pressure to duplicate it. If R²
rises toward the control, the branch has swapped which neighbour it
duplicates, and the next design is option C (fold sub-frame samples into
channels instead of pooling).

**P3 — performance: null.** At n=1 the eval AUC difference is within run-to-run
noise (|Δ| < 1 point). If stage 2 runs, the paired n=4 delta is not
significant. The prediction rests on f4 ≈ f1: nothing so far shows anomaly
information outside Log-Mel + Tgram.

**P4 — it trains.** On real clips at init, in train mode (BatchNorm on batch
statistics), `linear_avg`'s output std is 0.65 against `relu_max`'s 0.81, with
mean 0.00 against 0.43. It is concatenated before MobileFaceNet's first
conv+BN, which can absorb a scale difference that small. Expect validation loss and accuracy curves
within the seed-to-seed spread of `relu_max`.

A **P2-holds, P3-null** outcome is the expected one, and still a result: the
waveform branch *can* be made to carry information that neither Log-Mel nor
Tgram carries, and that information doesn't help detect anomalies. That goes
further than "TFgram is redundant".

## Gate: stage 1 → stage 2

**Stage 1**: both arms × seed 2024 × 300 epochs, probe, eval.

**Stage 2**: both arms × seeds 2025–2027 × 300 epochs (6 runs, ~3 days on two
A4000s), for an n=4 paired AUC comparison.

Run stage 2 only if **both** hold at stage 1 (fixed in advance):

- **G1, not a Tgram copy:** `linear_avg`'s trained-EMA frame R²(Tgram front →
  TFgram) < **0.50**. Above that it's mostly a second Tgram filterbank and the
  AUC question is moot (it would be asking whether two Tgrams beat one).
- **G2, not broken:** `linear_avg`'s eval AUC ≥ `relu_max`'s − **1.0** and
  its dev AUC ≥ `relu_max`'s − **1.0**. A larger single-seed loss means the arm
  is likely broken (scale, optimisation), and three more seeds would only
  measure a broken arm more precisely.

Both thresholds are judgement calls, not derived. 0.50 is a bit over half the
control's 0.86, so it sits between "independent" (0.01 at init) and "same
information" (0.86). 1.0 AUC is the run-to-run noise on this reproduction
(CLAUDE.md).

**Stage 2 decision rule (if it runs).** `linear_avg` counts as a gain only if
its paired delta vs `relu_max` is positive on dev *and* eval, survives
excluding ToyConveyor, and passes p < 0.025 (paired t, Bonferroni over the 2
sets). mAUC deltas under ~2 points are not read.

## Running it

```bash
bash ablation/tfgram-linear/run.sh stage1      # 2 arms x seed 2024 x 300 ep, then probe
bash ablation/tfgram-linear/run.sh stage2      # only if the gate passes; seeds 2025-2027
/home/f74134118/anaconda3/bin/python ablation/tfgram-linear/make_plots.py
```

`TFL_GPUS` overrides the card list (default `"1 2"`). Resumable: an arm whose
two eval CSVs exist is skipped. Each arm writes
`scores/<model_name>/{eval_dev,eval_eval,config}.*`, `run.sh` distils
`curve.csv` and writes `scores/branch_probe_<stage>.json`. At ~0.075 h/epoch on
an A4000 (`../tfgram-redesign/scores/pilot_timing.csv`), each 300-epoch arm takes
~22 h.

## Verification

Before any training, on smartlab1 (torch 2.6.0+cu124, RTX A4000):

```
full-model init identical (relu/max vs linear/avg, seed 2024):  True
linear == ../tfgram-redesign probe's slope-1 patch, bit-identical: True
odd up to a constant: max|TF(x) + TF(-x) - 2b| / std(TF) = 0.0   (eval mode)
```

So the trained `linear_avg` is exactly the variant `../tfgram-redesign`
probed at init, both arms start from the same weights, and the branch is odd
in the waveform up to a constant, which is the premise of P1. A 1-epoch smoke
run (`--student_snapshots 1`) saved `student_e1.pth` next to `model.pth`, and
`eval.py` refused it without `--tf_act linear --tf_pool avg` and scored it
with them.

## Results

### Stage 1 (seed 2024, n=1)

Both arms trained 300 epochs on smartlab1 (`relu_max` 21.3 h, `linear_avg`
22.2 h). `scores/branch_probe_stage1.json`, frame level:

```
                       init   e10    e30    e90    e300   EMA
CKA(Log-Mel, TFgram)
  relu_max             0.736  0.667  0.646  0.731  0.796  0.792
  linear_avg           0.000  0.000  0.000  0.001  0.008  0.007
R²(Tgram front -> TFgram)                         control (2 random fronts): 0.86
  relu_max             0.007  0.007  0.007  0.007  0.006  0.005
  linear_avg           0.013  0.012  0.014  0.028  0.028  0.029
```

Every other cross-branch number for `linear_avg` stays ≤ 0.011: R²(TFgram →
Tgram front), CKA with Tgram, and R²(Log-Mel → TFgram) at the EMA.

```
EMA checkpoint, AUC (%)     dev     eval    eval mAUC
relu_max                   94.97   95.20   90.51
linear_avg                 95.12   95.45   91.87
Δ (linear_avg − relu_max)  +0.15   +0.24   +1.36
```

Cross-host check: the seed-sweep's f1 at seed 2024 (other host, torch 2.8)
scored 94.70 dev / 95.85 eval, so `relu_max` here is 0.65 lower on eval.
That is within run-to-run noise but not paired.

### Against the predictions

- **P1: holds for `linear_avg`, misses slightly for `relu_max`.**
  `linear_avg` stays ≤ 0.008 against Log-Mel at every state. That confirms the
  odd-vs-even argument on real data: the input distribution is sign-symmetric
  enough. `relu_max` dips *below* the predicted 0.70 floor early in training
  (0.65 at e30) before rising to 0.80. Early training moves the paper's branch
  slightly away from the spectrogram, and later training moves it back past its
  init value. The old host's trained f1 (0.81) is the same endpoint.
- **P2: holds.** `linear_avg` gains a little Tgram-front predictability
  between e30 and e90 (0.014 → 0.028) and then stops. 0.029 is a thirtieth of
  the control and well under the 0.20 prediction. Training does not turn it
  into a second Tgram.
- **P3: holds at n=1.** Both overall deltas are under 1 AUC. Per machine
  (fig2), every dev and eval delta is inside ±1 except ToyConveyor eval
  (+1.7), whose run-to-run noise is > 4 points.
- **P4: holds.** The validation curves overlap (fig3). `linear_avg`'s
  best validation loss is slightly lower between epochs 150 and 200, and both
  end at 98.2% validation accuracy.

A side observation in fig3: both EMA nets sit at 0% validation accuracy until
epochs 106–108. This is the alpha-0.9999 horizon that voided
`../tfgram-redesign`'s 30-epoch pilot, now seen directly.

### Gate

- **G1 (not a Tgram copy): pass.** R²(Tgram front → TFgram) = 0.029 < 0.50.
- **G2 (not broken): pass.** `linear_avg` is ahead on both dev (+0.15) and
  eval (+0.24).

Stage 2 (seeds 2025–2027, both arms) was launched on 2026-10-10.

**What stage 1 already shows, independent of stage 2:** a waveform branch can
be built that carries information *neither* Log-Mel nor Tgram carries
(frame R² ≤ 0.03 from either), and it stays that way through 300 epochs of
training. Whether that information helps detection is what stage 2's n=4
paired comparison is for.

## Figures

Regenerate with `/home/f74134118/anaconda3/bin/python ablation/tfgram-linear/make_plots.py`
(matplotlib lives in `base`). Reads `scores/` only and picks up every seed
present, so the same script redraws them once stage 2 lands. `relu_max` is the
paper's branch and the baseline, so it is drawn achromatic.

| figure | shows |
|--------|-------|
| `plots/fig1_branch_trajectory.png` | frame CKA with Log-Mel and frame R² from Tgram's front end, at init, each student snapshot and the EMA: `linear_avg` stays near 0 on both, far below the G1 line and the two-random-fronts control |
| `plots/fig2_paired_auc.png` | per-machine AUC difference `linear_avg − relu_max`, dev and eval, against the ±1-point noise band |
| `plots/fig3_valid_curves.png` | EMA validation loss and accuracy (best-so-far rows): the two arms train alike |
