# Distill — can one OS-SCL network learn what the 4-seed ensemble knows?

> **Status: pre-registered 2026-10-13 (a521f92). Nothing trained, nothing scored.**
> The only measurements so far are on training clips, recorded under *What has
> already been seen*.
>
> **Resume:**
> 1. Implement `--kd_teachers/--kd_beta/--kd_tau` in `train.py`/`trainer.py`. They are
>    off by default, so the main behaviour is unchanged.
> 2. Run gate G1 (10 epochs, β = 0, teachers on, seed 2024; it must bit-match
>    `tfl_relu_max_s2024/student_e10.pth`) and G2 (timing) on the first free GPU.
>    GPU 0 frees up after the memory-bank run; GPUs 1–2 after confirm-seeds.
> 3. Then the 4 students, one per GPU. This worktree has no `data/`; symlink it
>    as `../OS-SCL-confirm-seeds` does (`ln -s /home/f74134118/OS-SCL/data data`).

## Why this exists

`../ensemble-backend` (branch `ablation/ensemble-backend`) found that summing
the per-ID z-scored R0 of the four `tfl_relu_max` seeds (E1[relu4]) beats every
member. The gain over the mean member is +0.96 dev and +0.54 eval AUC, and
most of it is ToyConveyor (+4.48 dev), the machine with the largest
seed-to-seed swings. It exceeds the paper's dev AUC, dev pAUC and eval number
by point estimate. But it is four networks, and the paper's table compares
single networks.

Each seed has its own quirks. Averaging cancels them and keeps what the seeds
agree on. This ablation asks whether **one network of the paper's size** can be
trained to carry that consensus, by learning from the other seeds' outputs
(knowledge distillation; Hinton et al. 2015).

**Why not a weight average ("model soup").** Seeds trained from different
random inits cannot be weight-averaged: their units are permuted relative to
each other. The workable soup shares a trunk and branches late, so all
branches share the trunk's init, and that init is where much of the seed
variance lives. The model also already keeps an EMA of its own weights. Weight
averaging is therefore not tested here.

## What has already been seen

All of it was measured on **training clips only**. No test clip was used to
design this.

1. **On clean training clips the teachers are saturated and agree.** These are
   the 4 `relu_max` teachers' margin-free ArcFace distributions, softmax(30·cos/τ)
   averaged over the teachers, on all 36,283 training clips. They come from the
   cached pre-head embeddings of `../ensemble-backend`, and its cached margin CE
   is reproduced to ≤ 3.5e-5.

   ```
   tau   own-ID prob, median (p10)   entropy, median (max ln 41 = 3.71)
   1     0.993 (0.991)               0.07 nats
   2     0.653 (0.632)               1.92
   4     0.179 (0.172)               3.50
   8     0.069 (0.067)               3.69
   ```

   The own-ID cosine is 0.9956 on average, with a median per-clip sd across the
   4 teachers of 0.0008. On the data they were trained on, the seeds agree
   almost perfectly. That is no surprise: they fit it. So soft labels
   precomputed on clean training clips are nearly one-hot at τ = 1, and nearly
   identical for every clip at τ ≥ 4. They would carry class geometry, not the
   per-clip consensus that makes the ensemble work.

2. **On mixup'd training clips they disagree more.** On 1,024 random training
   clips with mixup as `utils.mixup_data` does it (λ ~ Beta(0.5, 0.5)), at τ = 1:

   ```
            mean KL(teacher || ensemble mean)   all 4 teachers' argmax agree
   clean    0.0002                              100.0%
   mixed    0.0020                              98.6%
   ```

   Mixing gives 10× the disagreement. Inputs off the training manifold are
   where the seeds' functions differ, and that is where a student can learn
   their consensus. This matches "function matching" (Beyer et al., CVPR 2022):
   the teacher scores the *same* augmented input the student sees, computed
   online.

3. **Cost.** Four teachers' forward pass took about 6 ms per clip in fp32, but
   GPU 0 was shared at the time, so this is an upper bound. A baseline training
   step costs about 7 ms per clip (2.27 it/s at batch 64). So online teachers
   add up to about 1×. Three teachers under fp16 autocast should add much less.
   The gate (below) measures it.

## Design

### Teachers: leave-one-seed-out

The student at seed *s* is distilled from the **three `tfl_relu_max` seeds
other than *s*** (EMA weights, eval mode, frozen, fp16 autocast). Not all
four, for two reasons:

- The fourth would be the student's own twin: the same init, split, batch
  order and mixup draws. Distilling from it is self-distillation, which is
  known to help on its own (Furlanello et al. 2018, "born-again networks").
  That would confound "learns the cross-seed consensus" with "is regularised by
  a copy of itself".
- One teacher fewer is cheaper.

### Students: paired with the baseline at the same seed

Students use seeds 2024–2027 with **exactly** `../tfgram-linear`'s `relu_max`
recipe: `--m 0.4 --fussion 1 --ht basic --num_workers 8 --tf_act relu
--tf_pool max`, 300 epochs, same snapshots.

The teachers' forward consumes no random numbers. So a student shares with
`tfl_relu_max_s<seed>`:

- its initialisation;
- its 95/5 split;
- every batch order and mixup draw.

The only difference is the extra loss term. Each student − baseline difference
is therefore paired, which is the strongest comparison n=4 allows. The gate
verifies this.

### Loss

```
L = os_scl(...)                                   unchanged: mixup ArcFace CE + SupCon
  + beta * tau^2 * KL( p_T || p_S )
p_S = softmax(30 * cos_S(x_mix) / tau)            student, margin-free scaled cosines
p_T = mean over 3 teachers of softmax(30 * cos_T(x_mix) / tau)
```

- **beta = 1, tau = 2, fixed now.**
- τ = 2 because of measurement 1: τ = 1 is saturated (0.993 on the own ID),
  and τ ≥ 4 is close to uniform. τ = 2 is the only power of two that is neither.
- β = 1 adds the term at the scale Hinton's τ² factor was designed for. The
  original loss is kept intact, so the recipe stays the paper's plus one term.
- Margin-free cosines are used because the margin is a training device on the
  own-ID logit only. The 41 cosines are comparable across teachers because each
  index is the same machine ID.

Validation and checkpoint selection are unchanged: the EMA's hard CE on the
5% split. So model selection does not see the teachers.

### Scoring

- **Primary rule: eval.py's R0**, unchanged. The claim being tested is a
  single-model one, and R0 is the paper's rule.
- **Secondary, reported only:** S1′ from `../ensemble-backend`, and E1 over the
  4 students.

## Gate (before any 300-epoch run)

**G1, reproduction.** With β = 0 and the teachers loaded and running, 10
epochs at seed 2024 must reproduce `tfl_relu_max_s2024/student_e10.pth`
**bit-exactly** (max |Δ| over all parameters = 0). This proves the KD
plumbing changes nothing else: no RNG draw, no batch order, no BatchNorm
update leaks through the teachers. If G1 fails, nothing runs until the cause is
found.

**G2, cost.** If a β = 1 epoch takes more than 1.6× a baseline epoch, the
teachers drop to bf16 or two teachers. That change is recorded here before the
run starts.

### Gate result, 2026-10-13 (`scores/gate.json`)

- **G2 passes:** 1.49 vs 2.15 it/s, a 1.44× slowdown. By epoch time it is
  366 s vs 265 s, 1.38×.
- **G1 fails.** At epoch 10, 188 of 315 tensors differ: every trained one. The
  127 that match are the TFgram modules `forward()` never uses. The largest
  difference is 12.1, in `TFgramNet.pre_block3.bn1.running_var`. The trained
  weights differ by about 1% relative. The runs already part in epoch 0:
  train loss is 13.05942 here against 13.06018 for the baseline.
- **Reading, before diagnosis.** An RNG leak would change the mixup draws and
  move the epoch-0 loss by around 1e-2, not 6e-5. Nothing in the model draws
  random numbers in eval mode (there is no dropout). So the cause is
  numerical. There are three candidates:
  - (a) training is not bit-deterministic on smartlab1 even without teachers;
  - (b) the 0 × KD backward path changes a gradient;
  - (c) cuDNN or cuBLAS picks different kernels with the teachers on the GPU.

  A tiny numerical difference is then amplified by training. That is not the
  RNG unpairing G1 was meant to catch, but it still decorrelates the student
  from its baseline. So the paired design holds only if (a) is ruled out.
- **Diagnosis (`run.sh diag`, `run.sh null`, before any student).** `diag`
  runs 20 steps in fresh processes under five conditions: no teachers twice,
  teachers loaded only, teachers run with KD not added, and the gate's
  loss + 0 × KD. It compares the losses and weights; the output goes to
  `scores/diag_g1.txt`. `null` repeats the 10-epoch gate with no teachers
  against the baseline (`scores/gate_null.json`).
- **What follows from each outcome.**
  - If `null` is not bit-exact, (a) holds: no run on this host is reproducible.
    Each student is then compared with a **fresh baseline trained alongside it**
    instead of the archived `tfl_relu_max_s<seed>`. That costs 4 more runs. The
    predictions are unchanged.
  - If `null` is bit-exact, the plumbing is fixed until the gate passes.

## Predictions

Written before any student exists. Δ means student − same-seed `tfl_relu_max`
baseline, R0, overall AUC.

**P1. Distillation helps on dev, the decision-bearing prediction.** Δ > 0 on ≥ 3
of 4 seeds and mean Δ ≥ +0.3 on dev. Confidence: low-moderate (35%).
Measurements 1 and 2 show the teachers mostly agree even on mixed inputs, so
the signal is thin.

**P2. Eval confirms.** Mean Δ > 0 and Δ > 0 on ≥ 3 of 4 seeds on eval.

**P3. A minority of the ensemble's gain is recovered.** (mean student − mean
baseline) / (E1[relu4] − mean baseline) lies in [0.1, 0.5] on dev, where the
ensemble's gain is +0.96. Born-again and function-matching results usually
recover a fraction, not all of it.

**P4. Where it lands.** If P1 holds, the largest per-machine mean Δ is
ToyConveyor's (50%). Pump moves < +0.5 (70%), since the ensemble itself did
not fix pump.

**P5. Against the paper (single model, mean of 4 students).** Dev AUC ≥ 95.71:
10%. It needs about +0.9 over the baseline mean of 94.82. Eval mean(AUC, pAUC)
≥ 93.22: 45%. It needs +0.18 over 93.04.

**P6. Seed spread.** The sd of ToyConveyor dev AUC across the 4 students is
below the 4 baselines'. Reported, not read, at n=4.

## Decision rules

Fixed now. Dev selects, eval confirms. Nothing is tuned on eval.

- **D1, adopt the recipe.** KD (β = 1, τ = 2, 3 leave-one-out teachers) is
  adopted iff P1 **and** P2 hold.
- **D2, single-model claim against the paper.** Allowed per metric only where
  the 4-student mean exceeds the paper's number. It is reported with the seed
  sd and worded "a single OS-SCL network (1.38 M parameters) trained with
  distillation from 3 OS-SCL teachers". That wording is honest about the extra
  training compute. "Clearly" is used only if the worst student also exceeds.
- **D3, no rescue on these seeds.** If D1 fails, β and τ are not retuned and
  rescored on these students' eval. A second attempt picks its settings on dev
  only and is pre-registered again.
- Single-model mAUC deltas under ~2 points are not read (`CLAUDE.md`).

## What this cannot answer

- **Whether the consensus is the mechanism.** A gain could come from the KD
  term acting as a generic regulariser. The leave-one-out design removes the
  self-distillation route, but not label-smoothing-like effects. A control with
  uniform targets in place of p_T would separate them. It is not run unless D1
  holds.
- **Teachers trained on these clips.** They are scored on mixed versions of
  their own training data. Out-of-fold teachers would be cleaner but would cost
  4 more training runs.
- **Generality across checkpoints.** The teachers are the same 4 checkpoints
  whose eval scores `../ensemble-backend` already reported. The students are
  new networks, so their eval is unseen.

## Planned files

```
trainer.py / train.py   --kd_teachers <names> --kd_beta --kd_tau; default off, main behaviour unchanged
run.sh                  gate (G1, G2), then 4 students; resumable; KD_GPUS
make_plots.py           reads scores/ only
scores/<model_name>/    eval_dev.csv, eval_eval.csv, config.json, curve.csv
scores/gate.json        G1 parameter diff and G2 timing
results/                logs, gitignored
```

Students: `SCLTFSTgramMFN_kd_relu_max_s{2024..2027}`, about 20–30 h each,
one per GPU. They start when a GPU frees up: GPU 0 after the memory-bank probe,
GPUs 1–2 after `../confirm-seeds` (≈ Thursday).
