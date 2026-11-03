# Margin warm-up: are the single-branch mel-free runs an optimisation failure?

> **Status: done 2026-10-23 (n=4).** All 12 warm-up runs PASS the mechanism
> criterion, so the warm-up removes the trap every time.
>
> **V1 (optimisation failure) holds for both trapped arms:**
> - **f6 (TF only):** W6 − W2 = **+1.59 dev** / +3.73 eval.
> - **f7 (T only):** W7 − W2 = **+10.13 dev** / +6.34 eval.
>
> **V4 fails.** Even converged, TFgram alone stays well below Log-Mel alone: R1 =
> −6.77 dev / −9.08 eval. The control is unchanged (W2 = −0.11 / +0.29). See
> [Stage 2](#stage-2-n4-2026-10-23).
>
> Pre-registered 2026-10-20, before any run; stage 2 user-approved 2026-10-22.
> Branch `ablation/margin-warmup`, from `ablation/no-logmel` at `10e07e7`.

## Question

`../no-logmel` trains the model without Log-Mel. Its single-branch arms, `nl_f6`
(TFgram only) and `nl_f7` (Tgram only), end far below everything else. The
5-epoch probe gives dev AUC 85.9/85.4 for f6 and 77.0/77.6 for f7 (s2040/s2041).
For comparison, f5 (T+TF) gives 89.7/90.3, f2 (Log-Mel only) 94.0, and `fr_f1`
94.5. `diag_margin.py` (`../no-logmel/scores/diag_margin_s2040*.txt`) shows that
f6 and f7 are caught in an **ArcFace margin trap**:

- Train and valid accuracy, both with the margin applied, are exactly 0 from
  epoch ~55 to the end.
- Every embedding points away from every class weight. The cosine to its own
  class is −0.74 to −0.83.
- There the target logit is cos(θ+m) with θ+m ≈ π, and its gradient
  −sin(θ+m) ≈ 0.
- `nl_f7` still ranks 87% of train clips correctly without the margin, and
  `nl_f6` ranks 99.9% (by epoch 250).

**Is the single-branch mel-free deficit an optimisation failure (the trap), or
missing information?** If it is the trap, a fix that keeps the network out of it
should raise f6/f7 substantially. It should do so without raising a model that
was never trapped.

## When the trap forms (from the registered logs)

The logged `Train_accuracy` is the margin-0.4 top-1 on mixup batches. Its
trajectory is nearly identical in all four trapped runs:

| run | first > 0.001 | peak (epoch) | ≤ 0.002 from |
|---|---|---|---|
| `nl_f6_s2040` | 21 | 0.050 (47) | 54 |
| `nl_f6_s2041` | ≤ 38 (lines missing before) | 0.044 (46) | 55 |
| `nl_f7_s2040` | 10 | 0.072 (38) | 54 |
| `nl_f7_s2041` | 11 | 0.082 (37) | 55 |

(EPOCH lines are printed only when validation loss improves, so some epochs are
missing.) `nl_f6_s2042` peaked at 0.058 (epoch 46) and was still falling when this was written. The untrapped
references reach margin accuracy ≈ 0.10 by epoch 31 (`nl_f5`) and 0.20 by
epoch 5 (`nl_f2`), and then plateau at the mixup ceiling of ~0.4. So the trap
closes in a narrow window: margin accuracy peaks at epochs 37–47 and is gone
by **epochs 54–55**, while the single-branch nets are
still at 4–8% margin accuracy. They never got far enough into the target cone
for cos(θ+m) to have a usable gradient.

## Mechanism and the fix

Take a clip whose embedding sits at angle θ from its class weight. Its target
logit is s·cos(θ+m), and the pull towards the class has magnitude
s·sin(θ+m). The non-target logits s·cos θ_j push the embedding away from
all the other class weights. When those weights sit in one cone, as here (the
nearest other class is at cos −0.85 to −0.89), the pushes add up coherently.
Take the `nl_f7` trap point, cos θ_y = −0.83 (θ = 2.55 rad):

| | target pull ∝ | value |
|---|---|---|
| margin 0.4 | sin(θ + 0.4) | **0.19** |
| margin 0 | sin θ | 0.56 |

With m = 0, the target there already has the largest logit (no-margin top-1
0.87), and its pull is 3× stronger. So the antipodal point is not a stable
attractor without the margin. The margin is what makes it one.

**Chosen fix: margin warm-up** (`--m_warmup 60,120`):

- The student's margin is **0 for epochs 0–59**.
- It then ramps **linearly per step** to **0.4 at epoch 120**.
- It stays at **0.4 for epochs 120–299**.

This follows three precedents:

- **SphereFace** (Liu et al., CVPR 2017, arXiv 1704.08063) anneals from plain
  softmax into its angular margin, because training with the margin alone
  converges poorly.
- **AirFace** (Li et al., arXiv 1907.12256) reports that ArcFace with
  **MobileFaceNet's 128-d embedding**, the backbone and width used here, fails
  to converge from scratch without softmax pre-training. This is contested:
  MobileFaceNets (arXiv 1804.07573) trained it from scratch.
- **WeSpeaker's** AAM-softmax recipe codifies the idea as a scheduler
  (`examples/voxceleb/v2/conf/resnet.yaml`: margin 0 → 0.2 between epochs 20
  and 40 of 150).

Schedule, from the observed timing:

- **Hold to 60.** The hold covers the whole trap window (peak 37–47, closed by 54–55) plus margin.
  It is not shorter, because a hold that ends inside the window would test
  the window, not the fix.
- **Ramp of 60 epochs, equal to the hold.** WeSpeaker uses hold = ramp
  (20/20), here at 20%/40% of training against its 13%/27%. A gradual ramp lets
  embeddings that already sit inside their class cone (cos > 0, where
  sin(θ+m) is large) absorb the margin.
- **180 epochs at the final margin.** The EMA (α 0.9999, ~540 steps/epoch)
  has a time constant of ~19 epochs, so the saved EMA net has seen ~10 time
  constants of the exact registered objective.

**Why warm-up and not easy-margin.** Easy-margin
(`phi = cos θ` wherever cos θ < 0) removes the trap region by construction.
But it does so permanently: it changes the training objective to the end of the
run, for every clip and mixup pair with cos < 0. It also adds a discontinuity
at cos = 0, where phi jumps from −sin m to 0. After epoch 120, the warm-up
trains exactly the registered loss, so it changes only the **path** and not
the **objective**. A positive result then says "the registered objective was
reachable, the optimiser just didn't get there". Easy-margin is the
pre-registered **fallback** (see D-pilot) and is not implemented now.

### What changes, and what does not

- Only the **student's** `ArcMarginProduct` margin moves, via
  `set_margin(m)`, which recomputes `cos_m/sin_m/th/mm` together.
- `ema_net` keeps `cfg['m'] = 0.4` (the margin is a Python attribute and not
  in the state_dict, so `WeightEMA` never copies it). So validation loss,
  checkpoint selection, the probe and `model.pth` all score at **m = 0.4**,
  exactly as the registered runs do and as `eval.py --m 0.4` does.
- State_dict keys are unchanged, so checkpoints load strictly.
- `config.json` records `m_warmup`. `cfg['m']` stays 0.4, the final margin.
- **Logging.** When the flag is on, each EPOCH line also carries
  `| m: <student margin> | Train_acc_mfinal: <train top-1 at m = 0.4>`, which is
  the registered runs' `Train_accuracy` definition, so the trap signal stays
  comparable. `probe.csv` gains an `m` column. `run.sh score` writes both to
  `scores/<run>/curve_mw.csv`. `Train_accuracy` itself is then at the
  *current* student margin.
- **Default off, verified.** On CPU, a fussion 7 and a fussion 2 model with
  the same seed: logits before training, and every tensor of both nets after
  two training epochs, are **bit-identical** between the pristine code
  (`10e07e7`) and this branch without the flag (180/180 and 170/170 tensors).
  With `--m_warmup 0.5,1.5`, 145–153 tensors differ and the EMA margin stays
  0.4. `set_margin(0.17)` matches a fresh `ArcMarginProduct(m=0.17)` in all
  five constants, and at m = 0 the logits equal 30·cos.

## Design

The fast recipe of `../no-logmel/run_one.sh`, unchanged
(`--m 0.4 --ht basic --num_workers 8 --amp fp16 --cudnn_bench 1 --lean 1
--cache_dir data/cache --probe_every 5`, 300 epochs). Added: `--m_warmup 60,120`,
plus `--state_snapshots 60,100`, which saves full state (student + EMA) after
epochs 60 and 100 for the mechanism check. Snapshots draw no random numbers.
Runs go through `../train_one.sh` under ablation name `margin-warmup` (descs
`mw_<arm>_s<seed>`).

| arm | fussion | inputs | registered twin | role |
|---|---|---|---|---|
| `mw_f6` | 6 | TFgram | `nl_f6` (trapped) | fix on a trapped arm |
| `mw_f7` | 7 | Tgram | `nl_f7` (trapped) | fix on a trapped arm |
| `mw_f2` | 2 | Log-Mel | `nl_f2` (never trapped) | **control**: does warm-up help a model that was never stuck? |
| `mw_f5` | 5 | T+TF | `nl_f5` (healthy, lags 5 pts) | optional (stage 2b) |

**Why a control, and why f2.** If warm-up simply helps every model, a gain on
f6/f7 says nothing about the trap. The control must be a model that was never
trapped. `nl_f2` is the cheapest such model (60 s/epoch). It has a 4-seed
registered twin at the same seeds and recipe, and it is the reference cell for
the C2b/C4 questions that this ablation re-asks. `fr_f1` would also work, but it
costs 2.2× more and adds nothing the question needs. `mw_f5` asks a different
question (does warm-up rescue the healthy-but-lagging mel-free net?), so it is
optional.

## Stages and budget

Per-epoch cost on one A4000, alone: f6 155 s, f5 133 s, f7 72 s, f2 60 s.
Measured walls: `nl_f6_s2040` 13.0 h, `nl_f7_s2040` 6.1 h.

| stage | runs | GPU-h | wall | launch |
|---|---|---|---|---|
| 0 gate | `mw_f6/f7_s1_e2` (MW=0.5,1.5, SNAPS=1) + eval.py + diag | ~0.5 | ~40 min, GPU 0 | auto (`chain.sh`) |
| 1 pilot | `mw_f6_s2040`, `mw_f7_s2040`, `mw_f2_s2040` | 13.0 + 6.1 + 5.0 = **24** | **~13 h** on GPUs 0+2 (f2 follows f7) | auto (`chain.sh`) |
| 1 score/diag/plots | eval.py dev+eval ×3, diag ×3 | ~1 | ~1 h, GPU 0 | auto (`chain.sh`) |
| 2 | `mw_{f6,f7,f2}` × seeds 2041–2043 | 39 + 18 + 15 = **72** | **~26 h** on 3 GPUs | by hand: `run.sh stage2` |
| 2b (optional) | `mw_f5` × 2040–2043 | **44** | ~15 h on 3 GPUs | by hand: `run.sh stage2b` |

The pilot's seed-2040 runs count as seed 1 of stage 2, provided the code and
schedule are not changed after the pilot. If they are changed, stage 2 reruns
s2040 as well (+24 GPU-h). GPU 1 is left free during the pilot for
`../no-logmel`'s `run.sh score`.

## Contrasts

Δ = mean difference of eval.py R0 overall AUC, in points, at matched seeds.

| contrast | runs | question |
|---|---|---|
| W6 | `mw_f6` − `nl_f6` | warm-up effect, TF only (trapped) |
| W7 | `mw_f7` − `nl_f7` | warm-up effect, T only (trapped) |
| W2 | `mw_f2` − `nl_f2` | warm-up effect on the never-trapped control |
| **W6 − W2, W7 − W2** | | the part of the effect that is specific to the trap |
| R1 | `mw_f6` − `mw_f2` | C2b redone with both nets converging: does TFgram replace Log-Mel alone? |
| R2 | `mw_f7` − `mw_f6` | C4 redone: Tgram vs TFgram as the only branch |
| W5 (2b) | `mw_f5` − `nl_f5` | does warm-up close f5's lag? |

## Pre-registered mechanism criterion (stage 1 pass/fail)

`diag_margin.py` is run on the same 1024 training clips as `../no-logmel`, always
at m = 0.4.

- **PASS (per arm)**: the final best-EMA `model.pth` has acc(margin 0.4)
  **≥ 0.5** *and* mean cos to own class **> 0**. The registered values are
  0.000 and −0.74/−0.83; `fr_f1` has 0.999 and +0.996.
- **TRAPPED**: acc(margin 0.4) < 0.05 *or* cos to own class < 0.
- In between: **partial**.
- **Supporting checks** (reported, not gating):
  - the student at e60 (end of hold) has cos to own class > 0 and no-margin
    top-1 ≥ 0.5;
  - the student at e100 (m = 0.267) has acc(margin 0.4) > 0;
  - `Train_acc_mfinal` never sits at 0 after epoch 120 (fig2).

## Predictions (written before any run)

Pilot, dev AUC (eval.py R0), against the registered twin at the same seed:

- **P0 (mechanism).** Both `mw_f6` and `mw_f7` PASS. Confidence ~75% for f7,
  which already separated IDs in the trap, and ~65% for f6, which also
  collapsed early (all embeddings at cos 0.998 at epoch 110).
- **P1.** `mw_f6` − `nl_f6` = **+3.5**, range [+1.5, +7]. That puts `mw_f6` near
  89, around `nl_f5`'s level. Reason: inside the trap the score
  CE(cos(θ+m)) responds to a change in θ only through sin(θ+m) ≈ 0.2–0.4, so
  anomalies are compressed. Escaping restores that sensitivity. The ceiling is
  TFgram's information (PANNs' Wavegram-CNN was 0.389 vs log-mel's 0.431 mAP).
- **P2.** `mw_f7` − `nl_f7` = **+4**, range [+1, +9], ending near 81.
  STgram-MFN's Tgram-only model was 12.5 below its log-mel model under plain
  CE; 94 − 12.5 ≈ 81.5.
- **P3 (control).** `mw_f2` − `nl_f2` = **0.0**, range [−0.6, +0.4]. f2 is
  never trapped, and its EMA AUC only takes off after epoch ~100, after which
  both runs train the same objective.

Stage 2 (n=4):

- **P4.** W6 − W2 ≥ +1.0 and W7 − W2 ≥ +1.0 on dev.
- **P5 (residual).** Even converged, the mel-free single branches stay below
  Log-Mel: R1 = `mw_f6` − `mw_f2` ≤ −2 (central −5). R2 < 0: Tgram stays
  weakest (central −8).

## Decision rules

**D-pilot (do I recommend launching stage 2?)** The pilot is n=1, so its AUC does
not gate stage 2. Only the mechanism does.

- PASS (or partial) in ≥ 1 of `mw_f6`/`mw_f7` → **launch stage 2 as
  registered**.
- TRAPPED in both → **do not launch stage 2.** The trap re-forms during or after
  the ramp, so warm-up is not a fix. Write an amendment before any further
  run. Candidates: easy-margin on the student (the fallback above), or a longer
  hold.
- A crash or non-finite blow-up is a gate/infrastructure failure, not a
  result. Rerun.

**Stage 2 verdicts** (4-seed means, dev and eval, eval.py R0):

- **V1, optimisation failure**, for arm X ∈ {f6, f7}: PASS in ≥ 3/4 seeds
  *and* W_X − W2 ≥ **+1.0** on dev and > 0 on eval.
- **V2, trap real but not the bottleneck**: PASS in ≥ 3/4 seeds but
  W_X − W2 < +1.0 on dev. The deficit is then information or inductive bias.
- **V3, inconclusive**: PASS in < 3/4 seeds. The fix does not reliably
  remove the trap.
- **V4, TFgram replaces Log-Mel alone** under a converging recipe: R1 ≥ −0.20
  on dev and on eval (`../no-logmel`'s D2 threshold). Otherwise the C2b gap
  survives the fix.
- **Control flag**: if W2 ≥ +0.5 on dev and eval, warm-up is a general
  improvement. It is reported as a candidate recipe change, to be tested on
  `fr_f1` in its own ablation. It is not adopted here. If W2 ≤ −0.5, warm-up
  costs a healthy model, and that is reported.

**Noise.** Fast-recipe dev AUC seed sd ≈ **0.34** (`fr_f1`, n=4), so a
4-vs-4 difference has se ≈ 0.24. A difference of differences (W_X − W2, four
groups of 4) has se ≈ 0.34, so +1.0 is ≈ 3 se. Single-branch arms may be
noisier, and their registered sd is reported when it lands. Eval is noisier
than dev. mAUC deltas under ~2 points are unresolvable (CLAUDE.md). Every
verdict is reported with per-seed values.

## What this does not test

- One schedule only (60/120, linear). No sweep of hold length or ramp shape.
  WeSpeaker uses an exp ramp.
- Easy-margin, sub-center ArcFace, and margin-free (cos-softmax) training to
  the end are not run.
- Nothing else is retuned (lr, epochs, SupCon τ, mixup).
- `fr_f1` with warm-up is not run. f2 stands in as the control.

## Files

- `run.sh`: stages gate | pilot | stage2 | stage2b | score | diag | plots;
  resumable.
- `run_one.sh`: one run.
- `chain.sh`: the detached stage-1 chain.
- `diag_margin.py`: copied from `../no-logmel`, extended to read
  `state_eN.pth:net|ema_net`.
- `make_plots.py`: reads `scores/` only. Registered `nl_*` runs are resolved via
  `../scores_path.py`, or, until the no-logmel branch is merged, from
  `$MW_EXTRA_SCORES` (the no-logmel worktree's `scores/`).
- `scores/`: `gate.txt`, `diag_margin_<run>.txt`, `summary.txt`, and per run
  `eval_{dev,eval}.csv`, `config.json`, `probe.csv`, `wall.csv`, `curve.csv`,
  `curve_mw.csv`.

## Results

### Stage 1 pilot (seed 2040, n=1), 2026-10-22

Trained 2026-10-21 19:30 to 2026-10-22 06:51 on GPUs 0 and 2: f6 10.9 h,
f7 5.9 h, f2 5.4 h. All runs had 0 non-finite steps, 0 crashes and 0
RAM-cache recomputes. In every run `model.pth` is the epoch-299 EMA, because
validation loss was still falling at the end.

**Mechanism (gating).** `scores/diag_margin_mw_*.txt`, fig3.

| run | acc (no margin) | acc (margin 0.4) | cos to own class | verdict | registered twin `nl_*_s2040` |
|---|---|---|---|---|---|
| `mw_f6` | 0.999 | 0.991 | +0.995 | **PASS** | 0.000 / −0.74, trapped |
| `mw_f7` | 0.990 | 0.975 | +0.993 | **PASS** | 0.000 / −0.83, trapped |
| `mw_f2` (control) | 0.999 | 0.999 | +0.995 | PASS | never trapped |

Supporting checks (not gating) all hold:
- **e60 student** (end of the m = 0 hold): no-margin top-1 is 0.82 (f6) and
  0.85 (f7), and cos to own class is +0.19 and +0.17.
- **e100 student** (m = 0.267): margin accuracy is 0.052 (f6) and 0.055 (f7),
  so above 0.
- **`Train_acc_mfinal`** never returns to 0 once the ramp ends (fig2).

**AUC (eval.py R0, n=1, not a verdict).**

| arm | dev: reg. n=4 mean [seed range] | dev: warm-up | dev: paired Δ (s2040) | eval: reg. n=4 mean | eval: warm-up | eval: paired Δ |
|---|---|---|---|---|---|---|
| TF only (f6) | 85.77 [85.52, 86.24] | **88.13** | +1.89 | 82.66 | **85.76** | +3.78 |
| T only (f7) | 69.73 [66.91, 72.85] | **78.51** | +8.60 | 74.79 | **80.78** | +6.20 |
| Log-Mel only (f2) | 94.13 [93.98, 94.31] | 93.98 | +0.01 | 95.47 | 95.59 | −0.05 |

- Both mel-free warm-up runs land above every registered seed of their arm, on
  dev and on eval.
- Against the n=4 means, W6 − W2 = +2.52 dev / +2.98 eval and W7 − W2 = +8.93
  dev / +5.87 eval. Both are past the +1.0 V1 bar, but each rests on one seed.

**Predictions.**

| | predicted | pilot | |
|---|---|---|---|
| P0 | f6 and f7 PASS | both PASS | ✓ |
| P1 | `mw_f6` − `nl_f6` = +3.5 [+1.5, +7] | +1.89 (paired) | ✓ in range, below central |
| P2 | `mw_f7` − `nl_f7` = +4 [+1, +9], near 81 | +8.60 (paired), 78.5 | ✓ in range, high end |
| P3 | `mw_f2` − `nl_f2` = 0.0 [−0.6, +0.4] | +0.01 | ✓ |

**Where the f7 gain comes from: mostly checkpoint selection.** fig1 shows the
two arms gain in different ways.

- **f7.** Its final-epoch probe barely moves: 78.5 against the registered
  77.8 mean (`nl_f7_s2040` 77.0). The registered runs reached the same end
  state. Inside the trap, though, their validation loss bottomed out at
  epoch ~150, so `model.pth` froze there at 69.7. Out of the trap, validation
  loss falls to the last epoch, so `model.pth` is the end state. The trap's
  cost to f7 was **which net got saved**, not how good the net could get.
- **f6.** The registered `model.pth` (85.8) already sat near its own final
  probe (85.5; s2040 85.9). The warm-up run ends higher (88.1), so for f6 the
  gain is in the end state itself.

The per-machine dev change for f6 at s2040 is mixed:
- **up:** slider +7.1, valve +5.8, pump +3.4, ToyCar +0.3
- **down:** ToyConveyor −4.7, fan −0.6

ToyConveyor's seed noise is above 4 points (CLAUDE.md), so none of these
changes is resolvable at n=1.

**Not pre-registered: EMA lag.**
- At e100 the EMA net is still collapsed in all three arms, including the
  control: all embeddings sit at cos ≥ 0.997 to each other. Meanwhile the
  student already reaches no-margin top-1 ≥ 0.96.
- The EMA net's time constant is 1/(1 − 0.9999) = 10⁴ steps, about 18.5
  epochs at 539 steps per epoch. The EMA also averages over the early
  collapsed weights.
- This explains why the f6 probe stayed at chance until about epoch 150 even
  though its student was healthy at epoch 100 (fig1). It does not change the
  verdict.

**D-pilot.** Both trapped arms PASS, so the rule says **launch stage 2 as
registered**: `run.sh stage2`, `mw_{f6,f7,f2}` × seeds 2041–2043.
- Measured wall time is 22.2 GPU-h per seed, so about 67 GPU-h in total, or
  ~23 h on three cards.
- Stage 2b (`mw_f5` × 4, ~44 GPU-h) stays optional.

### Stage 2 (n=4), 2026-10-23

**Run.** Seeds 2041–2043 trained 2026-10-22 10:17 to 2026-10-23 12:30 on GPUs
0–2 and were then scored, diagnosed and plotted by the detached
`results/stage2_chain.sh`. The tables pool them with the s2040 pilot.

**Health.** 0 crashes, 0 non-finite steps and 0 RAM-cache recomputes.
`model.pth` is a late EMA in every run: last saved epoch 294–299, except
`mw_f7_s2041` at 264. The registered `nl_f7` runs saved at 144–154, and
`nl_f6` at 234–239.

**Mechanism: 12 of 12 PASS** (`scores/summary.txt`, fig3).

| arm | acc (margin 0.4), seeds 2040–43 | cos to own class | registered |
|---|---|---|---|
| f6 | 0.991 0.990 0.990 0.983 | +0.995 (all) | 0.000, −0.74 |
| f7 | 0.975 0.976 0.979 0.979 | +0.993/+0.994 | 0.000, −0.83 |
| f2 | 0.999 1.000 0.998 0.998 | +0.994/+0.995 | (never trapped) |

**AUC per seed** (eval.py R0, seeds 2040, 2041, 2042, 2043).

| arm | ds | registered `nl_*` | warm-up `mw_*` | paired Δ | mean reg. → warm-up |
|---|---|---|---|---|---|
| f6 TF only | dev | 86.24 85.76 85.52 85.54 | 88.13 87.08 86.54 87.25 | +1.89 +1.31 +1.02 +1.71 | 85.77 → **87.25** |
| f6 TF only | eval | 81.98 82.53 83.49 82.63 | 85.76 86.07 86.74 88.14 | +3.78 +3.54 +3.25 +5.51 | 82.66 → **86.68** |
| f7 T only | dev | 69.92 72.85 66.91 69.26 | 78.51 79.89 81.37 79.22 | +8.60 +7.04 +14.47 +9.96 | 69.73 → **79.75** |
| f7 T only | eval | 74.58 77.20 72.48 74.88 | 80.78 82.43 80.70 81.73 | +6.20 +5.23 +8.21 +6.86 | 74.79 → **81.41** |
| f2 LM (control) | dev | 93.98 94.18 94.06 94.32 | 93.98 94.16 94.04 93.91 | +0.01 −0.03 −0.02 −0.40 | 94.13 → 94.02 |
| f2 LM (control) | eval | 95.64 95.71 95.38 95.14 | 95.59 95.86 95.89 95.68 | −0.05 +0.15 +0.51 +0.53 | 95.47 → 95.76 |

Every paired Δ for f6 and f7 is positive, on both sets. The warm-up also
narrows the seed spread of f7: dev sd 1.22, against 2.45 registered.

**Contrasts**: means of 4 vs 4, Welch se, bootstrap 95% (20k resamples).

| contrast | dev | eval |
|---|---|---|
| W6 − W2 | **+1.59** (se 0.38) [+0.98, +2.25] | **+3.73** (se 0.63) [+2.70, +4.83] |
| W7 − W2 | **+10.13** (se 1.37) [+7.78, +12.38] | **+6.34** (se 1.06) [+4.55, +8.11] |
| W2 (control) | −0.11 (se 0.09) | +0.29 (se 0.15) |
| R1 = `mw_f6` − `mw_f2` | **−6.77** (se 0.33) [−7.30, −6.15] | **−9.08** (se 0.53) [−9.87, −8.11] |
| R2 = `mw_f7` − `mw_f6` | −7.50 (se 0.69) [−8.61, −6.29] | −5.27 (se 0.67) [−6.44, −4.19] |

**Verdicts.**

| | rule | result | |
|---|---|---|---|
| **V1, f6** | PASS ≥ 3/4, W6 − W2 ≥ +1.0 dev and > 0 eval | 4/4; +1.59 dev, +3.73 eval | **holds**, with a thin dev margin: 1.5 se above the bar, and the bootstrap lower bound is +0.98 |
| **V1, f7** | same | 4/4; +10.13 dev, +6.34 eval | **holds**, decisively |
| V2 / V3 | | | not reached |
| **V4** | R1 ≥ −0.20 dev and eval | −6.77 / −9.08 | **fails**: TFgram alone does not replace Log-Mel, even converged |
| Control flag | \|W2\| ≥ 0.5 on dev and eval | −0.11 / +0.29 | neither: warm-up neither helps nor hurts a healthy net |

**Predictions.**

| | predicted | result | |
|---|---|---|---|
| P0 | f6 and f7 PASS | 8/8 PASS | ✓ |
| P1 | W6 = +3.5 [+1.5, +7] (dev) | W6 = +1.48 (n=4) | ✗ just below the range; the pilot seed (+1.89) was the best of four |
| P2 | W7 = +4 [+1, +9], near 81 | W7 = +10.02, mean 79.75 | ✗ above the range. The gain was under-predicted because checkpoint selection was not foreseen (below). The level, 79.8 against 81, is close |
| P3 | W2 = 0.0 [−0.6, +0.4] (dev) | −0.11 | ✓ |
| P4 | W6 − W2 ≥ +1.0 and W7 − W2 ≥ +1.0 (dev) | +1.59, +10.13 | ✓ |
| P5 | R1 ≤ −2 (central −5); R2 < 0 (central −8) | −6.77; −7.50 | ✓ ✓ |

**Two different gains.** Where does each gain come from: the final net, or
which epoch gets saved? fig1 shows each arm's final-epoch probe, which is
dev, at m = 0.4, on the EMA net.

| arm | final probe: reg. → warm-up | `model.pth` dev: reg. → warm-up | share from checkpoint selection |
|---|---|---|---|
| f6 | 85.50 → 87.25 (+1.75) | 85.77 → 87.25 (+1.48) | none: the registered `model.pth` already sat at its end state |
| f7 | 77.83 → 79.82 (+1.99) | 69.73 → 79.75 (+10.02) | **~8 of the 10 points** |

- **f7: the trap mostly cost checkpoint selection, not capacity.**
  - The registered runs' validation loss bottomed out at epoch 144–154. The
    trap kept the margin logit pinned while the net kept improving, so
    `model.pth` froze about 150 epochs early, at 69.7, even though the nets
    reached ~77.8 by the end.
  - Out of the trap, validation loss tracks the net to the end. The end state
    itself is also +2.0 better.
- **f6: the gain is in the end state.**
  - The registered f6 un-collapsed inside the trap (`../no-logmel`, interim
    2026-10-20), so its `model.pth` was already late. Escaping the trap adds
    +1.5 to +1.75 to where the net ends up.
  - **Per machine** (Δ at n=4, dev):
    - up: slider +5.9, valve +4.9, pump +4.0, ToyCar +0.4
    - down: fan −3.1, ToyConveyor −3.2

    On eval, valve gains +19.3 and slider +10.4, while ToyCar loses −2.6 and
    ToyConveyor −3.5.
  - The gain is in the stationary machines (valve, slider, pump). ToyConveyor
    moves by less than its >4-point seed noise. The fan loss on dev is not
    resolvable per machine at n=4, and on eval it is −0.3.

**EMA lag** (not pre-registered).
- The e100 EMA net is collapsed in all 12 warm-up runs: embedding cos ≥ 0.990
  even between different IDs (`diag_margin_*.txt`), while the student is already healthy.
- The probe takes off 20–50 epochs after the student (fig1).
- This explains why the f6 probe sits at chance until about epoch 150. It
  does not change any verdict.

**What this changes for `../no-logmel`.**
- **Its headline stands.**
  - Its C1 (paper − T+TF) and C2a contrasts rest on f5, which was never
    trapped. D1 and D2 still fail.
  - LM-only is still the cheapest cell within 0.2 of the paper model. TF-only
    is still 6.8 / 9.1 points below LM-only (R1 here), the same verdict as
    its C2b (−8.37 / −12.81).
- **Its single-branch deficits were inflated by the trap:**
  - C2b (TF − LM): from −8.37 dev / −12.81 eval to **−6.77 / −9.08**.
  - C4 (T − TF): from −16.03 / −7.87 to **−7.50 / −5.27**.

  The no-logmel README carries an annotation pointing here.
- **For anyone training a single-branch or weak-input ArcFace net on this
  recipe:**
  - Use `--m_warmup 60,120`.
  - Better, check that `model.pth`'s epoch is late. A last-saved epoch far
    before the end, with probe AUC still rising, is the trap's signature.

**Not done.** Stage 2b (`mw_f5` × 4, ~44 GPU-h) would test whether warm-up
also closes f5's lag. f5 was never trapped, and the control flag is neutral,
so the expected gain is small. It is left for a separate decision.

## Figures

| file | shows |
|---|---|
| `plots/fig1_probe_curves.png` | probe dev AUC vs epoch: warm-up vs registered, per arm |
| `plots/fig2_train_margin_acc.png` | train top-1 at m = 0.4 vs epoch, with the student-margin schedule |
| `plots/fig3_geometry.png` | diag geometry: cos to own class vs margin accuracy, e60/e100/final, against the registered trap points |
| `plots/fig4_auc.png` | eval.py dev/eval AUC per seed, warm-up vs registered |
