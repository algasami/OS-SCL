# PerMix: does predicting the perturbation ratio of multi-sample mixes beat Noisy-ArcMix + SupCon?

> **Status: PRE-REGISTERED 2026-11-05, stage 1 (pm s2040–2043) launched after this
> commit.** No pm run had trained past the 2-epoch smoke when the design,
> predictions and decision rules below were written.

## Why this exists

Our training objective is OS-SCL's (`utils.os_scl`). It is Noisy-ArcMix plus
SupCon:

- **Noisy-ArcMix.** Two clips are mixed with λ ~ Beta(0.5, 0.5). The ArcFace
  margin goes on the first clip's class, and the loss is λ·CE(y_a) +
  (1−λ)·CE(y_b).
- **SupCon** on the mixed embedding uses the first clip's label `y_a`.

Because λ is not folded to max(λ, 1−λ), **half the mixes are labelled with
their minority clip**: SupCon's positives and the margin's target are then
the clip that contributes less. The margin is also the same whether the clip
is 95% or 55% target.

PerMix (below) is a direct fix for both, from the Noisy-ArcMix authors. On the
same TFSTgram-MFN backbone it reports +1.6 dev AUC over their OS-SCL rerun,
concentrated on ToyCar and ToyConveyor. Those two machines carry ~80% of
`ao_nogain`'s eval error mass (ToyConveyor ~60%, ToyCar 22%). It changes
only the objective, so it stacks on our recipe at no extra cost (bench below).

## Literature (read 2026-11-05)

**J. Choi and J.-W. Choi, "PerMix: Anomalous sound detection by predicting the
perturbation ratio of mixed samples", arXiv:2610.31434v1 (eess.AS, 22 Oct
2026), KAIST.** I read the PDF itself. Verified:

- The paper exists and the method is fully specified (below).
- **The official code is not out.** `github.com/kaistss/PerMix` holds only
  `LICENSE` (GPL-2.0) and a one-line `README.md` (1 commit, checked
  2026-11-05). Every implementation choice the paper leaves open is listed
  under *Deviations and open choices*.
- The research subagent's summary was accurate on the numbers.

**Method** (paper §2):

- **Mixing (Eq. 1).**
  - For each anchor i in the batch, R ~ U{1, 2, 3} perturbing clips J_i ⊂ B \ {i}.
  - R+1 coefficients are drawn from a symmetric Dirichlet, a = 0.5. The largest
    goes to the anchor, and the rest go to the perturbers in random order.
  - x̃_i = α_ii x_i + Σ_j α_ij x_j.
- **Dominant-class ratio (Eq. 2).** The anchor's class y_i is the *dominant
  class*. λ_i = Σ α_ij over all mixed clips (anchor included) with y_j = y_i,
  and 1 − λ_i is the *perturbation ratio*.
- **Logit (Eq. 3).**
  - Dominant class: ℓ_{i,y_i} = s(λ_i cos(θ̃+m) + (1−λ_i) cos θ̃).
  - Every other class: s cos θ̃.
  - This equals s·γ_i·cos(θ̃ + m_i), with an effective margin
    m_i = atan(sin m / (cos m + (1−λ_i)/λ_i)) and an effective scale
    γ_i = √(1 − 2λ_i(1−λ_i)(1−cos m)) (Eq. 4–5). Both shrink as λ_i falls, and
    λ_i = 1 gives ArcFace.
- **L_per (Eq. 6).** The BCE between λ_i and p_i, the softmax probability of
  the dominant class.
- **L_dom (Eq. 7).** SupCon, τ = 0.02, on the mixed embeddings. The positives
  are the other clips with the same dominant class.
- **Total (Eq. 8).** L = L_per + L_dom.
- **Score (Eq. 9).** The cosine distance to the nearest training embedding in
  a memory bank (kNN, k = 1).
- **Setup.** OS-SCL's MobileFaceNet network "with its feature perturbation
  head removed", 300 epochs, batch 64, lr 1e-4, AdamW, weight EMA, m 0.4,
  s 30, τ 0.02. One RTX 4090.

**Reported numbers.** TFSTgram-MFN, DCASE 2020 Task 2:

| | dev AUC | dev pAUC | eval AUC | eval pAUC |
|---|---|---|---|---|
| ArcFace (Table 3, 3 seeds, kNN) | 91.74 ± 0.14 | 85.30 | 92.17 | 86.25 |
| NAMix | 94.10 ± 0.37 | 88.49 | 94.59 | 89.67 |
| OS-SCL, their rerun | 94.28 ± 0.06 | 89.22 | 95.58 | 90.73 |
| **PerMix** | **95.89 ± 0.07** | **90.98 ± 0.17** | **96.45 ± 0.16** | **91.49 ± 0.15** |
| OS-SCL as published (Table 1, one network, CE score) | 95.71 | 90.23 | (eval mean 93.22) | |
| *ours: `ao_nogain`, 2040–43, R0* | *96.17* | *91.16* | *95.17* | *90.90* |
| *ours: `ao_nogain`, 2040–43, S1′* | *96.47* | *91.62* | *96.05* | *91.40* |

What the numbers imply for us:

- **The +1.6 is measured against a weak baseline.**
  - Table 3's OS-SCL is a rerun without the FPH head and scored by kNN: 94.28.
    Table 1 lists OS-SCL's published 95.71 for the same name. The paper does
    not explain the gap.
  - Against the published 95.71, PerMix is only **+0.18 dev AUC**.
  - Per machine against the published OS-SCL (Table 1, AUC): ToyCar +2.10,
    ToyConveyor +1.48, pump **−1.59**, fan −0.55, valve −0.28, slider −0.05.
- **Our baseline is already above PerMix's absolute numbers.** `ao_nogain`
  reaches 96.17 dev AUC under R0 and 96.47 under S1′. It is also above
  PerMix's dev ToyConveyor (85.96 R0, 87.35 S1′, vs 85.04). Only its eval
  AUC under R0 sits below PerMix's.
- **Where the gain comes from (their Table 2).**
  - L_per alone vs ArcFace: 94.54 vs 91.74. Adding L_dom: 95.89.
  - The NAMix → PerMix step, the one that matches our change, is +1.8 in
    their setting.

## Implementation (`--permix 1`, off by default)

- `utils.permix_data`:
  - Per anchor: R ~ U{1..`--permix_r`} (default 3), perturbers without
    replacement from B \ {i}, and α ~ Dir(`--permix_a`) (default 0.5).
  - The largest α goes to the anchor; the rest keep their draw order, which is
    exchangeable, so that is the paper's random order.
  - The mix is an exact weighted sum, with unused slots at weight 0. It
    applies to both the wave and the mel.
  - It returns λ_i (Eq. 2).
  - Draws come from numpy's global RNG, which `--crash_resume` restores.
- `ArcMarginProduct.forward_permix` computes Eq. 3. It uses the same
  cos(θ+m) as `forward()`, including its θ+m > π fallback. With λ = 1 it is
  `torch.equal` to `forward()` (checked).
- `utils.permix_loss`:
  - L_per is the BCE of Eq. 6. log(1−p) is computed stably as the log-mass of
    the other 40 classes. It matches the naive formula to float precision, and
    equals CE at λ = 1 (checked).
  - L_dom is our `SupConLoss` on the mixed, normalised embedding, labelled
    with the dominant class. That is the same call `os_scl` makes with `y_a`,
    which is the anchor label in both.
- `trainer.py`:
  - Under `--permix 1` the branch replaces `mixup_data` + `os_scl`. The net is
    called with the anchor labels. The PerMix logits are recomputed from the
    returned feature in fp32, outside autocast.
  - The EPOCH line adds `L_per`, `L_dom` and the batch-mean λ.
  - `--permix` refuses `--kd_teachers` and `--m_warmup`.
- **Untouched:**
  - wave_aug runs before mixing, as in `ao_nogain`.
  - `valid()` (EMA, unmixed, CE), checkpoint selection, the probe, `eval.py`
    and `extract.py` are unchanged.
  - At λ = 1 the training logit *is* the ArcFace logit, so R0 is the
    objective's own clean-clip score.
- **Checked numerically** (CPU unit test): the Eq. 4–5 identity holds. The
  mixing weights and λ, replayed from the RNG, match for every anchor. The
  anchor's weight is the largest, and the weights sum to 1.
  - With 41 classes at batch 64: mean λ 0.716, median 0.71, 10–90% range
    [0.48, 0.97], and λ = 1 for 1.1% of anchors.

### Deviations and open choices (the paper is silent, or our pipeline forces them)

1. **The FPH head is kept.**
   - PerMix removes OS-SCL's feature-perturbation head (`BasicHead`,
     128→64→128). We keep it, so the arm changes one thing: the objective.
   - Table 3, the paper's evidence, compares objectives at a fixed backbone.
   - OS-SCL's own Table VI puts "w/o FPH" at 94.94 against 95.71 with it.
     `../fast-recipe` matched 94.86–94.90 without our faster recipe.
   - L_per and L_dom act on the normalised **post-head** embedding, which is
     what ArcFace sees in our net.
2. **L_dom weight.**
   - `SupConLoss` multiplies by τ/base_τ = 0.02/0.07 ≈ 0.29, as OS-SCL's code
     (and SupCon's reference code) does. Eq. 7 has no such factor.
   - So with "equal weights" (Eq. 8), our L_dom counts ~0.29× relative to the
     paper's formula.
   - PerMix's code is unreleased, and it "follows OS-SCL", whose code has the
     factor. Kept, so the SupCon term is the same as `ao_nogain`'s.
3. **What is mixed.**
   - The wave and the cached **dB log-mel** are both mixed linearly with the
     same weights, as `mixup_data` does. OS-SCL's code does the same, which is
     probably what PerMix inherits.
   - The Tgram/TFgram branches see the true mixed wave; the Log-Mel channel
     sees a mixture of dB values.
4. **wave_aug noise,shift** runs per clip before mixing (`ao_nogain`'s
   recipe). The paper has no augmentation.
5. **Scoring.**
   - The paper scores by 1-NN cosine distance (Eq. 9). It does not say
     whether the memory bank is per ID or global, or which embedding it uses.
     Its encoder has no head, so its z is the MobileFaceNet output.
   - We gate on **S1′**, and **K** is reported. K = the 1-NN cosine distance
     of the L2-normalised **pre-head** embedding to the clip's own ID's
     training-part clips, i.e. S1′'s kNN component without z-scoring. That
     makes K the closest analogue of the paper's rule.
6. **Checkpoint selection.** The minimum CE on the 5% validation split (EMA,
   unmixed), as in every run here. The paper does not say how it selects.
7. **Fast recipe.** fp16 autocast for the branches + MobileFaceNet, with the
   head, ArcFace and both losses in fp32. Also `cudnn.benchmark`, `--lean`,
   cached mels and `--num_workers 8`.
8. **The short last batch** (36 clips) is mixed like the others, with
   R ≤ B − 1.

### Flag-off equality (2026-11-05)

`eqtest.py` is `../spectrum-branch/eqtest.py` with `--permix`. It seeds
2040, then runs 3 real `Trainer.train` batches from `data/cache`: wave_aug
noise+shift, mixup, SupCon, EMA, then one validation. It runs once in this
worktree with the flag off and once in main's checkout (bb02ced):

- CPU, batch 8: **444/444 tensors `torch.equal`** (net + EMA), and the valid
  loss is identical (18.789770).
- GPU 1, fp16 autocast, batch 16, deterministic cudnn: **444/444 tensors
  `torch.equal`**, and the valid loss is identical (18.322233).

With the flag on (GPU, batch 64), the loss is finite: L_per 11.00,
L_dom 3.61, mean λ 0.715.

### Smoke and bench (2026-11-05)

- **Smoke** (`pm_smoke_s1_e2`, GPU 2): seed 1, the full `pm` recipe, 2
  epochs, probe every epoch.
  - Training finished (rc 0, 329 s) with no skipped fp16 steps.
  - Epoch 0: train loss 11.47 = L_per 8.66 + L_dom 2.82, and mean λ 0.719
    (the simulation gave 0.716). The probe logged both epochs.
  - Epoch 1 did not improve the valid loss, so `model.pth` is epoch 0.
  - `eval.py --d` on it (batch-1 `train=False` path) gives dev AUC / pAUC /
    mAUC 0.48505 / 0.50024 / 0.42476. The probe's epoch-0 row is 0.48547 /
    0.49953 / 0.42383.
  - Chance level is expected, because the EMA net is still near init. The
    ~5e-4 gap is float noise on an almost constant embedding, not a scoring
    difference. `../spectrum-branch`'s smoke agreed to 1e-5, so the first
    scored stage-1 run is cross-checked the same way (S0 vs `eval.py`, < 5e-4
    per ID).
- **Bench** (`scores/bench.txt`, GPU 2): the full `Trainer.train` step at
  batch 64 on random inputs. That covers wave_aug, mixing, forward, loss,
  backward, AdamW and EMA, under fp16, cudnn.benchmark and lean.
  - 2 reps per arm: **4.39 it/s with PerMix vs 4.39 without**.
  - Peak memory is the same (5.7 GB).
  - The mixing loop (64 numpy draws per step) and the 4-slot weighted sum
    are lost in the noise. The paper reports +3.3% per epoch.

## Design

- **Arm `pm`:** the `ao_nogain` recipe plus `--permix 1 --permix_a 0.5
  --permix_r 3`. That is `--m 0.4 --fussion 1 --ht basic --num_workers 8
  --amp fp16 --cudnn_bench 1 --lean 1 --cache_dir data/cache --probe_every 5
  --wave_aug 1 --wave_aug_ops noise,shift`, 300 epochs. `run_one.sh` has the
  exact command.
- **Stage 1: seeds 2040–2043.** Each is paired with the existing
  `ao_nogain_s2040–2043` (`../aug-ops/scores/`; trained on this host with this
  recipe, not retrained).
  - The pairing is by **seed: init, split and epoch-0 batch order**.
  - PerMix draws from numpy instead of mixup's CPU `torch.randperm`, so the
    torch stream (batch order) diverges from epoch 1 on.
  - smartlab1 is not bit-deterministic over a full run anyway.
- **Stage 2** (only if D1 passes): fresh seeds **2044–2047**, paired with
  `ao_nogain_s2044–2047`. Their scores and S1′ dumps already exist.
- **Scoring:**
  - **S1′ is primary.** It is the repo's adopted single-model rule
    (`../confirm-seeds`). It contains the kNN term the paper scores with, so a
    gain that shows only under kNN scoring is not missed.
  - `../spectrum-branch` showed that R0 can mis-calibrate an embedding change
    (eval ToyCar −1.7 under R0, +0.15 under S1′). PerMix reshapes the target
    logit itself, so that risk is larger here.
  - **R0 is a guard.** `ao_nogain`'s standing claim, "beats the paper's dev
    numbers as one network under R0", must not be lost.
  - **K** (the paper-style kNN) is reported only.
  - S1′ and K come from `../ensemble-backend/extract.py` dumps through
    `score_s1p.py`, against `ao_nogain`'s dumps from `../aug-ops`. S0
    recomputed from the dumps must match `eval.py` per ID (< 5e-4).
- **Cost:** the bench shows no step-time change, so a run takes about as long
  as `ao_nogain`: 11.2–12.0 h on GPUs 0/2, 13.6–14.2 h on GPU 1.
  - Stage 1 (4 runs on 3 GPUs, the fourth queued) is about **24–26 h wall,
    ~50 GPU-hours**.
  - Stage 2 costs the same.
  - Lean checkpoints are small.

## Noise reference

`ao_nogain`, 8 seeds, per-seed sd:

| | S1′ | R0 |
|---|---|---|
| dev AUC | 0.19 | 0.23 |
| dev pAUC | 0.48 | 0.47 |
| dev mean(AUC, pAUC) | 0.32 | 0.33 |
| eval mean(AUC, pAUC) | 0.58 | 0.60 |
| dev ToyConveyor (4 seeds) | 0.72 | 0.57 |
| eval ToyConveyor (4 seeds) | 1.75 | 2.15 |
| dev pump (4 seeds) | 0.59 | 0.65 |

The inits are not paired, so a 4-pair mean Δ has se ≈ √2·sd/2 = 0.71·sd:

- S1′ dev AUC ≈ 0.13
- dev mean ≈ 0.23
- eval mean ≈ 0.41
- dev ToyConveyor ≈ 0.5
- eval ToyConveyor ≈ 1.2–1.5

## Predictions (written 2026-11-05, before any training run)

Δ = `pm` − `ao_nogain`, the mean over the 4 seed pairs 2040–2043, in AUC
points. Reference values for `ao_nogain` are in the Literature table. Under
S1′: dev ToyConveyor 87.35, ToyCar 98.07, pump 95.50; eval mean 93.73, eval
ToyCar 94.98, eval ToyConveyor 87.19.

The paper's +1.6 dev AUC is shrunk hard, for three reasons:

- Its baseline is 2 points below ours.
- Against the *published* OS-SCL its gain is +0.18.
- Wave augmentation already gives our net a perturbation pretext. Recent
  changes on this recipe moved it by tenths at most: `../tta` +0.05 S1′ eval;
  `../spectrum-branch` and `../trunk-soup` failed their gates.

| | quantity | central | 80% interval |
|---|---|---|---|
| **P1** | S1′ dev AUC Δ | **+0.10** | [−0.35, +0.50] |
| P2 | S1′ dev pAUC Δ | +0.15 | [−0.50, +0.80] |
| P3 | S1′ dev ToyConveyor Δ | +0.5 | [−1.5, +2.5] |
| P3 | S1′ dev ToyCar Δ | +0.1 | [−0.5, +0.6] |
| P4 | S1′ dev pump Δ | −0.2 | [−1.0, +0.6] |
| P5 | S1′ eval mean(AUC, pAUC) Δ | +0.1 | [−0.7, +0.9] |
| P5 | S1′ eval ToyCar Δ | +0.3 | [−0.8, +1.4] |
| P5 | S1′ eval ToyConveyor Δ | +0.5 | [−2.0, +3.0] |
| P6 | R0 dev AUC Δ | −0.05 | [−0.45, +0.35] |
| P6 | R0 eval mean Δ | 0.0 | [−0.8, +0.8] |
| P7 | K dev AUC Δ minus S1′ dev AUC Δ | +0.15 | [−0.3, +0.6] |

- **P1.** P(Δ ≥ +0.20, i.e. D1(a)) ≈ 30%.
- **P3.** The paper's gains are in ToyCar and ToyConveyor. Our ToyCar dev is
  at 98, near its ceiling.
- **P4.** In the paper, PerMix's pump is 1.6 below the published OS-SCL.
- **P7.** The objective is built for kNN scoring, so it should help K more
  than S1′.
- **P8, training.** The batch-mean λ in the EPOCH lines stays at
  0.72 ± 0.01 (a property of the mixing, not of learning). The EMA valid CE
  at the selected epoch ends within ±15% of `ao_nogain`'s on the same seed.
  No fp16 `skip` lines appear after epoch 5.

## Decision rules

- **D1, stage 1 → stage 2.** All of these must hold (4 seed pairs; `score.py`
  prints the checklist):
  - (a) S1′ dev AUC Δ ≥ **+0.20** (≈ 1.5 se), Δ > 0 on ≥ 3 of 4 seeds, and
    S1′ dev pAUC Δ ≥ 0;
  - (b) S1′ eval mean(AUC, pAUC) Δ ≥ −0.20;
  - (c) R0 guard: R0 dev AUC Δ ≥ −0.20, and R0 eval mean(AUC, pAUC)
    Δ ≥ −0.30;
  - (d) S1′ dev ToyConveyor Δ ≥ −0.80.

  Under no effect, (a) passes about 8% of the time; stage 2 guards against
  that.
- **D0.** If D1 fails, stop. `--permix` is not adopted. Report which
  criterion failed, and the per-machine Δ under S1′, R0 and K.
- **D2, adoption** (stage 2, fixed now). `--permix 1` joins the single-model
  recipe if all hold:
  - fresh seeds 2044–2047: S1′ dev AUC Δ ≥ +0.10;
  - all 8 pairs: S1′ dev AUC Δ has mean − 1.96·se > 0;
  - all 8 pairs: S1′ eval mean Δ ≥ 0 (point estimate);
  - (b), (c) and (d) hold on the fresh seeds.

  Report the 8-seed `pm` numbers against the PerMix paper's TFSTgram-MFN row.
  Those are numbers, not claim criteria.
- **D3, no claims.** K, eval per ID, and whether `pm` models help an E2′
  ensemble are reported only. An ensemble with them would need its own
  pre-registration.

## Confounds

- **Two changes in one arm.** PerMix bundles the mixing (1–3 clips,
  Dirichlet) with the objective (λ-scaled margin BCE + dominance SupCon). A
  D1 pass credits the package, not either part.
  - The paper's Table 2 separates L_per from L_dom but not the mixing from
    L_per.
  - A follow-up could run "PerMix mixing + os_scl", or "pairwise mixup with
    max(λ, 1−λ) + L_per", if the package wins.
- **Kept head, kept SupCon factor** (Deviations 1–2). A null here does not
  refute the paper's headless setting.

## Files

| file | what |
|---|---|
| `run.sh` | `train` / `score` / `s1p` / `plots`; all resumable |
| `run_one.sh` | one `pm` run on `../train_one.sh` (locks under `/tmp/os-scl-permix-locks`) |
| `score_s1p.py` | S0/S1′/K on `extract.py` dumps, paired with `../aug-ops` dumps → `scores/s1p.csv` |
| `score.py` | 4-seed means, paired Δ with se and per-seed Δ under S1′, R0 and K; D1 checklist → `scores/summary.txt` |
| `make_plots.py` | figures from `scores/` only; skips any figure whose inputs are missing |
| `eqtest.py` | the flag-off equality test |
| `bench.py` | full-step rate with and without PerMix → `scores/bench.txt` |

## Figures

| file | shows |
|---|---|
| `plots/fig1_machines.png` | paired Δ AUC per machine and overall, dev and eval, under S1′ and R0 (seeds and mean) |
| `plots/fig2_overall.png` | overall dev/eval AUC and pAUC for both arms under S1′, R0 and K, against the PerMix paper's numbers |
| `plots/fig3_probe.png` | dev-test probe over training (R0): overall, ToyCar, ToyConveyor; mean over seeds |

(Rendered once stage-1 scores exist.)

## Results

Pending: stage 1 is training.
