# Time-pooled head: does tying `linear7` over time help, now that we know position enters there?

> **Status: DONE 2026-11-07. D1 FAILS on (a) and (d) → D0: stop, `--tpool` not adopted, no stage 2.**
> Under C1, dev Amean Δ −0.21 (2/4 seeds ≥ 0), eval +0.10; dev ToyConveyor AUC −1.73. Tying `linear7`
> over time cuts ToyCar's roll penalty from 27 / 41 sd to 1.7 / 3.7 sd, but loses accuracy
> rather than gaining it: eval ToyCar id_05 *falls* (−1.58 C1, −5.22 R0). See *Results*.
> Predictions and decision rules below were committed (416a24d) before any `tp` training run.

## Why this exists

`../pooled-knn` (merged at e919b3d; `scores/diag.json`) found where ToyCar's dependence on
absolute time position enters the network:

- Circularly rolling a ToyCar test clip moves the current single model's score (S1′) by
  **27 sd (dev) and 44 sd (eval)** at the farthest rolls, and dev ToyCar AUC falls 98.0 → 54.1.
- kNN on the **time mean** of `mobilefacenet.conv2`'s output (the 512 × 8 × 20 map,
  channels × frequency × time, that feeds `linear7`) moves only **0.13 / 0.20 sd**.
- Between that map and the embedding there are only `linear7`, a 1×1 conv and the head.
  `linear7 = ConvBlock(512, 512, (8, 20), dw=True, linear=True)` is a depthwise conv whose
  kernel has **one weight per (frequency, time) cell**. It is the only layer there that can
  depend on position.

Adding that time-mean kNN to the score gave the current single-model rule,
**C1 = S1′ + zGmean**: +0.19 dev / +0.21 eval Amean (`../pooled-knn`), mostly ToyCar
(eval id_05 +2.5) and ToyConveyor. That is a scoring-side fix: C1 *adds* a position-free view
next to two terms that still carry the position weighting (C1's own roll penalty is 24 / 36 sd).

This ablation makes the fix on the network side instead. It removes the time axis from
`linear7`, so the network never sees absolute time position through that layer.

**It is a real bet, not a sure thing.** Position also *helps* some IDs. On dev ToyCar id_03,
S1′ scores 99.7, but time-mean kNN alone scores 88.6 (8-seed means). A network that cannot
weight time positions may lose that, unless it finds another cue.

## The arm: `tp` (`--tpool 1`, off by default)

`linear7` with its weights **tied across time**. Per channel c:

```
flag off:  y_c = BN( Σ_f Σ_t W[c, f, t] · x[c, f, t] )      W: 512 × 8 × 20
flag on:   y_c = BN( Σ_f w[c, f] · mean_t x[c, f, t] )        w: 512 × 8
```

Implementation (`model/net.py`, `MobileFaceNet`): `forward` takes `x.mean(3, keepdim=True)`
after `conv2`, and `linear7` becomes a depthwise `(8, 1)` conv on that, followed by the same
BN. The output is still 512 × 1 × 1, so `linear1`, the FPH head, ArcFace, SupCon and the EMA
are unchanged. Equivalently, it is an `(8, 1)` conv applied first and then averaged over time
(the two are linear and commute).

**Why this form, and only this one:**

- **Minimal change.** Among per-channel linear functionals of the map, the ones invariant to a
  circular time shift are exactly those whose weights are constant over t. So this is the
  `(8, 20)` layer with only the time-position degrees of freedom removed. Frequency weighting
  per channel stays. In `../pooled-knn`, per-band time-mean kNN alone was 1.0 dev AUC above
  the global time mean (94.43 vs 93.40), so the band axis carries something worth keeping.
- **Mean, not max.** In `../pooled-knn`, the time max of this map was 4 AUC below the time
  mean as a kNN feature, and only *reduced* the roll sensitivity (1.3 / 2.5 sd) where the mean
  *removed* it.
- **Not attentive statistics pooling.** ASP's weights come from the content, softmaxed over
  t. That is shift-equivariant, so it would not bring position back by itself. But it changes
  three things at once: a learned nonlinear weighting, an added std statistic, and doubled
  width into `linear1` (or a new projection). It is a second hypothesis ("content-weighted
  pooling helps"), not this one ("position weighting hurts or helps"). It is left for later,
  and only if `tp` is informative.

**Parameters.** 1,377,535 → 1,299,711 (−77,824 = 512 × 8 × 19, all of them time-position
weights): 5.6% fewer.

**Initialisation keeps the pairing as close as it can be.** The `(8, 20)` layer is still built
and drawn as usual. The tied kernel is then set, deterministically, to that draw summed over
time, i.e. the `(8, 20)` kernel's response to a time-constant map. Its own construction draws
happen inside `fork_rng` and are discarded. Checked on CPU at seed 2040 (net + EMA net): every
other tensor is `torch.equal` to the flag-off init, and the RNG state after both nets is
identical. So the train/valid split, every other init, and the first batch order and mixup
draws are the same as `ao_nogain`'s at the same seed. (smartlab1 is not bit-deterministic
over a full run anyway, and `cudnn_bench` is on, so the trajectories still diverge; see
`CLAUDE.md`.)

**Robustness of the claim "position-free".** Only `linear7` is made position-free. The map
before it is not exactly roll-equivariant:

- Every conv zero-pads at the clip edges.
- Tgram's `LayerNorm(313)` has a **per-frame affine**, a learned absolute-position input.
- The roll moves a splice into the clip.

In the `ao_nogain` nets these contributed almost nothing (the 0.13 sd above). A network
*trained* without `linear7`'s position weights could learn to route position through them
instead. That is why the roll diagnostic is repeated on `tp` (P10).

Other code: `eval.py` reads `tpool` from the checkpoint's `config.json`, and `soup.py`
lists it as an architecture key. `../ensemble-backend/extract.py` and `../pooled-knn/extract.py`
build the net from `config.json`, so they follow the flag. The state_dict keys are unchanged;
only `mobilefacenet.linear7.conv.weight` changes shape (512, 1, 8, 1). Loading an `ao_nogain`
checkpoint into a `tp` net, or the other way round, fails loudly.

### Flag-off equality (2026-11-06)

`eqtest.py` (copied from `../spectrum-branch`) seeds as `train.py` does (seed 2040). It builds
net + EMA, runs `Trainer.train` for 3 real batches from `data/cache` (wave_aug noise+shift,
mixup, SupCon, EMA, fp16 autocast), then validates once, and saves both state_dicts. Run once
on this branch with the flag off and once on main's checkout (e919b3d):

- **GPU 0, batch 16, deterministic cudnn: 444/444 tensors `torch.equal`**, and the valid loss
  is identical (18.322233, the same value `../spectrum-branch` got on main f57b58a). A second
  main-vs-main run is also 444/444.
- **CPU is not usable as an equality test on this host today.** Main against main itself
  gives 173/444 equal tensors with `OMP_NUM_THREADS=1`. The diffs are ~1e-8, and the valid
  loss agrees to 1e-6. `../spectrum-branch` got 444/444 on CPU on 11-03. This was not
  investigated (CPU kernels or the faulty host RAM).

With the flag on, the same 3 CPU steps give a finite loss (train 14.66, valid 18.79).

### Smoke and bench (GPU 1 and 2, shared with `permix` training, 2026-11-06)

- **Smoke** (`tp_smoke_s1_e2`): seed 1, the full `tp` recipe, 2 epochs, probe every epoch.
  - **First attempt (10:01–10:29): failed, attributed to the host.** Two attempts each
    finished both epochs and both probe rows, then died with a segfault at interpreter exit;
    the third died in epoch 1 with `cuDNN error: CUDNN_STATUS_BAD_PARAM` in backward. `dmesg`
    shows general-protection faults inside the `python3.11` binary (ip 0x4fa0da, 0x538ba8),
    the same signature and addresses as the host's earlier RAM crashes (0x4fa0df, 0x538ca0;
    15 such traps in the log). Before rerunning: a 400-step fp16 / `cudnn_bench` stress test
    of the `tp` net with batches 64, 46 and 2, plus eval-mode batch 1/7/32, ran clean and
    exited cleanly.
  - **Rerun (10:33–10:44): rc 0**, no skipped fp16 steps, finite losses (train 12.16, valid
    19.15 at epoch 0), probe logged both epochs.
  - `eval.py --d` on the checkpoint (batch-1 `train=False` path, `tpool` read from
    `config.json`) loads strict and wrote its CSV: dev AUC / pAUC / mAUC 0.47987 / 0.50335 /
    0.38417. The process then segfaulted at exit (GP fault in `python3.11` again,
    ip 0x538b41), after the CSV was written.
  - The probe row for the same weights (epoch 0) is 0.47623 / 0.50070 / 0.35831. This does
    **not** match to 1e-5 as `../spectrum-branch`'s smoke did, and that was checked: on 48
    dev-test clips the batched and batch-1 R0 differ by ≤ 1.5e-5, while R0's sd across clips
    is 1.4e-5 (the 2-epoch EMA net's embedding is almost constant, sd 1.6e-6). So both AUCs
    rank numerical noise. On a trained `ao_nogain` net the same check gives 1.2e-3 against an
    sd of 4.4. The real path check is `score.py`'s P0 on the trained `tp` nets (dumped R0 vs
    eval.py per ID, < 5e-4).
- **Bench** (`scores/bench.txt`; random inputs, batch 64, fp16, lean, GPU 2 while `permix`
  trained on it, so absolute rates are halved and noisy): 2.56 / 2.82 it/s with `--tpool`
  against 2.54 / 2.49 without, interleaved. **Not slower**; peak memory is the same (5.7 GB).
  The removed compute (one depthwise conv over 160 cells) is negligible, so wall-clock should
  match `ao_nogain`'s.

## Design

- **Arm `tp`:** the `ao_nogain` recipe plus `--tpool 1`: `--m 0.4 --fussion 1 --ht basic
  --num_workers 8 --amp fp16 --cudnn_bench 1 --lean 1 --cache_dir data/cache --probe_every 5
  --wave_aug 1 --wave_aug_ops noise,shift`, 300 epochs. `run_one.sh` has the exact command.
- **Stage 1: seeds 2040–2043**, each paired with the existing `ao_nogain_s2040–2043` (trained
  on this host with this recipe and the same `num_workers`; scores in `../aug-ops/scores`,
  not retrained).
- **Stage 2** (only if D1 passes): seeds **2044–2047**, paired with `ao_nogain_s2044–2047`.
- **Scoring.** All three rules, computed by `score.py` with `../pooled-knn/score.py`'s `Bank`
  (imported unchanged) for both arms, from dumps made by the same extractor
  (`../pooled-knn/extract.py`; `ao_nogain`'s dumps are `../pooled-knn`'s own):
  - **C1 = S1′ + zGmean: the primary rule.** zGmean is computed for `tp` exactly as for
    `ao_nogain`: the global mean of `conv2`'s 512 × 8 × 20 output, the same tensor in both
    nets.
  - S1′ = zR0′ + zK′, and R0 (eval.py's score; `score.py` checks it against the committed
    eval.py CSVs per ID, < 5e-4). Both reported.
  - Amean = mean(AUC, pAUC), each averaged over the 6 machines. Per machine, and per ID for
    ToyCar and ToyConveyor.
  - The `ao_nogain` S1′ and C1 are recomputed and checked against `../pooled-knn`'s committed
    `confirm_metrics.csv` (< 1e-3 overall AUC; the identity test below gave ≤ 3.7e-5).
- **What tpool does to the C1 comparison.** In `tp`, the 128-d embedding is itself computed
  from the time mean: K is a learned per-band weighting of the same 512 × 8 band means, then
  projected by `linear1`. Gmean is the uniform-band mean of those same band means.
  - So for `tp`, C1's third term is no longer a *different* (position-free) view. It is a
    re-weighted, unprojected copy of what K already sees. The C1 − S1′ gain within `tp`
    should shrink (P9).
  - Under C1, the comparison is therefore "a position-free net, three position-free terms"
    against "a position-weighted net, two position terms plus one position-free term".
    `ao_nogain` already gets part of the position-free benefit at scoring time, so `tp`'s
    headroom is smaller under C1 than under R0 or S1′. That is why all three rules are
    reported.
- **Diagnostic (stage 1, all 4 seeds):** `../pooled-knn`'s ToyCar roll test on both arms. Every
  ToyCar test clip is circularly rolled by k/8 of the clip, k = 0…7; bank and z-statistics
  stay unrolled. Penalty = rise of the normal clips' score, in sd of the unrolled score;
  headline = mean over k = 3, 4, 5. Verdict on `tp`'s S1′, against `ao_nogain`'s S1′ on the
  same seeds: *removes* (≤ 1 sd), *reduces* (≤ 25% of `ao_nogain`'s), *does not* (otherwise).
- **Cost.** `ao_nogain` took 11.2–12.0 h per run on GPUs 0/2 and 13.6–14.2 h on GPU 1. `tp`
  removes a negligible amount of compute (see bench). Stage 1 ≈ 4 × 12–14 h ≈ 50 GPU-hours.
  It runs as GPUs free up behind `permix`.

### Scoring pipeline check (2026-11-06)

`score.py` was run with `ao_nogain_s2040`'s dumps standing in for both arms (scratch, not
kept): every Δ is 0.00, R0 matches eval.py to 1.1e-4 per ID, and S1′ / C1 match
`../pooled-knn` to 3.7e-5. The diagnostic gives 27.78 / 40.54 sd for seed 2040 S1′, which is
`../pooled-knn`'s `diag_roll.csv` for that seed.

## Noise reference

`ao_nogain`, seeds 2040–2043 (from `../pooled-knn`'s dumps), per-seed sd:

| | C1 dev | C1 eval | S1′ dev | S1′ eval | R0 dev | R0 eval |
|---|---|---|---|---|---|---|
| Amean | 94.18 (0.31) | 93.91 (0.80) | 94.05 (0.35) | 93.73 (0.76) | 93.67 (0.37) | 93.04 (0.70) |
| ToyCar AUC | 98.45 (0.10) | 95.79 (0.26) | 98.07 (0.14) | 94.98 (0.29) | 97.88 (0.21) | 93.48 (0.41) |
| ToyConveyor AUC | 87.58 (0.71) | 88.15 (1.57) | 87.35 (0.72) | 87.19 (1.75) | 85.96 (0.57) | 83.18 (2.15) |
| ToyCar id_03 (dev) / id_05 (eval) | 99.63 (0.08) | 87.05 (0.56) | 99.67 (0.08) | 84.50 (0.40) | 99.59 (0.13) | 81.60 (1.30) |

The two arms do not share a trajectory, so a 4-pair mean Δ has se ≈ √2 · sd / 2 ≈ 0.71 · sd:
**≈ 0.22 for dev Amean and ≈ 0.55 for eval Amean under C1.** Per-ID ToyConveyor is noisier
still (eval per-seed sd 2–3).

## Predictions (written 2026-11-06, before any `tp` training run)

Δ = `tp` − `ao_nogain`, mean over the 4 seed pairs (2040–2043), in points. Intervals are 80%.

- **P1, dev Amean, C1 (primary).** Δ **−0.10** [−0.55, +0.30]. The dev ToyCar loss (P4, P5)
  outweighs a small ToyConveyor/ToyCar-id_01 gain.
- **P2, eval Amean, C1.** Δ **+0.05** [−0.65, +0.70]. Eval ToyCar gains (id_05, P6), and the
  interval is mostly seed noise.
- **P3, R0 and S1′ (secondary).**
  - R0: dev Amean Δ −0.05 [−0.55, +0.40]; eval Amean Δ **+0.25** [−0.50, +1.00]. R0 had the
    largest roll penalty on eval (50 sd), so it has the most to gain there.
  - S1′: dev Δ −0.05 [−0.50, +0.35]; eval Δ +0.15 [−0.55, +0.85].
  - Δ under R0 > Δ under C1 on eval ToyCar AUC (70%).
- **P4, ToyCar AUC, C1.**
  - dev: `tp` **97.6** [95.8, 98.6] (ao 98.45).
  - eval: `tp` **96.0** [94.6, 97.0] (ao 95.79).
  - Under R0, eval ToyCar Δ +1.0 [−0.5, +2.5].
- **P5, dev ToyCar id_03, C1.** `tp` **95.5** [88.0, 99.6] (ao 99.63). It falls by more than
  1 point: 65%. This is the ID where position helped most.
- **P6, eval ToyCar id_05, C1.** `tp` **88.0** [85.0, 90.5] (ao 87.05). Under R0: 85.0
  [81.0, 89.0] (ao 81.60).
- **P7, ToyConveyor AUC, C1.** dev Δ 0.0 [−1.2, +1.2]; eval Δ +0.3 [−2.5, +3.0].
- **P8, non-Toy machines** (fan, pump, slider, valve; mean of their Amean), C1: dev Δ 0.00
  [−0.25, +0.25]; eval Δ 0.0 [−0.6, +0.6]. These machines are near-stationary or have events
  at random times, so position weights should carry little for them.
- **P9, the pooled term is redundant in `tp`.** Within-arm C1 − S1′ dev Amean: `tp`
  **+0.04** [−0.08, +0.15], against `ao_nogain`'s +0.13 on the same seeds. `tp`'s is the
  smaller: 80%.
- **P10, diagnostic.** `tp`'s own S1′ ToyCar headline roll penalty: dev **0.5 sd** [0.05, 4],
  eval **0.6 sd** [0.05, 5] (`ao_nogain`: ≈ 27 / 44).
  - Verdict: *removes* 65%, *reduces* 30%, *does not* 5%.
  - `tp`'s S1′ ToyCar AUC at k = 4 is within 1.5 points of k = 0 on both sets: 70%.
  - Residual penalty, if any, comes from Tgram's per-frame LayerNorm affine and the edge
    padding (see *Robustness* above).

## Decision rules

- **D1, stage 1 → stage 2** (4 pairs, rule C1). All of:
  - (a) dev Amean Δ ≥ **+0.05**, or (dev Amean Δ ≥ −0.10 **and** eval Amean Δ ≥ +0.30);
  - (b) eval Amean Δ ≥ −0.20;
  - (c) non-Toy (fan, pump, slider, valve) mean Amean Δ ≥ −0.20 on dev and ≥ −0.40 on eval;
  - (d) dev ToyConveyor AUC Δ ≥ −1.0.

  `score.py main` prints the checklist. Under no effect, (a) passes about 40% of the time.
  Stage 2 is the guard against that.
- **D0.** If D1 fails, stop. `--tpool` is not adopted. Report which criterion failed, the
  diagnostic, and ToyCar/ToyConveyor per ID under all three rules.
- **D2, adoption** (after stage 2, seeds 2044–2047; 8 pairs, rule C1). `--tpool 1` joins the
  single-model recipe if all hold:
  - (a) fresh-seed (2044–2047) dev Amean Δ > 0;
  - (b) 8-pair dev Amean Δ ≥ **+0.10**, positive on ≥ 5 of 8 seeds;
  - (c) 8-pair eval Amean Δ ≥ **0.00** (point estimate);
  - (d) 8-pair non-Toy mean Amean Δ ≥ −0.15 on dev and ≥ −0.30 on eval;
  - (e) 8-pair S1′ Amean Δ ≥ −0.10 on dev and on eval. The E2′ ensembles sum S1′, so a
    network that wins only through the pooled term must not lose there.
- **D3, no further claims.** mAUC is reported but not read (single-model deltas under ~2
  points; `CLAUDE.md`). The scoring rule for `tp` nets stays C1 whatever P9 shows: C1 vs S1′
  on `tp` nets would need its own pre-registration. Ensembles with `tp` members, and composition
  with TTA, are not tested here.

## Results (stage 1, seeds 2040–2043, 2026-11-07)

Four `tp` runs, 300 epochs each, no crashes (40.5–48.8 ks wall; s2043 on the slow GPU 1).
Scored with `score.py main` on the dumps from `extract.py`. P0 check: R0 recomputed from the
dumps matches `eval.py` to ≤ 1.4e-4 per-ID AUC, and `ao_nogain`'s S1′/C1 match `../pooled-knn`
to ≤ 4.8e-5. Full table: `scores/scorecard_2040_2041_2042_2043.txt`.

### D1 (rule C1, `tp` − `ao_nogain`, 4 pairs)

| criterion | value | pass |
|---|---|---|
| (a) dev Amean Δ ≥ +0.05, or (dev ≥ −0.10 and eval ≥ +0.30) | dev **−0.21** (se 0.23), eval +0.10 (se 0.29) | **no** |
| (b) eval Amean Δ ≥ −0.20 | +0.10 | yes |
| (c) non-Toy Amean Δ ≥ −0.20 dev / ≥ −0.40 eval | +0.21 / +0.30 | yes |
| (d) dev ToyConveyor AUC Δ ≥ −1.0 | **−1.73** (se 0.85; per seed −3.49 −1.26 +0.42 −2.59) | **no** |

**D0 applies.** `--tpool` stays off by default and stage 2 is not run.

### Amean by rule (Δ, se)

| rule | dev | eval |
|---|---|---|
| R0 | **−0.51** (0.28) | −0.34 (0.23) |
| S1′ | −0.22 (0.22) | +0.02 (0.24) |
| C1 (primary) | −0.21 (0.23) | +0.10 (0.29) |
| zK′ alone (embedding kNN) | **+0.26** (0.11), 4/4 seeds | +0.27 (0.39) |
| zGmean alone (time-mean kNN) | +0.24 (0.17) | −0.42 (0.27) |

The loss is in the **classifier score**. R0, the CE of the own-ID logit, loses 0.5 dev Amean and
4.06 points on dev ToyConveyor. The kNN on `tp`'s embedding is *better* than `ao_nogain`'s
(+0.26 dev, 4/4 seeds; +1.00 dev mAUC). S1′ = zR0′ + zK′ averages the two to a wash. So
position weighting at `linear7` helps the network tell machine IDs apart, which is what the
CE score measures. It does not help the geometry of the embedding.

### Per machine (C1 AUC Δ)

| | fan | pump | slider | ToyCar | ToyConveyor | valve |
|---|---|---|---|---|---|---|
| dev | −0.02 | +0.67 | −0.07 | −0.26 | **−1.73** | +0.05 |
| eval | −0.13 | +0.05 | +0.01 | −0.49 | −0.65 | +0.84 |

- **ToyCar loses on both sets and every seed** (dev −0.19…−0.30, eval −0.23…−0.91). The ID
  the bet was about went the wrong way. Eval **id_05 is 85.48 vs 87.05** under C1 (−1.58, 4/4
  seeds negative). Under R0 it is 76.39 vs 81.60 (−5.22). Dev id_03 falls 99.63 → 98.82
  (−0.81, 4/4 seeds).
- **ToyConveyor loses on five of six IDs under every rule.** Eval id_04, the easy one, is level
  (C1 +0.06). It is worst on dev id_02/id_03 (R0 −5.2 / −6.0).
- Non-Toy machines gain slightly: pump +0.67 dev (C1), valve +0.84 eval (seed spread ±2).
  Both are inside noise.

### Diagnostic: the roll penalty (`score.py diag`, `scores/diag.json`)

ToyCar normal-clip penalty at rolls of 37.5–62.5% of the clip, in sd of the unrolled score:

| | dev S1′ | eval S1′ | dev C1 | eval C1 |
|---|---|---|---|---|
| `ao_nogain` | 27.29 | 40.56 | 24.55 | 33.12 |
| `tp` | **1.65** | **3.66** | 1.66 | 3.06 |

The network change does what it was built to do: it cuts the penalty 16× on dev and 11× on eval.
It doesn't remove it. ToyCar S1′ AUC still falls from 97.9 to 89.5 (dev) and from 94.1 to 83.8
(eval) at the farthest roll. `ao_nogain` falls to 54.1 / 71.2. The residual is presumably (not tested) the Tgram
branch's per-frame LayerNorm affine, the edge padding of the convolutions, and the frame
borders the roll creates (README *Robustness*). The pooled map itself is nearly roll-free
(`tp` zGmean 0.56 / 0.67 sd).

### Predictions vs outcome

| | predicted (80% interval) | outcome | in interval |
|---|---|---|---|
| P1 dev Amean C1 | −0.10 [−0.55, +0.30] | −0.21 | yes |
| P2 eval Amean C1 | +0.05 [−0.65, +0.70] | +0.10 | yes |
| P3 R0 dev / eval | −0.05 [−0.55, +0.40] / +0.25 [−0.50, +1.00] | −0.51 / −0.34 | yes / yes |
| P3 S1′ dev / eval | −0.05 [−0.50, +0.35] / +0.15 [−0.55, +0.85] | −0.22 / +0.02 | yes / yes |
| P3 R0 Δ > C1 Δ on eval ToyCar (70%) | yes | −2.30 vs −0.49 | **wrong** |
| P4 ToyCar C1 dev / eval | 97.6 [95.8, 98.6] / 96.0 [94.6, 97.0] | 98.19 / 95.31 | yes / yes |
| P4 R0 eval ToyCar Δ | +1.0 [−0.5, +2.5] | −2.30 | **no** |
| P5 dev id_03 C1 | 95.5 [88.0, 99.6]; drop > 1 point (65%) | 98.82; drop 0.81 | yes; wrong |
| P6 eval id_05 C1 / R0 | 88.0 [85.0, 90.5] / 85.0 [81.0, 89.0] | 85.48 / 76.39 | yes (edge) / **no** |
| P7 ToyConveyor C1 dev / eval | 0.0 [−1.2, +1.2] / +0.3 [−2.5, +3.0] | −1.73 / −0.65 | **no** / yes |
| P8 non-Toy C1 dev / eval | 0.00 [−0.25, +0.25] / 0.0 [−0.6, +0.6] | +0.21 / +0.30 | yes / yes |
| P9 `tp` C1 − S1′ dev; smaller than `ao`'s (80%) | +0.04 [−0.08, +0.15] | +0.14 (ao +0.13) | yes; wrong |
| P10 penalty dev / eval | 0.5 [0.05, 4] / 0.6 [0.05, 5] sd | 1.65 / 3.66 | yes / yes |
| P10 verdict | removes 65%, reduces 30% | reduces | (the 30% branch) |
| P10 k = 4 within 1.5 points of k = 0 (70%) | yes | −8.4 / −10.2 | **wrong** |

Most interval predictions held, but nearly every **directional** bet about the Toy machines
was wrong. I expected eval ToyCar, and id_05 above all, to gain once position was removed, with
R0 gaining most. Instead id_05 lost under every rule and R0 lost most. The pooled term was not
made redundant either (P9): C1 − S1′ is as large in `tp` (+0.14 dev, +0.26 eval) as in
`ao_nogain`.

### What this shows

- **Position at `linear7` is a cue the classifier uses well, not a bug.** Removing it hurts the
  ID-classification score (R0) on both Toy machines and both sets. The embedding-kNN score gets
  slightly better (dev +0.26, 4/4 seeds), and the composite S1′/C1 ends up at about zero.
- **ToyCar's eval id_05 failure is not caused by positional latching.** A network that is
  ~11× less position-sensitive does worse on id_05. `../pooled-knn`'s C1 gain on id_05 came
  from *adding* a position-free view next to the positional one, not from replacing it.
- **Scoring-side fusion beats the architectural fix.** C1 on `ao_nogain` (94.18 / 93.91) stays
  the best single-network rule. `tp` C1 is 93.97 / 94.01.

### What next (not pre-registered here)

- `tp` nets as **ensemble members**. Their embedding kNN is better and their errors differ
  (ToyCar penalty profile, pump/valve gains), so a `tp` + `ao_nogain` E2′/C1 mix may beat
  E2′[A8]. That needs its own pre-registration; D3 rules it out here.
- Nothing more on the architecture of the last layer. Roll-invariance is reached and buys nothing.

## Files

| file | what |
|---|---|
| `run.sh` | `train` / `score` / `extract` / `main` / `diag` / `plots`; all resumable |
| `run_one.sh` | one `tp` run on `../train_one.sh` (locks under `/tmp/os-scl-timepool-head-locks`, crash-retry) |
| `eqtest.py` | the flag-off equality test |
| `bench.py` | step rate with and without `--tpool` → `scores/bench.txt` |
| `extract.py` | `../pooled-knn/extract.py` pointed at this worktree's `check_points/` (CRC-verified dumps) |
| `score.py` | `main`: R0 / S1′ / C1 for both arms, paired Δ, D1 → `scores/metrics.csv`, `perid.csv`, `summary_<seeds>.json`, `scorecard_<seeds>.txt`; `diag`: ToyCar rolls → `scores/diag*.{csv,json}`, `scorecard_diag.txt` |
| `make_plots.py` | figures from `scores/` only; skips any figure whose inputs are missing |
| `scores/` | committed record: per-run `eval_{dev,eval}.csv`, `config.json`, `probe.csv`, `wall.csv`, `curve.csv`; ablation-level files above |
| `results/` | logs and dumps: gitignored scratch |

## Figures

All read `scores/` only (`make_plots.py`).

| file | shows |
|---|---|
| `plots/fig1_rules.png` | paired Δ Amean under R0, S1′, C1, dev and eval (dots = seeds) |
| `plots/fig2_machines.png` | paired Δ AUC per machine under C1, dev and eval |
| `plots/fig3_toy_ids.png` | ToyCar and ToyConveyor per-ID AUC, both arms, S1′ and C1 |
| `plots/fig4_toycar_roll.png` | ToyCar penalty and AUC vs circular roll: `tp` S1′, `tp` zGmean, `ao_nogain` S1′ |
| `plots/fig5_probe.png` | dev-test R0 probe over training, both arms (overall and ToyCar); `tp` trails from epoch 130 on |
