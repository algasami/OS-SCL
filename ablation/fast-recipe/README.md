# Fast recipe: a 2–4× faster training recipe, and how much test-set selection is worth

> **Status: done (runs scored 2026-10-19; written up 2026-10-23).**
>
> - **D1, adopt the fast recipe.** It leaves the paper model unchanged: `fr_f1`
>   − relu8 = +0.03 dev / −0.13 eval AUC, and P2 holds. It is 1.9× faster for
>   `fr_f1` and 3.0× for `fr_f4`. Every later ablation already uses it.
> - **D2 → Phase 1 builds on `fussion 1`, not 4.**
>   - Dropping TFgram costs −0.40 dev but gains +0.81 eval.
>   - D2 needs both sets to be ≥ −0.15, so the dev cost fails it.
> - **D3: test-set epoch selection is worth almost nothing.** It adds +0.02
>   dev to `fr_f1`, and the best probed epoch of any seed is 95.28, below the
>   paper's 95.71.
>
> Phase 0 of the post-confirm-seeds plan, pre-registered 2026-10-17.

## Why this exists

Two questions gate everything after confirm-seeds.

1. **Can a run take ~5 h instead of 21.5 h without changing what it learns?**
   Training is GPU-bound (CLAUDE.md). Step rates from `bench.py`
   (`scores/bench.txt`; GPU 1, random inputs, batch 64, forward + backward +
   AdamW, no data loading):

   | recipe | it/s | vs today | params |
   |---|---|---|---|
   | `--fussion 1`, cudnn deterministic, fp32 (today) | 2.51 | 1.0× | 81.2 M |
   | `--fussion 1`, cudnn.benchmark | 3.03 | 1.2× | 81.2 M |
   | `--fussion 1`, cudnn.benchmark + fp16 autocast | 5.08 | 2.0× | 81.2 M |
   | same, `--lean 1` (**`fr_f1`**) | 5.07 | 2.0× | 1.4 M |
   | `--fussion 4` (Log-Mel + Tgram), lean, benchmark + fp16 (**`fr_f4`**) | 10.75 | 4.3× | 1.2 M |
   | `--fussion 2` (Log-Mel), lean, benchmark + fp16 | 11.95 | 4.8× | 0.9 M |

   The lean parameter counts match the paper's Table V (TFSTgram 1.38 M,
   Log-Mel 0.88 M), so `--lean` drops only dead weight. The 80 M dead
   parameters cost no compute; they only inflate checkpoints and the EMA loop.

   TFgram is ~60% of the step time. Five ablations found it redundant with
   Log-Mel (n=4, `../seed-sweep`, `../tfgram-linear`).

2. **Is the paper's 95.71 dev AUC reachable with an honest protocol?** Our
   faithful reproduction gives 94.86 ± 0.19 dev AUC (relu8, n=8). The paper's
   own Table VI gives **94.94 for "w/o FPH"**, but the repo's head *is* the FPH
   (128→64→128, R=64), and SupCon τ=0.02 also matches. The paper reports one
   run, with no seeds and no checkpoint-selection rule. The upstream
   STgram-MFN trainer saves the epoch with the best **test** AUC+pAUC; we
   select on a 5% validation split. Logging dev-test AUC during training (never
   used for selection) measures how much picking the best epoch on the test
   set would add.

## Code (all off by default; old runs and commands are unchanged)

New modules: `cache_data.py` (root) builds and verifies the arrays, and
`probe.py` (root) holds the batched dev-test metrics. `trainer.py` and this
directory's `gate.py` both use it. `dataloader.py` gains module-level
`machine_ids`, `load_wav`, `test_index` and `cached_item_ok`. The two dataset
classes now call these instead of keeping their own copies. Flags:

| flag | effect |
|---|---|
| `--amp fp16` | autocast the branches and MobileFaceNet **in training steps only**; validation and the probe stay fp32. The early EMA net's BN statistics are still near init, so its activations overflow fp16 (the gate's smoke run got NaN scores). The head, ArcFace and both losses stay fp32: the margin math takes sqrt(1 − cos²) near \|cos\| = 1, and SupCon's 1/0.02 logits would overflow fp16. A GradScaler is used. |
| `--cudnn_bench 1` | `cudnn.benchmark=True`, `deterministic=False` |
| `--lean 1` | build only the branches the fussion mode uses, and none of TFgram's ~80 M dead parameters. Checkpoints shrink from 310 MB. The state_dict keys and the init draw order change. `eval.py` reads `lean` from the checkpoint's `config.json`. |
| `--cache_dir data/cache` | read the arrays `cache_data.py` wrote instead of decoding each wav and computing its mel per item. The values are identical (checked in G1). |
| `--probe_every 5` | every 5 epochs and at the last one, append the EMA net's dev-test metrics under eval.py's score (R0) to `check_points/<run>/probe.csv`. **Diagnostic only**: selection still uses validation loss. |

## Host RAM flips bits (found by the gate, 2026-10-17)

The first G3 smoke run of `fr_f1` NaN'd at step 59. MobileFaceNet's first conv
received a mel channel with |x| = 1.3e11. Two cached training mels were
corrupt: `ToyConveyor/train/normal_id_01_00000372` and
`ToyCar/train/normal_id_06_00000804`. Their wavs were intact, and a recompute
gave normal values.

A full verify pass (`cache_data.py --verify`: recompute every item, compare bit
for bit) then found 14 mismatched training items. 12 of them were in the
*waveform*, which is a plain file decode. A second pass found 3 more. Each
mismatch is a set of **single-bit flips**: 20–40 per item, spaced every 64 or
128 bytes inside a 5–16 KB span. The decode is deterministic: 3 × 8 reloads in
pool workers and the main process all match. After a correct rewrite, fsync,
and eviction of the pages, a re-read from disk is clean. That points to
**faulty host RAM**, which is non-ECC here. The flips land on whatever data
occupies the bad region; the GPUs report ECC on with 0 errors.

The old pipeline was exposed too, but less. Wavs sit in the page cache as
16-bit PCM, where a flip adds a bounded click, and mels were transient. A
float32 cache that sits in RAM for hours exposes exponent bits. So:

- every cached item is range-checked when read (`cached_item_ok`: finite,
  |mel| ≤ 200 dB, |wav| ≤ 1). On failure the item is recomputed from its wav and
  the event is logged. Mantissa flips pass this check; they are below the
  signal's own noise.
- under `--amp`, a training step with a non-finite loss is skipped and counted
  (`skip |` lines in the log).
- the probe applies the same check and catches its own errors, so it can't
  kill a run.

Whether these flips also contribute to smartlab1's run-to-run divergence
(CLAUDE.md, the distill Gate) is untested.

## Gate (before any 300-epoch run)

- **G1, cache identity.** `__getitem__` from the cache equals the live decode +
  mel, bit for bit, on 64 random training items and 64 dev-test clips. The live
  side runs single-threaded, like the DataLoader workers (`nw=8`) that compute
  it in training; multi-threaded mels differ by ~1e-5 dB.
- **G2, probe = eval.py.** The probe's batched path on an existing checkpoint
  (`cs_relu_max_s2028`, fp32) reproduces its committed `eval_dev.csv`: every
  machine AUC within 5e-4.
- **G3, smoke.** 2 epochs of each arm run, finish, write `probe.csv`, and
  `eval.py` loads the lean checkpoint strictly.

**Gate result (2026-10-17, `scores/gate.txt`, `scores/cache_verify.txt`): passed.**
G1: 0 difference on 128 items, right after a verify pass. G2: per-machine AUC
within 1.2e-5 of `eval.py` (the fp16 probe would be 2.7e-4 off; the probe runs
fp32). G3: both arms train, probe and reload strictly, with 5.4 MB and 4.7 MB
checkpoints. Epoch 0 took 162 s (`fr_f1`) and 103 s (`fr_f4`), against
257 s/epoch for relu8.

The verify pass right before G1 repaired **49 items: 42 train, 7 dev-test**.
They had all been clean ~30 min earlier, so the flips keep coming. A verify
pass cannot tell which side flipped: the cached copy, or the freshly computed
one held in a worker's RAM. That is why the range check on read, not the
verify pass, is the safeguard. Mantissa-level flips on ~0.1% of items are
far below the data's own noise.

## Arms

Fresh seeds 2040–2043, 300 epochs, `--num_workers 8`, `--m 0.4 --ht basic`,
all with `--amp fp16 --cudnn_bench 1 --lean 1 --cache_dir data/cache --probe_every 5`.

| arm | fussion | what it answers |
|---|---|---|
| `fr_f1` | 1 (paper, TFSTgram) | does the fast recipe change the paper model? |
| `fr_f4` | 4 (Log-Mel + Tgram) | is TFgram still redundant here, so Phase 1 can build on the 4× faster model? |

The reference is **relu8**: `tfl_relu_max_s2024–2027` and
`cs_relu_max_s2028–2031`, the paper model under today's recipe on this host.

```
relu8 R0     dev AUC  94.60 .. 95.15   mean 94.86  sd 0.19
             eval AUC 95.02 .. 95.86   mean 95.38
```

## Predictions (written before any run)

- **P1, speed.** Wall clock per epoch, from the logs: `fr_f1` ≥ 1.8× faster
  than relu8 (~4.3 min/epoch), and `fr_f4` ≥ 3.5× faster.
- **P2, the fast recipe is harmless.** `fr_f1` mean R0 dev AUC lies in
  [94.56, 95.16] (relu8 ± 0.3). Its eval AUC lies in [94.98, 95.78] (± 0.4).
  No seed is below the relu8 minimum by more than 0.3.
- **P3, TFgram stays redundant.** `fr_f4` − `fr_f1`, mean R0 dev AUC, lies in
  [−0.3, +0.4]. For reference, `../seed-sweep` measured f4 − f1 = +0.20 dev and
  +0.38 eval on the other host.
- **P4, test-set selection is worth little.** Per run, the dev AUC at the best
  probed epoch minus the dev AUC of the saved checkpoint (eval.py) averages
  +0.1 to +0.5. The best probed epoch over the four `fr_f1` seeds stays
  **below 95.71**: test-set selection alone does not explain the paper's
  number. EMA alpha 0.9999 is a ~19-epoch time constant, which smooths
  epoch-to-epoch jumps.
- **P5, the gap lives in ToyConveyor.** In P4's oracle gain, ToyConveyor
  contributes more than any other machine.

## Decision rules

- **D1.** If P2 holds, every later arm uses the fast recipe. Its paired baseline
  is trained on the fast recipe, never relu8.
- **D2.** If `fr_f4` mean dev and eval R0 AUC are both ≥ `fr_f1` − 0.15,
  Phase 1 builds on `fussion 4`. Otherwise it builds on `fussion 1` under the
  fast recipe.
- **D3.** P4 is reported, not decided on. If the oracle best-of-4 reaches 95.71,
  the README says so plainly: the paper's dev number is then within reach of
  test-set selection.

## Results

**Runs.** All 8 runs trained 2026-10-17 19:06 to 2026-10-19 01:22, one per
GPU (`run.sh` queue via `run_one.sh`).

- **Crash.** `fr_f4_s2040` died at epoch 214 with a general-protection fault
  in python (rc 139). That is the faulty host RAM; see above and CLAUDE.md.
  - This recipe has no crash-resume, so the run was redone from scratch with
    the same seed, and the recorded `fr_f4_s2040` is that rerun.
  - The first attempt's probe is kept in
    `scores/crashes/fr_f4_s2040_attempt1_probe.csv`. It was at 95.10 dev AUC
    (probe) when it died.
  - Because smartlab1 is not bit-deterministic, the rerun is another noise
    draw, not a copy.
- **Health.** The other 7 runs and the rerun logged 0 skipped (non-finite)
  steps and 0 cache range-check repairs.
- **Reconstructed timing.** `fr_f4_s2043`'s `wall.csv` was never written: the
  queue ended before its done line. It was rebuilt from its `config.json` and
  `probe.csv` mtimes (30 071 s), which agrees with the queue log, 17:00:39 to
  01:21:54.

### Speed (P1)

| arm | per run (h), by GPU | median | per epoch | vs relu8 (21.47 h) |
|---|---|---|---|---|
| relu8 (today's recipe) | 21.2–22.9 (GPUs not recorded) | 21.47 | 4.29 min | 1.0× |
| `fr_f1` | 11.11 (0), **13.48 (1)**, 11.79 (2), 11.07 (0) | 11.45 | 2.29 min | **1.87×** |
| `fr_f4` | 5.95 (0), **8.32 (1)**, 6.14 (2), **8.35 (1)** | 7.23 | 1.45 min | **2.97×** |

**P1 is half right.**
- `fr_f1` is 1.87× faster, which meets the ≥ 1.8× bar.
- `fr_f4` is 2.97× faster, short of the ≥ 3.5× bar.
- The bench's step rates (4.3× for f4) overstate the end-to-end gain:
  validation, the probe and data loading do not shrink with fp16.

**GPU 1 is slow** (found here, not pre-registered).
- Every run on GPU 1 took 20–40% longer than the same arm on GPUs 0 or 2.
  `margin-warmup` repeats the pattern: f6 took 13.1 h on GPU 1 against
  10.7–10.9 h elsewhere.
- `nvidia-smi` reports GPU 1's PCIe link at **x1** (max x16). GPU 2 is at x4
  and GPU 0 at x16. All three have the same power limit and clocks.
- This was read at idle and not re-checked under load. A batch of 64
  ten-second waveforms is ~41 MB per step, so an x1 link plausibly explains
  the gap.
- On GPUs 0 and 2 alone, `fr_f1` is 1.93× and `fr_f4` 3.55× faster, which
  would pass P1. As registered, with all four runs, P1 fails for `fr_f4`.
- **Practical rule:** put the longest arm on GPU 0, and don't compare
  wall-clock across GPUs.

### Accuracy (P2, P3)

eval.py R0 overall AUC; per-seed values in seed order (relu8 is s2024–2031).

| arm | dev AUC | mean (sd) | eval AUC | mean (sd) |
|---|---|---|---|---|
| relu8 | 94.97 94.60 95.06 94.66 95.15 94.94 94.79 94.74 | 94.86 (0.20) | 95.20 95.53 95.27 95.86 95.44 95.09 95.02 95.57 | 95.38 (0.28) |
| `fr_f1` | 94.52 95.26 94.70 95.10 | **94.90** (0.34) | 95.07 95.59 95.31 95.02 | **95.25** (0.26) |
| `fr_f4` | 94.10 94.63 94.86 94.39 | **94.49** (0.33) | 96.25 96.25 96.18 95.56 | **96.06** (0.34) |

Contrasts are differences of means, with Welch se and a bootstrap 95%
interval (20 000 resamples).

| contrast | dev AUC | eval AUC | dev pAUC | eval pAUC |
|---|---|---|---|---|
| `fr_f1` − relu8 | **+0.03** (0.19) [−0.28, +0.35] | **−0.13** (0.16) [−0.41, +0.17] | −0.10 (0.25) | −0.51 (0.21) |
| `fr_f4` − `fr_f1` | **−0.40** (0.24) [−0.80, −0.01] | **+0.81** (0.21) [+0.42, +1.13] | −0.15 (0.25) [−0.55, +0.30] | +1.09 (0.23) [+0.69, +1.45] |

mAUC is left out of the contrasts: its seed sd is 0.4–1.2, and per CLAUDE.md,
deltas under ~2 points are unresolvable. The means are in `make_plots.py`'s
inputs.

**P2 holds.**
- `fr_f1` dev 94.90 is inside [94.56, 95.16], and eval 95.25 is inside
  [94.98, 95.78].
- The lowest seed, 94.52, is 0.08 below relu8's minimum (94.60), inside the
  0.3 allowance.
- The fast recipe is indistinguishable from today's on AUC.
- The one soft spot is eval pAUC at −0.51 (se 0.21), about 2.4 se. It was
  not a registered criterion, and with 4 vs 8 runs on one secondary metric
  it is reported, not acted on.

**P3 fails.** `fr_f4` − `fr_f1` = −0.40 dev, just outside [−0.3, +0.4], and
+0.81 eval. Dropping TFgram on the fast recipe splits by dataset.
- **Mostly ToyConveyor.** Per-machine means show it moving both ways: dev
  −2.0 (80.13 vs 82.14), eval +4.2 (85.05 vs 80.81). ToyConveyor's seed noise
  is above 4 points.
- **Elsewhere, small moves:**
  - eval: valve +0.7, slider +0.4
  - dev: ToyCar −0.5
- **Earlier host.** `../seed-sweep` measured f4 − f1 = +0.20 dev and +0.38
  eval there. The eval direction replicates; the dev direction does not.

### Decisions

- **D1 → adopt the fast recipe.** Every later arm uses `--amp fp16
  --cudnn_bench 1 --lean 1 --cache_dir data/cache --probe_every 5`, with
  baselines trained on that same recipe. `../no-logmel` and `../margin-warmup`
  already used `fr_f1`/`fr_f4` as their paired baselines.
- **D2 → Phase 1 builds on `fussion 1`.**
  - The rule needs `fr_f4` ≥ `fr_f1` − 0.15 on dev *and* eval. Eval passes by
    a wide margin (+0.81); dev fails (−0.40).
  - So `../lit-arms` launches with `LA_FUSSION=1`. The paper model also keeps
    D1's comparison like-for-like.
  - On the eval set, f4 is the better single model: 96.06 vs 95.25. That is
    worth noting for any eval-facing number, but it does not override the
    registered rule.
  - `../no-logmel` later found that Log-Mel alone (f2) is within ~0.8 of
    `fr_f1` and faster still.

### Test-set selection (P4, P5, D3)

`probe.csv` scores the EMA net on the dev test set every 5 epochs, never for
selection. Oracle gain = the best probed epoch's dev AUC − the saved
checkpoint's (eval.py).

| run | saved (val loss) | best probed (epoch) | oracle gain |
|---|---|---|---|
| `fr_f1_s2040` | 94.52 | 94.53 (289) | +0.01 |
| `fr_f1_s2041` | 95.26 | **95.28** (269) | +0.01 |
| `fr_f1_s2042` | 94.70 | 94.74 (279) | +0.04 |
| `fr_f1_s2043` | 95.10 | 95.12 (284) | +0.03 |
| `fr_f4_s2040` | 94.10 | 94.46 (204) | +0.36 |
| `fr_f4_s2041` | 94.63 | 94.64 (284) | +0.02 |
| `fr_f4_s2042` | 94.86 | 94.98 (244) | +0.12 |
| `fr_f4_s2043` | 94.39 | 94.51 (224) | +0.12 |

**P4 splits:**
- **The gain is smaller than predicted.** `fr_f1`'s mean oracle gain is
  **+0.02**, below the predicted +0.1 to +0.5. Its saved checkpoint is within
  0.04 of the best epoch it ever reaches. `fr_f4`'s is +0.16, with its best
  epochs earlier, at 204–284: it peaks and then drifts down by ~0.1–0.4.
- **The ceiling part holds.** The best probed epoch over all `fr_f1` seeds is
  **95.28 < 95.71**.

**P5 holds.** ToyConveyor has the largest mean per-machine oracle gain in both
arms: +0.12 for `fr_f1` and +0.83 for `fr_f4`. Pump is next at +0.06 and
+0.32 (fig4).

**D3.** No seed, with no selection rule at all, reaches the paper's 95.71.
- Test-set epoch selection does not explain the paper's dev number.
- Neither does the EMA smoothing: alpha 0.9999 is a ~19-epoch time constant,
  so the EMA net changes little from one 5-epoch probe to the next.
- The gap to 95.71 (~0.4–0.8 points) has to come from somewhere else: seed
  luck at n=1, a different recipe, or selection across runs rather than
  across epochs. That is consistent with Table VI's "w/o FPH" 94.94, which
  matches our 94.86–94.90.

### What does not replicate or was not tested

- **Determinism.** smartlab1 is not bit-deterministic (CLAUDE.md), and
  `cudnn.benchmark` adds more non-determinism. The fast recipe is compared to
  relu8 as a distribution (n=4 vs n=8), not run-for-run.
- **Cross-host.** relu8 ran on the same host but its GPU assignment was not
  recorded, so the speed ratio mixes GPUs.
- **Untested components.** `torch.compile` crashed in planning and was not
  pursued. bf16 autocast was not tried.

## Figures

| file | shows |
|---|---|
| `plots/fig1_speed.png` | wall clock per 300-epoch run: relu8, `fr_f1`, `fr_f4`; hollow = GPU 1 (PCIe x1) |
| `plots/fig2_auc.png` | per-seed dev/eval AUC (eval.py R0) for relu8, `fr_f1`, `fr_f4`, with the paper's 95.71 |
| `plots/fig3_probe.png` | probed dev AUC vs epoch per seed; best probed epoch (dot) vs saved checkpoint (square) |
| `plots/fig4_oracle_by_machine.png` | oracle gain by machine: best probed epoch − saved checkpoint |
