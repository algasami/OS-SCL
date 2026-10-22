# Fast recipe: a 2–4× faster training recipe, and how much test-set selection is worth

> **Status: pre-registered 2026-10-17. Nothing trained yet.** Phase 0 of the
> post-confirm-seeds plan. Phases 1–3 (Log-Mel gating, waveform augmentation,
> ECA, a shared-trunk weight soup, a pretrained teacher) need many 4-seed arms.
> At 21.5 h per run on this host's A4000s, that is weeks.

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

*(empty until the runs finish)*

## Figures

*(empty until the runs finish)*
