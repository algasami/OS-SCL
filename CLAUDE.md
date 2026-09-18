# CLAUDE.md

Official code for **"Noise Supervised Contrastive Learning and Feature-Perturbed for Anomalous Sound Detection"** (ICASSP 2025, arXiv:2509.13853). Unsupervised ASD on DCASE 2020 Task 2, trained on normal audio only. Builds on STgram-MFN and Noisy-ArcMix.

## Environment

Conda env `os-scl` (Python 3.11): `conda env create -f environment.yml`. `torch`/`torchaudio` are pinned to **2.8.0+cu128** on purpose — 2.9+ routes `torchaudio.load()` through TorchCodec/FFmpeg and breaks `dataloader.py`.

## Commands

All paths are relative to the repo root (`data/dataset`, `config.yaml`) — always run from there. No tests, linter, or formatter.

```bash
bash datadownload.sh                                     # dev + eval sets -> data/dataset, data/eval_dataset
cp Preparation/{*.csv,split.py} data/eval_dataset/ && (cd data/eval_dataset && python split.py)
                                                         # renames eval test wavs to normal_*/anomaly_*; run once
python train.py --m 0.4 --gpu_num 0 --fussion 1 --ht basic --desc main
python model_prune.py --input check_points/<name>/model.pth [--overwrite]
python eval.py --m 0.4 --gpu_num 0 --fussion 1 --ht basic --model_path <ckpt> --d   # --d dev set, --e eval set
python eval.py ... --d --csv [path]                      # export eval matrix; bare --csv -> <ckpt_dir>/eval_<dev|eval>.csv
```

## Config

`config.yaml` defaults; non-`None` CLI args override. `trainer.py` snapshots the merged config to `check_points/<model_name>/config.json`, where `model_name = "SCLTFSTgramMFN"` + `_<desc>` unless `desc` is the string `'None'`.

- `--fussion 1` = Log-Mel + Tgram + TFgram (paper's model); `2` = Log-Mel only
- `--ht basic|leaky_relu` = projection head; `--m` = ArcFace margin

## Architecture

`SCLTFSTgramMFN` (`model/net.py`) is a 41-way machine-ID classifier. Anomaly score = per-sample CE of the ArcFace logits against the sample's own ID (`ASDLoss(reduction=False)` in `eval.py`); no separate detector head.

Three branches, each `128 × 313`, concatenated as channels → `MobileFaceNet` → 128-d embedding → `head` → L2-norm → `ArcMarginProduct`:

- Log-Mel: `Wave2Mel` in `dataloader.py` (n_fft 2048, hop 512, 128 mels), precomputed on CPU
- Tgram: `TgramNet` (`model/net.py`), 1-D conv on raw wave
- TFgram: `TFgram` (`model/waveq.py`), deeper 1-D conv stack — the paper's contribution

Loss (`utils.os_scl`): mixup-interpolated ArcFace CE + `SupConLoss` on the embedding using the *unmixed* `y_a` labels (the "noisy" part). `net.forward` gets the original `labels` for the margin, not `y_a`/`y_b`.

**EMA net is the deliverable.** `WeightEMA` (`losses.py`, alpha 0.9999) tracks `ema_net`; `trainer.py` validates and saves `ema_net.state_dict()` at each new min validation loss.

## Traps

- `TFgram.forward(x, train)` reshapes by flag: `train=True` → `squeeze()`, `False` → `squeeze().unsqueeze(0)`. So `train=False` needs batch size 1 (`eval.py`) and `train=True` breaks on batch 1 (`trainer.valid()` uses `train=True`).
- Pruned checkpoints fail in `eval.py` by default: `model_prune.py` strips unused `TFgramNet.*` keys, but `load_state_dict` is `strict=True`. Pass `--no-strict` (or drop the dead modules from `TFgram.__init__`).
- Shapes assume 10 s @ 16 kHz: `LayerNorm(313)` in `TgramNet`, `adaptive_max_pool1d(626/313)` in `waveq.py`, `linear7`'s `(8, 20)` kernel in `MobileFaceNet`.
- Class IDs depend on `name_list = ['fan','pump','slider','ToyCar','ToyConveyor','valve']` order and per-machine ID ranges (`ToyConveyor` id_01–06, `ToyCar` id_01–07, others id_00–06). Reordering invalidates all checkpoints.
- `Preparation/split.py` renames in place, must run inside `data/eval_dataset`, once per download.
- `eval.py` `mAUC` = mean over machine types of the *minimum per-ID AUC*.
- `--csv` writes rows `model,dataset,machine,id,scope,AUC,pAUC,mAUC`; `scope` is `id` | `machine` | `overall`.
- **`--num_workers` changes the run at a fixed seed.** `persistent_workers=nw>0` makes the DataLoader iterator draw its base seed once instead of once per epoch, desynchronising the global torch RNG from epoch 1 on: batch order and mixup pairings differ (the split and the weight init do not — both are drawn before iteration). So `nw=0` and `nw=8` at the same seed are *different runs*. Never compare a baseline trained at one `nw` against arms trained at another; this silently invalidated the `f1` baseline in two ablations.
- Training is otherwise **bit-deterministic** at a fixed seed (`random_seed()` sets `cudnn.deterministic=True, benchmark=False`), verified by three independent reruns reproducing their checkpoints to 5 decimals. So a rerun with identical code, seed and `nw` is redundant — reuse the checkpoint.
- `WeightEMA.__init__` copies **ema_net → net** (`losses.py`), so the student's own init is discarded. `net` and `ema_net` draw different random inits; the second one wins.
- `mAUC` is a *minimum* per machine, so it is an order statistic and by far the noisiest metric. Seed-to-seed swings of ~2 points are normal; treat mAUC deltas under ~2 points as unresolvable.
- Run-to-run noise on this reproduction is ~1 point of eval AUC and >4 points on ToyConveyor. Any single-seed difference below that is not a result — see `ablation/seed-sweep/`.
- Checkpoints are **310 MB** each because `TFgram.__init__` allocates ~80 M params of `conv_block1-6`/`fc*` that `forward` never uses. Budget disk before a sweep (16 runs ≈ 5 GB), or prune.
- These runs are **GPU-bound**, not dataloader-bound, despite `train_dataset.__getitem__` decoding the wav and computing the mel per item. Stacking jobs on one card *lowers* aggregate throughput (measured: 13.2 it/s alone vs 3×3.1 shared). Run one per GPU.

## Ablations

Each ablation is a self-contained directory, `ablation/<name>/`:

```
ablation/<name>/
  README.md              question, setup, results, and what does NOT replicate
  run.sh                 reproduces the runs end to end; resumable
  make_plots.py          figures; reads scores/, never check_points/
  plots/*.png            committed -- the only copy that survives a log cleanup
  scores/                THE COMMITTED RECORD -- a few KB per run
      <model_name>/        one subdirectory per trained run:
          eval_dev.csv       from `eval.py --csv <path>`
          eval_eval.csv
          config.json        copied from the checkpoint dir
          curve.csv          distilled training curve (ablation/distill_curves.py)
      *.json | *.txt | *.csv   ablation-level artifacts: probe output, derived
                               summaries (cka.txt, probe.json, summary.csv)
  results/               raw logs -- gitignored scratch, NOT durable
```

**Nothing under `results/` may be tracked.** If an artifact is worth committing
it goes in `scores/`. Only `*.log` is gitignored by pattern, so a `.json` or
`.txt` dropped in `results/` *looks* safe and is not -- it sits in a directory
everyone treats as disposable. An ablation that trains nothing (a probe) still
has a `scores/`; it just holds flat files instead of per-run subdirectories.

`check_points/` is **gitignored scratch holding only `model.pth`**. Nothing in it
is durable: don't read it from a plot script and don't put results there.
Resolve a run's scores with `ablation/scores_path.py`:

```python
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from scores_path import scores_csv, scores_dir
```

It globs `ablation/*/scores/<model_name>`, so cross-ablation reads (every set
plots the shared `SCLTFSTgramMFN_test` baseline, which lives in
`ablation/baselines/scores/`) need no relative paths. Scores belong to the
ablation that *produced* the run.

Rules that are not optional:

- **Write eval CSVs straight to `scores/`** via `eval.py --csv <path>`. The bare
  `--csv` flag writes next to the weights, which are gitignored.
- **Distil the training curve** with `python ablation/distill_curves.py <name>`
  after a run. `trainer.py` only prints an EPOCH line when validation loss
  *improves*, so the curve is a best-so-far trace with fewer rows than epochs —
  that is expected, not data loss. A 300-epoch log is ~10 MB of tqdm spam around
  ~200 useful lines, and `*.log` is gitignored: two ablations already lost their
  curves this way and their fig4 can never be regenerated.
- **Every ablation ships figures.** Run `make_plots.py` with
  `/home/f74134118/anaconda3/bin/python` (matplotlib is in `base`, not `os-scl`),
  read only from `scores/` so a clone can regenerate them, and add a Figures
  table to the README. Degrade gracefully — skip a figure whose inputs are
  missing rather than crashing the whole script.
- **Supersede by annotation.** When a later ablation corrects an earlier one, add
  a note at the top and inline `**[n=4]**`-style corrections. Never rewrite the
  original numbers away.

Figures follow the shared palette in the existing `make_plots.py` files:
categorical slots `#2a78d6` / `#eb6834` / `#1baf7a` on surface `#fcfcfb`, and an
arm keeps its colour across figure sets. Only **three** categorical slots
validate under `--pairs all`; draw a baseline series achromatic rather than
taking a fourth slot. Aqua is below 3:1 on the light surface, so aqua marks need
a direct label. Always render and actually look at the output before shipping,
and check each title states something the data supports.

## Working style

- Long-running work (training, eval sweeps, downloads) should be launched in the background and monitored — don't stop to ask whether to wait. Run it with `run_in_background`, or set up a monitor/watcher, and notify me when it finishes or when something needs a decision. Only interrupt me for choices I actually have to make.
