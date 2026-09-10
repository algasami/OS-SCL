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

## Working style

- Long-running work (training, eval sweeps, downloads) should be launched in the background and monitored — don't stop to ask whether to wait. Run it with `run_in_background`, or set up a monitor/watcher, and notify me when it finishes or when something needs a decision. Only interrupt me for choices I actually have to make.
