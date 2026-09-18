#!/usr/bin/env bash
# Probe: is the TFgram branch starved of gradient, or architecturally constrained?
#
# No training -- this measures an existing checkpoint plus 400 real training
# steps, ~20 min. Writes the durable record to scores/probe.json, which is what
# make_plots.py reads.
#
# Run from the repo root:  bash ablation/tfgram-gradient/run.sh [gpu]
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU=${1:-${PROBE_GPU:-0}}
PY=/home/f74134118/anaconda3/envs/os-scl/bin/python
mkdir -p ablation/tfgram-gradient/scores ablation/tfgram-gradient/results

# The drift measurement replays train.py's RNG order against this checkpoint,
# so the checkpoint and its seed must match.
$PY ablation/tfgram-gradient/probe.py \
    --gpu "$GPU" --steps 400 --seed 2024 \
    --ckpt check_points/SCLTFSTgramMFN_sweep_f1_s2024/model.pth \
    --out ablation/tfgram-gradient/scores/probe.json \
    2>&1 | tee ablation/tfgram-gradient/results/probe.log

/home/f74134118/anaconda3/bin/python ablation/tfgram-gradient/make_plots.py
