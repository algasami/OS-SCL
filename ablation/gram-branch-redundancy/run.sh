#!/usr/bin/env bash
# Ablation: which learned gram branch actually carries the fusion gain?
# Run from the repo root:  bash ablation/gram-branch-redundancy/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p ablation/gram-branch-redundancy/results

# fussion 3 = Log-Mel + TFgram   (the row the paper never reports)
# fussion 4 = Log-Mel + Tgram    (STgram under OS-SCL; paper reports 94.63 AUC)
python train.py --m 0.4 --fussion 3 --ht basic --gpu_num 1 --num_workers 8 \
    --desc f3_mel_tfgram > ablation/gram-branch-redundancy/results/train_f3.log 2>&1 &
python train.py --m 0.4 --fussion 4 --ht basic --gpu_num 2 --num_workers 8 \
    --desc f4_mel_tgram  > ablation/gram-branch-redundancy/results/train_f4.log 2>&1 &
wait
