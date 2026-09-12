#!/usr/bin/env bash
# Ablation: is TFgram a *learned* feature, or just a fixed random projection?
# Both arms are fussion=1 (all three channels); only the TFgram branch is frozen
# at its random init, so any drop vs f1 is attributable to TFgram's learning.
# Run from the repo root:  bash ablation/frozen-tfgram/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p ablation/frozen-tfgram/results

# strict   = frozen weights AND BatchNorm pinned to eval() (running stats stay at
#            init, so BN is an identity) -- a pure fixed random projection
# bn_adapt = frozen weights, BatchNorm still tracks the data, so the random
#            features stay sanely scaled (branch output std 0.67 vs 0.024 strict)
python train.py --m 0.4 --fussion 1 --ht basic --gpu_num 0 --num_workers 8 \
    --freeze_tfgram strict   --seed 2024 --desc ft_strict \
    > ablation/frozen-tfgram/results/train_ft_strict.log 2>&1 &
python train.py --m 0.4 --fussion 1 --ht basic --gpu_num 1 --num_workers 8 \
    --freeze_tfgram bn_adapt --seed 2024 --desc ft_bnadapt \
    > ablation/frozen-tfgram/results/train_ft_bnadapt.log 2>&1 &
wait
