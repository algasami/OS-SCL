#!/usr/bin/env bash
# One permix run: run_one.sh <seed> [epochs]
# Base = the single-model recipe (ao_nogain: fast recipe, fussion 1, --wave_aug_ops noise,shift)
# plus --permix 1 (Dirichlet a 0.5, R ~ U{1,2,3}: the paper's values, also the code defaults).
# GPU locking, crash-retry and records: ../train_one.sh.
set -uo pipefail
seed=$1; epochs=${2:-300}
desc=pm_s${seed}; [ "$epochs" != 300 ] && desc=${desc}_e${epochs}
exec "$(dirname "$0")/../train_one.sh" permix "$desc" "$epochs" \
    --m 0.4 --fussion 1 --ht basic --num_workers 8 --seed "$seed" \
    --amp fp16 --cudnn_bench 1 --lean 1 --cache_dir data/cache --probe_every 5 \
    --wave_aug 1 --wave_aug_ops noise,shift --permix 1 --permix_a 0.5 --permix_r 3
