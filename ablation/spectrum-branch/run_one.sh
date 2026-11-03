#!/usr/bin/env bash
# One spectrum-branch run: run_one.sh <seed> [epochs]
# Base = the single-model recipe (ao_nogain: fast recipe, fussion 1, --wave_aug_ops noise,shift)
# plus --spec_branch 1. GPU locking, crash-retry and records: ../train_one.sh.
set -uo pipefail
seed=$1; epochs=${2:-300}
desc=sb_spec_s${seed}; [ "$epochs" != 300 ] && desc=${desc}_e${epochs}
exec "$(dirname "$0")/../train_one.sh" spectrum-branch "$desc" "$epochs" \
    --m 0.4 --fussion 1 --ht basic --num_workers 8 --seed "$seed" \
    --amp fp16 --cudnn_bench 1 --lean 1 --cache_dir data/cache --probe_every 5 \
    --wave_aug 1 --wave_aug_ops noise,shift --spec_branch 1
