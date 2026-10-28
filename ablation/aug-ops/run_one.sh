#!/usr/bin/env bash
# One aug-ops run: run_one.sh <arm> <seed> [epochs]
#   arm = nonoise (shift+gain) | noshift (noise+gain) | nogain (noise+shift)
# Base = the fast recipe at fussion 1 with ../lit-arms' --wave_aug 1, minus one op.
# GPU locking, crash-retry and records: ../train_one.sh.
set -uo pipefail
arm=$1; seed=$2; epochs=${3:-300}
case $arm in
    nonoise) ops=shift,gain ;;
    noshift) ops=noise,gain ;;
    nogain)  ops=noise,shift ;;
    *) echo "unknown arm $arm"; exit 1 ;;
esac
desc=ao_${arm}_s${seed}; [ "$epochs" != 300 ] && desc=${desc}_e${epochs}
exec "$(dirname "$0")/../train_one.sh" aug-ops "$desc" "$epochs" \
    --m 0.4 --fussion 1 --ht basic --num_workers 8 --seed "$seed" \
    --amp fp16 --cudnn_bench 1 --lean 1 --cache_dir data/cache --probe_every 5 \
    --wave_aug 1 --wave_aug_ops "$ops"
