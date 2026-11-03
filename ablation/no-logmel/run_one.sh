#!/usr/bin/env bash
# One no-logmel run: run_one.sh <arm> <seed> [epochs]
#   arm = f5 (Tgram+TFgram) | f6 (TFgram) | f7 (Tgram) | f2 (Log-Mel, reference)
#
# The fast recipe exactly as ../fast-recipe's fr_f1/fr_f4 ran it; only --fussion changes.
# GPU locking, crash-retry and records: ../train_one.sh.
set -uo pipefail
arm=$1; seed=$2; epochs=${3:-300}
case $arm in
    f5|f6|f7|f2) fussion=${arm#f} ;;
    *) echo "unknown arm $arm"; exit 1 ;;
esac
desc=nl_${arm}_s${seed}; [ "$epochs" != 300 ] && desc=${desc}_e${epochs}
exec "$(dirname "$0")/../train_one.sh" no-logmel "$desc" "$epochs" \
    --m 0.4 --fussion "$fussion" --ht basic --num_workers 8 --seed "$seed" \
    --amp fp16 --cudnn_bench 1 --lean 1 --cache_dir data/cache --probe_every 5
