#!/usr/bin/env bash
# One margin-warmup run: run_one.sh <arm> <seed> [epochs]
#   arm = f6 (TFgram) | f7 (Tgram) | f2 (Log-Mel, control) | f5 (Tgram+TFgram, optional stage 2b)
#
# ../no-logmel's fast recipe exactly (run_one.sh there), plus the fix:
#   --m_warmup 60,120      student margin 0 for epochs 0-59, linear per step to 0.4 at 120, then 0.4
#   --state_snapshots 60,100   full state (student + EMA) after epochs 60 and 100, for diag_margin.py
# Env MW / SNAPS override both (the gate uses MW=0.5,1.5 SNAPS=1).
# GPU locking, crash-retry and records: ../train_one.sh.
set -uo pipefail
arm=$1; seed=$2; epochs=${3:-300}
case $arm in
    f2|f5|f6|f7) fussion=${arm#f} ;;
    *) echo "unknown arm $arm"; exit 1 ;;
esac
desc=mw_${arm}_s${seed}; [ "$epochs" != 300 ] && desc=${desc}_e${epochs}
exec "$(dirname "$0")/../train_one.sh" margin-warmup "$desc" "$epochs" \
    --m 0.4 --fussion "$fussion" --ht basic --num_workers 8 --seed "$seed" \
    --amp fp16 --cudnn_bench 1 --lean 1 --cache_dir data/cache --probe_every 5 \
    --m_warmup "${MW:-60,120}" --state_snapshots "${SNAPS:-60,100}"
