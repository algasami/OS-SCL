#!/usr/bin/env bash
# One lit-arms run: run_one.sh <arm> <seed>   arm = aug | gate | eca | combo
#
# Base = ../fast-recipe's recipe at LA_FUSSION (its decision D2); LA_COMBO holds the
# flags of the arms that passed D1 (stage 2). GPU locking, crash-retry and records:
# ../train_one.sh.
set -uo pipefail
arm=$1; seed=$2; epochs=${3:-300}
FUSSION=${LA_FUSSION:?set LA_FUSSION to fast-recipe D2 base (4 or 1)}
case $arm in
    aug)   flags="--wave_aug 1" ;;
    gate)  flags="--mel_gate tristat" ;;
    eca)   flags="--eca 1" ;;
    combo) flags=${LA_COMBO:?set LA_COMBO to the passing arms\' flags} ;;
    *) echo "unknown arm $arm"; exit 1 ;;
esac
desc=la_${arm}_s${seed}; [ "$epochs" != 300 ] && desc=${desc}_e${epochs}
# shellcheck disable=SC2086
exec "$(dirname "$0")/../train_one.sh" lit-arms "$desc" "$epochs" \
    --m 0.4 --fussion "$FUSSION" --ht basic --num_workers 8 --seed "$seed" \
    --amp fp16 --cudnn_bench 1 --lean 1 --cache_dir data/cache --probe_every 5 $flags
