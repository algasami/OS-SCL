#!/usr/bin/env bash
# Train one fast-recipe run: run_one.sh <arm> <seed> [epochs]   arm = f1 | f4
#
# Resumable: a run whose model.pth exists and whose probe.csv reaches the last
# epoch is skipped. One run per GPU: takes a GPU only if no ablation's lock is
# on it and it stays under 1 GB used for 3 checks a minute apart.
set -uo pipefail
cd "$(dirname "$0")/../.."
arm=$1; seed=$2; epochs=${3:-300}
PY=${FR_PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
OUT=ablation/fast-recipe/results
LOCKS=${FR_LOCKS:-/tmp/os-scl-fr-locks}
GPUS=${FR_GPUS:-"0 1 2"}
desc=fr_${arm}_s${seed}; [ "$epochs" != 300 ] && desc=${desc}_e${epochs}
name=SCLTFSTgramMFN_${desc}
mkdir -p "$OUT" "$LOCKS"

if [ -f "check_points/${name}/model.pth" ] && grep -q "^$((epochs - 1))," "check_points/${name}/probe.csv" 2>/dev/null; then
    echo "[$(date +%T)] skip ${desc} (trained)"; exit 0
fi
idle() {
    for _ in 1 2 3; do
        for l in /tmp/os-scl-*-locks/gpu"$1"; do
            [ -d "$l" ] && [ "${l%/gpu*}" != "$LOCKS" ] && return 1
        done
        [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] || return 1
        sleep "${FR_IDLE_SLEEP:-60}"
    done
}
gpu=""
while [ -z "$gpu" ]; do
    for g in $GPUS; do
        if mkdir "${LOCKS}/gpu${g}" 2>/dev/null; then
            if idle "$g"; then gpu=$g; break; fi
            rmdir "${LOCKS}/gpu${g}"
        fi
    done
    [ -z "$gpu" ] && sleep 60
done
trap 'rmdir "${LOCKS}/gpu${gpu}" 2>/dev/null' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
rm -f "check_points/${name}/probe.csv"   # a restarted run starts its probe log over

fussion=${arm#f}
echo "[$(date +%T)] TRAIN start ${desc} gpu=${gpu}"
t0=$(date +%s)
$PY train.py --m 0.4 --fussion "$fussion" --ht basic --gpu_num "$gpu" --num_workers 8 \
    --seed "$seed" --epoch "$epochs" --desc "$desc" \
    --amp fp16 --cudnn_bench 1 --lean 1 --cache_dir data/cache --probe_every 5 \
    > "${OUT}/train_${desc}.log" 2>&1
rc=$?
if [ $rc -ne 0 ] || [ ! -f "check_points/${name}/model.pth" ]; then
    echo "[$(date +%T)] FAIL train ${desc} rc=${rc}"; exit 1
fi
echo "start,end,seconds,epochs,gpu" > "check_points/${name}/wall.csv"
echo "$t0,$(date +%s),$(( $(date +%s) - t0 )),$epochs,$gpu" >> "check_points/${name}/wall.csv"
echo "[$(date +%T)] TRAIN done ${desc}"
