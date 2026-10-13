#!/usr/bin/env bash
# Train one fresh paper-model seed: run_one.sh <seed>   (stage 'train' only)
#
# Identical recipe to ../tfgram-linear's relu_max arm (same flags, nw=8, 300
# epochs, same snapshots), only the seed and desc differ. Resumable: a run whose
# last student snapshot exists is skipped. One run per GPU via a lock directory.
set -uo pipefail
cd "$(dirname "$0")/../.."
seed=$1; epochs=300
PY=${CS_PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
OUT=ablation/confirm-seeds/results
LOCKS=${CS_LOCKS:-/tmp/os-scl-cs-locks}
GPUS=${CS_GPUS:-"1 2"}
desc=cs_relu_max_s${seed}
name=SCLTFSTgramMFN_${desc}
mkdir -p "$OUT" "$LOCKS"

if [ -f "check_points/${name}/student_e${epochs}.pth" ]; then
    echo "[$(date +%T)] skip ${desc} (trained)"; exit 0
fi
gpu=""
while [ -z "$gpu" ]; do
    for g in $GPUS; do
        if mkdir "${LOCKS}/gpu${g}" 2>/dev/null; then gpu=$g; break; fi
    done
    [ -z "$gpu" ] && sleep 20
done
trap 'rmdir "${LOCKS}/gpu${gpu}" 2>/dev/null' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "[$(date +%T)] TRAIN start ${desc} gpu=${gpu}"
$PY train.py --m 0.4 --fussion 1 --ht basic --gpu_num "$gpu" --num_workers 8 \
    --seed "$seed" --epoch "$epochs" --tf_act relu --tf_pool max --desc "$desc" \
    --student_snapshots "10,30,90,${epochs}" \
    > "${OUT}/train_${desc}.log" 2>&1
rc=$?
if [ $rc -ne 0 ] || [ ! -f "check_points/${name}/model.pth" ]; then
    echo "[$(date +%T)] FAIL train ${desc} rc=${rc}"; exit 1
fi
echo "[$(date +%T)] TRAIN done ${desc}"
