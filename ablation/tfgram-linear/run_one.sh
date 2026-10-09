#!/usr/bin/env bash
# Train and evaluate one arm: run_one.sh <seed> <tf_act> <tf_pool> [epochs]
#
# Resumable: an arm whose two eval CSVs already exist is skipped.
# One run per GPU via a lock directory -- runs are GPU-bound (see CLAUDE.md).
set -uo pipefail
cd "$(dirname "$0")/../.."
seed=$1; act=$2; pool=$3; epochs=${4:-300}
PY=${TFL_PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
OUT=ablation/tfgram-linear/results
SCORES=ablation/tfgram-linear/scores
LOCKS=${TFL_LOCKS:-/tmp/os-scl-tfl-locks}
GPUS=${TFL_GPUS:-"1 2"}
desc=tfl_${act}_${pool}_s${seed}
name=SCLTFSTgramMFN_${desc}
ckpt=check_points/${name}/model.pth
mkdir -p "$OUT" "$LOCKS"

if [ -f "${SCORES}/${name}/eval_dev.csv" ] && [ -f "${SCORES}/${name}/eval_eval.csv" ]; then
    echo "[$(date +%T)] skip ${desc} (already complete)"; exit 0
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

arch=(--tf_act "$act" --tf_pool "$pool")
# a train run is complete only once its last student snapshot exists;
# model.pth alone appears at the first validation improvement
if [ ! -f "check_points/${name}/student_e${epochs}.pth" ]; then
    echo "[$(date +%T)] TRAIN start ${desc} gpu=${gpu}"
    # --num_workers 8 to match seed-sweep; nw changes the run at a fixed seed
    $PY train.py --m 0.4 --fussion 1 --ht basic --gpu_num "$gpu" --num_workers 8 \
        --seed "$seed" --epoch "$epochs" "${arch[@]}" --desc "$desc" \
        --student_snapshots "10,30,90,${epochs}" \
        > "${OUT}/train_${desc}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "$ckpt" ]; then
        echo "[$(date +%T)] FAIL train ${desc} rc=${rc}"; exit 1
    fi
    echo "[$(date +%T)] TRAIN done ${desc}"
fi

mkdir -p "${SCORES}/${name}"
cp -p "check_points/${name}/config.json" "${SCORES}/${name}/"
for split in d e; do
    ds=dev; [ "$split" = e ] && ds=eval
    $PY eval.py --m 0.4 --fussion 1 --ht basic --gpu_num "$gpu" "${arch[@]}" \
        --model_path "$ckpt" --${split} --csv "${SCORES}/${name}/eval_${ds}.csv" \
        > "${OUT}/eval_${desc}_${split}.log" 2>&1 \
        || { echo "[$(date +%T)] FAIL eval ${desc} --${split}"; exit 1; }
done
echo "[$(date +%T)] ARM done ${desc}"
