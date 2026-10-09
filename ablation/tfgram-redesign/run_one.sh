#!/usr/bin/env bash
# Train and evaluate one cell: run_one.sh <tag> <seed> <tf_act> <tf_pool> [epochs]
#
# <tag> is 'pilot' (30-epoch CKA gate) or 'tfr' (the 300-epoch sweep).
# Resumable: a cell whose two eval CSVs already exist is skipped.
# One run per GPU via a lock directory -- runs are GPU-bound (see CLAUDE.md),
# and GPU 0 on smartlab1 is usually taken by another user, hence GPUS="1 2".
set -uo pipefail
cd "$(dirname "$0")/../.."
tag=$1; seed=$2; act=$3; pool=$4; epochs=${5:-300}
PY=${TFR_PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
OUT=ablation/tfgram-redesign/results
SCORES=ablation/tfgram-redesign/scores
LOCKS=${TFR_LOCKS:-/tmp/os-scl-tfr-locks}
GPUS=${TFR_GPUS:-"1 2"}
desc=${tag}_${act}_${pool}_s${seed}
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
if [ ! -f "$ckpt" ]; then
    echo "[$(date +%T)] TRAIN start ${desc} gpu=${gpu}"
    # --num_workers 8 to match seed-sweep; nw changes the run at a fixed seed
    $PY train.py --m 0.4 --fussion 1 --ht basic --gpu_num "$gpu" --num_workers 8 \
        --seed "$seed" --epoch "$epochs" "${arch[@]}" --desc "$desc" \
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
echo "[$(date +%T)] CELL done ${desc}"
