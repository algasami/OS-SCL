#!/usr/bin/env bash
# Train and evaluate one (arm, seed) cell of the sweep.
#
# Resumable: a cell whose two eval CSVs already exist is skipped, so run.sh can
# be relaunched after an interruption without redoing finished work.
#
# GPU pinning is dynamic: the runs are GPU-bound (measured 99-100% util), and
# stacking jobs on one card LOWERS aggregate throughput (13.2 it/s alone vs
# 3x3.1 shared), so each cell claims a whole card via a lock directory.
set -uo pipefail
cd "$(dirname "$0")/../.."
arm=$1; seed=$2; fussion=$3; freeze=$4
PY=/home/f74134118/anaconda3/envs/os-scl/bin/python
OUT=ablation/seed-sweep/results
SCORES=ablation/seed-sweep/scores
LOCKS=${SWEEP_LOCKS:-/tmp/os-scl-sweep-locks}
GPUS=${SWEEP_GPUS:-"0 1"}
desc=sweep_${arm}_s${seed}
name=SCLTFSTgramMFN_${desc}
ckpt=check_points/${name}/model.pth

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

extra=()
[ "$freeze" != "none" ] && extra=(--freeze_tfgram "$freeze")
export OMP_NUM_THREADS=${SWEEP_OMP:-1} MKL_NUM_THREADS=${SWEEP_OMP:-1}

if [ ! -f "$ckpt" ]; then
    echo "[$(date +%T)] TRAIN start ${desc} gpu=${gpu}"
    ep=(); [ -n "${SWEEP_EPOCH:-}" ] && ep=(--epoch "$SWEEP_EPOCH")
    $PY train.py --m 0.4 --fussion "$fussion" --ht basic --gpu_num "$gpu" \
        --num_workers "${SWEEP_NW:-8}" --seed "$seed" "${extra[@]}" "${ep[@]}" --desc "$desc" \
        > "${OUT}/train_${desc}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "$ckpt" ]; then
        echo "[$(date +%T)] FAIL train ${desc} rc=${rc}"; exit 1
    fi
    echo "[$(date +%T)] TRAIN done ${desc}"
fi

# CSVs go to scores/, not next to the weights: check_points/ is gitignored
# scratch, while the scores are the committed record the figures read.
mkdir -p "${SCORES}/${name}"
cp -p "check_points/${name}/config.json" "${SCORES}/${name}/" 2>/dev/null
for split in d e; do
    ds=dev; [ "$split" = e ] && ds=eval
    $PY eval.py --m 0.4 --fussion "$fussion" --ht basic --gpu_num "$gpu" \
        --model_path "$ckpt" --${split} --csv "${SCORES}/${name}/eval_${ds}.csv" \
        > "${OUT}/eval_${desc}_${split}.log" 2>&1 \
        || { echo "[$(date +%T)] FAIL eval ${desc} --${split}"; exit 1; }
done
echo "[$(date +%T)] CELL done ${desc}"
