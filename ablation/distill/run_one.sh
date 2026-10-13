#!/usr/bin/env bash
# Train and evaluate one distill run: run_one.sh <seed> <beta|none>
#
#   beta  student SCLTFSTgramMFN_kd_relu_max_s<seed>, distilled from the 3 other seeds
#   none  rerun   SCLTFSTgramMFN_kdnull_s<seed>: the baseline recipe again (noise floor)
#
# Resumable: training is skipped once student_e300.pth exists, the whole run once both
# eval CSVs exist. Takes a GPU only when it holds the lock AND the card is idle, since
# ../confirm-seeds keeps its own locks on the same cards.
set -uo pipefail
cd "$(dirname "$0")/../.."
seed=$1; beta=$2
D=ablation/distill
PY=${KD_PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
TDIR=${KD_TEACHER_DIR:-/home/f74134118/OS-SCL/check_points}
LOCKS=${KD_LOCKS:-/tmp/os-scl-kd-locks}
GPUS=${KD_GPUS:-"0 1 2"}
if [ "$beta" = none ]; then desc=kdnull_s$seed; else desc=kd_relu_max_s$seed; fi
name=SCLTFSTgramMFN_$desc
ck=check_points/$name
mkdir -p $D/results "$LOCKS"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

if [ -f $D/scores/$name/eval_dev.csv ] && [ -f $D/scores/$name/eval_eval.csv ]; then
    echo "[$(date +%T)] skip $desc (already complete)"; exit 0
fi

idle() { [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; }
gpu=""
while [ -z "$gpu" ]; do
    for g in $GPUS; do
        if mkdir "$LOCKS/gpu$g" 2>/dev/null; then
            if idle "$g"; then gpu=$g; break; fi
            rmdir "$LOCKS/gpu$g"
        fi
    done
    [ -z "$gpu" ] && sleep 120
done
trap 'rmdir "$LOCKS/gpu$gpu" 2>/dev/null' EXIT

kd=()
if [ "$beta" != none ]; then
    t=""; for s in 2024 2025 2026 2027; do [ $s != $seed ] && t="$t,SCLTFSTgramMFN_tfl_relu_max_s$s"; done
    kd=(--kd_teachers "${t#,}" --kd_teacher_dir "$TDIR" --kd_beta "$beta" --kd_tau 2)
fi

if [ ! -f $ck/student_e300.pth ]; then
    echo "[$(date +%T)] TRAIN start $desc gpu=$gpu"
    # the tfl_relu_max recipe exactly (../tfgram-linear/run_one.sh)
    $PY train.py --m 0.4 --fussion 1 --ht basic --gpu_num "$gpu" --num_workers 8 \
        --seed "$seed" --epoch 300 --tf_act relu --tf_pool max --desc "$desc" \
        --student_snapshots "10,30,90,300" ${kd[@]+"${kd[@]}"} \
        > $D/results/train_$desc.log 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f $ck/model.pth ]; then echo "[$(date +%T)] FAIL train $desc rc=$rc"; exit 1; fi
    echo "[$(date +%T)] TRAIN done $desc"
fi

mkdir -p $D/scores/$name
cp -p $ck/config.json $D/scores/$name/
for split in d e; do
    ds=dev; [ $split = e ] && ds=eval
    $PY eval.py --m 0.4 --fussion 1 --ht basic --gpu_num "$gpu" --tf_act relu --tf_pool max \
        --model_path $ck/model.pth --$split --csv $D/scores/$name/eval_$ds.csv \
        > $D/results/eval_${desc}_$split.log 2>&1 \
        || { echo "[$(date +%T)] FAIL eval $desc --$split"; exit 1; }
done
echo "[$(date +%T)] RUN done $desc"
