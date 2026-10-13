#!/usr/bin/env bash
# Distillation: gate, then 4 students paired with the tfl_relu_max baselines.
#
#   KD_GPU=0 bash ablation/distill/run.sh gate         # G1 bit-exact repro + G2 timing (~45 min)
#   KD_GPUS="0 1 2" bash ablation/distill/run.sh students   # only after the gate passes
#
# Teachers are leave-one-seed-out: the student at seed s learns from the other three
# tfl_relu_max seeds. They and the baselines live in KD_TEACHER_DIR (the main checkout).
set -uo pipefail
cd "$(dirname "$0")/../.."
D=ablation/distill
PY=${KD_PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
TDIR=${KD_TEACHER_DIR:-/home/f74134118/OS-SCL/check_points}
BASE_LOGS=${KD_BASE_LOGS:-/home/f74134118/OS-SCL/ablation/tfgram-linear/results}
SEEDS="2024 2025 2026 2027"
mkdir -p $D/results $D/scores
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

teachers() {  # leave-one-seed-out
    local out=""
    for t in $SEEDS; do [ "$t" != "$1" ] && out="$out,SCLTFSTgramMFN_tfl_relu_max_s$t"; done
    echo "${out#,}"
}

# the baseline recipe exactly (../tfgram-linear/run_one.sh); --epoch must stay 300 even
# for the gate, since the cosine schedule's T_max is the epoch count
train() {  # gpu seed desc beta
    $PY train.py --m 0.4 --fussion 1 --ht basic --gpu_num "$1" --num_workers 8 \
        --seed "$2" --epoch 300 --tf_act relu --tf_pool max --desc "$3" \
        --student_snapshots "10,30,90,300" \
        --kd_teachers "$(teachers "$2")" --kd_teacher_dir "$TDIR" --kd_beta "$4" --kd_tau 2
}

case ${1:?usage: run.sh gate|students} in
gate)
    GPU=${KD_GPU:-0}; name=SCLTFSTgramMFN_kdgate_s2024; ck=check_points/$name
    rm -rf "$ck"
    echo "=== gate start $(date) | gpu=$GPU ==="
    train "$GPU" 2024 kdgate_s2024 0 > $D/results/train_kdgate_s2024.log 2>&1 &
    pid=$!
    # stop once the epoch-10 student snapshot is fully written
    until [ -f "$ck/student_e10.pth" ] || ! kill -0 $pid 2>/dev/null; do sleep 20; done
    sleep 30; kill $pid 2>/dev/null; wait $pid 2>/dev/null
    $PY $D/gate.py "$ck" "$TDIR/SCLTFSTgramMFN_tfl_relu_max_s2024" \
        $D/results/train_kdgate_s2024.log "$BASE_LOGS/train_tfl_relu_max_s2024.log" $D/scores/gate.json
    echo "=== gate end $(date) ===" ;;
students)
    python3 -c "import json,sys; sys.exit(0 if json.load(open('$D/scores/gate.json'))['G1_bit_exact'] else 1)" \
        || { echo "gate G1 has not passed; refusing to train students"; exit 1; }
    echo "students: not wired yet (lands after the gate result is recorded)"; exit 1 ;;
*) echo "unknown stage"; exit 1 ;;
esac
