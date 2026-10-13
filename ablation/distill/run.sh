#!/usr/bin/env bash
# Distillation: gate, then 4 students paired with the tfl_relu_max baselines.
#
#   KD_GPU=0 bash ablation/distill/run.sh gate         # G1 bit-exact repro + G2 timing (~1 h)
#   KD_GPU=0 bash ablation/distill/run.sh diag         # G1 diagnosis: 20 steps x 5 modes (~15 min)
#   KD_GPU=0 bash ablation/distill/run.sh null         # G1 diagnosis: 10 epochs, no teachers (~45 min)
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
train() {  # gpu seed desc [beta: omitted = no teachers, the plain baseline recipe]
    local kd=()
    [ $# -ge 4 ] && kd=(--kd_teachers "$(teachers "$2")" --kd_teacher_dir "$TDIR" --kd_beta "$4" --kd_tau 2)
    $PY train.py --m 0.4 --fussion 1 --ht basic --gpu_num "$1" --num_workers 8 \
        --seed "$2" --epoch 300 --tf_act relu --tf_pool max --desc "$3" \
        --student_snapshots "10,30,90,300" ${kd[@]+"${kd[@]}"}
}

# 10 epochs of seed 2024 at the full 300-epoch schedule, then compare student_e10.pth with
# the baseline's. `train` is a shell function, so $! is a subshell: kill the python by name.
ten_epochs() {  # desc json [beta: omit for no teachers at all]
    local ck=check_points/SCLTFSTgramMFN_$1
    rm -rf "$ck"
    if [ $# -ge 3 ]; then train "$GPU" 2024 "$1" "$3"
    else train "$GPU" 2024 "$1"; fi > $D/results/train_$1.log 2>&1 &
    local pid=$!
    until [ -f "$ck/student_e10.pth" ] || ! kill -0 $pid 2>/dev/null; do sleep 20; done
    sleep 30; pkill -f -- "--desc $1 " ; wait $pid 2>/dev/null
    $PY $D/gate.py "$ck" "$TDIR/SCLTFSTgramMFN_tfl_relu_max_s2024" \
        $D/results/train_$1.log "$BASE_LOGS/train_tfl_relu_max_s2024.log" "$2"
}

case ${1:?usage: run.sh gate|null|diag|students} in
gate)
    GPU=${KD_GPU:-0}
    echo "=== gate start $(date) | gpu=$GPU ==="
    ten_epochs kdgate_s2024 $D/scores/gate.json 0
    echo "=== gate end $(date) ===" ;;
null)  # G1 diagnosis: the baseline recipe with no teachers -- does this host reproduce at all?
    GPU=${KD_GPU:-0}
    echo "=== null start $(date) | gpu=$GPU ==="
    ten_epochs kdnull_s2024 $D/scores/gate_null.json
    echo "=== null end $(date) ===" ;;
diag)  # G1 diagnosis: 20 steps per mode, each in a fresh process
    GPU=${KD_GPU:-0}
    echo "=== diag start $(date) | gpu=$GPU ==="
    for m in none none2 loaded side add0; do
        $PY $D/diag_g1.py $m "$GPU" $D/results/diag_$m.pt 2>&1 | grep -v "it/s\]" | tail -2
        rm -rf check_points/SCLTFSTgramMFN_kddiag_$m
    done
    $PY $D/diag_g1.py compare $D/results | tee $D/scores/diag_g1.txt
    echo "=== diag end $(date) ===" ;;
students)
    python3 -c "import json,sys; sys.exit(0 if json.load(open('$D/scores/gate.json'))['G1_bit_exact'] else 1)" \
        || { echo "gate G1 has not passed; refusing to train students"; exit 1; }
    echo "students: not wired yet (lands after the gate result is recorded)"; exit 1 ;;
*) echo "unknown stage"; exit 1 ;;
esac
