#!/usr/bin/env bash
# Fresh paper-model seeds for confirmation.
#
#   bash ablation/confirm-seeds/run.sh train   # 4 x 300 epochs, 2 GPUs at a time
#   bash ablation/confirm-seeds/run.sh score   # only after the README pre-registration is committed
#
# Training is split from scoring on purpose: no score of any kind, R0 included,
# is computed on these checkpoints until the predictions are in git.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=ablation/confirm-seeds
PY=${CS_PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
SEEDS="2028 2029 2030 2031"
export CS_GPUS=${CS_GPUS:-"1 2"}
SLOTS=$(echo $CS_GPUS | wc -w)
rmdir "${CS_LOCKS:-/tmp/os-scl-cs-locks}"/gpu* 2>/dev/null

case ${1:?usage: run.sh train|score} in
train)
    echo "=== train start $(date) | gpus=$CS_GPUS ==="
    printf '%s\n' $SEEDS | xargs -P "$SLOTS" -L1 $D/run_one.sh
    for s in $SEEDS; do
        n=SCLTFSTgramMFN_cs_relu_max_s$s
        mkdir -p $D/scores/$n && cp -p check_points/$n/config.json $D/scores/$n/
    done
    $PY ablation/distill_curves.py confirm-seeds
    echo "=== train end $(date) ===" ;;
score)  # the pre-registration (README) is committed; nothing below may tune on its output
    OLD=${CS_OLD_CKPT:-/home/f74134118/OS-SCL/check_points}
    RUNS=""; for s in $SEEDS; do RUNS="$RUNS SCLTFSTgramMFN_cs_relu_max_s$s"; done
    for s in 2024 2025 2026 2027; do
        n=SCLTFSTgramMFN_tfl_relu_max_s$s; RUNS="$RUNS $n"
        [ -e check_points/$n ] || ln -s "$OLD/$n" check_points/$n
    done
    # one idle GPU: no other ablation's lock, under 1 GB used on 3 checks a minute apart
    LOCKS=${CS_LOCKS:-/tmp/os-scl-cs-locks}; mkdir -p "$LOCKS"; gpu=""
    idle() {
        for _ in 1 2 3; do
            for l in /tmp/os-scl-*-locks/gpu"$1"; do
                [ -d "$l" ] && [ "${l%/gpu*}" != "$LOCKS" ] && return 1
            done
            [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] || return 1
            sleep 60
        done
    }
    while [ -z "$gpu" ]; do
        for g in ${CS_SCORE_GPUS:-0 1 2}; do
            if mkdir "$LOCKS/gpu$g" 2>/dev/null; then
                if idle "$g"; then gpu=$g; break; fi
                rmdir "$LOCKS/gpu$g"
            fi
        done
        [ -z "$gpu" ] && sleep 120
    done
    trap 'rmdir "$LOCKS/gpu$gpu" 2>/dev/null' EXIT
    echo "=== score start $(date) | gpu=$gpu ==="
    for s in $SEEDS; do  # R0 from eval.py itself, the P0 reference
        n=SCLTFSTgramMFN_cs_relu_max_s$s
        for split in d e; do
            ds=dev; [ $split = e ] && ds=eval
            [ -f $D/scores/$n/eval_$ds.csv ] && continue
            $PY eval.py --m 0.4 --fussion 1 --ht basic --gpu_num "$gpu" --tf_act relu --tf_pool max \
                --model_path check_points/$n/model.pth --$split --csv $D/scores/$n/eval_$ds.csv \
                > $D/results/eval_${n}_$split.log 2>&1 || { echo "FAIL eval $n --$split"; exit 1; }
        done
    done
    $PY ablation/ensemble-backend/extract.py --gpu "$gpu" --repo "$PWD" --out $D/results/emb --runs $RUNS \
        || { echo "FAIL extract"; exit 1; }
    $PY $D/score.py $D/results/emb || { echo "FAIL score"; exit 1; }
    echo "=== score end $(date) ===" ;;
*) echo "unknown stage"; exit 1 ;;
esac
