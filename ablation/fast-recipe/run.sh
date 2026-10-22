#!/usr/bin/env bash
# Fast recipe (see README).
#
#   bash ablation/fast-recipe/run.sh gate    # cache verify + G1-G3, ~30 min, one GPU
#   bash ablation/fast-recipe/run.sh train   # fr_f1 + fr_f4, seeds 2040-2043, 300 epochs
#   bash ablation/fast-recipe/run.sh score   # eval.py dev + eval CSVs, probe.csv and curves into scores/
set -uo pipefail
cd "$(dirname "$0")/../.."
D=ablation/fast-recipe
PY=${FR_PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
SEEDS="2040 2041 2042 2043"
mkdir -p $D/results $D/scores
[ -f data/cache/devtest_mel.npy ] || $PY cache_data.py || { echo "FAIL cache"; exit 1; }

case ${1:?usage: run.sh gate|train|score} in
gate)
    $PY cache_data.py --verify 2>&1 | grep -v "class :" > $D/scores/cache_verify.txt || { echo "FAIL verify"; exit 1; }
    cat $D/scores/cache_verify.txt
    $PY $D/gate.py --gpu "${FR_GATE_GPU:-1}" || { echo "FAIL gate"; exit 1; }
    for arm in f1 f4; do   # G3 smoke: 2 epochs, then a strict eval.py load
        FR_IDLE_SLEEP=1 $D/run_one.sh $arm 1 2 || exit 1
        n=SCLTFSTgramMFN_fr_${arm}_s1_e2
        $PY eval.py --m 0.4 --fussion ${arm#f} --ht basic --gpu_num "${FR_GATE_GPU:-1}" \
            --model_path check_points/$n/model.pth --d > $D/results/eval_${n}_d.log 2>&1 \
            || { echo "FAIL eval $n"; exit 1; }
        echo "G3 $arm: $(grep -c . check_points/$n/probe.csv) probe lines, $(du -h check_points/$n/model.pth | cut -f1) ckpt," \
             "eval.py: $(grep 'Average AUC' $D/results/eval_${n}_d.log)" | tee -a $D/scores/gate.txt
    done ;;
train)
    echo "=== train start $(date) ==="
    # f1 runs first: they are the long ones (~2x f4), so the queue drains evenly
    for arm in f1 f4; do for s in $SEEDS; do echo "$arm $s"; done; done \
        | xargs -P 3 -L1 $D/run_one.sh
    echo "=== train end $(date) ===" ;;
score)
    for arm in f1 f4; do for s in $SEEDS; do
        n=SCLTFSTgramMFN_fr_${arm}_s$s; mkdir -p $D/scores/$n
        cp -p check_points/$n/{config.json,probe.csv,wall.csv} $D/scores/$n/
        for split in d e; do
            ds=dev; [ $split = e ] && ds=eval
            [ -f $D/scores/$n/eval_$ds.csv ] && continue
            $PY eval.py --m 0.4 --fussion ${arm#f} --ht basic --gpu_num "${FR_GATE_GPU:-1}" \
                --model_path check_points/$n/model.pth --$split --csv $D/scores/$n/eval_$ds.csv \
                > $D/results/eval_${n}_$split.log 2>&1 || { echo "FAIL eval $n --$split"; exit 1; }
        done
    done; done
    $PY ablation/distill_curves.py fast-recipe ;;
*) echo "unknown stage"; exit 1 ;;
esac
