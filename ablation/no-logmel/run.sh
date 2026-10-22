#!/usr/bin/env bash
# No Log-Mel (see README). Resumable: every stage skips work already done.
#
#   bash ablation/no-logmel/run.sh bench   # cost benchmark -> scores/bench.{csv,txt}, ~15 min, one idle GPU
#   bash ablation/no-logmel/run.sh gate    # 2-epoch smoke of f5/f6/f7 + strict eval.py load -> scores/gate.txt
#   bash ablation/no-logmel/run.sh train   # nl_{f5,f6,f7,f2} x seeds 2040-2043, 300 epochs, 3 GPUs
#   bash ablation/no-logmel/run.sh score   # eval.py dev + eval CSVs, probe, wall, curves -> scores/
#   bash ablation/no-logmel/run.sh plots   # summarise.py -> scores/summary.csv, make_plots.py (base python)
set -uo pipefail
cd "$(dirname "$0")/../.."
D=ablation/no-logmel
PY=${PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
SEEDS=${NL_SEEDS:-"2040 2041 2042 2043"}
ARMS=${NL_ARMS:-"f5 f6 f7 f2"}
GPU=${NL_GPU:-1}
mkdir -p $D/results $D/scores
[ -f data/cache/devtest_mel.npy ] || { echo "no data/cache: run cache_data.py (../fast-recipe)"; exit 1; }

score_run() {   # score_run <name>
    local n=$1 f
    f=$(sed -n 's/.*"fussion": \([0-9]\).*/\1/p' check_points/$n/config.json)
    mkdir -p $D/scores/$n
    cp -p check_points/$n/{config.json,probe.csv,wall.csv} $D/scores/$n/
    if [ -f check_points/$n/crashes.csv ]; then cp -p check_points/$n/crashes.csv $D/scores/$n/; fi
    for split in d e; do
        ds=dev; [ $split = e ] && ds=eval
        [ -f $D/scores/$n/eval_$ds.csv ] && continue
        $PY eval.py --m 0.4 --fussion "$f" --ht basic --gpu_num "$GPU" \
            --model_path check_points/$n/model.pth --$split --csv $D/scores/$n/eval_$ds.csv \
            > $D/results/eval_${n}_$split.log 2>&1 || { echo "FAIL eval $n --$split"; return 1; }
    done
}

case ${1:?usage: run.sh bench|gate|train|score|plots} in
bench)
    $PY $D/bench.py --gpu "$GPU" ;;
gate)
    for arm in f5 f6 f7; do
        GPUS=$GPU IDLE_SLEEP=1 $D/run_one.sh $arm 1 2 || exit 1
        n=SCLTFSTgramMFN_nl_${arm}_s1_e2
        $PY eval.py --m 0.4 --fussion ${arm#f} --ht basic --gpu_num "$GPU" \
            --model_path check_points/$n/model.pth --d > $D/results/eval_${n}_d.log 2>&1 \
            || { echo "FAIL eval $n"; exit 1; }
        echo "G $arm: $(grep -c . check_points/$n/probe.csv) probe lines," \
             "$(grep -c 'non-finite' $D/results/train_nl_${arm}_s1_e2.log) skipped steps," \
             "$(du -h check_points/$n/model.pth | cut -f1) ckpt, eval.py: $(grep 'Average AUC' $D/results/eval_${n}_d.log)" \
            | tee -a $D/scores/gate.txt
    done ;;
train)
    echo "=== train start $(date) | arms=$ARMS seeds=$SEEDS ==="
    # seed-major: at any stopping point every arm has the same seeds done
    for s in $SEEDS; do for arm in $ARMS; do echo "$arm $s"; done; done | xargs -P 3 -L1 $D/run_one.sh
    echo "=== train end $(date) ===" ;;
score)
    for arm in $ARMS; do for s in $SEEDS; do
        n=SCLTFSTgramMFN_nl_${arm}_s$s
        if [ -f check_points/$n/done ]; then score_run $n || exit 1; else echo "not trained: $n"; fi
    done; done
    $PY ablation/distill_curves.py no-logmel ;;
plots)
    /home/f74134118/anaconda3/bin/python $D/summarise.py | tee $D/scores/summary.txt
    /home/f74134118/anaconda3/bin/python $D/make_plots.py ;;
*) echo "unknown stage"; exit 1 ;;
esac
