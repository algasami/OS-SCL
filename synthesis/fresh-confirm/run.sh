#!/usr/bin/env bash
# fresh-confirm (see README).
#
#   FC_TYPES="nogain" FC_SEEDS="2060 2061 2062 2063" bash synthesis/fresh-confirm/run.sh train
#   bash synthesis/fresh-confirm/run.sh records     # config/wall -> scores/<run>/ (needed by extract.py)
#   FC_GPU=0 bash synthesis/fresh-confirm/run.sh extract   # one-pass dumps -> results/emb (scratch)
#   bash synthesis/fresh-confirm/run.sh selftest    # scorer reproduces committed phase-1 numbers
#   bash synthesis/fresh-confirm/run.sh score       # ONCE: frozen rules -> scores/ (F1-F6)
#   FC_GPU=0 bash synthesis/fresh-confirm/run.sh evalpy    # eval.py CSVs + probe + curves (after score)
# Scoring is done by the frozen rules of ../single-final and ../grand-ensemble (README, Scoring).
set -uo pipefail
cd "$(dirname "$0")/../.."
D=synthesis/fresh-confirm
SEEDS=${FC_SEEDS:-"2060 2061 2062 2063"}
TYPES=${FC_TYPES:-"nogain"}
PY=${PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
mkdir -p $D/results $D/scores
case ${1:?usage: run.sh train|records} in
train)
    echo "=== train start $(date) | types=$TYPES seeds=$SEEDS ==="
    for s in $SEEDS; do for t in $TYPES; do echo "$t $s"; done; done | GPUS=${GPUS:-"0 2 1"} xargs -P 3 -L1 $D/run_one.sh
    echo "=== train end $(date) ===" ;;
records)
    for d in check_points/SCLTFSTgramMFN_fc_*; do
        [ -f "$d/done" ] || continue
        n=$(basename "$d"); mkdir -p $D/scores/$n
        cp -p "$d"/{config.json,wall.csv} $D/scores/$n/
        [ -f "$d/crashes.csv" ] && cp -p "$d/crashes.csv" $D/scores/$n/
    done ;;
extract)
    runs=$(for d in check_points/SCLTFSTgramMFN_fc_*; do [ -f "$d/done" ] && basename "$d"; done)
    for split in train dev eval; do
        $PY $D/extract.py --gpu "${FC_GPU:-0}" --split $split --out $D/results/emb --runs $runs \
            >> $D/results/extract.log 2>&1 || { echo "FAIL extract $split"; exit 1; }
    done ;;
selftest)
    OMP_NUM_THREADS=8 $PY $D/score.py selftest 2>&1 | tee $D/results/selftest.log ;;
score)
    OMP_NUM_THREADS=8 $PY $D/score.py score 2>&1 | tee $D/results/score.log ;;
evalpy)
    for d in check_points/SCLTFSTgramMFN_fc_*; do
        [ -f "$d/done" ] || continue
        n=$(basename "$d"); cp -p "$d/probe.csv" $D/scores/$n/ 2>/dev/null
        for split in d e; do
            ds=dev; [ $split = e ] && ds=eval
            [ -f $D/scores/$n/eval_$ds.csv ] && continue
            $PY eval.py --m 0.4 --fussion 1 --ht basic --gpu_num "${FC_GPU:-0}" \
                --model_path "$d/model.pth" --$split --csv $D/scores/$n/eval_$ds.csv \
                > $D/results/eval_${n}_$split.log 2>&1 || { echo "FAIL eval $n --$split"; exit 1; }
        done
    done
    $PY ablation/distill_curves.py ../synthesis/fresh-confirm ;;
*) echo "unknown stage"; exit 1 ;;
esac
