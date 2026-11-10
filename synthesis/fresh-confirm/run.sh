#!/usr/bin/env bash
# fresh-confirm (see README).
#
#   FC_TYPES="nogain" FC_SEEDS="2060 2061 2062 2063" bash synthesis/fresh-confirm/run.sh train
#   bash synthesis/fresh-confirm/run.sh records     # config/wall -> scores/<run>/ (needed by extract.py)
#   FC_GPU=0 bash synthesis/fresh-confirm/run.sh extract   # one-pass dumps -> results/emb (scratch)
#   bash synthesis/fresh-confirm/run.sh selftest    # scorer reproduces committed phase-1 numbers
#   bash synthesis/fresh-confirm/run.sh score       # ONCE: frozen rules -> scores/ (F1-F6)
#   FC_GPU=0 bash synthesis/fresh-confirm/run.sh evalpy    # eval.py CSVs + probe + curves (after score)
#   nohup setsid bash synthesis/fresh-confirm/run.sh post > .../results/post.log 2>&1 < /dev/null &   # detached: wait, records, extract
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
post)
    # Session-independent: waits for training to end, then records + extract (retried; faulty RAM).
    # Stops before `score`, which is run once by hand. Status line -> results/post.status.
    # pgrep patterns are anchored so shells that merely mention them in their command line don't match.
    st=$D/results/post.status
    echo "WAITING for training $(date)" > $st
    while pgrep -f "^\S*python train\.py .*--desc fc_" > /dev/null \
       || pgrep -f "^bash synthesis/fresh-confirm/run\.sh train" > /dev/null; do sleep 300; done
    n=$(ls check_points/SCLTFSTgramMFN_fc_*/done 2>/dev/null | wc -l)
    if [ "$n" -lt 8 ]; then echo "FAIL only $n/8 runs done $(date); see results/run_train*.log" > $st; exit 1; fi
    echo "RECORDS $(date)" > $st
    bash $0 records
    for try in 1 2 3; do
        echo "EXTRACT try $try $(date)" > $st
        bash $0 extract && break
        [ $try -eq 3 ] && { echo "FAIL extract after 3 tries $(date); see results/extract.log" > $st; exit 1; }
        sleep 30
    done
    echo "READY_TO_SCORE $(date) (8/8 trained, records + extract done; next: run.sh score, ONCE)" > $st ;;
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
