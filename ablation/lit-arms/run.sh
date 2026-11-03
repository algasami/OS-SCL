#!/usr/bin/env bash
# Literature arms (see README).
#
#   LA_FUSSION=4 bash ablation/lit-arms/run.sh train          # la_aug / la_gate / la_eca x seeds 2040-2043
#   LA_FUSSION=4 bash ablation/lit-arms/run.sh score          # eval.py dev+eval CSVs, probe, wall, curves -> scores/
#   LA_FUSSION=4 bash ablation/lit-arms/run.sh plots          # make_plots.py -> plots/, summary -> scores/summary.txt
#   LA_FUSSION=4 LA_COMBO="..." bash ablation/lit-arms/run.sh combo   # stage 2, only after D1 is recorded
#   LA_FUSSION=1 LA_ARMS=aug LA_SEEDS="2044 2045 2046 2047" bash ablation/lit-arms/run.sh train   # D2, one arm passed
set -uo pipefail
cd "$(dirname "$0")/../.."
D=ablation/lit-arms
PY=${PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
FUSSION=${LA_FUSSION:?set LA_FUSSION}
SEEDS=${LA_SEEDS:-"2040 2041 2042 2043"}   # D2 replication: LA_SEEDS="2044 2045 2046 2047"
ARMS=${LA_ARMS:-"aug gate eca"}
mkdir -p $D/results $D/scores

score_run() {   # score_run <name>
    local n=$1
    mkdir -p $D/scores/$n
    cp -p check_points/$n/{config.json,probe.csv,wall.csv} $D/scores/$n/
    if [ -f check_points/$n/crashes.csv ]; then cp -p check_points/$n/crashes.csv $D/scores/$n/; fi
    for split in d e; do
        ds=dev; [ $split = e ] && ds=eval
        [ -f $D/scores/$n/eval_$ds.csv ] && continue
        $PY eval.py --m 0.4 --fussion "$FUSSION" --ht basic --gpu_num "${LA_SCORE_GPU:-0}" \
            --model_path check_points/$n/model.pth --$split --csv $D/scores/$n/eval_$ds.csv \
            > $D/results/eval_${n}_$split.log 2>&1 || { echo "FAIL eval $n --$split"; return 1; }
    done
}

case ${1:?usage: run.sh train|score|combo|plots} in
train)
    echo "=== train start $(date) | fussion=$FUSSION arms=$ARMS ==="
    for arm in $ARMS; do for s in $SEEDS; do echo "$arm $s"; done; done | xargs -P 3 -L1 $D/run_one.sh
    echo "=== train end $(date) ===" ;;
combo)
    echo "=== combo start $(date) | $LA_COMBO ==="
    for s in ${LA_COMBO_SEEDS:-$SEEDS}; do echo "combo $s"; done | xargs -P 3 -L1 $D/run_one.sh
    echo "=== combo end $(date) ===" ;;
score)
    for d in check_points/SCLTFSTgramMFN_la_*_s20[0-9][0-9]; do score_run "$(basename $d)" || exit 1; done
    $PY ablation/distill_curves.py lit-arms ;;
plots)
    /home/f74134118/anaconda3/bin/python $D/make_plots.py | tee $D/scores/summary.txt ;;
*) echo "unknown stage"; exit 1 ;;
esac
