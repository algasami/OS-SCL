#!/usr/bin/env bash
# aug-ops (see README).
#
#   bash ablation/aug-ops/run.sh train     # 3 leave-one-out arms x seeds 2040-2043, 3 GPUs
#   bash ablation/aug-ops/run.sh score     # eval.py dev+eval CSVs, probe, wall, curves -> scores/
#   bash ablation/aug-ops/run.sh plots     # make_plots.py -> plots/, summary -> scores/summary.txt
#   AO_ARMS=<arm> AO_SEEDS="2060 ... 2067" bash ablation/aug-ops/run.sh train   # stage 2, after D1 (not run: D0)
#   AO_ARMS=nogain AO_SEEDS="2044 2045 2046 2047" bash ablation/aug-ops/run.sh train   # ao_nogain replication
set -uo pipefail
cd "$(dirname "$0")/../.."
D=ablation/aug-ops
PY=${PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
SEEDS=${AO_SEEDS:-"2040 2041 2042 2043"}
ARMS=${AO_ARMS:-"nonoise noshift nogain"}
mkdir -p $D/results $D/scores

score_run() {   # score_run <name>
    local n=$1
    mkdir -p $D/scores/$n
    cp -p check_points/$n/{config.json,probe.csv,wall.csv} $D/scores/$n/
    if [ -f check_points/$n/crashes.csv ]; then cp -p check_points/$n/crashes.csv $D/scores/$n/; fi
    for split in d e; do
        ds=dev; [ $split = e ] && ds=eval
        [ -f $D/scores/$n/eval_$ds.csv ] && continue
        $PY eval.py --m 0.4 --fussion 1 --ht basic --gpu_num "${AO_SCORE_GPU:-0}" \
            --model_path check_points/$n/model.pth --$split --csv $D/scores/$n/eval_$ds.csv \
            > $D/results/eval_${n}_$split.log 2>&1 || { echo "FAIL eval $n --$split"; return 1; }
    done
}

case ${1:?usage: run.sh train|score|plots} in
train)
    echo "=== train start $(date) | arms=$ARMS seeds=$SEEDS ==="
    # seed-major, so every arm has early seeds to read
    for s in $SEEDS; do for arm in $ARMS; do echo "$arm $s"; done; done | xargs -P 3 -L1 $D/run_one.sh
    echo "=== train end $(date) ===" ;;
score)
    for d in check_points/SCLTFSTgramMFN_ao_*_s20[0-9][0-9]; do
        [ -f "$d/done" ] || continue
        score_run "$(basename $d)" || exit 1
    done
    $PY ablation/distill_curves.py aug-ops ;;
plots)
    /home/f74134118/anaconda3/bin/python $D/make_plots.py | tee $D/scores/summary.txt ;;
*) echo "unknown stage"; exit 1 ;;
esac
