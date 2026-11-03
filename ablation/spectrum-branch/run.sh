#!/usr/bin/env bash
# spectrum-branch (see README).
#
#   bash ablation/spectrum-branch/run.sh train   # sb_spec x seeds 2040-2043 (GPUS="0 1 2" by default)
#   bash ablation/spectrum-branch/run.sh score   # eval.py dev+eval CSVs, probe, wall, curves -> scores/
#   bash ablation/spectrum-branch/run.sh plots   # score.py summary + make_plots.py -> scores/, plots/
#   SB_SEEDS="2044 2045 2046 2047" bash ablation/spectrum-branch/run.sh train   # stage 2, only if D1 passes
#
# Every stage is resumable: train_one.sh skips finished runs, score skips scored splits.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=ablation/spectrum-branch
PY=${PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
SEEDS=${SB_SEEDS:-"2040 2041 2042 2043"}
mkdir -p $D/results $D/scores

score_run() {   # score_run <name>
    local n=$1
    mkdir -p $D/scores/$n
    cp -p check_points/$n/{config.json,probe.csv,wall.csv} $D/scores/$n/
    if [ -f check_points/$n/crashes.csv ]; then cp -p check_points/$n/crashes.csv $D/scores/$n/; fi
    for split in d e; do
        ds=dev; [ $split = e ] && ds=eval
        [ -f $D/scores/$n/eval_$ds.csv ] && continue
        # eval.py reads lean / spec_branch from the checkpoint's config.json
        $PY eval.py --m 0.4 --fussion 1 --ht basic --gpu_num "${SB_SCORE_GPU:-0}" \
            --model_path check_points/$n/model.pth --$split --csv $D/scores/$n/eval_$ds.csv \
            > $D/results/eval_${n}_$split.log 2>&1 || { echo "FAIL eval $n --$split"; return 1; }
    done
}

case ${1:?usage: run.sh train|score|plots} in
train)
    echo "=== train start $(date) | seeds=$SEEDS ==="
    for s in $SEEDS; do echo "$s"; done | xargs -P 3 -L1 $D/run_one.sh
    echo "=== train end $(date) ===" ;;
score)
    for d in check_points/SCLTFSTgramMFN_sb_*_s20[0-9][0-9]; do
        [ -f "$d/done" ] || continue
        score_run "$(basename $d)" || exit 1
    done
    runs=$(for d in check_points/SCLTFSTgramMFN_sb_*_s20[0-9][0-9]; do [ -f "$d/done" ] && basename $d; done)
    [ -n "$runs" ] && { $PY $D/head_share.py --gpu "${SB_SCORE_GPU:-0}" $runs || exit 1; }
    $PY ablation/distill_curves.py spectrum-branch ;;
plots)
    /home/f74134118/anaconda3/bin/python $D/score.py | tee $D/scores/summary.txt
    /home/f74134118/anaconda3/bin/python $D/make_plots.py ;;
*) echo "unknown stage"; exit 1 ;;
esac
