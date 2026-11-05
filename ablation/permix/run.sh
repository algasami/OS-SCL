#!/usr/bin/env bash
# permix (see README).
#
#   GPUS="0 2 1" bash ablation/permix/run.sh train   # pm x seeds 2040-2043 (3 at a time)
#   bash ablation/permix/run.sh score   # eval.py dev+eval CSVs (R0), probe, wall, curves -> scores/
#   bash ablation/permix/run.sh s1p     # extract.py dumps (results/emb, scratch) -> score_s1p.py -> scores/s1p.csv
#                                       # needs ../aug-ops dumps: AO_EMB=<aug-ops results/emb>
#   bash ablation/permix/run.sh plots   # score.py summary (S1′, R0, K; D1) + make_plots.py -> scores/, plots/
#   PM_SEEDS="2044 2045 2046 2047" bash ablation/permix/run.sh train   # stage 2, only if D1 passes
#
# Every stage is resumable: train_one.sh skips finished runs, score skips scored splits,
# extract.py skips dumped runs.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=ablation/permix
PY=${PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
SEEDS=${PM_SEEDS:-"2040 2041 2042 2043"}
mkdir -p $D/results $D/scores

score_run() {   # score_run <name>
    local n=$1
    mkdir -p $D/scores/$n
    cp -p check_points/$n/{config.json,probe.csv,wall.csv} $D/scores/$n/
    if [ -f check_points/$n/crashes.csv ]; then cp -p check_points/$n/crashes.csv $D/scores/$n/; fi
    for split in d e; do
        ds=dev; [ $split = e ] && ds=eval
        [ -f $D/scores/$n/eval_$ds.csv ] && continue
        # eval.py reads lean from the checkpoint's config.json; --permix is training-only
        $PY eval.py --m 0.4 --fussion 1 --ht basic --gpu_num "${PM_SCORE_GPU:-0}" \
            --model_path check_points/$n/model.pth --$split --csv $D/scores/$n/eval_$ds.csv \
            > $D/results/eval_${n}_$split.log 2>&1 || { echo "FAIL eval $n --$split"; return 1; }
    done
}

runs_done() { for d in check_points/SCLTFSTgramMFN_pm_s20[0-9][0-9]; do [ -f "$d/done" ] && basename $d; done; }

case ${1:?usage: run.sh train|score|s1p|plots} in
train)
    echo "=== train start $(date) | seeds=$SEEDS | GPUS=${GPUS:-0 1 2} ==="
    for s in $SEEDS; do echo "$s"; done | xargs -P 3 -L1 $D/run_one.sh
    echo "=== train end $(date) ===" ;;
score)
    for n in $(runs_done); do score_run "$n" || exit 1; done
    $PY ablation/distill_curves.py permix ;;
s1p)
    runs=$(runs_done)
    $PY ablation/ensemble-backend/extract.py --gpu "${PM_SCORE_GPU:-0}" --repo "$PWD" --out $D/results/emb \
        --runs $runs > $D/results/extract.log 2>&1 || { echo "FAIL extract"; exit 1; }
    /home/f74134118/anaconda3/bin/python $D/score_s1p.py $D/results/emb \
        "${AO_EMB:-/home/f74134118/OS-SCL-augops/ablation/aug-ops/results/emb}" ;;
plots)
    /home/f74134118/anaconda3/bin/python $D/score.py | tee $D/scores/summary.txt
    /home/f74134118/anaconda3/bin/python $D/make_plots.py ;;
*) echo "unknown stage"; exit 1 ;;
esac
