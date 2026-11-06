#!/usr/bin/env bash
# timepool-head (see README). Every stage is resumable.
#
#   GPUS="0 2 1" bash ablation/timepool-head/run.sh train    # tp x seeds 2040-2043 on ../train_one.sh (locks, crash-retry)
#   bash ablation/timepool-head/run.sh score                 # eval.py dev+eval CSVs, config/probe/wall, curves -> scores/
#   bash ablation/timepool-head/run.sh extract               # pooled-map + R0 + embedding dumps -> results/emb (scratch)
#   bash ablation/timepool-head/run.sh main                  # CPU: R0 / S1′ / C1 vs ao_nogain, D1 -> scores/
#   bash ablation/timepool-head/run.sh diag                  # CPU: ToyCar roll diagnostic -> scores/
#   bash ablation/timepool-head/run.sh plots
#   TP_SEEDS="2044 2045 2046 2047" bash ablation/timepool-head/run.sh train   # stage 2, only if D1 passes
#
# ao_nogain's dumps come from ../pooled-knn (AO_EMB); they are gitignored scratch in that worktree.
# If they are gone, rebuild them there with `bash ablation/pooled-knn/run.sh extractA` / `extractB`.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=ablation/timepool-head
PY=${PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
BPY=/home/f74134118/anaconda3/bin/python
SEEDS=${TP_SEEDS:-"2040 2041 2042 2043"}
GPU=${TP_GPU:-1}
EMB=$D/results/emb
AO_EMB=${AO_EMB:-/home/f74134118/OS-SCL-pknn/ablation/pooled-knn/results/emb}
export TP_SEEDS="$SEEDS"
mkdir -p $D/results $D/scores

done_runs() { for s in $SEEDS; do d=check_points/SCLTFSTgramMFN_tp_s$s; [ -f $d/done ] && basename $d; done; }

score_run() {   # score_run <name>
    local n=$1
    mkdir -p $D/scores/$n
    cp -p check_points/$n/{config.json,probe.csv,wall.csv} $D/scores/$n/
    if [ -f check_points/$n/crashes.csv ]; then cp -p check_points/$n/crashes.csv $D/scores/$n/; fi
    for split in d e; do
        ds=dev; [ $split = e ] && ds=eval
        [ -f $D/scores/$n/eval_$ds.csv ] && continue
        # eval.py reads lean / tpool from the checkpoint's config.json
        $PY eval.py --m 0.4 --fussion 1 --ht basic --gpu_num "$GPU" \
            --model_path check_points/$n/model.pth --$split --csv $D/scores/$n/eval_$ds.csv \
            > $D/results/eval_${n}_$split.log 2>&1 || { echo "FAIL eval $n --$split"; return 1; }
    done
}

# TP_NOLOCK=1 shares the GPU with whatever holds its lock (a short eval/extraction next to a training run)
lock() { [ "${TP_NOLOCK:-0}" = 1 ] && return 0
         mkdir -p /tmp/os-scl-tphead-score-locks
         until ! ls -d /tmp/os-scl-*-locks/gpu$GPU >/dev/null 2>&1 && mkdir /tmp/os-scl-tphead-score-locks/gpu$GPU 2>/dev/null; do sleep 60; done
         trap 'rmdir /tmp/os-scl-tphead-score-locks/gpu$GPU 2>/dev/null' EXIT; }
# smartlab1's faulty RAM crashes processes: retry (each finished run's dump is kept)
ext() { for t in 1 2 3; do
          OMP_NUM_THREADS=1 $PY $D/extract.py --gpu $GPU --out $EMB "$@" >> $D/results/extract.log 2>&1 && return 0
          echo "extract $* failed (try $t)" | tee -a $D/results/extract.log; done
        echo "FAIL extract $*"; exit 1; }

case ${1:?usage: run.sh train|score|extract|main|diag|plots} in
train)
    echo "=== train start $(date) | seeds=$SEEDS | GPUS=${GPUS:-0 1 2} ==="
    for s in $SEEDS; do echo "$s"; done | xargs -P 4 -L1 $D/run_one.sh
    echo "=== train end $(date) ===" ;;
score)
    lock
    for n in $(done_runs); do score_run "$n" || exit 1; done
    $PY ablation/distill_curves.py timepool-head ;;
extract)
    runs=$(done_runs); [ -n "$runs" ] || { echo "no finished runs"; exit 1; }
    for n in $runs; do [ -f $D/scores/$n/config.json ] || { echo "run.sh score first ($n)"; exit 1; }; done
    lock
    ext --split train --runs $runs
    ext --split dev --runs $runs
    ext --split eval --runs $runs
    ext --split dev --machine ToyCar --rolls 1,2,3,4,5,6,7 --group 4 --runs $runs
    ext --split eval --machine ToyCar --rolls 1,2,3,4,5,6,7 --group 4 --runs $runs ;;
main|diag)
    OMP_NUM_THREADS=8 $BPY $D/score.py $1 $EMB $AO_EMB | tee $D/results/score_$1.log ;;
plots)
    $BPY $D/make_plots.py ;;
*) echo "unknown stage"; exit 1 ;;
esac
