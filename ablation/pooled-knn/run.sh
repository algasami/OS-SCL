#!/usr/bin/env bash
# Pooled-kNN ablation (see README). Every stage is resumable; GPU stages take a lock dir
# under /tmp/os-scl-*-locks so other runners skip that GPU.
#
#   bash ablation/pooled-knn/run.sh extractA   # ao_nogain x8: train bank + dev clips (no eval)
#   bash ablation/pooled-knn/run.sh select     # CPU: P0 + the six candidates on seeds 2040-2043 dev -> scores/select.json
#   bash ablation/pooled-knn/run.sh extractB   # only once scores/select.json is committed: eval clips + ToyCar rolls
#   bash ablation/pooled-knn/run.sh confirm    # CPU: confirm seeds dev, eval x8, ensemble, bootstrap
#   bash ablation/pooled-knn/run.sh diag       # CPU: ToyCar penalty / AUC vs roll
#   bash ablation/pooled-knn/run.sh plots
set -uo pipefail
cd "$(dirname "$0")/../.."
D=ablation/pooled-knn
PY=${PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
BPY=/home/f74134118/anaconda3/bin/python
GPU=${PK_GPU:-1}
EMB=$D/results/emb
RUNS=""; for s in 2040 2041 2042 2043 2044 2045 2046 2047; do RUNS="$RUNS SCLTFSTgramMFN_ao_nogain_s$s"; done
mkdir -p $D/results $D/scores

lock() { mkdir -p /tmp/os-scl-pknn-locks
         until ! ls -d /tmp/os-scl-*-locks/gpu$GPU >/dev/null 2>&1 && mkdir /tmp/os-scl-pknn-locks/gpu$GPU 2>/dev/null; do sleep 60; done
         trap 'rmdir /tmp/os-scl-pknn-locks/gpu$GPU 2>/dev/null' EXIT; }
# smartlab1's faulty RAM crashes processes: retry (each finished run's dump is kept)
ext() { for t in 1 2 3; do
          OMP_NUM_THREADS=1 $PY $D/extract.py --gpu $GPU --out $EMB "$@" >> $D/results/extract.log 2>&1 && return 0
          echo "extract $* failed (try $t)" | tee -a $D/results/extract.log; done
        echo "FAIL extract $*"; exit 1; }

case ${1:?usage: run.sh extractA|select|extractB|confirm|diag|plots} in
extractA) lock; ext --split train --runs $RUNS; ext --split dev --runs $RUNS ;;
select)   $BPY $D/score.py select $EMB | tee $D/results/score_select.log ;;
extractB) git ls-files --error-unmatch $D/scores/select.json >/dev/null 2>&1 \
              || { echo "select.json not committed: eval stays unextracted"; exit 1; }
          lock; ext --split eval --runs $RUNS
          ext --split dev --machine ToyCar --rolls 1,2,3,4,5,6,7 --group 4 --runs $RUNS
          ext --split eval --machine ToyCar --rolls 1,2,3,4,5,6,7 --group 4 --runs $RUNS ;;
confirm|diag) $BPY $D/score.py $1 $EMB | tee $D/results/score_$1.log ;;
plots)    $BPY $D/make_plots.py ;;
*) echo "unknown stage"; exit 1 ;;
esac
