#!/usr/bin/env bash
# single-final (see README). CPU only, from existing dumps; every stage refuses to run until the
# previous stage's output is committed, and `eval` refuses once eval.json is committed.
#
#   bash synthesis/single-final/run.sh check    # reproduce committed C1 / S1′ / S1′ᵀ -> scores/repro_check.json
#   bash synthesis/single-final/run.sh stageA   # dev 2040-2043: C1ᵀ vs C1 (gate)
#   bash synthesis/single-final/run.sh stageB   # dev 2044-2047 + 8-seed dev
#   bash synthesis/single-final/run.sh eval     # eval x8, once; decision + scores/final_rule.json
#   bash synthesis/single-final/run.sh summary  # from scores/ only: summary.csv, perid_delta.csv
#   bash synthesis/single-final/run.sh plots
#
# Dumps: SF_TTA_EMB (default ../OS-SCL-tta/ablation/tta/results/emb, from ablation/tta/run.sh) and
# SF_PK_EMB (default ../OS-SCL-pknn/ablation/pooled-knn/results/emb, from ablation/pooled-knn/run.sh).
set -uo pipefail
cd "$(dirname "$0")/../.."
D=synthesis/single-final
PY=${PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
BPY=/home/f74134118/anaconda3/bin/python
mkdir -p $D/results $D/scores
# smartlab1's faulty RAM can crash a process: retry (score.py itself guards against silent flips)
sc() { for t in 1 2 3; do
         OMP_NUM_THREADS=8 $PY $D/score.py "$1" > $D/results/score_$1.log 2>&1 && { cat $D/results/score_$1.log; return 0; }
         grep -q ABORT $D/results/score_$1.log && { tail -5 $D/results/score_$1.log; exit 1; }
         echo "score $1 crashed (try $t)"; done
       exit 1; }
case ${1:?usage: run.sh check|stageA|stageB|eval|summary|plots} in
check|stageA|stageB|eval) sc $1 ;;
summary) $PY $D/score.py summary ;;
plots) $BPY $D/make_plots.py ;;
*) echo "unknown stage"; exit 1 ;;
esac
