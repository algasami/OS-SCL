#!/usr/bin/env bash
# grand-ensemble (see README). CPU only, from existing extract.py dumps in other worktrees' results/emb.
#
#   bash synthesis/grand-ensemble/run.sh repro   # P0: reproduce A8, G8, F1k ... from the dumps
#   bash synthesis/grand-ensemble/run.sh dev     # selection on dev only; writes selection_dev.json, dev_pick.json
#   bash synthesis/grand-ensemble/run.sh eval    # only after selection_dev.json is committed; scores eval once
#   bash synthesis/grand-ensemble/run.sh plots
set -uo pipefail
cd "$(dirname "$0")/../.."
D=synthesis/grand-ensemble
PY=/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4   # GPUs' dataloaders share these CPUs
mkdir -p $D/results
case ${1:?usage: run.sh repro|dev|eval|plots} in
repro|dev|eval) $PY $D/score.py "$1" 2>&1 | tee $D/results/score_$1.log ;;
plots)          /home/f74134118/anaconda3/bin/python $D/make_plots.py ;;
*) echo "unknown stage"; exit 1 ;;
esac
