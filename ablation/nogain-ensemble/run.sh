#!/usr/bin/env bash
# E2′ ensembles of ao_nogain vs E2′[A8] (see README). CPU only; every member is already trained
# and dumped by ../ensemble-backend/extract.py (aug-ops, lit-arms, confirm-seeds results/emb).
#
#   bash ablation/nogain-ensemble/run.sh dev     # stage A
#   bash ablation/nogain-ensemble/run.sh eval    # stage B: only after the eval predictions are committed
#   bash ablation/nogain-ensemble/run.sh plots
set -uo pipefail
cd "$(dirname "$0")/../.."
D=ablation/nogain-ensemble
PY=/home/f74134118/anaconda3/bin/python
mkdir -p $D/results
case ${1:?usage: run.sh dev|eval|plots} in
dev|eval) $PY $D/score.py "$1" | tee $D/results/score_$1.log ;;
plots)    $PY $D/make_plots.py ;;
*) echo "unknown stage"; exit 1 ;;
esac
