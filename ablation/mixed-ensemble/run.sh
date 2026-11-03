#!/usr/bin/env bash
# Mixed-type E2′ ensembles (see README). CPU only; members are ../lit-arms, ../confirm-seeds
# and ../ensemble-backend extract.py dumps (regenerate with ablation/ensemble-backend/extract.py).
#
#   bash ablation/mixed-ensemble/run.sh dev     # stage A
#   bash ablation/mixed-ensemble/run.sh eval    # stage B: only after the eval predictions are committed
#   bash ablation/mixed-ensemble/run.sh plots
set -uo pipefail
cd "$(dirname "$0")/../.."
D=ablation/mixed-ensemble
PY=/home/f74134118/anaconda3/bin/python
case ${1:?usage: run.sh dev|eval|plots} in
dev|eval) $PY $D/score.py "$1" | tee $D/results/score_$1.log ;;
plots)    $PY $D/make_plots.py ;;
*) echo "unknown stage"; exit 1 ;;
esac
