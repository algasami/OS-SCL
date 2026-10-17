#!/usr/bin/env bash
# Ensemble + back-end: re-score the 8 tfgram-linear checkpoints. No training.
#
#   bash ablation/ensemble-backend/run.sh
#
# Env: EB_GPU (default 0), EB_REPO (checkout holding data/ and check_points/;
# default this checkout). Resumable: extract.py skips runs already dumped;
# scoring and figures are cheap and always redone.
set -euo pipefail
cd "$(dirname "$0")/../.."
D=ablation/ensemble-backend
PY=${EB_PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
PLOT_PY=/home/f74134118/anaconda3/bin/python
GPU=${EB_GPU:-0}
REPO=${EB_REPO:-$PWD}
RUNS=""
for arm in relu_max linear_avg; do
    for s in 2024 2025 2026 2027; do RUNS="$RUNS SCLTFSTgramMFN_tfl_${arm}_s$s"; done
done
mkdir -p $D/results/emb

echo "=== ensemble-backend start $(date) | gpu=$GPU ==="
$PY $D/extract.py --gpu "$GPU" --repo "$REPO" --out $D/results/emb --runs $RUNS
$PY $D/score.py $D/results/emb
$PLOT_PY $D/make_plots.py
echo "=== ensemble-backend end $(date) ==="
