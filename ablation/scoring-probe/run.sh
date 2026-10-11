#!/usr/bin/env bash
# Scoring probe: re-score finished checkpoints under R0-R4. No training.
#
#   bash ablation/scoring-probe/run.sh            # every finished tfgram-linear run
#   bash ablation/scoring-probe/run.sh RUN ...    # explicit model names
#
# A run is "finished" when its eval_dev.csv and eval_eval.csv are committed
# under some ablation/*/scores/ and its model.pth exists, so stage-2 runs that
# are still training are skipped and picked up on a later invocation.
# Resumable: dump_cosines.py skips (run, dataset) pairs already dumped, and the
# scoring + figures are cheap and always redone from all dumps.
#
# Env: PROBE_GPU (default 0), PROBE_REPO (where data/ and check_points/ live;
# default this checkout).
set -euo pipefail
cd "$(dirname "$0")/../.."
D=ablation/scoring-probe
PY=${PROBE_PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
PLOT_PY=/home/f74134118/anaconda3/bin/python
GPU=${PROBE_GPU:-0}
REPO=${PROBE_REPO:-$PWD}
mkdir -p "$D/results/cosines"

if [ $# -gt 0 ]; then
    RUNS="$*"
else
    RUNS=""
    for dir in ablation/*/scores/SCLTFSTgramMFN_tfl_*_s*; do
        name=$(basename "$dir")
        if [ -f "$dir/eval_dev.csv" ] && [ -f "$dir/eval_eval.csv" ] \
           && [ -f "$REPO/check_points/$name/model.pth" ]; then
            RUNS="$RUNS $name"
        else
            echo "skip $name (not finished, or no checkpoint)"
        fi
    done
fi
[ -n "$RUNS" ] || { echo "no runs to score"; exit 1; }

echo "=== scoring-probe start $(date) | gpu=$GPU | runs:$RUNS ==="
$PY $D/dump_cosines.py --gpu "$GPU" --data_root "$REPO/data" --ckpt_root "$REPO/check_points" \
    --out $D/results/cosines --datasets dev eval --runs $RUNS
$PY $D/score_rules.py $D/results/cosines
$PLOT_PY $D/make_plots.py
echo "=== scoring-probe end $(date) ==="
