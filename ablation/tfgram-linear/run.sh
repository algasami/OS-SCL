#!/usr/bin/env bash
# TFgram linear: relu_max (paper) vs linear_avg, paired at each seed.
#
#   bash ablation/tfgram-linear/run.sh stage1   # seed 2024, 300 epochs, + branch probe
#   bash ablation/tfgram-linear/run.sh stage2   # seeds 2025-2027; only if the gate passes
#
# The gate (README.md) is read by a person from stage 1's probe and eval CSVs;
# this script does not check it. Resumable: finished arms are skipped.
set -uo pipefail
cd "$(dirname "$0")/../.."
stage=${1:?usage: run.sh stage1|stage2}
D=ablation/tfgram-linear
PY=${TFL_PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
export TFL_GPUS=${TFL_GPUS:-"1 2"}
SLOTS=$(echo $TFL_GPUS | wc -w)
mkdir -p "$D/results"
rmdir "${TFL_LOCKS:-/tmp/os-scl-tfl-locks}"/gpu* 2>/dev/null

case $stage in
stage1) SEEDS="2024" ;;
stage2) SEEDS="2025 2026 2027" ;;
*) echo "unknown stage $stage"; exit 1 ;;
esac

arms() {  # seed act pool
    for seed in $SEEDS; do
        echo "$seed relu max"
        echo "$seed linear avg"
    done
}

echo "=== $stage start $(date) | gpus=$TFL_GPUS ==="
arms | xargs -P "$SLOTS" -L1 $D/run_one.sh
$PY $D/branch_probe.py ${TFL_GPUS%% *} $D/scores/branch_probe_${stage}.json \
    $(arms | awk '{print "SCLTFSTgramMFN_tfl_"$2"_"$3"_s"$1}')
$PY ablation/distill_curves.py tfgram-linear
echo "=== $stage end $(date) ==="
