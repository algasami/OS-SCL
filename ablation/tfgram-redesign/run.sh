#!/usr/bin/env bash
# TFgram redesign: 2x2 factorial {relu, leaky} x {max, avg}, run in two stages.
#
#   bash ablation/tfgram-redesign/run.sh pilot   # 4 cells x 30 epochs, seed 2024, + CKA gate
#   bash ablation/tfgram-redesign/run.sh sweep   # 4 cells x 4 seeds x 300 epochs
#
# The sweep is only launched if the pilot passes the gate stated in README.md.
# Resumable at both stages: finished cells are skipped.
set -uo pipefail
cd "$(dirname "$0")/../.."
stage=${1:?usage: run.sh pilot|sweep}
D=ablation/tfgram-redesign
PY=${TFR_PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
export TFR_GPUS=${TFR_GPUS:-"1 2"}
SLOTS=$(echo $TFR_GPUS | wc -w)
mkdir -p "$D/results"
rmdir "${TFR_LOCKS:-/tmp/os-scl-tfr-locks}"/gpu* 2>/dev/null

cells() {  # tag seed act pool epochs
    for seed in $2; do for act in relu leaky; do for pool in max avg; do
        echo "$1 $seed $act $pool $3"
    done; done; done
}

echo "=== $stage start $(date) | gpus=$TFR_GPUS ==="
case $stage in
pilot)
    cells pilot 2024 30 | xargs -P "$SLOTS" -L1 $D/run_one.sh
    $PY $D/cka_probe.py ${TFR_GPUS%% *} $D/scores/cka_pilot.json \
        $(cells pilot 2024 30 | awk '{print "SCLTFSTgramMFN_"$1"_"$3"_"$4"_s"$2}')
    ;;
sweep)
    cells tfr "2024 2025 2026 2027" 300 | xargs -P "$SLOTS" -L1 $D/run_one.sh
    $PY $D/cka_probe.py ${TFR_GPUS%% *} $D/scores/cka_sweep.json \
        $(cells tfr "2024 2025 2026 2027" 300 | awk '{print "SCLTFSTgramMFN_"$1"_"$3"_"$4"_s"$2}')
    ;;
*) echo "unknown stage $stage"; exit 1 ;;
esac
$PY ablation/distill_curves.py tfgram-redesign
echo "=== $stage end $(date) ==="
