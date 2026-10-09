#!/usr/bin/env bash
# TFgram redesign: 2x2 factorial {relu, leaky} x {max, avg}.
#
#   bash ablation/tfgram-redesign/run.sh pilot    # 4 cells x 30 epochs, seed 2024, + CKA
#   bash ablation/tfgram-redesign/run.sh probes   # init-only probes, no training
#
# The planned 300-epoch sweep was cancelled (README.md, "What this means for the
# sweep") and its stage removed; ../tfgram-linear continues the question.
# The pilot is resumable: finished cells are skipped.
set -uo pipefail
cd "$(dirname "$0")/../.."
stage=${1:?usage: run.sh pilot|probes}
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
probes)
    G=${TFR_GPUS%% *}
    $PY $D/init_probe.py "$G" $D/scores/init_probe.json
    $PY $D/tgram_probe.py "$G" $D/scores/tgram_probe.json
    $PY $D/band_probe.py "$G" $D/scores/band_probe.json
    echo "=== $stage end $(date) ==="; exit 0
    ;;
*) echo "unknown stage $stage"; exit 1 ;;
esac
$PY ablation/distill_curves.py tfgram-redesign
echo "=== $stage end $(date) ==="
