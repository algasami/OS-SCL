#!/usr/bin/env bash
# Fresh paper-model seeds for confirmation.
#
#   bash ablation/confirm-seeds/run.sh train   # 4 x 300 epochs, 2 GPUs at a time
#   bash ablation/confirm-seeds/run.sh score   # only after the README pre-registration is committed
#
# Training is split from scoring on purpose: no score of any kind, R0 included,
# is computed on these checkpoints until the predictions are in git.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=ablation/confirm-seeds
PY=${CS_PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
SEEDS="2028 2029 2030 2031"
export CS_GPUS=${CS_GPUS:-"1 2"}
SLOTS=$(echo $CS_GPUS | wc -w)
rmdir "${CS_LOCKS:-/tmp/os-scl-cs-locks}"/gpu* 2>/dev/null

case ${1:?usage: run.sh train|score} in
train)
    echo "=== train start $(date) | gpus=$CS_GPUS ==="
    printf '%s\n' $SEEDS | xargs -P "$SLOTS" -L1 $D/run_one.sh
    for s in $SEEDS; do
        n=SCLTFSTgramMFN_cs_relu_max_s$s
        mkdir -p $D/scores/$n && cp -p check_points/$n/config.json $D/scores/$n/
    done
    $PY ablation/distill_curves.py confirm-seeds
    echo "=== train end $(date) ===" ;;
score)
    echo "score stage not written yet: it lands with the pre-registration"; exit 1 ;;
*) echo "unknown stage"; exit 1 ;;
esac
