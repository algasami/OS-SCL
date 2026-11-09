#!/usr/bin/env bash
# fresh-confirm (see README).
#
#   FC_TYPES="nogain" FC_SEEDS="2060 2061 2062 2063" bash synthesis/fresh-confirm/run.sh train
#   bash synthesis/fresh-confirm/run.sh records     # config/probe/wall -> scores/, distilled curves
# Scoring is done by the frozen rules of ../single-final and ../grand-ensemble (README, Scoring).
set -uo pipefail
cd "$(dirname "$0")/../.."
D=synthesis/fresh-confirm
SEEDS=${FC_SEEDS:-"2060 2061 2062 2063"}
TYPES=${FC_TYPES:-"nogain"}
mkdir -p $D/results $D/scores
case ${1:?usage: run.sh train|records} in
train)
    echo "=== train start $(date) | types=$TYPES seeds=$SEEDS ==="
    for s in $SEEDS; do for t in $TYPES; do echo "$t $s"; done; done | GPUS=${GPUS:-"0 2 1"} xargs -P 3 -L1 $D/run_one.sh
    echo "=== train end $(date) ===" ;;
records)
    for d in check_points/SCLTFSTgramMFN_fc_*; do
        [ -f "$d/done" ] || continue
        n=$(basename "$d"); mkdir -p $D/scores/$n
        cp -p "$d"/{config.json,wall.csv} $D/scores/$n/
        [ -f "$d/crashes.csv" ] && cp -p "$d/crashes.csv" $D/scores/$n/
    done ;;
*) echo "unknown stage"; exit 1 ;;
esac
