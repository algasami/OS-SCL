#!/usr/bin/env bash
# One trunk-soup run (see README):
#   run_one.sh trunk <trunk_seed>              300 epochs, full state saved at TS_FORK_AT
#   run_one.sh fork  <trunk_seed> <fork_seed>  resumes the trunk's state_e<TS_FORK_AT>.pth
#
# TS_FUSSION / TS_FLAGS: the base recipe fixed in the README addendum.
# GPU locking, crash-retry and records: ../train_one.sh.
set -uo pipefail
kind=$1; tseed=$2; fseed=${3:-}
FUSSION=${TS_FUSSION:?set TS_FUSSION from the README addendum}
FLAGS=${TS_FLAGS-}
AT=${TS_FORK_AT:-150}
base=(--m 0.4 --fussion "$FUSSION" --ht basic --num_workers 8 --seed "$tseed"
      --amp fp16 --cudnn_bench 1 --lean 1 --cache_dir data/cache --probe_every 5)
trunk=SCLTFSTgramMFN_ts_trunk_s${tseed}
case $kind in
trunk)
    # shellcheck disable=SC2086
    exec "$(dirname "$0")/../train_one.sh" trunk-soup "ts_trunk_s${tseed}" 300 "${base[@]}" \
        --state_snapshots "$AT" $FLAGS ;;
fork)
    state=check_points/${trunk}/state_e${AT}.pth
    cd "$(dirname "$0")/../.."
    until [ -f "$state" ]; do sleep 120; done   # forks queue behind their trunk
    # shellcheck disable=SC2086
    exec ablation/train_one.sh trunk-soup "ts_fork_s${tseed}_f${fseed}" 300 "${base[@]}" \
        --resume_state "$state" --fork_seed "$fseed" $FLAGS ;;
*) echo "unknown kind $kind"; exit 1 ;;
esac
