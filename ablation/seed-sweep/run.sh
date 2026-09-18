#!/usr/bin/env bash
# Seed sweep over f1 / f4 / ft_strict / ft_bnadapt, 4 seeds each (16 runs).
#
# Every arm is run fresh under identical settings -- including seed 2024, which
# already has checkpoints from the earlier ablations. Those were fine, but the
# f1 baseline they were compared against was not: check_points/SCLTFSTgramMFN_test
# predates --num_workers and ran with 0, which desynchronises the torch RNG
# stream from epoch 1 on (batch order and mixup pairings differ; the split and
# the weight init do not). Rerunning 2024 here puts all four arms on one stream
# and doubles as a reproducibility check against the old checkpoints.
#
# Run from the repo root:  bash ablation/seed-sweep/run.sh
# Resumable: cells with both eval CSVs present are skipped.
set -uo pipefail
cd "$(dirname "$0")/../.."

SLOTS=${SWEEP_SLOTS:-2}       # concurrent runs = number of free GPUs
export SWEEP_NW=${SWEEP_NW:-8}
export SWEEP_OMP=${SWEEP_OMP:-1}

JOBS=ablation/seed-sweep/jobs.txt
OUT=ablation/seed-sweep/results
mkdir -p "$OUT"

echo "=== sweep start $(date) | slots=$SLOTS nw=$SWEEP_NW omp=$SWEEP_OMP ==="
mkdir -p "${SWEEP_LOCKS:-/tmp/os-scl-sweep-locks}"
rmdir "${SWEEP_LOCKS:-/tmp/os-scl-sweep-locks}"/gpu* 2>/dev/null
xargs -a "$JOBS" -P "$SLOTS" -L1 ablation/seed-sweep/run_one.sh
echo "=== sweep end $(date) ==="
