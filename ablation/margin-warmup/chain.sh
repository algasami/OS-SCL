#!/usr/bin/env bash
# Detached stage-1 chain. Waits for ../no-logmel's training queue to end (all 3 GPUs are
# busy until then), then: gate (abort on failure) -> pilot on GPUs 0+2 -> score -> diag -> plots.
# Stage 2 is NOT run here. Launch:
#   setsid nohup bash ablation/margin-warmup/chain.sh > ablation/margin-warmup/results/chain.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
D=ablation/margin-warmup
L=${NL_TRAIN_LOG:-/home/f74134118/OS-SCL-nomel/ablation/no-logmel/results/run_train.log}
echo "[$(date '+%F %T')] chain pid $$: waiting for '=== train end' in $L"
until grep -q '=== train end' "$L" 2>/dev/null; do sleep 300; done
echo "[$(date '+%F %T')] no-logmel training ended; gate"
bash $D/run.sh gate || { echo "[$(date '+%F %T')] GATE FAILED: chain aborted, no pilot"; exit 1; }
bash $D/run.sh pilot
echo "[$(date '+%F %T')] pilot finished; scoring"
bash $D/run.sh score || echo "[$(date '+%F %T')] score stage failed"
bash $D/run.sh diag || echo "[$(date '+%F %T')] diag stage failed"
bash $D/run.sh plots || echo "[$(date '+%F %T')] plots stage failed"
echo "[$(date '+%F %T')] chain done: stage 1 complete, stage 2 awaits review"
