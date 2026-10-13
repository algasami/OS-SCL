#!/usr/bin/env bash
# Memory bank: patch-level kNN on the 8 ../tfgram-linear checkpoints. No training.
#
#   bash ablation/memory-bank/run.sh bank      # GPU; MB_RUNS picks checkpoints, so several
#                                              # shells can split the 8 across GPUs
#   bash ablation/memory-bank/run.sh score     # CPU, once every bank is done
#
# Env: MB_GPU (default 0), MB_REPO (checkout holding data/ and check_points/; default
# this one), MB_EMB (../ensemble-backend's cached results/emb), MB_RUNS (default all 8),
# MB_MEL (1 = also run the seed-free imagenet_mel arm; give it to exactly one shell),
# MB_ARMS (default all; e.g. "mfn tfgram imagenet" first, the sensitivity arms after).
# Resumable per (arm, checkpoint, ID): finished files are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
D=ablation/memory-bank
PY=${MB_PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
GPU=${MB_GPU:-0}
REPO=${MB_REPO:-$PWD}
EMB=${MB_EMB:-/home/f74134118/OS-SCL-scoring-probe/ablation/ensemble-backend/results/emb}
ALL_RUNS=""
for arm in relu_max linear_avg; do
    for s in 2024 2025 2026 2027; do ALL_RUNS="$ALL_RUNS SCLTFSTgramMFN_tfl_${arm}_s$s"; done
done
RUNS=${MB_RUNS:-$ALL_RUNS}
ARMS=${MB_ARMS:-"mfn mfn_l2 mfn_l3 mfn_cs10 tfgram imagenet"}
MEL=""
[ "${MB_MEL:-0}" = 1 ] && MEL=--imagenet-mel
mkdir -p $D/results

case ${1:?usage: run.sh bank|score} in
bank)
    echo "=== memory-bank bank start $(date) | gpu=$GPU runs=$RUNS arms=$ARMS $MEL ==="
    $PY $D/bank.py --gpu "$GPU" --repo "$REPO" --emb "$EMB" --out $D/results --runs $RUNS --arms $ARMS $MEL
    echo "=== memory-bank bank end $(date) ===" ;;
score)
    $PY $D/score.py $D/results/full "$EMB" ;;
*) echo "unknown stage"; exit 1 ;;
esac
