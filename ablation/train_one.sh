#!/usr/bin/env bash
# Shared runner for one training run (used by lit-arms, trunk-soup, ...).
#
#   ablation/train_one.sh <ablation> <desc> <epochs> <train.py args...>
#
# - skips a run that already finished (model.pth + a probe row for the last epoch,
#   or model.pth + done marker when the run has no probe)
# - takes one GPU from $GPUS (default "0 1 2"): only one that no other ablation's
#   lock holds and that stays under 1 GB used for 3 checks a minute apart
# - passes --state_every 10 and, after a crash, retries up to 3 times with
#   --crash_resume (smartlab1's RAM crashes processes; see ablation/fast-recipe)
# - writes check_points/<run>/wall.csv, and crashes.csv if any attempt died
# - logs to ablation/<ablation>/results/train_<desc>.log
#
# Env: PY (python), GPUS, IDLE_SLEEP (seconds between idle checks), IDLE_MB (busy threshold, 1000);
# IGNORE_LOCKS=1 with a high IDLE_MB forces a busy GPU (smoke tests only).
set -uo pipefail
cd "$(dirname "$0")/.."
abl=$1; desc=$2; epochs=$3; shift 3
PY=${PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
GPUS=${GPUS:-"0 1 2"}
LOCKS=/tmp/os-scl-${abl}-locks
OUT=ablation/${abl}/results
name=SCLTFSTgramMFN_${desc}
ck=check_points/${name}
mkdir -p "$OUT" "$LOCKS"

if [ -f "$ck/model.pth" ] && [ -f "$ck/done" ]; then
    echo "[$(date +%T)] skip ${desc} (trained)"; exit 0
fi
idle() {
    for _ in 1 2 3; do
        for l in /tmp/os-scl-*-locks/gpu"$1"; do
            [ -z "${IGNORE_LOCKS:-}" ] && [ -d "$l" ] && [ "${l%/gpu*}" != "$LOCKS" ] && return 1
        done
        [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt "${IDLE_MB:-1000}" ] || return 1
        sleep "${IDLE_SLEEP:-60}"
    done
}
gpu=""
while [ -z "$gpu" ]; do
    for g in $GPUS; do
        if mkdir "${LOCKS}/gpu${g}" 2>/dev/null; then
            if idle "$g"; then gpu=$g; break; fi
            rmdir "${LOCKS}/gpu${g}"
        fi
    done
    [ -z "$gpu" ] && sleep 60
done
trap 'rmdir "${LOCKS}/gpu${gpu}" 2>/dev/null' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "[$(date +%T)] TRAIN start ${desc} gpu=${gpu} $*"
t0=$(date +%s)
rm -f "$ck/probe.csv" "$ck/state_last.pth" "$ck/crashes.csv"
resume=""; attempt=1
: > "${OUT}/train_${desc}.log"
while :; do
    $PY train.py --gpu_num "$gpu" --epoch "$epochs" --desc "$desc" --state_every 10 "$@" $resume \
        >> "${OUT}/train_${desc}.log" 2>&1
    rc=$?
    [ $rc -eq 0 ] && [ -f "$ck/model.pth" ] && break
    if [ $attempt -ge 3 ]; then echo "[$(date +%T)] FAIL train ${desc} rc=${rc} after ${attempt} attempts"; exit 1; fi
    attempt=$((attempt + 1))
    if [ -f "$ck/state_last.pth" ]; then
        resume="--crash_resume $ck/state_last.pth"
    else
        resume=""; rm -f "$ck/probe.csv"
    fi
    [ -f "$ck/crashes.csv" ] || echo "time,rc,next_attempt" > "$ck/crashes.csv"
    echo "$(date +%s),${rc},${attempt}" >> "$ck/crashes.csv"
    echo "[$(date +%T)] RETRY ${desc} rc=${rc} attempt ${attempt} ${resume:-from scratch}"
done
rm -f "$ck/state_last.pth"
echo "start,end,seconds,epochs,gpu" > "$ck/wall.csv"
echo "$t0,$(date +%s),$(( $(date +%s) - t0 )),$epochs,$gpu" >> "$ck/wall.csv"
touch "$ck/done"
echo "[$(date +%T)] TRAIN done ${desc}"
