#!/usr/bin/env bash
# One BEATs fine-tuning run, with ../train_one.sh's GPU protocol (lock, idle check, crash retry).
#
#   ablation/beats/run_one.sh <desc> <train_beats.py args...>
#
# - skips a run that already finished (model.pth + done marker)
# - takes one GPU from $GPUS (default "0 2 1"): one no other ablation's lock holds and that stays
#   under $IDLE_MB (1000) MiB for 3 checks $IDLE_SLEEP (60) s apart
# - retries a crash up to 3 times with --crash_resume state_last.pth (smartlab1's RAM crashes processes)
# - writes check_points/BEATsASD_<desc>/{wall.csv, crashes.csv, done}; log results/train_BEATsASD_<desc>.log
set -uo pipefail
cd "$(dirname "$0")/../.."
desc=$1; shift
PY=${PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
GPUS=${GPUS:-"0 2 1"}
LOCKS=/tmp/os-scl-beats-locks
OUT=ablation/beats/results
ck=check_points/BEATsASD_${desc}
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
rm -f "$ck/state_last.pth" "$ck/crashes.csv"
resume=""; attempt=1
: > "${OUT}/train_BEATsASD_${desc}.log"
while :; do
    $PY ablation/beats/train_beats.py --gpu_num "$gpu" --desc "$desc" "$@" $resume >> "${OUT}/train_BEATsASD_${desc}.log" 2>&1
    rc=$?
    [ $rc -eq 0 ] && [ -f "$ck/model.pth" ] && break
    if [ $attempt -ge 3 ]; then echo "[$(date +%T)] FAIL train ${desc} rc=${rc} after ${attempt} attempts"; exit 1; fi
    attempt=$((attempt + 1))
    if [ -f "$ck/state_last.pth" ]; then resume="--crash_resume $ck/state_last.pth"; else resume=""; fi
    [ -f "$ck/crashes.csv" ] || echo "time,rc,next_attempt" > "$ck/crashes.csv"
    echo "$(date +%s),${rc},${attempt}" >> "$ck/crashes.csv"
    echo "[$(date +%T)] RETRY ${desc} rc=${rc} attempt ${attempt} ${resume:-from scratch}"
done
rm -f "$ck/state_last.pth"
echo "start,end,seconds,gpu" > "$ck/wall.csv"
echo "$t0,$(date +%s),$(( $(date +%s) - t0 )),$gpu" >> "$ck/wall.csv"
touch "$ck/done"
echo "[$(date +%T)] TRAIN done ${desc}"
