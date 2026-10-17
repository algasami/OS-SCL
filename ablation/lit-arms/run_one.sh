#!/usr/bin/env bash
# Train one lit-arms run: run_one.sh <arm> <seed>   arm = aug | gate | eca | combo
#
# Base recipe = ../fast-recipe's (fp16, cudnn.benchmark, lean, cache, probe). LA_FUSSION
# is the base fussion fixed by fast-recipe's D2; LA_COMBO the flags of the arms that
# passed D1 (stage 2). Resumable: skips a run whose probe.csv reaches the last epoch.
# One run per GPU, and only on a GPU no other ablation holds and that stays idle.
set -uo pipefail
cd "$(dirname "$0")/../.."
arm=$1; seed=$2; epochs=${3:-300}
PY=${LA_PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
FUSSION=${LA_FUSSION:?set LA_FUSSION to fast-recipe D2 base (4 or 1)}
OUT=ablation/lit-arms/results
LOCKS=${LA_LOCKS:-/tmp/os-scl-la-locks}
GPUS=${LA_GPUS:-"0 1 2"}
case $arm in
    aug)   flags="--wave_aug 1" ;;
    gate)  flags="--mel_gate tristat" ;;
    eca)   flags="--eca 1" ;;
    combo) flags=${LA_COMBO:?set LA_COMBO to the passing arms\' flags} ;;
    *) echo "unknown arm $arm"; exit 1 ;;
esac
desc=la_${arm}_s${seed}; [ "$epochs" != 300 ] && desc=${desc}_e${epochs}
name=SCLTFSTgramMFN_${desc}
mkdir -p "$OUT" "$LOCKS"

if [ -f "check_points/${name}/model.pth" ] && grep -q "^$((epochs - 1))," "check_points/${name}/probe.csv" 2>/dev/null; then
    echo "[$(date +%T)] skip ${desc} (trained)"; exit 0
fi
idle() {
    for _ in 1 2 3; do
        for l in /tmp/os-scl-*-locks/gpu"$1"; do
            [ -d "$l" ] && [ "${l%/gpu*}" != "$LOCKS" ] && return 1
        done
        [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] || return 1
        sleep "${LA_IDLE_SLEEP:-60}"
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
rm -f "check_points/${name}/probe.csv"

echo "[$(date +%T)] TRAIN start ${desc} gpu=${gpu} fussion=${FUSSION} ${flags}"
t0=$(date +%s)
$PY train.py --m 0.4 --fussion "$FUSSION" --ht basic --gpu_num "$gpu" --num_workers 8 \
    --seed "$seed" --epoch "$epochs" --desc "$desc" \
    --amp fp16 --cudnn_bench 1 --lean 1 --cache_dir data/cache --probe_every 5 $flags \
    > "${OUT}/train_${desc}.log" 2>&1
rc=$?
if [ $rc -ne 0 ] || [ ! -f "check_points/${name}/model.pth" ]; then
    echo "[$(date +%T)] FAIL train ${desc} rc=${rc}"; exit 1
fi
echo "start,end,seconds,epochs,gpu" > "check_points/${name}/wall.csv"
echo "$t0,$(date +%s),$(( $(date +%s) - t0 )),$epochs,$gpu" >> "check_points/${name}/wall.csv"
echo "[$(date +%T)] TRAIN done ${desc}"
