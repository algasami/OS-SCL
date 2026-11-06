#!/usr/bin/env bash
# ablation/beats -- WITH EXTERNAL PRETRAINING (BEATs_iter3). Run from anywhere; resumable.
#
#   bash ablation/beats/run.sh fetch          # BEATs_iter3.pt -> data/pretrained/hf/, sha256-checked
#   GPU=0 bash ablation/beats/run.sh bench    # step time / peak memory per config (~10 min, GPU must be idle)
#   GPU=0 bash ablation/beats/run.sh smoke    # 200-step train + crash-resume + full extraction + scoring
#   GPUS="0 2 1" bash ablation/beats/run.sh train     # the 3 pre-registered runs, one per GPU (parallel)
#   GPU=0 bash ablation/beats/run.sh extract  # dumps -> results/emb (scratch)
#   bash ablation/beats/run.sh dev            # stage A: dev scores, rule + fusion selection -> scores/
#   bash ablation/beats/run.sh eval           # stage B (only after stage A is committed)
#   bash ablation/beats/run.sh curves         # scores/<run>/{config.json,curve.csv}, plots
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
PLOT_PY=/home/f74134118/anaconda3/bin/python
A=ablation/beats
SEEDS="2040 2041 2042"
CKPT=data/pretrained/hf/BEATs_iter3.pt
SHA=8d1b234032a9ccff353612dc6c20982346dc2968b205b79d97303eb5e77bfb34
# the pre-registered recipe (README *Recipe*); the bench picked 16 x 16 accumulated clips, fused attention,
# no activation checkpointing (38.8 clips/s, peak 6.9 GiB on GPU 2; 32 x 8 is as fast at 12.1 GiB)
RECIPE="--steps 10000 --warmup 960 --lr 1e-4 --wd 0.01 --micro_bs 16 --accum 16 --m 0.4 --arc_s 30
        --freq_mask 40 --time_mask 80 --amp bf16 --grad_ckpt 0 --sdpa 1 --num_workers 6 --cache_dir data/cache
        --beats_ckpt $CKPT --valid_every 250 --state_every 500"
RUNS=$(for s in $SEEDS; do printf 'BEATsASD_bt_arc_s%s ' "$s"; done)
mkdir -p $A/results

case ${1:-} in
fetch)
    mkdir -p data/pretrained/hf
    [ -f $CKPT ] || curl -fL -o $CKPT https://huggingface.co/lpepino/beats_ckpts/resolve/main/BEATs_iter3.pt
    echo "$SHA  $CKPT" | sha256sum -c ;;
bench)
    # one GPU, idle; each config (micro_bs accum amp grad_ckpt sdpa) 4 timed optimizer steps after 2 warm-up
    for cfg in ${BENCH_CFGS:-"32,8,bf16,1,1" "16,16,bf16,0,1" "32,8,bf16,0,1" "32,8,bf16,1,0" "16,16,bf16,0,0"}; do
        IFS=, read -r mb ac amp gc sd <<< "$cfg"
        $PY $A/train_beats.py --gpu_num "${GPU:?}" --seed 2040 --desc bench --bench 4 --micro_bs $mb --accum $ac \
            --amp $amp --grad_ckpt $gc --sdpa $sd --cache_dir data/cache --beats_ckpt $CKPT 2>&1 | grep -E 'BENCH|Error' | tail -1
    done 2>&1 | tee -a $A/results/bench_gpu${GPU}.log ;;
smoke)
    # 200 steps (no accumulation), state at 100; then resume 100 -> 200; then extraction + both score stages
    S="--steps 200 --warmup 50 --micro_bs 16 --accum 1 --amp bf16 --grad_ckpt 0 --sdpa 1 --cache_dir data/cache
       --beats_ckpt $CKPT --valid_every 100 --state_every 100 --seed 2040"
    $PY $A/train_beats.py --gpu_num "${GPU:?}" --desc smoke $S > $A/results/train_smoke.log 2>&1 || exit 1
    $PY $A/train_beats.py --gpu_num "$GPU" --desc smoke_resume $S \
        --crash_resume check_points/BEATsASD_smoke/state_last.pth > $A/results/train_smoke_resume.log 2>&1 || exit 1
    /usr/bin/time -v $PY $A/extract.py --gpu "$GPU" --out $A/results/emb_smoke --runs BEATsASD_smoke \
        > $A/results/extract_smoke.log 2>&1 || exit 1
    for st in dev eval; do
        BT_RUNS=BEATsASD_smoke BT_SCORES=$A/results/smoke_scores $PY $A/score.py $st $A/results/emb_smoke || exit 1
    done ;;
train)
    for s in $SEEDS; do
        GPUS=${GPUS:-"0 2 1"} bash $A/run_one.sh bt_arc_s$s --seed $s $RECIPE &
        sleep 240     # let each runner take its GPU before the next one looks
    done
    wait ;;
extract)
    $PY $A/extract.py --gpu "${GPU:?}" --out $A/results/emb --runs $RUNS ;;
dev|eval)
    $PY $A/score.py "$1" $A/results/emb ;;
curves)
    for r in $RUNS; do mkdir -p $A/scores/$r && cp check_points/$r/config.json $A/scores/$r/; done
    $PY ablation/distill_curves.py beats
    $PLOT_PY $A/make_plots.py ;;
*)
    sed -n 2,12p "$0"; exit 1 ;;
esac
