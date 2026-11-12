#!/usr/bin/env bash
# ensemble-g8 (see README): four more fresh ao_nogain networks (seeds 2064-2067), then the fresh
# ensembles G8f / GS12f against the already scored G4 / GS8.
#
#   nohup setsid bash synthesis/ensemble-g8/run.sh train > synthesis/ensemble-g8/results/run_train.log 2>&1 < /dev/null &
#   nohup setsid bash synthesis/ensemble-g8/run.sh post  > synthesis/ensemble-g8/results/post.log 2>&1 < /dev/null &
#                                                    # detached: waits for training, then records + extract
#   bash synthesis/ensemble-g8/run.sh records        # config/wall -> scores/<run>/ (needed by extract.py)
#   EG_GPU=0 bash synthesis/ensemble-g8/run.sh extract    # dumps of the 4 new runs -> results/emb (scratch)
#   bash synthesis/ensemble-g8/run.sh dryrun         # all of score() on seen seeds 2040-47 -> scores/dryrun_p1
#   bash synthesis/ensemble-g8/run.sh score          # ONCE, after the README's predictions are committed
#   EG_GPU=0 bash synthesis/ensemble-g8/run.sh evalpy     # eval.py CSVs + probe + curves + R0 cross-check (after score)
#
# The recipe is ../fresh-confirm/run_one.sh's `nogain` line, flag for flag; only the seed and the study
# name passed to ablation/train_one.sh (log directory, GPU lock) differ. Run from the worktree that holds
# check_points/SCLTFSTgramMFN_fc_* and synthesis/fresh-confirm/results/emb (../OS-SCL-syn-fresh).
# Edit this file only by writing a new one and renaming it over: `post` re-enters it with `bash $0`.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=synthesis/ensemble-g8
FC=synthesis/fresh-confirm
SEEDS=${EG_SEEDS:-"2064 2065 2066 2067"}
PY=${PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
mkdir -p $D/results $D/scores
new_runs() { for s in $SEEDS; do echo "SCLTFSTgramMFN_fc_nogain_s$s"; done; }

# score.py <stage> as independent processes until two write byte-identical output (smartlab1's RAM
# flips bits; one process cannot vouch for itself). Same guard as ../fresh-confirm/run.sh.
replicated() {
    local stage=$1 final=$2 k j
    local base=$D/results/${stage}_tries
    mkdir -p $base
    for k in 1 2 3 4 5 6; do
        rm -rf $base/$k
        if ! OMP_NUM_THREADS=8 $PY $D/score.py $stage "$PWD/$base/$k" > $base/$k.log 2>&1; then
            echo "try $k failed: $(tail -1 $base/$k.log | cut -c1-150)"; rm -rf $base/$k; continue
        fi
        for j in $(seq 1 $((k - 1))); do
            if [ -d $base/$j ] && diff -rq $base/$j $base/$k > /dev/null; then
                mkdir -p $final; cp -p $base/$k/* $final/
                echo "tries $j and $k agree byte for byte -> $final ($(date))"
                return 0
            fi
        done
        echo "try $k done, no agreeing partner yet"
    done
    echo "FAIL $stage: no two of 6 tries agreed; see $base/*.log"; return 1
}

case ${1:?usage: run.sh train|post|records|extract|dryrun|score|evalpy} in
train_one)
    exec env ABL_ROOT=synthesis ablation/train_one.sh ensemble-g8 "fc_nogain_s$2" 300 \
        --m 0.4 --fussion 1 --ht basic --num_workers 8 --seed "$2" \
        --amp fp16 --cudnn_bench 1 --lean 1 --cache_dir data/cache --probe_every 5 \
        --wave_aug 1 --wave_aug_ops noise,shift ;;
train)
    echo "=== train start $(date) | seeds=$SEEDS ==="
    for s in $SEEDS; do echo "$s"; done | GPUS=${GPUS:-"0 2 1"} xargs -P 3 -L1 bash $D/run.sh train_one
    echo "=== train end $(date) ===" ;;
records)
    for n in $(new_runs); do
        d=check_points/$n
        [ -f "$d/done" ] || continue
        mkdir -p $D/scores/$n
        cp -p "$d"/{config.json,wall.csv} $D/scores/$n/
        [ -f "$d/crashes.csv" ] && cp -p "$d/crashes.csv" $D/scores/$n/
    done ;;
extract)
    # New runs are extracted here; the eight fresh-confirm dumps are linked, not copied or re-extracted.
    runs=$(for n in $(new_runs); do [ -f "check_points/$n/done" ] && echo "$n"; done)
    for split in train dev eval; do
        mkdir -p $D/results/emb/$split
        for f in $FC/results/emb/$split/SCLTFSTgramMFN_fc_*.npz; do
            ln -sfn "$PWD/$f" $D/results/emb/$split/
        done
        $PY $FC/extract.py --gpu "${EG_GPU:-0}" --split $split --out $D/results/emb --runs $runs \
            >> $D/results/extract.log 2>&1 || { echo "FAIL extract $split"; exit 1; }
    done ;;
post)
    # Session-independent: waits for training to end, then records + extract (retried; faulty RAM).
    # Stops before `score`, which is run once by hand. pgrep patterns are anchored so that shells
    # which merely mention them in their command line do not match.
    st=$D/results/post.status
    echo "WAITING for training $(date)" > $st
    while pgrep -f "^\S*python train\.py .*--desc fc_nogain_s206[4-7]" > /dev/null \
       || pgrep -f "^bash synthesis/ensemble-g8/run\.sh train" > /dev/null; do sleep 300; done
    n=0; for r in $(new_runs); do [ -f "check_points/$r/done" ] && n=$((n + 1)); done
    want=$(new_runs | wc -l)
    if [ "$n" -lt "$want" ]; then echo "FAIL only $n/$want runs done $(date); see results/run_train.log" > $st; exit 1; fi
    echo "RECORDS $(date)" > $st
    bash $0 records
    for try in 1 2 3; do
        echo "EXTRACT try $try $(date)" > $st
        bash $0 extract && break
        [ $try -eq 3 ] && { echo "FAIL extract after 3 tries $(date); see results/extract.log" > $st; exit 1; }
        sleep 30
    done
    echo "READY_TO_SCORE $(date) ($n/$want trained, records + extract done; next: run.sh score, ONCE)" > $st ;;
dryrun) replicated dryrun $D/scores/dryrun_p1 ;;
score)  replicated score $D/scores ;;
evalpy)
    for n in $(new_runs); do
        d=check_points/$n
        [ -f "$d/done" ] || continue
        cp -p "$d/probe.csv" $D/scores/$n/ 2>/dev/null
        for split in d e; do
            ds=dev; [ $split = e ] && ds=eval
            [ -f $D/scores/$n/eval_$ds.csv ] && continue
            $PY eval.py --m 0.4 --fussion 1 --ht basic --gpu_num "${EG_GPU:-0}" \
                --model_path "$d/model.pth" --$split --csv $D/scores/$n/eval_$ds.csv \
                > $D/results/eval_${n}_$split.log 2>&1 || { echo "FAIL eval $n --$split"; exit 1; }
        done
    done
    $PY ablation/distill_curves.py ../synthesis/ensemble-g8
    $PY $D/crosscheck.py ;;
*) echo "unknown stage"; exit 1 ;;
esac
