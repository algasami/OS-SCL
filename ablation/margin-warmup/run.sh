#!/usr/bin/env bash
# Margin warm-up (see README). Resumable: every stage skips work already done.
#
#   bash ablation/margin-warmup/run.sh gate     # 2-epoch smoke of f6/f7 with a compressed warm-up, strict
#                                              #   eval.py load, diag on a state snapshot -> scores/gate.txt
#   bash ablation/margin-warmup/run.sh pilot    # stage 1: mw_{f6,f7,f2}_s2040 on GPUs 0 and 2 (MW_GPUS)
#   bash ablation/margin-warmup/run.sh stage2   # stage 2: mw_{f6,f7,f2} x seeds 2041-2043 on 3 GPUs. NOT auto-run
#   bash ablation/margin-warmup/run.sh stage2b  # optional: mw_f5 x seeds 2040-2043
#   bash ablation/margin-warmup/run.sh score    # eval.py dev + eval CSVs, probe, wall, curves -> scores/
#   bash ablation/margin-warmup/run.sh diag     # diag_margin.py at e60/e100 (student + EMA) and final -> scores/
#   bash ablation/margin-warmup/run.sh plots    # make_plots.py (base python) -> plots/, scores/summary.txt
#
# chain.sh runs gate -> pilot -> score -> diag -> plots once ../no-logmel's queue has ended.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=ablation/margin-warmup
PY=${PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
GPU=${MW_GPU:-0}                 # gate, eval.py and diag
PILOT_GPUS=${MW_GPUS:-"0 2"}     # GPU 1 is left to ../no-logmel's score stage
NL_SCORES=${NL_SCORES:-/home/f74134118/OS-SCL-nomel/ablation/no-logmel/scores}   # registered runs, read-only
mkdir -p $D/results $D/scores
[ -f data/cache/devtest_mel.npy ] || { echo "no data/cache: run cache_data.py (../fast-recipe)"; exit 1; }

score_run() {   # score_run <name>
    local n=$1 f split ds log
    f=$(sed -n 's/.*"fussion": \([0-9]\).*/\1/p' check_points/$n/config.json)
    mkdir -p $D/scores/$n
    cp -p check_points/$n/{config.json,probe.csv,wall.csv} $D/scores/$n/
    if [ -f check_points/$n/crashes.csv ]; then cp -p check_points/$n/crashes.csv $D/scores/$n/; fi
    for split in d e; do
        ds=dev; [ $split = e ] && ds=eval
        [ -f $D/scores/$n/eval_$ds.csv ] && continue
        # --m 0.4: scored with the final margin, exactly as the registered runs
        $PY eval.py --m 0.4 --fussion "$f" --ht basic --gpu_num "$GPU" \
            --model_path check_points/$n/model.pth --$split --csv $D/scores/$n/eval_$ds.csv \
            > $D/results/eval_${n}_$split.log 2>&1 || { echo "FAIL eval $n --$split"; return 1; }
    done
    # the warm-up columns distill_curves.py does not keep: student margin and train acc at m = 0.4
    log=$D/results/train_${n#SCLTFSTgramMFN_}.log
    { echo "epoch,m,train_acc_mfinal"
      tr '\r' '\n' < "$log" | sed -nE 's/.*EPOCH: ([0-9]+) \|.*\| m: ([0-9.]+) \| Train_acc_mfinal: ([0-9.]+).*/\1,\2,\3/p'
    } > $D/scores/$n/curve_mw.csv
}

runs_done() {   # trained 300-epoch runs of this ablation
    for d in check_points/SCLTFSTgramMFN_mw_f?_s20??; do
        [ -f "$d/done" ] && basename "$d"
    done
}

case ${1:?usage: run.sh gate|pilot|stage2|stage2b|score|diag|plots} in
gate)
    for arm in f6 f7; do
        n=SCLTFSTgramMFN_mw_${arm}_s1_e2
        GPUS=$GPU IDLE_SLEEP=1 MW=0.5,1.5 SNAPS=1 $D/run_one.sh $arm 1 2 || exit 1
        log=$D/results/train_mw_${arm}_s1_e2.log
        $PY eval.py --m 0.4 --fussion ${arm#f} --ht basic --gpu_num "$GPU" \
            --model_path check_points/$n/model.pth --d > $D/results/eval_${n}_d.log 2>&1 \
            || { echo "FAIL eval.py strict load $n"; exit 1; }
        DEV=cuda:$GPU $PY $D/diag_margin.py check_points/$n/state_e1.pth:net > $D/results/diag_${n}.txt 2>&1 \
            || { echo "FAIL diag on state snapshot $n"; exit 1; }
        nf=$(grep -ac 'non-finite' $log)
        mlog=$(tr '\r' '\n' < $log | grep -o '| m: [0-9.]*' | tr '\n' ' ')
        mcol=$(head -1 check_points/$n/probe.csv | tr ',' '\n' | grep -cx m)
        probe_m=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="m") c=i} NR>1{printf "%s ", $c}' check_points/$n/probe.csv)
        echo "G $arm: $(($(grep -c . check_points/$n/probe.csv) - 1)) probe rows, student m at probes [$probe_m]," \
             "EPOCH-line m [${mlog}], $nf non-finite steps, $(du -h check_points/$n/model.pth | cut -f1) ckpt," \
             "eval.py strict: $(grep 'Average AUC' $D/results/eval_${n}_d.log)" | tee -a $D/scores/gate.txt
        echo "  diag(state_e1:net): $(tail -1 $D/results/diag_${n}.txt)" | tee -a $D/scores/gate.txt
        [ "$mcol" = 1 ] && [ -n "$mlog" ] && [ "$nf" -le 20 ] \
            || { echo "FAIL gate $arm: m not logged or $nf non-finite steps" | tee -a $D/scores/gate.txt; exit 1; }
    done
    echo "gate passed $(date '+%F %T')" | tee -a $D/scores/gate.txt ;;
pilot)
    echo "=== pilot start $(date) | GPUs $PILOT_GPUS ==="
    # f6 (13 h) holds one GPU; f7 (6 h) then the f2 control (5 h) share the other
    printf '%s\n' "f6 2040" "f7 2040" "f2 2040" | GPUS=$PILOT_GPUS xargs -P 2 -L1 $D/run_one.sh
    echo "=== pilot end $(date) ===" ;;
stage2)
    echo "=== stage2 start $(date) ==="
    for s in 2041 2042 2043; do for arm in f6 f7 f2; do echo "$arm $s"; done; done \
        | GPUS=${MW_GPUS2:-"0 1 2"} xargs -P 3 -L1 $D/run_one.sh
    echo "=== stage2 end $(date) ===" ;;
stage2b)
    for s in 2040 2041 2042 2043; do echo "f5 $s"; done | GPUS=${MW_GPUS2:-"0 1 2"} xargs -P 3 -L1 $D/run_one.sh ;;
score)
    for n in $(runs_done); do score_run $n || exit 1; done
    $PY ablation/distill_curves.py margin-warmup ;;
diag)
    for n in $(runs_done); do
        out=$D/scores/diag_margin_${n#SCLTFSTgramMFN_}.txt
        [ -f $out ] && continue
        t=()
        for e in 60 100; do
            [ -f check_points/$n/state_e$e.pth ] && t+=(check_points/$n/state_e$e.pth:net check_points/$n/state_e$e.pth:ema_net)
        done
        { echo "# diag_margin.py $(date '+%F %T'), DEV=cuda:$GPU, same 1024 train clips as ../no-logmel's diag;"
          echo "# @e60 = end of the m=0 hold, @e100 = mid-ramp (student m 0.267); last row = best-EMA model.pth."
          echo "# All rows measured at the final margin 0.4. Pass (README): final acc(margin) >= 0.5 and cos tgt > 0."
          DEV=cuda:$GPU $PY $D/diag_margin.py ${t[@]+"${t[@]}"} check_points/$n 2>&1; } > $out.tmp && mv $out.tmp $out
        cat $out
    done ;;
plots)
    MW_EXTRA_SCORES=$NL_SCORES /home/f74134118/anaconda3/bin/python $D/make_plots.py | tee $D/scores/summary.txt ;;
*) echo "unknown stage"; exit 1 ;;
esac
