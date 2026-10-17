#!/usr/bin/env bash
# Trunk soup (see README). TS_FUSSION / TS_FLAGS come from the README addendum.
#
#   bash ablation/trunk-soup/run.sh train   # trunks T1, T2 + 4 forks each (forks wait for state_e150)
#   bash ablation/trunk-soup/run.sh soup    # soup.py per trunk (4 forks; 5 with the trunk itself)
#   bash ablation/trunk-soup/run.sh score   # eval.py dev+eval CSVs, probe/wall/config -> scores/
set -uo pipefail
cd "$(dirname "$0")/../.."
D=ablation/trunk-soup
PY=${PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
TRUNKS=${TS_TRUNKS:-"2050 2051"}
FORKS="1 2 3 4"
mkdir -p $D/results $D/scores

case ${1:?usage: run.sh train|soup|score} in
train)
    echo "=== train start $(date) | fussion=${TS_FUSSION:?} flags=${TS_FLAGS-} ==="
    { for t in $TRUNKS; do echo "trunk $t"; done
      for t in $TRUNKS; do for f in $FORKS; do echo "fork $t $f"; done; done; } \
        | xargs -P 3 -L1 $D/run_one.sh
    echo "=== train end $(date) ===" ;;
soup)
    for t in $TRUNKS; do
        forks=""; for f in $FORKS; do forks="$forks check_points/SCLTFSTgramMFN_ts_fork_s${t}_f$f"; done
        $PY soup.py --out SCLTFSTgramMFN_ts_soup4_s$t $forks || exit 1
        # the trunk's own second half is a fork without a re-seed; its config has no
        # resume_state, hence --force
        $PY soup.py --force --out SCLTFSTgramMFN_ts_soup5_s$t $forks check_points/SCLTFSTgramMFN_ts_trunk_s$t || exit 1
    done ;;
score)
    for d in check_points/SCLTFSTgramMFN_ts_*; do
        n=$(basename $d); mkdir -p $D/scores/$n
        for f in config.json probe.csv wall.csv crashes.csv; do [ -f $d/$f ] && cp -p $d/$f $D/scores/$n/; done
        for split in d e; do
            ds=dev; [ $split = e ] && ds=eval
            [ -f $D/scores/$n/eval_$ds.csv ] && continue
            $PY eval.py --m 0.4 --fussion "${TS_FUSSION:?}" --ht basic --gpu_num "${TS_SCORE_GPU:-0}" \
                --model_path $d/model.pth --$split --csv $D/scores/$n/eval_$ds.csv \
                > $D/results/eval_${n}_$split.log 2>&1 || { echo "FAIL eval $n --$split"; exit 1; }
        done
    done
    $PY ablation/distill_curves.py trunk-soup ;;
*) echo "unknown stage"; exit 1 ;;
esac
