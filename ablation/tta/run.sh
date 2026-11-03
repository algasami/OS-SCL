#!/usr/bin/env bash
# TTA ablation (see README). GPU stages take the shared lock dir, so train_one.sh runs skip that GPU.
#
#   bash ablation/tta/run.sh pilot      # held-out training clips, 16 shifts, 5 runs; then pilot.py
#   bash ablation/tta/run.sh extractA   # ao_nogain x8: train bank + dev clips, 9 shifts
#   bash ablation/tta/run.sh dev        # stage A scoring (CPU)
#   bash ablation/tta/run.sh extractB   # only after gate A passes and the eval addendum is committed; then rerun `dev` (adds A8)
#   bash ablation/tta/run.sh eval       # stage B scoring (CPU)
#   bash ablation/tta/run.sh plots
set -uo pipefail
cd "$(dirname "$0")/../.."
D=ablation/tta
PY=${PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
BPY=/home/f74134118/anaconda3/bin/python
GPU=${TTA_GPU:-1}
EMB=$D/results/emb
SH9=0,0.01,-0.01,0.03,-0.03,0.055,-0.055,0.08,-0.08
SH3=0,0.01,-0.01   # stage B: only the scheme that passed gate A (small)
SH16=0,0.01,-0.01,0.03,-0.03,0.055,-0.055,0.08,-0.08,0.125,0.25,0.375,0.5,0.625,0.75,0.875
NG=""; AUG=""; for s in 2040 2041 2042 2043 2044 2045 2046 2047; do
    NG="$NG SCLTFSTgramMFN_ao_nogain_s$s"; AUG="$AUG SCLTFSTgramMFN_la_aug_s$s"; done
mkdir -p $D/results $D/scores

lock() { mkdir -p /tmp/os-scl-tta-locks
         until ! ls -d /tmp/os-scl-*-locks/gpu$GPU >/dev/null 2>&1 && mkdir /tmp/os-scl-tta-locks/gpu$GPU 2>/dev/null; do sleep 60; done
         trap 'rmdir /tmp/os-scl-tta-locks/gpu$GPU 2>/dev/null' EXIT; }
ext() { OMP_NUM_THREADS=1 $PY $D/extract_tta.py --gpu $GPU --out $EMB "$@" >> $D/results/extract.log 2>&1 \
        || { echo "FAIL extract $*"; exit 1; }; }

case ${1:?usage: run.sh pilot|extractA|dev|extractB|eval|plots} in
pilot)    lock
          ext --split valid --shifts $SH16 --runs SCLTFSTgramMFN_ao_nogain_s2040 SCLTFSTgramMFN_ao_nogain_s2041 \
              SCLTFSTgramMFN_la_aug_s2040 SCLTFSTgramMFN_fr_f1_s2040 SCLTFSTgramMFN_fr_f1_s2041
          $BPY $D/pilot.py $EMB ;;
extractA) lock; ext --split train --shifts $SH9 --runs $NG; ext --split dev --shifts $SH9 --runs $NG ;;
extractB) lock; ext --split eval --shifts $SH3 --runs $NG
          for sp in train dev eval; do ext --split $sp --shifts $SH3 --runs $AUG; done ;;
dev|eval) $BPY $D/score.py "$1" | tee $D/results/score_$1.log ;;
plots)    $BPY $D/make_plots.py ;;
*) echo "unknown stage"; exit 1 ;;
esac
