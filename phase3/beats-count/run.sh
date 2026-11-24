#!/usr/bin/env bash
# beats-count (see README): X1f with 1 or 2 BEATs networks instead of 3. CPU only, no training.
#
#   bash phase3/beats-count/run.sh links   # symlink tree onto ensemble-g8's dumps -> results/emb (scratch)
#   bash phase3/beats-count/run.sh dev     # DEV ONLY: dev_systems.csv, selection.json -> scores/  (runs `links`)
#   (commit scores/selection.json + scores/dev_systems.csv before the next stage; score.py eval refuses otherwise)
#   bash phase3/beats-count/run.sh eval    # systems.csv, bootstrap.json, decisions.json -> scores/  (runs `links`)
#   /home/f74134118/anaconda3/bin/python phase3/beats-count/make_plots.py
#
# The dumps live in ../OS-SCL-syn-fresh (read-only here). fresh-confirm's flat_path writes a flat cache
# under the emb dir it is given, so it is given results/emb, a tree of symlinks to the split dumps, and
# nothing in ../OS-SCL-syn-fresh changes.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=phase3/beats-count
SRC=/home/f74134118/OS-SCL-syn-fresh/synthesis/ensemble-g8/results/emb
PY=${PY:-/home/f74134118/anaconda3/envs/os-scl-cu124/bin/python}
mkdir -p $D/results $D/scores

# score.py <stage> as independent processes until two write byte-identical output (smartlab1's RAM
# flips bits; one process cannot vouch for itself). Same guard as synthesis/ensemble-g8/run.sh.
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

links() {
    local E=$D/results/emb split f
    mkdir -p $E/flat
    ln -sfn "$(readlink -f $SRC/meta.npz)" $E/meta.npz
    for split in train dev eval; do
        mkdir -p $E/$split
        for f in $SRC/$split/SCLTFSTgramMFN_fc_*.npz; do
            ln -sfn "$(readlink -f "$f")" $E/$split/
        done
    done
}

case ${1:?usage: run.sh links|dev|eval} in
links) links ;;
dev)   links && replicated dev $D/scores ;;
eval)  links && replicated eval $D/scores ;;
*) echo "unknown stage"; exit 1 ;;
esac
