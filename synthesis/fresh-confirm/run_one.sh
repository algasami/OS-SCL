#!/usr/bin/env bash
# One fresh-confirm run: run_one.sh <type> <seed>
#   type = nogain (= ablation/aug-ops ao_nogain) | aug (= ablation/lit-arms la_aug) | spec (= ablation/spectrum-branch sb_spec)
# Recipe is byte-for-byte the phase-1 recipe of that type; only the seed and the desc prefix (fc_) differ.
# GPU locking, crash-retry and records: ablation/train_one.sh with ABL_ROOT=synthesis.
set -uo pipefail
type=$1; seed=$2
case $type in
    nogain) flags="--wave_aug 1 --wave_aug_ops noise,shift" ;;
    aug)    flags="--wave_aug 1" ;;
    spec)   flags="--wave_aug 1 --wave_aug_ops noise,shift --spec_branch 1" ;;
    *) echo "unknown type $type"; exit 1 ;;
esac
# shellcheck disable=SC2086
ABL_ROOT=synthesis exec "$(dirname "$0")/../../ablation/train_one.sh" fresh-confirm "fc_${type}_s${seed}" 300 \
    --m 0.4 --fussion 1 --ht basic --num_workers 8 --seed "$seed" \
    --amp fp16 --cudnn_bench 1 --lean 1 --cache_dir data/cache --probe_every 5 $flags
