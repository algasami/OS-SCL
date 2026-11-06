#!/usr/bin/env python
"""Dump R0, the pre-head embedding and the pooled pre-linear7 map for the tp checkpoints.

    python ablation/timepool-head/extract.py --gpu 1 --split train|dev|eval --out ablation/timepool-head/results/emb \
        --runs SCLTFSTgramMFN_tp_s2040 ... [--machine ToyCar --rolls 1,2,3,4,5,6,7]

This is ../pooled-knn/extract.py, imported unchanged, with only its checkpoint root pointed
at this worktree's check_points/ (config.json is read from scores/<run>/ via scores_path).
Same forward pass (EMA weights, eval mode, batch 32), same hook on mobilefacenet.conv2 --
with --tpool the conv2 output is still the full 512 x 8 x 20 map; the time mean happens
after it, inside MobileFaceNet.forward -- same range checks and CRC-verified writes.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('pk_extract', os.path.join(HERE, '..', 'pooled-knn', 'extract.py'))
pk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pk)
pk.CKPT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', 'check_points'))

if __name__ == '__main__':
    pk.main()
