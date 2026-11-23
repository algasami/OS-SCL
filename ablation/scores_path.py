"""Locate an ablation run's eval CSVs.

Scores live with the ablation that produced them, at
`ablation/<ablation>/scores/<model_name>/eval_{dev,eval}.csv`, rather than in
`check_points/` -- which holds only weights and is gitignored. A run is resolved
by name across all ablations, so cross-ablation reads (every set plots the
shared `SCLTFSTgramMFN_test` baseline) need no relative paths.
"""
import glob
import os

ABLATION_ROOT = os.path.dirname(os.path.abspath(__file__))
# Phase 2 (synthesis/) keeps the same per-study layout one directory over.
SYNTHESIS_ROOT = os.path.join(os.path.dirname(ABLATION_ROOT), 'synthesis')
# Phase 3 (phase3/, studies after synthesis closed 2026-11-12), same layout again.
PHASE3_ROOT = os.path.join(os.path.dirname(ABLATION_ROOT), 'phase3')


def scores_dir(model_name):
    hits = sorted(glob.glob(os.path.join(ABLATION_ROOT, '*', 'scores', model_name)))
    hits += sorted(glob.glob(os.path.join(SYNTHESIS_ROOT, '*', 'scores', model_name)))
    hits += sorted(glob.glob(os.path.join(PHASE3_ROOT, '*', 'scores', model_name)))
    if not hits:
        raise FileNotFoundError(
            f'no scores for {model_name!r} under ablation/, synthesis/ or phase3/*/scores/ -- '
            f'run the ablation, or check the name')
    return hits[0]


def scores_csv(model_name, dataset):
    return os.path.join(scores_dir(model_name), f'eval_{dataset}.csv')
