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


def scores_dir(model_name):
    hits = sorted(glob.glob(os.path.join(ABLATION_ROOT, '*', 'scores', model_name)))
    if not hits:
        raise FileNotFoundError(
            f'no scores for {model_name!r} under ablation/*/scores/ -- '
            f'run the ablation, or check the name')
    return hits[0]


def scores_csv(model_name, dataset):
    return os.path.join(scores_dir(model_name), f'eval_{dataset}.csv')
