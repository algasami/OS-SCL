#!/usr/bin/env python
"""S1′ for sb_spec against its paired ao_nogain seeds (README: secondary, reported, no decision).

    python ablation/spectrum-branch/score_s1p.py <spectrum-branch emb dir> <aug-ops emb dir>

Dumps come from ../ensemble-backend/extract.py (sb_spec on this worktree; ao_nogain on
../aug-ops). Rules and metrics are ../ensemble-backend/score.py's, imported unchanged.
Checks that S0 recomputed from every dump matches eval.py's committed per-ID AUCs.

Writes scores/s1p.csv (run, rule, dataset, machine, AUC, pAUC) and prints paired deltas.
"""
import csv
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from scores_path import scores_csv  # noqa: E402

_spec = importlib.util.spec_from_file_location('eb_score', os.path.join(HERE, '..', 'ensemble-backend', 'score.py'))
eb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eb)

SEEDS = (2040, 2041, 2042, 2043)
ARMS = {'sb_spec': [f'SCLTFSTgramMFN_sb_spec_s{s}' for s in SEEDS],
        'ao_nogain': [f'SCLTFSTgramMFN_ao_nogain_s{s}' for s in SEEDS]}


def below_data(paths):
    return np.array([p.split('/data/', 1)[1] for p in paths])


def main():
    dirs = sys.argv[1:3]
    metas = [dict(np.load(os.path.join(d, 'meta.npz'))) for d in dirs]
    for k in metas[0]:
        a, b = metas[0][k], metas[1][k]
        if k.endswith('paths'):
            a, b = below_data(a), below_data(b)
        assert np.array_equal(a, b), f'meta.npz disagree on {k}'
    meta = metas[0]
    where = {}
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith('.npz') and f != 'meta.npz':
                where.setdefault(f[:-4], os.path.join(d, f))
    ev = {ds: eb.Evaluator(meta, ds) for ds in ('dev', 'eval')}
    val, rows, worst = {}, [], 0.0
    for runs in ARMS.values():
        for r in runs:
            S = eb.per_run_scores(meta, dict(np.load(where[r])))
            for ds in ev:
                committed = {}
                with open(scores_csv(r, ds), encoding='UTF-8') as f:
                    for row in csv.DictReader(f):
                        if row['scope'] == 'id':
                            committed[(row['machine'], row['id'])] = float(row['AUC'])
                for rule in ('S0', 'S1p'):
                    res = ev[ds](S[rule])
                    if rule == 'S0':
                        worst = max(worst, max(abs(res[m]['ids'][i] - committed[(m, i)])
                                               for m in eb.MACH for i in res[m]['ids']))
                    o = res['overall']
                    val[(r, rule, ds, 'ALL', 'AUC')] = o['AUC']
                    val[(r, rule, ds, 'ALL', 'mean')] = o['mean_AUC_pAUC']
                    rows.append([r, rule, ds, 'ALL', f"{o['AUC']:.4f}", f"{o['pAUC']:.4f}"])
                    for m in eb.MACH:
                        val[(r, rule, ds, m, 'AUC')] = 100 * res[m]['AUC']
                        rows.append([r, rule, ds, m, f"{100 * res[m]['AUC']:.4f}", f"{100 * res[m]['pAUC']:.4f}"])
                        for i, a in res[m]['ids'].items():
                            val[(r, rule, ds, f'{m} {i}', 'AUC')] = 100 * a
    assert worst < 5e-4, f'S0 does not reproduce eval.py: max per-ID |Δ| {worst:.2e}'
    print(f'check: max |S0 − eval.py| per-ID AUC {worst:.2e}')
    with open(os.path.join(HERE, 'scores', 's1p.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'rule', 'dataset', 'machine', 'AUC', 'pAUC'])
        w.writerows(rows)

    cells = [('dev', 'ALL', 'AUC'), ('dev', 'ALL', 'mean'), ('dev', 'pump', 'AUC'), ('dev', 'pump id_02', 'AUC'),
             ('dev', 'ToyConveyor', 'AUC'), ('eval', 'ALL', 'mean'), ('eval', 'pump', 'AUC'),
             ('eval', 'ToyCar', 'AUC'), ('eval', 'ToyConveyor', 'AUC')]
    for rule in ('S0', 'S1p'):
        print(f'\n== {rule}: sb_spec | ao_nogain | paired Δ (se), seeds {SEEDS}')
        for ds, m, k in cells:
            a = np.array([val[(r, rule, ds, m, k)] for r in ARMS['sb_spec']])
            b = np.array([val[(r, rule, ds, m, k)] for r in ARMS['ao_nogain']])
            d = a - b
            print(f'  {ds:4s} {m:13s} {k:4s} {a.mean():6.2f} | {b.mean():6.2f} | {d.mean():+.2f} '
                  f'({d.std(ddof=1) / np.sqrt(len(d)):.2f})')


if __name__ == '__main__':
    main()
