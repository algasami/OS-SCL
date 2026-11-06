#!/usr/bin/env python
"""S1′ (primary here) and the paper's kNN rule for pm and its paired ao_nogain seeds.

    python ablation/permix/score_s1p.py <permix emb dir> <aug-ops emb dir>

Dumps come from ../ensemble-backend/extract.py (pm on this worktree; ao_nogain on
../aug-ops). S0 and S1p are ../ensemble-backend/score.py's rules, imported unchanged.
K is the PerMix paper's score (Eq. 9): cosine distance of the L2-normalised pre-head
embedding to its nearest neighbour among the clip's own ID's training-part clips, raw
(no z-scoring), pooled per machine as eval.py pools. Checks that S0 recomputed from every
dump matches eval.py's committed per-ID AUCs.

Writes scores/s1p.csv (run, rule, dataset, machine, id, AUC, pAUC); score.py reads it.
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

SEEDS = [int(s) for s in os.environ.get('PM_SEEDS', '2040 2041 2042 2043 2044 2045 2046 2047').split()]
ARMS = {'pm': 'SCLTFSTgramMFN_pm_s{}', 'ao_nogain': 'SCLTFSTgramMFN_ao_nogain_s{}'}


def below_data(paths):
    return np.array([p.split('/data/', 1)[1] for p in paths])


def knn_raw(meta, a):
    """The paper's rule: 1-NN cosine distance to the own ID's training-part clips."""
    trl, tel = meta['tr_label'], meta['te_label']
    trp, tep = eb.l2(a['tr_pre'].astype(np.float64)), eb.l2(a['te_pre'].astype(np.float64))
    out = np.full(len(tel), np.nan)
    for c in np.unique(tel):
        ref, te = (trl == c) & ~a['valid'], tel == c
        out[te] = eb.nn_dist(tep[te], trp[ref])
    return out


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
    rows, worst = [], 0.0
    for s in SEEDS:
        runs = [t.format(s) for t in ARMS.values()]
        if not all(r in where for r in runs):
            continue                                  # score only complete pairs
        for r in runs:
            a = dict(np.load(where[r]))
            S = eb.per_run_scores(meta, a)
            S['K'] = knn_raw(meta, a)
            for ds in ev:
                committed = {}
                with open(scores_csv(r, ds), encoding='UTF-8') as f:
                    for row in csv.DictReader(f):
                        if row['scope'] == 'id':
                            committed[(row['machine'], row['id'])] = float(row['AUC'])
                for rule in ('S0', 'S1p', 'K'):
                    res = ev[ds](S[rule])
                    if rule == 'S0':
                        worst = max(worst, max(abs(res[m]['ids'][i] - committed[(m, i)])
                                               for m in eb.MACH for i in res[m]['ids']))
                    o = res['overall']
                    rows.append([r, rule, ds, 'ALL', 'ALL', f"{o['AUC']:.4f}", f"{o['pAUC']:.4f}"])
                    for m in eb.MACH:
                        rows.append([r, rule, ds, m, 'ALL', f"{100 * res[m]['AUC']:.4f}",
                                     f"{100 * res[m]['pAUC']:.4f}"])
                        for i, v in res[m]['ids'].items():
                            rows.append([r, rule, ds, m, i, f'{100 * v:.4f}', ''])
            print(f'scored {r}', flush=True)
    assert rows, 'no complete pm/ao_nogain pair in the dumps'
    assert worst < 5e-4, f'S0 does not reproduce eval.py: max per-ID |Δ| {worst:.2e}'
    print(f'check: max |S0 − eval.py| per-ID AUC {worst:.2e}')
    with open(os.path.join(HERE, 'scores', 's1p.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'rule', 'dataset', 'machine', 'id', 'AUC', 'pAUC'])
        w.writerows(rows)


if __name__ == '__main__':
    main()
