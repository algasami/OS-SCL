#!/usr/bin/env python
"""S1′ (../confirm-seeds' adopted single-model rule) for ao_nogain and its paired la_aug runs.

    python ablation/aug-ops/score_s1p.py <aug-ops emb dir> <lit-arms emb dir>

The ao_nogain dumps come from ../ensemble-backend/extract.py run on this worktree; the
la_aug dumps are ../lit-arms' (results/emb, scratch). The two metas must agree below
data/. Rule and metrics are ../ensemble-backend/score.py's, imported unchanged.
Checks first, for every run, that S0 (= R0) recomputed from the dump matches eval.py's
committed per-ID AUCs, and that the replayed 5% split gives the run's best validation loss.

Writes scores/s1p.csv (run, rule, dataset, AUC, pAUC, mAUC, mean_AUC_pAUC) and
scores/s1p_check.json; prints 4-seed means under S0 and S1′ against the paper.
"""
import csv
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from scores_path import scores_csv, scores_dir    # noqa: E402

_spec = importlib.util.spec_from_file_location('eb_score', os.path.join(HERE, '..', 'ensemble-backend', 'score.py'))
eb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eb)

S1, S2 = (2040, 2041, 2042, 2043), (2044, 2045, 2046, 2047)
RUNS = {'aug': [f'SCLTFSTgramMFN_la_aug_s{s}' for s in S1],
        'aug_rep': [f'SCLTFSTgramMFN_la_aug_s{s}' for s in S2],
        'nogain': [f'SCLTFSTgramMFN_ao_nogain_s{s}' for s in S1],
        'nogain_rep': [f'SCLTFSTgramMFN_ao_nogain_s{s}' for s in S2]}
KEYS = ('AUC', 'pAUC', 'mAUC', 'mean_AUC_pAUC')


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
    rows, check = [], {}
    for runs in RUNS.values():
        for r in runs:
            a = dict(np.load(where[r]))
            S = eb.per_run_scores(meta, a)
            c = {}
            for ds in ev:
                committed = {}
                with open(scores_csv(r, ds), encoding='UTF-8') as f:
                    for row in csv.DictReader(f):
                        if row['scope'] == 'id':
                            committed[(row['machine'], row['id'])] = float(row['AUC'])
                for rule in ('S0', 'S1p'):
                    res = ev[ds](S[rule])
                    rows.append([r, rule, ds] + [f"{res['overall'][k]:.4f}" for k in KEYS])
                    if rule == 'S0':
                        c[f'{ds}_max_abs_id_auc_diff'] = max(abs(res[m]['ids'][i] - committed[(m, i)])
                                                             for m in eb.MACH for i in res[m]['ids'])
            ce = a['tr_ce'][a['valid_order']].astype(np.float64)
            with open(os.path.join(scores_dir(r), 'curve.csv'), encoding='UTF-8') as f:
                best = float(list(csv.DictReader(f))[-1]['valid_loss'])
            c['valid_loss_replayed'] = float(np.mean([ce[i:i + 64].mean() for i in range(0, len(ce), 64)]))
            c['valid_loss_curve'] = best
            check[r] = c
    out = os.path.join(HERE, 'scores')
    with open(os.path.join(out, 's1p.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'rule', 'dataset'] + list(KEYS))
        w.writerows(rows)
    json.dump(check, open(os.path.join(out, 's1p_check.json'), 'w'), indent=2)

    worst = max(max(c['dev_max_abs_id_auc_diff'], c['eval_max_abs_id_auc_diff']) for c in check.values())
    vl = max(abs(c['valid_loss_replayed'] - c['valid_loss_curve']) for c in check.values())
    print(f'check: max |S0 − eval.py| per-ID AUC {worst:.2e}; max |replayed − curve| valid loss {vl:.2e}')

    val = {(r, rule, ds): dict(zip(KEYS, map(float, v))) for r, rule, ds, *v in rows}
    cells = [('dev', 'AUC'), ('dev', 'pAUC'), ('dev', 'mAUC'), ('eval', 'AUC'), ('eval', 'mean_AUC_pAUC')]
    for rule in ('S0', 'S1p'):
        print(f'\n== {rule}: 4-seed means (dev AUC / pAUC / mAUC, eval AUC, eval mean(AUC,pAUC))')
        for g, runs in RUNS.items():
            print(f'{g:11s}' + '  '.join(f'{ds[:3]} {k[:4]} {np.mean([val[(r, rule, ds)][k] for r in runs]):6.2f}'
                                         for ds, k in cells))
        print(f"paper      dev AUC {eb.PAPER['dev']['AUC']}  pAUC {eb.PAPER['dev']['pAUC']}  "
              f"mAUC {eb.PAPER['dev']['mAUC']}  eval mean {eb.PAPER['eval_mean']}")
        for g, b in (('nogain', 'aug'), ('nogain_rep', 'aug_rep')):
            d = [val[(x, rule, 'dev')]['AUC'] - val[(y, rule, 'dev')]['AUC'] for x, y in zip(RUNS[g], RUNS[b])]
            e = [val[(x, rule, 'eval')]['mean_AUC_pAUC'] - val[(y, rule, 'eval')]['mean_AUC_pAUC']
                 for x, y in zip(RUNS[g], RUNS[b])]
            print(f'{g} − {b}: dev AUC Δ per seed ' + ' '.join(f'{x:+.2f}' for x in d) +
                  f' (mean {np.mean(d):+.2f});  eval mean Δ ' + ' '.join(f'{x:+.2f}' for x in e) +
                  f' (mean {np.mean(e):+.2f})')
        n = {k: sum(val[(r, rule, 'dev')][k] > eb.PAPER['dev'][k] for r in RUNS['nogain'] + RUNS['nogain_rep'])
             for k in ('AUC', 'pAUC', 'mAUC')}
        ne = sum(val[(r, rule, 'eval')]['mean_AUC_pAUC'] > eb.PAPER['eval_mean']
                 for r in RUNS['nogain'] + RUNS['nogain_rep'])
        print(f'ao_nogain seeds above the paper (of 8): dev AUC {n["AUC"]}, pAUC {n["pAUC"]}, mAUC {n["mAUC"]}; '
              f'eval mean {ne}')


if __name__ == '__main__':
    main()
