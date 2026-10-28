#!/usr/bin/env python
"""S1′ (../confirm-seeds' adopted single-model rule) for every lit-arms run and the fr_f1 baseline.

    python ablation/lit-arms/score_s1p.py ablation/lit-arms/results/emb

Inputs come from ../ensemble-backend/extract.py; the rule and the metrics are
../ensemble-backend/score.py's, imported unchanged. Checks first that S0 (= R0)
recomputed from the dump matches eval.py's committed per-ID AUCs, and that the
replayed 5% split gives the run's best validation loss.

Writes scores/s1p.csv (run, rule, dataset, AUC, pAUC, mAUC, mean_AUC_pAUC) and
scores/s1p_check.json; prints the arm − fr_f1 contrasts under S0 and S1′.
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

SEEDS = (2040, 2041, 2042, 2043)
GROUPS = {'fr_f1': 'fr_f1', 'aug': 'la_aug', 'gate': 'la_gate', 'eca': 'la_eca'}
RUNS = {g: [f'SCLTFSTgramMFN_{p}_s{s}' for s in SEEDS] for g, p in GROUPS.items()}
RUNS['aug_rep'] = [f'SCLTFSTgramMFN_la_aug_s{s}' for s in (2044, 2045, 2046, 2047)]   # D2 replication
KEYS = ('AUC', 'pAUC', 'mAUC', 'mean_AUC_pAUC')


def main():
    d = sys.argv[1]
    meta = dict(np.load(os.path.join(d, 'meta.npz')))
    ev = {ds: eb.Evaluator(meta, ds) for ds in ('dev', 'eval')}
    rows, check = [], {}
    for g, runs in RUNS.items():
        for r in runs:
            a = dict(np.load(os.path.join(d, f'{r}.npz')))
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
    for rule in ('S0', 'S1p'):
        print(f'\n== {rule}: 4-seed means (dev AUC / pAUC / mAUC, eval mean(AUC,pAUC)); Δ vs fr_f1')
        b = {(ds, k): np.mean([val[(r, rule, ds)][k] for r in RUNS['fr_f1']]) for ds in ev for k in KEYS}
        for g, runs in RUNS.items():
            m = {(ds, k): np.mean([val[(r, rule, ds)][k] for r in runs]) for ds in ev for k in KEYS}
            cells = [('dev', 'AUC'), ('dev', 'pAUC'), ('dev', 'mAUC'), ('eval', 'AUC'), ('eval', 'mean_AUC_pAUC')]
            s = '  '.join(f'{ds[:3]} {k[:4]} {m[(ds, k)]:6.2f}' + (f' ({m[(ds, k)] - b[(ds, k)]:+.2f})' if g != 'fr_f1' else '')
                          for ds, k in cells)
            print(f'{g:7s} {s}')
        print(f"paper  dev AUC {eb.PAPER['dev']['AUC']}  pAUC {eb.PAPER['dev']['pAUC']}  "
              f"mAUC {eb.PAPER['dev']['mAUC']}  eval mean {eb.PAPER['eval_mean']}")
        if rule == 'S1p':
            for g in ('aug', 'gate', 'eca'):
                print(f'{g:7s} per seed dev AUC Δ',
                      ' '.join(f"{val[(a, rule, 'dev')]['AUC'] - val[(b, rule, 'dev')]['AUC']:+.2f}"
                               for a, b in zip(RUNS[g], RUNS['fr_f1'])))


if __name__ == '__main__':
    main()
