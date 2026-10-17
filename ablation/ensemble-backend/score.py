#!/usr/bin/env python
"""Score the extracted arrays under the pre-registered rules; write scores/.

    python ablation/ensemble-backend/score.py ablation/ensemble-backend/results/emb

Rules (README, *Rules*), per checkpoint m and machine ID i:
  S0 = R0                         eval.py's score
  S1 = zR0 + zK                   K = 1-NN cosine distance of the pre-head embedding
                                  to ID i's training-part clips
  z(.) uses mean/sd over ID i's HELD-OUT validation clips (never trained on);
  the primed variants (S1', E1') use training-part clips instead (P6).
  E0 = sum R0, E1 = sum zR0, E1' = sum zR0', E2 = sum (zR0 + zK)   over an ensemble

Writes: rule_metrics.csv, summary.csv, ensemble_curve.csv, bootstrap.json,
reproduction.json, scorecard.txt.
"""
import csv
import itertools
import json
import os
import sys

import numpy as np
from sklearn import metrics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from scores_path import scores_csv, scores_dir    # noqa: E402

SCORES = os.path.join(HERE, 'scores')
MACH = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
RELU = [f'SCLTFSTgramMFN_tfl_relu_max_s{s}' for s in (2024, 2025, 2026, 2027)]
LIN = [f'SCLTFSTgramMFN_tfl_linear_avg_s{s}' for s in (2024, 2025, 2026, 2027)]
ALL = RELU + LIN
FRESH = [r for r in ALL if r.endswith(('2026', '2027'))]
GROUPS = {'relu4': RELU, 'lin4': LIN, 'mix8': ALL}
PAPER = {'dev': {'AUC': 95.71, 'pAUC': 90.23, 'mAUC': 91.23}, 'eval_mean': 93.22,
         'dev_machine': {'fan': 98.41, 'pump': 96.25, 'slider': 99.63, 'ToyCar': 96.55,
                         'ToyConveyor': 83.56, 'valve': 99.85}}
N_BOOT = 2000


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def nn_dist(q, ref, loo=False):
    s = q @ ref.T
    if loo:
        np.fill_diagonal(s, -np.inf)
    return 1.0 - s.max(1)


def z(x, ref):
    return (x - ref.mean()) / ref.std()


def per_run_scores(meta, a):
    """All single-model score vectors for one checkpoint, over every test clip."""
    trl, tel = meta['tr_label'], meta['te_label']
    valid = a['valid']
    trp, tep = l2(a['tr_pre'].astype(np.float64)), l2(a['te_pre'].astype(np.float64))
    trc, tec = a['tr_ce'].astype(np.float64), a['te_ce'].astype(np.float64)
    out = {k: np.full(len(tel), np.nan) for k in ('zR0', 'zK', 'zR0p', 'zKp')}
    for c in np.unique(tel):
        ref, hold, te = (trl == c) & ~valid, (trl == c) & valid, tel == c
        k_te = nn_dist(tep[te], trp[ref])
        k_hold = nn_dist(trp[hold], trp[ref])
        k_ref = nn_dist(trp[ref], trp[ref], loo=True)
        out['zR0'][te] = z(tec[te], trc[hold])
        out['zK'][te] = z(k_te, k_hold)
        out['zR0p'][te] = z(tec[te], trc[ref])
        out['zKp'][te] = z(k_te, k_ref)
    out['S0'] = tec
    out['S1'] = out['zR0'] + out['zK']
    out['S1p'] = out['zR0p'] + out['zKp']
    return out


class Evaluator:
    """eval.py's metrics on one dataset, with optional stratified resampling."""

    def __init__(self, meta, ds):
        self.sel = np.where(meta['te_dataset'] == ds)[0]
        self.mach = meta['te_machine'][self.sel]
        self.id = meta['te_id'][self.sel]
        self.y = meta['te_y'][self.sel]
        self.strata = {}
        for m in MACH:
            for i in np.unique(self.id[self.mach == m]):
                for yy in (0, 1):
                    self.strata[(m, i, yy)] = np.where((self.mach == m) & (self.id == i) & (self.y == yy))[0]

    def resample(self, rng):
        return np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in self.strata.values()])

    def __call__(self, score, idx=None, per_id=False):
        s = score[self.sel]
        if idx is None:
            idx = np.arange(len(s))
        mach, ids, y, s = self.mach[idx], self.id[idx], self.y[idx], s[idx]
        res = {}
        for m in MACH:
            mm = mach == m
            ida = {i: metrics.roc_auc_score(y[mm & (ids == i)], s[mm & (ids == i)])
                   for i in np.unique(ids[mm])}
            res[m] = dict(AUC=metrics.roc_auc_score(y[mm], s[mm]),
                          pAUC=metrics.roc_auc_score(y[mm], s[mm], max_fpr=0.1),
                          mAUC=min(ida.values()), ids=ida)
        res['overall'] = {k: 100 * np.mean([res[m][k] for m in MACH]) for k in ('AUC', 'pAUC', 'mAUC')}
        res['overall']['mean_AUC_pAUC'] = (res['overall']['AUC'] + res['overall']['pAUC']) / 2
        return res


def main():
    d = sys.argv[1]
    os.makedirs(SCORES, exist_ok=True)
    meta = dict(np.load(os.path.join(d, 'meta.npz')))
    raw = {r: dict(np.load(os.path.join(d, f'{r}.npz'))) for r in ALL}
    S = {r: per_run_scores(meta, raw[r]) for r in ALL}
    ev = {ds: Evaluator(meta, ds) for ds in ('dev', 'eval')}

    # ---- reproduction: P0 and split replay -------------------------------------------
    repro = {}
    for r in ALL:
        e = {}
        for ds in ('dev', 'eval'):
            res = ev[ds](S[r]['S0'])
            committed = {}
            with open(scores_csv(r, ds), encoding='UTF-8') as f:
                for row in csv.DictReader(f):
                    if row['scope'] == 'id':
                        committed[(row['machine'], row['id'])] = float(row['AUC'])
            e[f'{ds}_max_abs_id_auc_diff'] = max(abs(res[m]['ids'][i] - committed[(m, i)])
                                                 for m in MACH for i in res[m]['ids'])
        vo = raw[r]['valid_order']
        ce = raw[r]['tr_ce'][vo].astype(np.float64)
        batches = [ce[i:i + 64].mean() for i in range(0, len(ce), 64)]
        with open(os.path.join(scores_dir(r), 'curve.csv'), encoding='UTF-8') as f:
            best = list(csv.DictReader(f))[-1]
        e.update(n_valid=int(raw[r]['valid'].sum()), valid_loss_replayed=float(np.mean(batches)),
                 valid_loss_curve=float(best['valid_loss']), best_epoch=int(best['epoch']))
        e['valid_loss_abs_diff'] = abs(e['valid_loss_replayed'] - e['valid_loss_curve'])
        repro[r] = e
    json.dump(repro, open(os.path.join(SCORES, 'reproduction.json'), 'w'), indent=2)

    # ---- every system x rule ----------------------------------------------------------
    systems = {}
    for r in ALL:
        for rule in ('S0', 'S1', 'S1p'):
            systems[(r, rule)] = S[r][rule]
    for g, mem in GROUPS.items():
        systems[(g, 'E0')] = sum(S[r]['S0'] for r in mem)
        systems[(g, 'E1')] = sum(S[r]['zR0'] for r in mem)
        systems[(g, 'E1p')] = sum(S[r]['zR0p'] for r in mem)
        systems[(g, 'E2')] = sum(S[r]['S1'] for r in mem)
    results = {(sys_, rule, ds): ev[ds](sc) for (sys_, rule), sc in systems.items() for ds in ev}

    with open(os.path.join(SCORES, 'rule_metrics.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['system', 'dataset', 'rule', 'machine', 'id', 'scope', 'AUC', 'pAUC', 'mAUC'])
        for (sys_, rule, ds), res in results.items():
            for m in MACH:
                for i, a in res[m]['ids'].items():
                    w.writerow([sys_, ds, rule, m, i, 'id', f'{a:.5f}', '', ''])
                w.writerow([sys_, ds, rule, m, 'ALL', 'machine'] + [f'{res[m][k]:.5f}' for k in ('AUC', 'pAUC', 'mAUC')])
            o = res['overall']
            w.writerow([sys_, ds, rule, 'ALL', 'ALL', 'overall'] + [f'{o[k] / 100:.5f}' for k in ('AUC', 'pAUC', 'mAUC')])
    with open(os.path.join(SCORES, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['system', 'dataset', 'rule', 'AUC', 'pAUC', 'mAUC', 'mean_AUC_pAUC'] + MACH)
        for (sys_, rule, ds), res in results.items():
            o = res['overall']
            w.writerow([sys_, ds, rule] + [f'{o[k]:.2f}' for k in ('AUC', 'pAUC', 'mAUC', 'mean_AUC_pAUC')]
                       + [f'{100 * res[m]["AUC"]:.2f}' for m in MACH])

    # ---- ensemble size curve: E1 and E2 over every subset of the 8 -------------------
    with open(os.path.join(SCORES, 'ensemble_curve.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rule', 'dataset', 'n', 'n_relu', 'members', 'AUC', 'pAUC', 'mAUC', 'mean_AUC_pAUC'])
        for rule, key in (('E1', 'zR0'), ('E2', 'S1')):
            for n in range(1, 9):
                for sub in itertools.combinations(ALL, n):
                    sc = sum(S[r][key] for r in sub)
                    for ds in ev:
                        o = ev[ds](sc)['overall']
                        w.writerow([rule, ds, n, sum(r in RELU for r in sub),
                                    '+'.join(r.split('tfl_')[1] for r in sub)]
                                   + [f'{o[k]:.3f}' for k in ('AUC', 'pAUC', 'mAUC', 'mean_AUC_pAUC')])

    # ---- bootstrap over test clips, paired ---------------------------------------------
    rng = np.random.default_rng(0)
    boot = {}
    for ds in ev:
        best = max(RELU, key=lambda r: results[(r, 'S0', ds)]['overall']['AUC'])
        vecs = {'E1': systems[('relu4', 'E1')], 'E2': systems[('relu4', 'E2')], 'best_member_S0': S[best]['S0']}
        draws = {k: [] for k in vecs}
        for _ in range(N_BOOT):
            idx = ev[ds].resample(rng)
            for k, v in vecs.items():
                draws[k].append(ev[ds](v, idx)['overall'])
        b = {'best_member': best}
        for k in ('AUC', 'pAUC', 'mAUC', 'mean_AUC_pAUC'):
            e1 = np.array([x[k] for x in draws['E1']])
            b[f'E1_{k}'] = [results[('relu4', 'E1', ds)]['overall'][k]] + list(np.percentile(e1, [2.5, 97.5]))
            for other in ('best_member_S0', 'E2'):
                dd = np.array([x[k] for x in draws[other]]) - e1
                pt = (results[(best, 'S0', ds)] if other == 'best_member_S0' else results[('relu4', 'E2', ds)])['overall'][k] \
                    - results[('relu4', 'E1', ds)]['overall'][k]
                b[f'{other}_minus_E1_{k}'] = [pt] + list(np.percentile(dd, [2.5, 97.5]))
        if ds == 'dev':
            for k in ('AUC', 'pAUC', 'mAUC'):
                e1 = np.array([x[k] for x in draws['E1']])
                b[f'frac_E1_{k}_above_paper'] = float((e1 > PAPER['dev'][k]).mean())
        else:
            e1 = np.array([x['mean_AUC_pAUC'] for x in draws['E1']])
            b['frac_E1_mean_above_paper'] = float((e1 > PAPER['eval_mean']).mean())
        boot[ds] = b
    json.dump(boot, open(os.path.join(SCORES, 'bootstrap.json'), 'w'), indent=2)

    # ---- scorecard ----------------------------------------------------------------------
    O = lambda s, rule, ds, k='AUC': results[(s, rule, ds)]['overall'][k]
    L = []
    p = L.append
    p('P0  max |per-ID AUC - committed| (dev, eval), |valid loss replayed - curve.csv|, n_valid:')
    for r, e in repro.items():
        p(f'    {r:38s} {e["dev_max_abs_id_auc_diff"]:.2e} {e["eval_max_abs_id_auc_diff"]:.2e}  '
          f'{e["valid_loss_replayed"]:.5f} vs {e["valid_loss_curve"]:.5f} (|d| {e["valid_loss_abs_diff"]:.1e})  {e["n_valid"]}')
    for ds in ev:
        mem = [O(r, 'S0', ds) for r in RELU]
        e1 = O('relu4', 'E1', ds)
        p(f'P1  {ds}: E1[relu4] {e1:.2f} | members {" ".join(f"{x:.2f}" for x in mem)} | '
          f'beats best member: {e1 > max(mem)} | gain over mean {e1 - np.mean(mem):+.2f}')
    curve = {}
    with open(os.path.join(SCORES, 'ensemble_curve.csv')) as f:
        for row in csv.DictReader(f):
            if row['rule'] == 'E1':
                curve.setdefault(row['dataset'], {}).setdefault(int(row['n']), []).append(float(row['AUC']))
    for ds, c in curve.items():
        means = [np.mean(c[n]) for n in range(1, 9)]
        p(f'P1  {ds}: size curve mean AUC n=1..8 {" ".join(f"{x:.2f}" for x in means)} | '
          f'non-decreasing: {all(b >= a for a, b in zip(means, means[1:]))}')
    r_ = results[('relu4', 'E1', 'dev')]['overall']
    p(f'P2  dev E1[relu4] AUC {r_["AUC"]:.2f} (>=95.71: {r_["AUC"] >= 95.71}) pAUC {r_["pAUC"]:.2f} '
      f'(>=90.23: {r_["pAUC"] >= 90.23}) mAUC {r_["mAUC"]:.2f} (>=91.23: {r_["mAUC"] >= 91.23})')
    em = O('relu4', 'E1', 'eval', 'mean_AUC_pAUC')
    p(f'P2  eval E1[relu4] mean(AUC,pAUC) {em:.2f} (>=93.22: {em >= 93.22})')
    pump = 100 * results[('relu4', 'E1', 'dev')]['pump']['AUC']
    p(f'P3  dev E1[relu4] pump {pump:.2f} (<96.25: {pump < 96.25}) | per machine vs paper: ' +
      ' '.join(f'{m} {100 * results[("relu4", "E1", "dev")][m]["AUC"] - PAPER["dev_machine"][m]:+.2f}' for m in MACH))
    for name, runs, ds in (('C-eval', ALL, 'eval'), ('C-dev-fresh', FRESH, 'dev'), ('C-dev-all', ALL, 'dev')):
        dl = [O(r, 'S1', ds) - O(r, 'S0', ds) for r in runs]
        dm = [O(r, 'S1', ds, 'mAUC') - O(r, 'S0', ds, 'mAUC') for r in runs]
        p(f'P4  {name}: S1-S0 AUC {" ".join(f"{x:+.2f}" for x in dl)} | positive {sum(x > 0 for x in dl)}/{len(dl)} '
          f'| mean {np.mean(dl):+.2f} | mAUC mean {np.mean(dm):+.2f}')
    for ds in ev:
        p(f'P5  {ds}: E2-E1 [relu4] {O("relu4", "E2", ds) - O("relu4", "E1", ds):+.2f} | '
          f'[lin4] {O("lin4", "E2", ds) - O("lin4", "E1", ds):+.2f} | [mix8] {O("mix8", "E2", ds) - O("mix8", "E1", ds):+.2f}')
    for ds in ev:
        s1 = max(abs(O(r, 'S1', ds) - O(r, 'S1p', ds)) for r in ALL)
        e1 = max(abs(O(g, 'E1', ds) - O(g, 'E1p', ds)) for g in GROUPS)
        e0 = max(abs(O(g, 'E1', ds) - O(g, 'E0', ds)) for g in GROUPS)
        p(f'P6  {ds}: max|S1-S1p| {s1:.2f}  max|E1-E1p| {e1:.2f}  max|E1-E0| {e0:.2f}')
    with open(os.path.join(SCORES, 'ensemble_curve.csv')) as f:
        rows = [r for r in csv.DictReader(f) if r['rule'] == 'E1' and r['n'] == '4']
    for ds in ev:
        mixed = [float(r['AUC']) for r in rows if r['dataset'] == ds and r['n_relu'] == '2']
        p(f'P7  {ds}: mean E1 over 2+2 subsets {np.mean(mixed):.2f} vs relu4 {O("relu4", "E1", ds):.2f} '
          f'(diff {np.mean(mixed) - O("relu4", "E1", ds):+.2f}) | mix8 {O("mix8", "E1", ds):.2f}')
    for ds, b in boot.items():
        p(f'BOOT {ds}: ' + json.dumps({k: (np.round(v, 2).tolist() if isinstance(v, list) else v) for k, v in b.items()}))
    txt = '\n'.join(L)
    print(txt)
    open(os.path.join(SCORES, 'scorecard.txt'), 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
