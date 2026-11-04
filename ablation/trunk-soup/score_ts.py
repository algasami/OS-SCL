#!/usr/bin/env python
"""Trunk-soup scorer: P1-P6 and D1 from the README addendum (2026-11-03).

    python ablation/trunk-soup/score_ts.py ablation/trunk-soup/results/emb

Dumps come from ../ensemble-backend/extract.py on this worktree's ts_* checkpoints.
Rules and metrics are ../ensemble-backend/score.py's, imported unchanged:
  S0 = R0 (eval.py), S1p = S1' (zR0' + zK', training-part stats)
  E1  = sum of members' zR0 (per-ID z of R0, held-out stats)   -> P3's fork ensemble
  E1p = sum of members' zR0p (training-part stats)              -> sensitivity
  E2p = sum of members' S1'                                     -> P2's fork ensemble
Independent references (ao_nogain s2040-2047, same recipe) are read from ../aug-ops/scores.

Checks first that S0 recomputed from each dump matches eval.py's committed per-ID AUCs.
Writes scores/systems.csv, scores/machines.csv, scores/check.json, scores/summary.txt.
"""
import csv
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from scores_path import scores_csv    # noqa: E402

_spec = importlib.util.spec_from_file_location('eb_score', os.path.join(HERE, '..', 'ensemble-backend', 'score.py'))
eb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eb)

TRUNKS = (2050, 2051)
FORKS = (1, 2, 3, 4)
P = 'SCLTFSTgramMFN_ts_'
INDEP = [f'SCLTFSTgramMFN_ao_nogain_s{s}' for s in range(2040, 2048)]
INDEP_E2_GAIN = 97.07 - 96.43   # hom4_nogain E2' minus single S1', dev AUC (addendum)
KEYS = ('AUC', 'pAUC', 'mAUC', 'mean_AUC_pAUC')


def forks(t):
    return [f'{P}fork_s{t}_f{f}' for f in FORKS]


def committed(run, ds, scope='overall'):
    """eval.py CSV row(s), in points."""
    out = {}
    with open(scores_csv(run, ds), encoding='UTF-8') as f:
        for r in csv.DictReader(f):
            if r['scope'] != scope:
                continue
            v = {k: 100 * float(r[k]) for k in ('AUC', 'pAUC', 'mAUC') if r.get(k) not in (None, '')}
            out[(r['machine'], r['id'])] = v
    return out


def main():
    d = sys.argv[1]
    meta = dict(np.load(os.path.join(d, 'meta.npz')))
    ev = {ds: eb.Evaluator(meta, ds) for ds in ('dev', 'eval')}
    runs = [f'{P}trunk_s{t}' for t in TRUNKS] + [r for t in TRUNKS for r in forks(t)] + \
           [f'{P}soup{k}_s{t}' for k in (4, 5) for t in TRUNKS]
    S, res, check = {}, {}, {}
    for r in runs:
        a = dict(np.load(os.path.join(d, r + '.npz')))
        S[r] = eb.per_run_scores(meta, a)
        c = {}
        for ds in ev:
            ids = committed(r, ds, 'id')
            for rule in ('S0', 'S1p'):
                res[(r, rule, ds)] = ev[ds](S[r][rule])
            got = res[(r, 'S0', ds)]
            c[ds] = max(abs(100 * got[m]['ids'][i] - ids[(m, i)]['AUC']) for m in eb.MACH for i in got[m]['ids'])
        check[r] = c
    worst = max(max(c.values()) for c in check.values())
    assert worst < 0.05, f'S0 does not reproduce eval.py: max per-ID |Δ| {worst:.3f} points'

    # fork ensembles per trunk (4 forks; 5 = with the trunk's own second half)
    for t in TRUNKS:
        for n, mem in ((4, forks(t)), (5, forks(t) + [f'{P}trunk_s{t}'])):
            for name, key in (('E1', 'zR0'), ('E1p', 'zR0p'), ('E2p', 'S1p')):
                sc = np.sum([S[m][key] for m in mem], axis=0)
                for ds in ev:
                    res[(f'{P}{name}{n}_s{t}', name, ds)] = ev[ds](sc)

    ov = {k: v['overall'] for k, v in res.items()}
    mach = {k: {m: 100 * v[m]['AUC'] for m in eb.MACH} for k, v in res.items()}

    # independent references (R0 from eval.py CSVs, S1' from aug-ops s1p.csv)
    ind = {ds: [committed(r, ds)[('ALL', 'ALL')] for r in INDEP] for ds in ('dev', 'eval')}
    for ds in ind:
        for v in ind[ds]:
            v['mean_AUC_pAUC'] = (v['AUC'] + v['pAUC']) / 2
    ind_mean = {ds: {k: np.mean([v[k] for v in ind[ds]]) for k in KEYS} for ds in ind}
    ind_mach = {ds: {m: np.mean([committed(r, ds, 'machine')[(m, 'ALL')]['AUC'] for r in INDEP])
                     for m in eb.MACH} for ds in ind}

    def g(r, rule, ds, k='AUC'):
        return ov[(r, rule, ds)][k]

    def mean_fork(t, rule, ds, k='AUC'):
        return np.mean([g(r, rule, ds, k) for r in forks(t)])

    L = ['Trunk soup: ao_nogain base, trunks s2050/s2051, forks 1-4 from epoch 150. Points x100.',
         f'check: S0 vs eval.py per-ID AUC, max |Δ| {worst:.4f} points', '']
    L.append(f"{'system':34s} {'rule':4s} | dev AUC  pAUC  mAUC  mean | eval AUC  pAUC  mAUC  mean")
    rows = []
    for k in sorted({(r, rule) for r, rule, _ in ov}):
        r, rule = k
        dv, evv = ov[(r, rule, 'dev')], ov[(r, rule, 'eval')]
        L.append(f'{r.replace("SCLTFSTgramMFN_", ""):34s} {rule:4s} | ' +
                 ' '.join(f'{dv[x]:6.2f}' for x in KEYS) + ' | ' + ' '.join(f'{evv[x]:6.2f}' for x in KEYS))
        for ds in ('dev', 'eval'):
            rows.append([r, rule, ds] + [f'{ov[(r, rule, ds)][x]:.4f}' for x in KEYS])
    L.append(f"{'independent ao_nogain, 8-seed mean':34s} {'S0':4s} | " +
             ' '.join(f'{ind_mean["dev"][x]:6.2f}' for x in KEYS) + ' | ' +
             ' '.join(f'{ind_mean["eval"][x]:6.2f}' for x in KEYS))
    L.append(f"{'paper':34s}      | {eb.PAPER['dev']['AUC']:6.2f} {eb.PAPER['dev']['pAUC']:6.2f} "
             f"{eb.PAPER['dev']['mAUC']:6.2f}        |                       {eb.PAPER['eval_mean']:6.2f}")

    # predictions
    L += ['', '== Predictions (addendum), R0 dev AUC unless stated']
    verdict = {}
    p1 = {t: (g(f'{P}soup4_s{t}', 'S0', 'dev') - mean_fork(t, 'S0', 'dev'),
              g(f'{P}soup4_s{t}', 'S0', 'eval') - mean_fork(t, 'S0', 'eval')) for t in TRUNKS}
    verdict['P1'] = all(0.10 <= a <= 0.40 and b >= 0 for a, b in p1.values())
    L.append('P1 soup4 − mean fork in [+0.10,+0.40] dev, ≥0 eval, both trunks: ' +
             '; '.join(f's{t} dev {a:+.2f} eval {b:+.2f}' for t, (a, b) in p1.items()) +
             f' -> {"holds" if verdict["P1"] else "fails"}')
    p2 = {t: g(f'{P}E2p4_s{t}', 'E2p', 'dev') - mean_fork(t, 'S1p', 'dev') for t in TRUNKS}
    verdict['P2'] = all(v <= 0.6 * INDEP_E2_GAIN for v in p2.values())
    L.append(f'P2 E2′(4 forks) − mean fork S1′ ≤ +{0.6 * INDEP_E2_GAIN:.2f} (60% of independent +{INDEP_E2_GAIN:.2f}): ' +
             '; '.join(f's{t} {v:+.2f} ({100 * v / INDEP_E2_GAIN:.0f}%)' for t, v in p2.items()) +
             f' -> {"holds" if verdict["P2"] else "fails"}')
    p3 = {t: g(f'{P}soup4_s{t}', 'S0', 'dev') - g(f'{P}E14_s{t}', 'E1', 'dev') for t in TRUNKS}
    verdict['P3'] = all(abs(v) <= 0.25 for v in p3.values())
    L.append('P3 |soup4 R0 − E1(4 forks)| ≤ 0.25 dev: ' +
             '; '.join(f's{t} {v:+.2f} (E1′ {g(f"{P}soup4_s{t}", "S0", "dev") - g(f"{P}E1p4_s{t}", "E1p", "dev"):+.2f})'
                       for t, v in p3.items()) + f' -> {"holds" if verdict["P3"] else "fails"}')
    p4 = {}
    for t in TRUNKS:
        dm = {m: mach[(f'{P}soup4_s{t}', 'S0', 'eval')][m] -
              np.mean([mach[(r, 'S0', 'eval')][m] for r in forks(t)]) for m in eb.MACH}
        p4[t] = dm
    verdict['P4'] = all(max(dm, key=dm.get) == 'ToyConveyor' for dm in p4.values())
    for t, dm in p4.items():
        L.append(f'P4 s{t} eval soup4 − mean fork per machine: ' + ' '.join(f'{m} {v:+.2f}' for m, v in dm.items()))
    L.append(f'P4 ToyConveyor largest on eval, both trunks -> {"holds" if verdict["P4"] else "fails"}')
    soup_dev = np.mean([g(f'{P}soup4_s{t}', 'S0', 'dev') for t in TRUNKS])
    soup_ev = np.mean([g(f'{P}soup4_s{t}', 'S0', 'eval') for t in TRUNKS])
    soup_evm = np.mean([g(f'{P}soup4_s{t}', 'S0', 'eval', 'mean_AUC_pAUC') for t in TRUNKS])
    p5 = soup_dev - ind_mean['dev']['AUC']
    verdict['P5'] = 0.10 <= p5 <= 0.40
    L.append(f'P5 mean soup4 − independent mean dev AUC in [+0.10,+0.40]: {soup_dev:.2f} − {ind_mean["dev"]["AUC"]:.2f}'
             f' = {p5:+.2f} -> {"holds" if verdict["P5"] else "fails"}')
    verdict['P6'] = soup_evm >= eb.PAPER['eval_mean']
    L.append(f'P6 mean soup4 eval mean(AUC,pAUC) ≥ 93.22: {soup_evm:.2f} -> {"holds" if verdict["P6"] else "fails"}')

    d1_dev, d1_ev = p5, soup_ev - ind_mean['eval']['AUC']
    d1 = d1_dev >= 0.20 and d1_ev >= 0 and verdict['P3']
    L += ['', f'D1 adopt soup recipe: dev Δ {d1_dev:+.2f} (≥ +0.20), eval AUC Δ {d1_ev:+.2f} (≥ 0), P3 both trunks '
              f'{"ok" if verdict["P3"] else "no"} -> {"PASSES" if d1 else "FAILS"}']
    d3 = any(v < -0.25 for v in p3.values())   # README D3: P3 fails *with the soup far below* its fork ensemble
    L.append(f'D3 trigger (soup more than 0.25 below E1 on dev, i.e. forks left the basin) -> {"yes" if d3 else "no"}'
             f'; soup4 S1′ − E2′(4 forks) dev: ' + '; '.join(
                 f's{t} {g(f"{P}soup4_s{t}", "S1p", "dev") - g(f"{P}E2p4_s{t}", "E2p", "dev"):+.2f}' for t in TRUNKS))

    # single networks vs paper
    L += ['', '== Single networks vs paper (dev AUC / pAUC / mAUC, eval mean)']
    for r in [f'{P}soup4_s{t}' for t in TRUNKS] + [f'{P}soup5_s{t}' for t in TRUNKS]:
        for rule in ('S0', 'S1p'):
            L.append(f'{r.replace("SCLTFSTgramMFN_", ""):18s} {rule:4s} ' +
                     ' / '.join(f'{g(r, rule, "dev", k):.2f}' for k in ('AUC', 'pAUC', 'mAUC')) +
                     f'  eval {g(r, rule, "eval", "mean_AUC_pAUC"):.2f}')
    for rule in ('S0', 'S1p'):
        L.append(f'mean fork, {rule}: ' + '  '.join(
            f's{t} dev {mean_fork(t, rule, "dev"):.2f} eval mean {mean_fork(t, rule, "eval", "mean_AUC_pAUC"):.2f}'
            for t in TRUNKS))

    out = os.path.join(HERE, 'scores')
    with open(os.path.join(out, 'systems.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['system', 'rule', 'dataset'] + list(KEYS))
        w.writerows(rows)
    with open(os.path.join(out, 'machines.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['system', 'rule', 'dataset'] + eb.MACH)
        for (r, rule, ds), v in sorted(mach.items()):
            w.writerow([r, rule, ds] + [f'{v[m]:.4f}' for m in eb.MACH])
        for ds in ind_mach:
            w.writerow(['independent_ao_nogain_mean8', 'S0', ds] + [f'{ind_mach[ds][m]:.4f}' for m in eb.MACH])
    json.dump({'s0_vs_evalpy_max_abs_id_auc': check, 'verdict': {k: bool(v) for k, v in verdict.items()}, 'D1': bool(d1)},
              open(os.path.join(out, 'check.json'), 'w'), indent=2)
    with open(os.path.join(out, 'summary.txt'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
