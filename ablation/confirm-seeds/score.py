#!/usr/bin/env python
"""Score the pre-registered rules on the fresh seeds (N4) and the old ones (O4); write scores/.

    python ablation/confirm-seeds/score.py ablation/confirm-seeds/results/emb

The rules, their code and the metrics are ../ensemble-backend's (per_run_scores,
Evaluator), imported unchanged. Per checkpoint: S0 = R0, S1 = zR0 + zK, S1p = zR0' + zK'
(training-part statistics). Ensembles: E0 = sum R0, E1 = sum zR0, E1p = sum zR0',
E2p = sum S1p. Predictions P0-P7 and decisions D1-D4 are the README's, checked as written.

Writes: rule_metrics.csv, summary.csv, ensemble_curve.csv, bootstrap.json,
reproduction.json, scorecard.txt.
"""
import csv
import importlib.util
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from scores_path import scores_csv, scores_dir    # noqa: E402

# ../ensemble-backend/score.py under another name: a plain `import score` finds this file
_spec = importlib.util.spec_from_file_location('eb_score', os.path.join(HERE, '..', 'ensemble-backend', 'score.py'))
eb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eb)
MACH, PAPER, Evaluator, per_run_scores = eb.MACH, eb.PAPER, eb.Evaluator, eb.per_run_scores

SCORES = os.path.join(HERE, 'scores')
N4 = [f'SCLTFSTgramMFN_cs_relu_max_s{s}' for s in (2028, 2029, 2030, 2031)]
O4 = [f'SCLTFSTgramMFN_tfl_relu_max_s{s}' for s in (2024, 2025, 2026, 2027)]
ALL = N4 + O4
GROUPS = {'N4': N4, 'O4': O4, 'relu8': ALL}
O4_S0 = {'dev': 94.82, 'eval': 95.47}     # the README's calibration, P1
N_BOOT = 2000
KEYS = ('AUC', 'pAUC', 'mAUC', 'mean_AUC_pAUC')


def short(r):
    return r.split('SCLTFSTgramMFN_')[1]


def main():
    d = sys.argv[1]
    os.makedirs(SCORES, exist_ok=True)
    meta = dict(np.load(os.path.join(d, 'meta.npz')))
    raw = {r: dict(np.load(os.path.join(d, f'{r}.npz'))) for r in ALL}
    S = {r: per_run_scores(meta, raw[r]) for r in ALL}
    ev = {ds: Evaluator(meta, ds) for ds in ('dev', 'eval')}

    # ---- P0: score reproduction and split replay ---------------------------------------
    repro = {}
    for r in ALL:
        e = {}
        for ds in ev:
            res = ev[ds](S[r]['S0'])
            committed = {}
            with open(scores_csv(r, ds), encoding='UTF-8') as f:
                for row in csv.DictReader(f):
                    if row['scope'] == 'id':
                        committed[(row['machine'], row['id'])] = float(row['AUC'])
            e[f'{ds}_max_abs_id_auc_diff'] = max(abs(res[m]['ids'][i] - committed[(m, i)])
                                                 for m in MACH for i in res[m]['ids'])
        ce = raw[r]['tr_ce'][raw[r]['valid_order']].astype(np.float64)
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
        systems[(g, 'E2p')] = sum(S[r]['S1p'] for r in mem)
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
            w.writerow([sys_, ds, rule] + [f'{o[k]:.2f}' for k in KEYS]
                       + [f'{100 * res[m]["AUC"]:.2f}' for m in MACH])

    # ---- P7: E1 and E2p over every subset of relu8 -------------------------------------
    with open(os.path.join(SCORES, 'ensemble_curve.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rule', 'dataset', 'n', 'n_new', 'members'] + list(KEYS))
        for rule, key in (('E1', 'zR0'), ('E2p', 'S1p')):
            for n in range(1, 9):
                for sub in itertools.combinations(ALL, n):
                    sc = sum(S[r][key] for r in sub)
                    for ds in ev:
                        o = ev[ds](sc)['overall']
                        w.writerow([rule, ds, n, sum(r in N4 for r in sub), '+'.join(short(r) for r in sub)]
                                   + [f'{o[k]:.3f}' for k in KEYS])

    # ---- bootstrap over test clips: one resample shared by every vector ----------------
    rng = np.random.default_rng(0)
    vecs = {f'{r}|{rule}': S[r][rule] for r in N4 for rule in ('S0', 'S1p')}
    vecs.update({'N4|E1': systems[('N4', 'E1')], 'N4|E2p': systems[('N4', 'E2p')]})
    boot = {}
    for ds in ev:
        draws = {k: {m: [] for m in KEYS} for k in vecs}
        for _ in range(N_BOOT):
            idx = ev[ds].resample(rng)
            for k, v in vecs.items():
                o = ev[ds](v, idx)['overall']
                for m in KEYS:
                    draws[k][m].append(o[m])
        D = {k: {m: np.array(v) for m, v in dd.items()} for k, dd in draws.items()}
        pt = lambda s, rule, m: results[(s, rule, ds)]['overall'][m]   # noqa: E731
        ci = lambda point, arr: [float(point)] + [float(x) for x in np.percentile(arr, [2.5, 97.5])]  # noqa: E731
        b = {}
        for m in KEYS:
            dl = np.mean([D[f'{r}|S1p'][m] - D[f'{r}|S0'][m] for r in N4], axis=0)
            b[f'mean_S1p_minus_S0_{m}'] = ci(np.mean([pt(r, 'S1p', m) - pt(r, 'S0', m) for r in N4]), dl)
            ms = np.mean([D[f'{r}|S1p'][m] for r in N4], axis=0)
            b[f'mean_S1p_{m}'] = ci(np.mean([pt(r, 'S1p', m) for r in N4]), ms)
            for e in ('E1', 'E2p'):
                b[f'{e}_{m}'] = ci(pt('N4', e, m), D[f'N4|{e}'][m])
            b[f'E2p_minus_E1_{m}'] = ci(pt('N4', 'E2p', m) - pt('N4', 'E1', m), D['N4|E2p'][m] - D['N4|E1'][m])
        if ds == 'dev':
            for m in ('AUC', 'pAUC', 'mAUC'):
                for e in ('E1', 'E2p'):
                    b[f'frac_{e}_{m}_above_paper'] = float((D[f'N4|{e}'][m] > PAPER['dev'][m]).mean())
        else:
            for e in ('E1', 'E2p'):
                b[f'frac_{e}_mean_above_paper'] = float((D[f'N4|{e}']['mean_AUC_pAUC'] > PAPER['eval_mean']).mean())
        boot[ds] = b
    json.dump(boot, open(os.path.join(SCORES, 'bootstrap.json'), 'w'), indent=2)

    # ---- scorecard ----------------------------------------------------------------------
    O = lambda s, rule, ds, k='AUC': results[(s, rule, ds)]['overall'][k]   # noqa: E731
    Mc = lambda s, rule, ds, m: 100 * results[(s, rule, ds)][m]['AUC']      # noqa: E731
    fmt = lambda xs: ' '.join(f'{x:+.2f}' for x in xs)                      # noqa: E731
    L = []
    p = L.append

    p('P0  max |per-ID AUC - eval.py CSV| (dev, eval) <= 5e-4, |valid loss replayed - curve.csv| <= 1e-4:')
    p0 = True
    for r, e in repro.items():
        ok = max(e['dev_max_abs_id_auc_diff'], e['eval_max_abs_id_auc_diff']) <= 5e-4 and e['valid_loss_abs_diff'] <= 1e-4
        p0 &= ok
        p(f'    {short(r):22s} {e["dev_max_abs_id_auc_diff"]:.2e} {e["eval_max_abs_id_auc_diff"]:.2e}  '
          f'{e["valid_loss_replayed"]:.5f} vs {e["valid_loss_curve"]:.5f} (|d| {e["valid_loss_abs_diff"]:.1e})  '
          f'n_valid {e["n_valid"]}  {"ok" if ok else "FAIL"}')
    p(f'P0  holds: {p0}')

    for ds in ev:
        n4 = [O(r, 'S0', ds) for r in N4]
        o4 = [O(r, 'S0', ds) for r in O4]
        ok = abs(np.mean(n4) - O4_S0[ds]) <= 0.4 and (ds != 'dev' or 0.1 <= np.std(n4, ddof=1) <= 0.5)
        p(f'P1  {ds}: S0 AUC N4 {" ".join(f"{x:.2f}" for x in n4)} mean {np.mean(n4):.2f} sd {np.std(n4, ddof=1):.2f} '
          f'| O4 mean {np.mean(o4):.2f} sd {np.std(o4, ddof=1):.2f} | holds: {ok}')

    d1 = {}
    for ds in ev:
        dl = [O(r, 'S1p', ds) - O(r, 'S0', ds) for r in N4]
        lo, hi = (0.3, 1.0) if ds == 'eval' else (0.25, 0.75)
        bm = boot[ds]['mean_S1p_minus_S0_AUC']
        d1[ds] = dict(pos=sum(x > 0 for x in dl), mean=np.mean(dl), lo=bm[1])
        dm = [O(r, 'S1p', ds, 'mAUC') - O(r, 'S0', ds, 'mAUC') for r in N4]
        p(f'P2  {ds}: S1p-S0 AUC {fmt(dl)} | positive {d1[ds]["pos"]}/4 | mean {np.mean(dl):+.2f} '
          f'[{bm[1]:+.2f}, {bm[2]:+.2f}] (predicted {lo:+.2f}..{hi:+.2f}: {lo <= np.mean(dl) <= hi}) '
          f'| mAUC mean {np.mean(dm):+.2f}')

    for ds in ev:
        cells = {m: [Mc(r, 'S1p', ds, m) - Mc(r, 'S0', ds, m) for r in N4] for m in MACH}
        p(f'P3  {ds}: S1p-S0 per machine: ' + ' | '.join(f'{m} {fmt(v)}' for m, v in cells.items()))
        tc = all(sum(x > 0 for x in cells[m]) >= 3 for m in ('ToyConveyor', 'ToyCar'))
        extra = (f'valve falls on {sum(x < 0 for x in cells["valve"])}/4 (>=3: {sum(x < 0 for x in cells["valve"]) >= 3})'
                 if ds == 'eval' else f'pump |d| < 0.5 on {sum(abs(x) < 0.5 for x in cells["pump"])}/4')
        p(f'P3  {ds}: ToyConveyor and ToyCar positive on >=3/4: {tc} | {extra}')

    for rule in ('S0', 'S1p'):
        em = [O(r, rule, 'eval', 'mean_AUC_pAUC') for r in N4]
        da = [O(r, rule, 'dev') for r in N4]
        dp = [O(r, rule, 'dev', 'pAUC') for r in N4]
        p(f'P4  {rule}: eval mean {" ".join(f"{x:.2f}" for x in em)} >=93.22 on {sum(x >= 93.22 for x in em)}/4 '
          f'| dev AUC >=95.71 on {sum(x >= 95.71 for x in da)}/4 | dev pAUC >=90.23 on {sum(x >= 90.23 for x in dp)}/4')

    for ds in ev:
        mem = [O(r, 'S0', ds) for r in N4]
        e1 = O('N4', 'E1', ds)
        lo, hi = (0.6, 1.2) if ds == 'dev' else (0.3, 0.9)
        p(f'P5  {ds}: E1[N4] {e1:.2f} | members {" ".join(f"{x:.2f}" for x in mem)} | beats best member: {e1 > max(mem)} '
          f'| gain over mean {e1 - np.mean(mem):+.2f} (predicted {lo:+.1f}..{hi:+.1f})')
    for e in ('E1', 'E2p'):
        r_ = results[('N4', e, 'dev')]['overall']
        em = O('N4', e, 'eval', 'mean_AUC_pAUC')
        p(f'P5  {e}[N4] vs paper: dev AUC {r_["AUC"]:.2f} ({r_["AUC"] >= 95.71}) pAUC {r_["pAUC"]:.2f} ({r_["pAUC"] >= 90.23}) '
          f'mAUC {r_["mAUC"]:.2f} ({r_["mAUC"] >= 91.23}) | eval mean {em:.2f} ({em >= 93.22})')

    d3 = {}
    for ds in ev:
        x = O('N4', 'E2p', ds) - O('N4', 'E1', ds)
        lo, hi = (0.0, 0.4) if ds == 'dev' else (0.2, 0.8)
        b = boot[ds]['E2p_minus_E1_AUC']
        d3[ds] = x
        p(f'P6  {ds}: E2p-E1 [N4] {x:+.2f} [{b[1]:+.2f}, {b[2]:+.2f}] (predicted {lo:+.1f}..{hi:+.1f}: {lo <= x <= hi}) '
          f'| E1p-E1 {O("N4", "E1p", ds) - O("N4", "E1", ds):+.2f} | E0-E1 {O("N4", "E0", ds) - O("N4", "E1", ds):+.2f}')

    curve = {}
    with open(os.path.join(SCORES, 'ensemble_curve.csv')) as f:
        for row in csv.DictReader(f):
            curve.setdefault((row['rule'], row['dataset']), {}).setdefault(int(row['n']), []).append(float(row['AUC']))
    for ds in ev:
        e8 = O('relu8', 'E1', ds)
        p(f'P7  {ds}: E1[relu8] {e8:.2f} vs N4 {O("N4", "E1", ds):.2f}, O4 {O("O4", "E1", ds):.2f} '
          f'(>= both: {e8 >= max(O("N4", "E1", ds), O("O4", "E1", ds))}) | E2p[relu8] {O("relu8", "E2p", ds):.2f}')
        for rule in ('E1', 'E2p'):
            means = [np.mean(curve[(rule, ds)][n]) for n in range(1, 9)]
            p(f'P7  {ds}: {rule} size curve mean AUC n=1..8 {" ".join(f"{x:.2f}" for x in means)} '
              f'| non-decreasing: {all(b >= a for a, b in zip(means, means[1:]))}')

    # ---- decisions ---------------------------------------------------------------------
    D1 = (d1['eval']['pos'] == 4 and d1['dev']['pos'] >= 3 and d1['eval']['mean'] >= 0.2
          and d1['dev']['mean'] >= 0.2 and d1['eval']['lo'] > 0)
    p(f'D1  adopt S1p: {D1} (eval {d1["eval"]["pos"]}/4, dev {d1["dev"]["pos"]}/4, means '
      f'{d1["eval"]["mean"]:+.2f} / {d1["dev"]["mean"]:+.2f}, eval CI low {d1["eval"]["lo"]:+.2f})')
    claims = [('dev', 'AUC', PAPER['dev']['AUC']), ('dev', 'pAUC', PAPER['dev']['pAUC']),
              ('dev', 'mAUC', PAPER['dev']['mAUC']), ('eval', 'mean_AUC_pAUC', PAPER['eval_mean'])]
    for ds, k, ref in claims:
        b = boot[ds][f'mean_S1p_{k}']
        seeds = sum(O(r, 'S1p', ds, k) > ref for r in N4)
        p(f'D2  {ds} {k}: mean S1p {b[0]:.2f} [{b[1]:.2f}, {b[2]:.2f}] vs paper {ref} | above: {D1 and b[0] > ref} '
          f'| clearly: {D1 and b[1] > ref} | seeds above {seeds}/4')
    ens = 'E2p' if d3['dev'] > 0 and d3['eval'] > 0 else 'E1'
    p(f'D3  reported ensemble: {ens} (E2p-E1 dev {d3["dev"]:+.2f}, eval {d3["eval"]:+.2f})')
    for ds, k, ref in claims:
        b = boot[ds][f'{ens}_{k}']
        p(f'D3  {ens}[N4] {ds} {k}: {b[0]:.2f} [{b[1]:.2f}, {b[2]:.2f}] vs paper {ref} | above: {b[0] > ref} '
          f'| clearly: {b[1] > ref} | O4 E1 {O("O4", "E1", ds, k):.2f}')
    for ds in ev:
        p(f'D4  relu8 (half seen) {ds}: ' + ' | '.join(
            f'{e} AUC {O("relu8", e, ds):.2f} pAUC {O("relu8", e, ds, "pAUC"):.2f} mAUC {O("relu8", e, ds, "mAUC"):.2f} '
            f'mean {O("relu8", e, ds, "mean_AUC_pAUC"):.2f}' for e in ('E1', 'E2p')))
    for ds, b in boot.items():
        p(f'BOOT {ds}: ' + json.dumps({k: (np.round(v, 2).tolist() if isinstance(v, list) else round(v, 3))
                                       for k, v in b.items()}))
    txt = '\n'.join(L)
    print(txt)
    open(os.path.join(SCORES, 'scorecard.txt'), 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
