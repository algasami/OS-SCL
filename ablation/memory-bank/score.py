#!/usr/bin/env python
"""Score the patch-level memory banks under the pre-registered rules; write scores/.

    python ablation/memory-bank/score.py ablation/memory-bank/results/full \
        /home/f74134118/OS-SCL-scoring-probe/ablation/ensemble-backend/results/emb

Rules (README, *Statistics and fused rules*), per checkpoint m and machine ID i:
  P_a  = zP_a                    clip score = max over patches of the NN distance, z-scored
                                 with mean/sd over 200 leave-one-out training-part clips
  F_a  = zR0' + zP_a             patch-level fusion (zR0' as in ../ensemble-backend)
  S1'  = zR0' + zK'              clip-level fusion, the reference (../ensemble-backend's S1p)
  E_a  = sum over relu4 of F_a   reported only
Sensitivity keys (D3): mfn_l2, mfn_l3, mfn_cs10, and <arm>_top10 (top-10-patch mean).

Writes: rule_metrics.csv, summary.csv, bootstrap.json, reproduction.json, scorecard.txt.
"""
import csv
import glob
import importlib.util
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
_spec = importlib.util.spec_from_file_location('eb_score', os.path.join(HERE, '..', 'ensemble-backend', 'score.py'))
eb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eb)
MACH, RELU, ALL, Evaluator, per_run_scores, N_BOOT = eb.MACH, eb.RELU, eb.ALL, eb.Evaluator, eb.per_run_scores, eb.N_BOOT

SCORES = os.environ.get('MB_SCORES', os.path.join(HERE, 'scores'))   # override only for dry runs
MODEL_ARMS = ['mfn', 'tfgram', 'imagenet']
SENS = ['mfn_l2', 'mfn_l3', 'mfn_cs10']
SEED_FREE = 'imagenet_mel'
EB_RULES = os.path.join(HERE, '..', 'ensemble-backend', 'scores', 'rule_metrics.csv')


def aggregate(d, how):
    d = d.astype(np.float64)
    return d.max(1) if how == 'max' else np.sort(d, 1)[:, -10:].mean(1)


def patch_z(res_dir, arm, run, n_te, how):
    """zP over every test clip for one arm and checkpoint, or None if any ID is missing."""
    files = sorted(glob.glob(os.path.join(res_dir, arm, run, 'id*.npz')))
    if len(files) != 41:
        return None
    out = np.full(n_te, np.nan)
    for f in files:
        a = np.load(f)
        st = aggregate(a['stats'], how)
        out[a['test_te_index']] = (aggregate(a['test'], how) - st.mean()) / st.std()
    assert not np.isnan(out).any(), f'{arm}/{run}: test clips missing'
    return out


def fast_auc(s, y, mach):
    """Mean over machines of the pooled AUC (Mann-Whitney on ranks); for the bootstrap."""
    out = []
    for m in MACH:
        k = mach == m
        r, yy = rankdata(s[k]), y[k]
        n1 = yy.sum()
        n0 = len(yy) - n1
        out.append((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))
    return 100 * np.mean(out)


def main():
    res_dir, emb = sys.argv[1], sys.argv[2]
    os.makedirs(SCORES, exist_ok=True)
    meta = dict(np.load(os.path.join(emb, 'meta.npz')))
    n_te = len(meta['te_label'])
    S = {r: per_run_scores(meta, dict(np.load(os.path.join(emb, f'{r}.npz')))) for r in ALL}
    ev = {ds: Evaluator(meta, ds) for ds in ('dev', 'eval')}

    # ---- systems: every (checkpoint, rule) score vector ----------------------------------
    systems = {}
    for r in ALL:
        systems[(r, 'S0')] = S[r]['S0']
        systems[(r, 'S1p')] = S[r]['S1p']
        systems[(r, 'Kp')] = S[r]['zKp']
    keys = []
    mel = {how: patch_z(res_dir, SEED_FREE, 'none', n_te, how) for how in ('max', 'top10')}
    for arm in MODEL_ARMS + SENS + [SEED_FREE]:
        for how in ('max', 'top10'):
            key = arm if how == 'max' else f'{arm}_top10'
            for r in ALL:
                zp = mel[how] if arm == SEED_FREE else patch_z(res_dir, arm, r, n_te, how)
                if zp is None:
                    continue
                systems[(r, f'P_{key}')] = zp
                systems[(r, f'F_{key}')] = S[r]['zR0p'] + zp
            if all((r, f'F_{key}') in systems for r in ALL):
                keys.append(key)
                systems[('relu4', f'E_{key}')] = sum(systems[(r, f'F_{key}')] for r in RELU)
    systems[('relu4', 'E_S1p')] = sum(S[r]['S1p'] for r in RELU)
    print('complete keys:', keys)
    results = {(s, rule, ds): ev[ds](v) for (s, rule), v in systems.items() for ds in ev}
    O = lambda s, rule, ds, k='AUC': results[(s, rule, ds)]['overall'][k]
    M = lambda s, rule, ds, m: 100 * results[(s, rule, ds)][m]['AUC']

    with open(os.path.join(SCORES, 'rule_metrics.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['system', 'dataset', 'rule', 'machine', 'id', 'scope', 'AUC', 'pAUC', 'mAUC'])
        for (s, rule, ds), res in results.items():
            for m in MACH:
                for i, a in res[m]['ids'].items():
                    w.writerow([s, ds, rule, m, i, 'id', f'{a:.5f}', '', ''])
                w.writerow([s, ds, rule, m, 'ALL', 'machine'] + [f'{res[m][k]:.5f}' for k in ('AUC', 'pAUC', 'mAUC')])
            o = res['overall']
            w.writerow([s, ds, rule, 'ALL', 'ALL', 'overall'] + [f'{o[k] / 100:.5f}' for k in ('AUC', 'pAUC', 'mAUC')])
    with open(os.path.join(SCORES, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['system', 'dataset', 'rule', 'AUC', 'pAUC', 'mAUC', 'mean_AUC_pAUC'] + MACH)
        for (s, rule, ds), res in results.items():
            o = res['overall']
            w.writerow([s, ds, rule] + [f'{o[k]:.2f}' for k in ('AUC', 'pAUC', 'mAUC', 'mean_AUC_pAUC')]
                       + [f'{100 * res[m]["AUC"]:.2f}' for m in MACH])

    # ---- P0: the clip-level reference reproduces ../ensemble-backend --------------------
    committed = {}
    with open(EB_RULES, encoding='UTF-8') as f:
        for row in csv.DictReader(f):
            if row['rule'] == 'S1p' and row['scope'] == 'overall':
                committed[(row['system'], row['dataset'])] = float(row['AUC'])
    repro = {f'{r}|{ds}': abs(O(r, 'S1p', ds) / 100 - committed[(r, ds)]) for r in ALL for ds in ev}
    ce_logs = {}
    for p in glob.glob(os.path.join(res_dir, 'log_*.json')):
        for k, v in json.load(open(p)).items():
            if k.startswith('ce_check'):
                ce_logs[k] = v['max_rel_err']
    json.dump({'S1p_overall_auc_abs_diff': repro, 'bank_R0_max_rel_err': ce_logs},
              open(os.path.join(SCORES, 'reproduction.json'), 'w'), indent=1)

    # ---- bootstrap: mean over checkpoints of F_a - S1', and E_mfn - E_S1' ----------------
    rng = np.random.default_rng(0)
    boot = {}
    fused = [k for k in keys if k in MODEL_ARMS + [SEED_FREE]]
    for ds in ev:
        e = ev[ds]
        y, mach = e.y, e.mach
        get = lambda v, idx: v[e.sel][idx]
        draws = {k: [] for k in fused + ['E_mfn']}
        for _ in range(N_BOOT):
            idx = e.resample(rng)
            yy, mm = y[idx], mach[idx]
            ref = {r: fast_auc(get(systems[(r, 'S1p')], idx), yy, mm) for r in ALL}
            for k in fused:
                draws[k].append(np.mean([fast_auc(get(systems[(r, f'F_{k}')], idx), yy, mm) - ref[r] for r in ALL]))
            if 'mfn' in keys:
                draws['E_mfn'].append(fast_auc(get(systems[('relu4', 'E_mfn')], idx), yy, mm)
                                      - fast_auc(get(systems[('relu4', 'E_S1p')], idx), yy, mm))
        b = {}
        for k in fused:
            pt = np.mean([O(r, f'F_{k}', ds) - O(r, 'S1p', ds) for r in ALL])
            b[f'F_{k}_minus_S1p_AUC'] = [pt] + list(np.percentile(draws[k], [2.5, 97.5]))
        if draws['E_mfn']:
            pt = O('relu4', 'E_mfn', ds) - O('relu4', 'E_S1p', ds)
            b['E_mfn_minus_E_S1p_AUC'] = [pt] + list(np.percentile(draws['E_mfn'], [2.5, 97.5]))
        boot[ds] = b
    json.dump(boot, open(os.path.join(SCORES, 'bootstrap.json'), 'w'), indent=2)

    # ---- scorecard ------------------------------------------------------------------------
    L = []
    p = L.append
    worst = max(repro.values())
    p(f'P0  S1\' overall AUC vs ../ensemble-backend: max |diff| {worst:.1e} (<=1e-3: {worst <= 1e-3}); '
      f'bank.py R0 vs cached, max rel err {max(ce_logs.values()) if ce_logs else float("nan"):.1e}')
    d = lambda a, b_, ds, runs=ALL: [O(r, a, ds) - O(r, b_, ds) for r in runs]
    fmt = lambda xs: ' '.join(f'{x:+.2f}' for x in xs)
    if 'mfn' in keys:
        x = d('P_mfn', 'Kp', 'dev')
        p(f'P1  dev P_mfn - zK\' per ckpt {fmt(x)} | positive {sum(v > 0 for v in x)}/8 (>=6: {sum(v > 0 for v in x) >= 6})')
        xd, xe = d('F_mfn', 'S1p', 'dev'), d('F_mfn', 'S1p', 'eval')
        p2 = np.mean(xd) >= 0.2 and sum(v > 0 for v in xe) >= 6
        p(f'P2  dev  F_mfn - S1\' {fmt(xd)} | mean {np.mean(xd):+.2f} (>=+0.2: {np.mean(xd) >= 0.2})')
        p(f'P2  eval F_mfn - S1\' {fmt(xe)} | positive {sum(v > 0 for v in xe)}/8 (>=6: {sum(v > 0 for v in xe) >= 6}) '
          f'| mean {np.mean(xe):+.2f}  => P2 {p2}')
        pump = [M(r, 'F_mfn', 'dev', 'pump') - M(r, 'S1p', 'dev', 'pump') for r in ALL]
        p(f'P3  dev pump F_mfn - S1\' {fmt(pump)} | mean {np.mean(pump):+.2f} (>=+0.5: {np.mean(pump) >= 0.5})')
        for ds in ev:
            p(f'    {ds} per machine, mean F_mfn - S1\': ' + ' '.join(
                f'{m} {np.mean([M(r, "F_mfn", ds, m) - M(r, "S1p", ds, m) for r in ALL]):+.2f}' for m in MACH))
    if SEED_FREE in keys:
        s0 = np.mean([O(r, 'S0', 'dev') for r in ALL])
        pm = O(ALL[0], f'P_{SEED_FREE}', 'dev')
        p(f'P4a dev P_imagenet_mel {pm:.2f} vs mean R0 {s0:.2f}: diff {pm - s0:+.2f} (<=-2: {pm - s0 <= -2})')
        x = d(f'F_{SEED_FREE}', 'S1p', 'dev')
        p(f'P4b dev F_imagenet_mel - S1\' {fmt(x)} | mean {np.mean(x):+.2f} (does not beat, mean<=0: {np.mean(x) <= 0})')
        if 'imagenet' in keys:
            pi = np.mean([O(r, 'P_imagenet', 'dev') for r in ALL])
            p(f'P4c dev mean P_imagenet {pi:.2f} vs P_imagenet_mel {pm:.2f}: |diff| {abs(pi - pm):.2f} (<0.5: {abs(pi - pm) < 0.5})')
    if 'tfgram' in keys and 'mfn' in keys:
        x = d('P_tfgram', 'P_mfn', 'dev')
        p(f'P5  dev P_tfgram - P_mfn {fmt(x)} | negative {sum(v < 0 for v in x)}/8 (>=6: {sum(v < 0 for v in x) >= 6})')
    if 'mfn' in keys:
        for ds in ev:
            v = O('relu4', 'E_mfn', ds) - O('relu4', 'E_S1p', ds)
            p(f'P6  {ds} E_mfn - E_S1\' [relu4] {v:+.2f} (in [-0.1,+0.2]: {-0.1 <= v <= 0.2})')
        p(f'D1  adopt F_mfn as the fusion rule for the next new checkpoints: {p2}')
    cands = [k for k in keys if k in ('tfgram', 'imagenet', SEED_FREE) and np.mean(d(f'F_{k}', 'S1p', 'dev')) >= 0.2]
    p(f'D2  other arms with mean dev F - S1\' >= +0.2 (candidates, not adopted): {cands or "none"}')
    p('D3  sensitivity (never replaces the primary), mean F - S1\' AUC dev / eval:')
    for k in keys:
        p(f'    {k:22s} {np.mean(d(f"F_{k}", "S1p", "dev")):+.2f} / {np.mean(d(f"F_{k}", "S1p", "eval")):+.2f}'
          f'   standalone P dev {np.mean([O(r, f"P_{k}", "dev") for r in ALL]):.2f}')
    p(f'    reference: mean R0 dev {np.mean([O(r, "S0", "dev") for r in ALL]):.2f} / eval '
      f'{np.mean([O(r, "S0", "eval") for r in ALL]):.2f}; mean S1\' dev {np.mean([O(r, "S1p", "dev") for r in ALL]):.2f}'
      f' / eval {np.mean([O(r, "S1p", "eval") for r in ALL]):.2f}')
    for ds, b in boot.items():
        p(f'BOOT {ds}: ' + json.dumps({k: np.round(v, 2).tolist() for k, v in b.items()}))
    txt = '\n'.join(L)
    print(txt)
    open(os.path.join(SCORES, 'scorecard.txt'), 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
