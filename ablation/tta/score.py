#!/usr/bin/env python
"""Score test-time shift averaging (TTA) against no TTA, one split per call (README).

    python ablation/tta/score.py dev      # stage A: dev only; eval clips are not even extracted yet
    python ablation/tta/score.py eval     # stage B: after the eval predictions are committed

Inputs: extract_tta.py dumps, <emb>/train/<run>.npz (the S1′ bank) and <emb>/<ds>/<run>.npz, with
the scheme's K shifts. Two versions of every checkpoint are built in extract.py's dump format and
scored by ../ensemble-backend/score.py's per_run_scores / Evaluator / resample, imported unchanged:
  T0  shift 0 only (= no TTA; must reproduce the committed dumps and eval.py)
  Tm  TTA, scheme `match` (primary): shifts 0, ±3%, ±5.5%, ±8% (K=7), the training shift range
  Ts  TTA, scheme `small` (secondary): shifts 0, ±1% (K=3)
  TTA: R0 = mean over the copies of the per-copy R0; embedding = mean over the copies of the
  L2-normalised pre-head embedding. Training clips (bank, z-stats) get the same treatment.
Rules S0 (= R0) and S1′ (S1p) per model; E2′ = sum of members' S1′.

Writes scores/<ds>_runs.csv (per run x version x rule), <ds>_systems.csv (groups, ensembles,
per machine), <ds>_perid.csv, <ds>_bootstrap.json, <ds>_check.json, scorecard_<ds>.txt.
"""
import csv
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('eb_score', os.path.join(HERE, '..', 'ensemble-backend', 'score.py'))
eb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eb)
sys.path.insert(0, os.path.join(HERE, '..'))
from scores_path import scores_csv  # noqa: E402

MACH, PAPER = eb.MACH, eb.PAPER
EMB = os.path.join(HERE, 'results', 'emb')
ORIG = {'ao_nogain': '/home/f74134118/OS-SCL-augops/ablation/aug-ops/results/emb',
        'la_aug': '/home/f74134118/OS-SCL-litarms/ablation/lit-arms/results/emb',
        'fr_f1': '/home/f74134118/OS-SCL-litarms/ablation/lit-arms/results/emb'}
N = 'SCLTFSTgramMFN_'
GROUPS = {'nogain': [f'{N}ao_nogain_s{s}' for s in range(2040, 2048)],
          'aug': [f'{N}la_aug_s{s}' for s in range(2040, 2048)],
          'plain': [f'{N}fr_f1_s{s}' for s in range(2040, 2044)]}
ENSEMBLES = {'G8': GROUPS['nogain'], 'A8': GROUPS['aug'], 'P4': GROUPS['plain']}
VERSIONS = {'T0': [0.0], 'Tm': [0.0, 0.03, -0.03, 0.055, -0.055, 0.08, -0.08], 'Ts': [0.0, 0.01, -0.01]}
KEYS = ('AUC', 'pAUC', 'mAUC', 'mean_AUC_pAUC')
RNG_SEED, N_BOOT = 0, 2000


def orig_path(r):
    for k, d in ORIG.items():
        if f'_{k}_' in r:
            return os.path.join(d, f'{r}.npz')


def finite(a, r):
    for k in ('ce', 'pre'):
        assert np.isfinite(a[k]).all() and np.abs(a[k]).max() < 1e4, f'{r}: bad {k}'


def versions(r, ds, meta_te_n):
    """{version: dump} in extract.py's format, test arrays over the full test list (NaN off-split)."""
    tr = dict(np.load(os.path.join(EMB, 'train', f'{r}.npz')))
    te = dict(np.load(os.path.join(EMB, ds, f'{r}.npz')))
    finite(tr, r), finite(te, r)
    assert tr['shifts'][0] == 0 and te['shifts'][0] == 0
    o = dict(np.load(orig_path(r)))
    out = {}
    # the train bank may hold more shifts than the test dump (stage B extracts only `small`);
    # columns are picked by shift value in each array separately
    sh = {id(tr): [round(float(x), 4) for x in tr['shifts']], id(te): [round(float(x), 4) for x in te['shifts']]}
    for v, keep in VERSIONS.items():
        if not all(round(x, 4) in s_ for s_ in sh.values() for x in keep):
            continue            # stage B extracts only the scheme that passed gate A

        def red(a):
            if v == 'T0':
                return a['ce'][:, 0], a['pre'][:, 0]
            cols = [sh[id(a)].index(round(x, 4)) for x in keep]
            p = a['pre'][:, cols] / np.linalg.norm(a['pre'][:, cols], axis=2, keepdims=True)
            return a['ce'][:, cols].mean(1), p.mean(1)
        trc, trp = red(tr)
        tc, tp = red(te)
        te_ce = np.full(meta_te_n, np.nan)
        te_pre = np.full((meta_te_n, trp.shape[1]), np.nan)
        te_ce[te['idx']], te_pre[te['idx']] = tc, tp
        out[v] = dict(tr_ce=trc, tr_pre=trp, te_ce=te_ce, te_pre=te_pre, valid=o['valid'],
                      valid_order=o['valid_order'])
    chk = dict(K=int(len(tr['shifts'])), shifts=[float(s) for s in tr['shifts']],
               max_abs_tr_ce_vs_orig=float(np.abs(tr['ce'][:, 0] - o['tr_ce']).max()),
               max_abs_te_ce_vs_orig=float(np.abs(te['ce'][:, 0] - o['te_ce'][te['idx']]).max()),
               max_abs_tr_pre_vs_orig=float(np.abs(tr['pre'][:, 0] - o['tr_pre']).max()))
    return out, chk


def sub_meta(meta, ds):
    """meta restricted to one test split, so per_run_scores never touches the other split's clips."""
    sel = np.where(meta['te_dataset'] == ds)[0]
    m = dict(meta)
    for k in [k for k in meta if k.startswith('te_')]:
        m[k] = meta[k][sel]
    return m, sel


def main():
    ds = sys.argv[1]
    assert ds in ('dev', 'eval')
    meta = dict(np.load(os.path.join(EMB, 'meta.npz')))
    om = dict(np.load(os.path.join(ORIG['ao_nogain'], 'meta.npz')))
    below = lambda a: [p.split('/data/', 1)[1] for p in a]   # noqa: E731
    for k in meta:
        same = below(meta[k]) == below(om[k]) if k.endswith('paths') else np.array_equal(meta[k], om[k])
        assert same, f'meta mismatch on {k}'
    runs = [r for g in GROUPS.values() for r in g
            if os.path.exists(os.path.join(EMB, ds, f'{r}.npz')) and os.path.exists(os.path.join(EMB, 'train', f'{r}.npz'))]
    smeta, sel = sub_meta(meta, ds)
    ev = eb.Evaluator(smeta, ds)
    S, check = {}, {}
    for r in runs:
        vs, chk = versions(r, ds, len(meta['te_label']))
        for v, a in vs.items():
            a = dict(a, te_ce=a['te_ce'][sel], te_pre=a['te_pre'][sel])
            sc = eb.per_run_scores(smeta, a)
            S[(r, v, 'S0')], S[(r, v, 'S1p')] = sc['S0'], sc['S1p']
        committed = {}
        with open(scores_csv(r, ds), encoding='UTF-8') as f:
            for row in csv.DictReader(f):
                if row['scope'] == 'id':
                    committed[(row['machine'], row['id'])] = float(row['AUC'])
        res = ev(S[(r, 'T0', 'S0')])
        chk['T0_S0_max_abs_id_auc_vs_evalpy'] = max(abs(res[m]['ids'][i] - committed[(m, i)])
                                                   for m in MACH for i in res[m]['ids'])
        # S1′ at T0 against ../aug-ops/scores/s1p.csv (committed), overall metrics
        with open(os.path.join(HERE, '..', 'aug-ops', 'scores', 's1p.csv'), encoding='UTF-8') as f:
            ref = [row for row in csv.DictReader(f) if row['run'] == r and row['rule'] == 'S1p' and row['dataset'] == ds]
        if ref:
            o = ev(S[(r, 'T0', 'S1p')])['overall']
            chk['T0_S1p_max_abs_overall_vs_augops'] = max(abs(o[k] - float(ref[0][k])) for k in KEYS)
        check[r] = chk
    out = os.path.join(HERE, 'scores')
    os.makedirs(out, exist_ok=True)
    json.dump(check, open(os.path.join(out, f'{ds}_check.json'), 'w'), indent=2)
    groups = {g: [r for r in m if r in runs] for g, m in GROUPS.items()}
    groups = {g: m for g, m in groups.items() if m}
    TV = [v for v in VERSIONS if v != 'T0']
    lines = [f'{ds.upper()} ONLY. T0 = no TTA; ' + '; '.join(f'{v} = mean over shifts {VERSIONS[v]}' for v in TV)
             + '. AUC/pAUC/mAUC x100; mean = mean(AUC, pAUC).',
             f'Paper: dev AUC {PAPER["dev"]["AUC"]} pAUC {PAPER["dev"]["pAUC"]} mAUC {PAPER["dev"]["mAUC"]}; '
             f'eval mean {PAPER["eval_mean"]}.',
             f'Reproduction: max |T0 R0 − original dump| train {max(c["max_abs_tr_ce_vs_orig"] for c in check.values()):.2e}, '
             f'test {max(c["max_abs_te_ce_vs_orig"] for c in check.values()):.2e}; max |T0 S0 − eval.py| per-ID AUC '
             f'{max(c["T0_S0_max_abs_id_auc_vs_evalpy"] for c in check.values()):.2e}', '']

    # ---- single models -------------------------------------------------------------------------
    R = {k: ev(v) for k, v in S.items()}
    with open(os.path.join(out, f'{ds}_runs.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'version', 'rule', 'dataset', 'scope'] + list(KEYS))
        for (r, v, rule), res in R.items():
            w.writerow([r, v, rule, ds, 'overall'] + [f'{res["overall"][k]:.4f}' for k in KEYS])
            for m in MACH:
                w.writerow([r, v, rule, ds, m] + [f'{100 * res[m][k]:.4f}' for k in ('AUC', 'pAUC', 'mAUC')] + [''])
    lines.append('Single models, group means (AUC pAUC mAUC mean); Δ = version − T0 '
                 '(seeds with Δ mean(AUC,pAUC) > 0)')
    for rule in ('S0', 'S1p'):
        for g, mem in groups.items():
            t0 = {k: np.mean([R[(r, 'T0', rule)]['overall'][k] for r in mem]) for k in KEYS}
            lines.append(f'{rule:4s} {g:7s} n={len(mem)}  T0 ' + ' '.join(f'{t0[k]:6.2f}' for k in KEYS))
            for v in [v for v in TV if all((r, v, rule) in R for r in mem)]:
                t = {k: np.mean([R[(r, v, rule)]['overall'][k] for r in mem]) for k in KEYS}
                pos = sum(R[(r, v, rule)]['overall']['mean_AUC_pAUC'] > R[(r, 'T0', rule)]['overall']['mean_AUC_pAUC']
                          for r in mem)
                lines.append(f'{"":19s}{v:2s} ' + ' '.join(f'{t[k]:6.2f}' for k in KEYS)
                             + '  | Δ ' + ' '.join(f'{t[k] - t0[k]:+.2f}' for k in KEYS) + f'  ({pos}/{len(mem)})'
                             + '  | Δ AUC by machine: ' + ' '.join(
                                 f'{m[:7]} {np.mean([100 * (R[(r, v, rule)][m]["AUC"] - R[(r, "T0", rule)][m]["AUC"]) for r in mem]):+.2f}'
                                 for m in MACH))

    # ---- ensembles (E2′) -----------------------------------------------------------------------
    ens = {f'{e}_{v}': sum(S[(r, v, 'S1p')] for r in mem) for e, mem in ENSEMBLES.items()
           if all(r in runs for r in mem) for v in VERSIONS if all((r, v, 'S1p') in S for r in mem)}
    ER = {k: ev(v) for k, v in ens.items()}
    with open(os.path.join(out, f'{ds}_systems.csv'), 'w', newline='') as f, \
            open(os.path.join(out, f'{ds}_perid.csv'), 'w', newline='') as g:
        w, wi = csv.writer(f), csv.writer(g)
        w.writerow(['system', 'dataset', 'scope', 'AUC', 'pAUC', 'mAUC', 'mean_AUC_pAUC'])
        wi.writerow(['system', 'dataset', 'machine', 'id', 'AUC'])
        for name, res in ER.items():
            w.writerow([name, ds, 'overall'] + [f'{res["overall"][k]:.4f}' for k in KEYS])
            for m in MACH:
                w.writerow([name, ds, m] + [f'{100 * res[m][k]:.4f}' for k in ('AUC', 'pAUC', 'mAUC')] + [''])
                for i, a in sorted(res[m]['ids'].items()):
                    wi.writerow([name, ds, m, i, f'{100 * a:.4f}'])
    lines += ['', 'E2′ ensembles (sum of members\' S1′): AUC pAUC mAUC mean | per-machine AUC']
    for name, res in ER.items():
        lines.append(f'{name:8s} ' + ' '.join(f'{res["overall"][k]:6.2f}' for k in KEYS) + ' | '
                     + ' '.join(f'{m[:7]} {100 * res[m]["AUC"]:6.2f}' for m in MACH))

    # ---- paired clip-bootstrap: version − T0 ----------------------------------------------------
    rng = np.random.default_rng(RNG_SEED)
    comps = {f'{e} E2′: {v}−T0': (f'{e}_{v}', f'{e}_T0') for e in ENSEMBLES for v in TV if f'{e}_{v}' in ens}
    for v in TV:
        if f'G8_{v}' in ens and 'A8_T0' in ens:
            comps[f'G8_{v} − A8_T0'] = (f'G8_{v}', 'A8_T0')
    single = {f'mean single {g} {rule}: {v}−T0': (g, rule, v) for g in ('nogain', 'aug') if g in groups
              for rule in ('S0', 'S1p') for v in TV if all((r, v, rule) in S for r in groups[g])}
    diffs = {n: {k: [] for k in KEYS} for n in list(comps) + list(single)}
    for _ in range(N_BOOT):
        idx = ev.resample(rng)
        cache = {}

        def get(key, vec):
            if key not in cache:
                cache[key] = ev(vec, idx)['overall']
            return cache[key]
        for n, (a, b) in comps.items():
            oa, ob = get(a, ens[a]), get(b, ens[b])
            for k in KEYS:
                diffs[n][k].append(oa[k] - ob[k])
        for n, (g, rule, v) in single.items():
            d = {k: [] for k in KEYS}
            for r in groups[g]:
                oa, ob = get((r, v, rule), S[(r, v, rule)]), get((r, 'T0', rule), S[(r, 'T0', rule)])
                for k in KEYS:
                    d[k].append(oa[k] - ob[k])
            for k in KEYS:
                diffs[n][k].append(np.mean(d[k]))
    boot = {n: {k: [float(np.mean(v)), *map(float, np.percentile(v, [2.5, 97.5]))] for k, v in d.items()}
            for n, d in diffs.items()}
    json.dump(boot, open(os.path.join(out, f'{ds}_bootstrap.json'), 'w'), indent=2)
    lines += ['', f'Paired clip-bootstrap ({N_BOOT}, seed {RNG_SEED}): mean [2.5%, 97.5%]']
    for n, d in boot.items():
        lines.append(f'{n:30s} ' + '  '.join(f'{k[:4]} {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]' for k, v in d.items()))
    txt = '\n'.join(lines)
    open(os.path.join(out, f'scorecard_{ds}.txt'), 'w').write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
