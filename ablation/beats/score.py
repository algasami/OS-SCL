#!/usr/bin/env python
"""Score the BEATs runs, alone and fused with E2′[A8]; write scores/ (README *Pre-registration*).

    python ablation/beats/score.py dev  ablation/beats/results/emb    # stage A: dev only, picks rule + fusion
    python ablation/beats/score.py eval ablation/beats/results/emb    # stage B: after stage A is committed

Env: BT_RUNS (space-separated BEATs run names; default the three pre-registered seeds),
BT_SCORES (output dir; default scores/ -- the smoke test writes to results/).

ALL RESULTS HERE ARE "WITH EXTERNAL PRETRAINING" (BEATs_iter3, AudioSet-2M, self-supervised).

Single-model rules per BEATs run and machine ID; bank = the ID's training-part clips (the
replayed 95%), 1-NN cosine distances on L2-normalised vectors, z over the bank scored
leave-one-out (the primed convention of ../ensemble-backend and ../pooled-knn):
  S0   R0 (eval.py's score)                      K    raw 1-NN distance on `pre` (AnoPatch's score)
  zKp  z(K)                                      S1p  zR0' + zK'  (../ensemble-backend per_run_scores)
  C1b  S1p + z(1-NN distance on gmean)          (pooled-knn's C1 with BEATs' token mean as Gmean)
Stage A picks the single rule among CANDS by 3-seed mean dev Amean.

Fusion (every member scored S1′, as E2′): E2′[A8] = Σ_A8 S1p (the bar), and
  F1 = E2′[A8] + Σ_B S1p          (plain E2′ over the 11 members)
  F2 = E2′[A8] + (8/|B|) Σ_B S1p  (block-balanced: BEATs block weighs as much as A8)
Stage A picks F1 or F2 by dev Amean; stage B reports eval and the decision rules once.
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
MACH = eb.MACH

A8_EMB = '/home/f74134118/OS-SCL-litarms/ablation/lit-arms/results/emb'
A8 = [f'SCLTFSTgramMFN_la_aug_s{s}' for s in range(2040, 2048)]
BT = os.environ.get('BT_RUNS', ' '.join(f'BEATsASD_bt_arc_s{s}' for s in (2040, 2041, 2042))).split()
SCORES = os.environ.get('BT_SCORES', os.path.join(HERE, 'scores'))
PKNN = os.path.join(HERE, '..', 'pooled-knn', 'scores', 'confirm_metrics.csv')
A8_COMMITTED = os.path.join(HERE, '..', 'nogain-ensemble', 'scores', 'systems_{}.csv')
CANDS = ('K', 'zKp', 'S1p', 'C1b')
FUSIONS = ('F1', 'F2')
C1_EVAL8, C1_DEV8 = 94.09, 94.16      # pooled-knn scorecard_confirm: C1 8-seed mean Amean
N_BOOT = 2000
KEYS = ('AUC', 'pAUC', 'mAUC', 'Amean')


def l2(x):
    x = x.astype(np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def nn1(q, ref, loo=False):
    s = q @ ref.T
    if loo:
        np.fill_diagonal(s, -np.inf)
    return 1.0 - s.max(1)


def load_npz(path, keys):
    a = dict(np.load(path))
    for k in keys:      # faulty host RAM: range-check on load
        v = a[k].astype(np.float32)
        assert np.isfinite(v).all() and np.abs(v).max() < (1e3 if k.endswith('ce') else 1e4), (path, k)
    return a


def beats_rules(meta, a):
    """{rule: score over every test clip} for one BEATs run."""
    out = eb.per_run_scores(meta, a)          # S0, S1p, zR0p, zKp (unchanged rule code)
    trl, tel = meta['tr_label'], meta['te_label']
    tp, ep = l2(a['tr_pre']), l2(a['te_pre'])
    tg, eg = l2(a['tr_gmean']), l2(a['te_gmean'])
    K, G = np.full(len(tel), np.nan), np.full(len(tel), np.nan)
    for c in np.unique(tel):
        ref, te = (trl == c) & ~a['valid'], tel == c
        K[te] = nn1(ep[te], tp[ref])
        g_ref = nn1(tg[ref], tg[ref], loo=True)
        G[te] = (nn1(eg[te], tg[ref]) - g_ref.mean()) / g_ref.std()
    out['K'], out['C1b'] = K, out['S1p'] + G
    for k in ('S0', 'K', 'zKp', 'S1p', 'C1b'):
        assert np.isfinite(out[k]).all(), k
    return out


def flat(res):
    o = res['overall']
    d = {'AUC': o['AUC'], 'pAUC': o['pAUC'], 'mAUC': o['mAUC'], 'Amean': o['mean_AUC_pAUC']}
    for m in MACH:
        d[m] = 100 * (res[m]['AUC'] + res[m]['pAUC']) / 2
    return d


def load_all(emb):
    meta = dict(np.load(os.path.join(emb, 'meta.npz')))
    old = dict(np.load(os.path.join(A8_EMB, 'meta.npz')))
    below = lambda a: [p.split('/data/', 1)[1] for p in a]   # noqa: E731
    for k in meta:
        same = below(meta[k]) == below(old[k]) if k.endswith('paths') else np.array_equal(meta[k], old[k])
        assert same, f'meta.npz disagrees with lit-arms on {k}'
    keys_b = ('tr_ce', 'tr_pre', 'tr_gmean', 'te_ce', 'te_pre', 'te_gmean')
    B = {r: beats_rules(meta, load_npz(os.path.join(emb, f'{r}.npz'), keys_b)) for r in BT}
    A = {r: eb.per_run_scores(meta, load_npz(os.path.join(A8_EMB, f'{r}.npz'), ('tr_ce', 'tr_pre', 'te_ce', 'te_pre')))
         for r in A8}
    e2 = sum(A[r]['S1p'] for r in A8)
    bsum = sum(B[r]['S1p'] for r in BT)
    sys_ = {'A8': e2, 'F1': e2 + bsum, 'F2': e2 + (len(A8) / len(BT)) * bsum}
    for r in BT:      # descriptive: one BEATs seed added, at F1 / F2 weight
        sys_[f'F1[{r[-5:]}]'] = e2 + B[r]['S1p']
        sys_[f'F2[{r[-5:]}]'] = e2 + len(A8) * B[r]['S1p']
    return meta, B, sys_


def write_rows(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def eval_csv(run, ds, res, ev, score, rule=None):
    """eval.py's --csv layout (plus a leading rule column when `rule`); per-ID pAUC as eval.py."""
    from sklearn import metrics
    s = score[ev.sel]
    rows = []
    for m in MACH:
        for i in sorted(res[m]['ids']):
            k = (ev.mach == m) & (ev.id == i)
            pa = metrics.roc_auc_score(ev.y[k], s[k], max_fpr=0.1)
            rows.append([run, ds, m, i, 'id', f'{res[m]["ids"][i]:.5f}', f'{pa:.5f}', ''])
        rows.append([run, ds, m, 'ALL', 'machine'] + [f'{res[m][k]:.5f}' for k in ('AUC', 'pAUC', 'mAUC')])
    o = res['overall']
    rows.append([run, ds, 'ALL', 'ALL', 'overall'] + [f'{o[k] / 100:.5f}' for k in ('AUC', 'pAUC', 'mAUC')])
    header = ['model', 'dataset', 'machine', 'id', 'scope', 'AUC', 'pAUC', 'mAUC']
    if rule:
        rows, header = [[rule] + r for r in rows], ['rule'] + header
    return rows, header


def boot(ev, a, b, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    d = []
    for _ in range(N_BOOT):
        idx = ev.resample(rng)
        d.append(ev(a, idx)['overall']['mean_AUC_pAUC'] - ev(b, idx)['overall']['mean_AUC_pAUC'])
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def c1_seed(ds):
    """Our single-model rule C1 per ao_nogain seed (pooled-knn's committed record)."""
    out = {}
    with open(PKNN) as f:
        for r in csv.DictReader(f):
            if r['rule'] == 'C1' and r['dataset'] == ds and r['machine'] == 'overall':
                out[int(r['system'][-4:])] = float(r['Amean'])
    return out


def stage(ds, emb):
    meta, B, S = load_all(emb)
    ev = eb.Evaluator(meta, ds)
    L = ['ALL NUMBERS: WITH EXTERNAL PRETRAINING (BEATs_iter3, AudioSet)', f'runs: {" ".join(BT)}', '']
    p = L.append

    # P0: E2′[A8] reproduces the committed number
    with open(A8_COMMITTED.format(ds)) as f:
        com = [float(r['mean_AUC_pAUC']) for r in csv.DictReader(f) if r['system'] == 'A8' and r['scope'] == 'overall'][0]
    a8 = flat(ev(S['A8']))
    p(f'P0  E2′[A8] {ds} Amean {a8["Amean"]:.4f} vs committed {com:.4f} (|d| {abs(a8["Amean"] - com):.1e})')
    assert abs(a8['Amean'] - com) < 5e-3

    # single models
    rows, single = [], {}
    for r in BT:
        per_rule = []
        for rule in ('S0', 'K', 'zKp', 'S1p', 'C1b'):
            res = ev(B[r][rule])
            single[(r, rule)] = flat(res)
            rows.append([r, ds, rule] + [f'{single[(r, rule)][k]:.4f}' for k in KEYS + tuple(MACH)])
            rr, hh = eval_csv(r, ds, res, ev, B[r][rule], rule)
            per_rule += rr
            if rule == 'S0':    # eval.py's R0, in eval.py's --csv layout
                write_rows(os.path.join(SCORES, r, f'eval_{ds}.csv'), *eval_csv(r, ds, res, ev, B[r][rule]))
        write_rows(os.path.join(SCORES, r, f'rules_{ds}.csv'), per_rule, hh)
    for rule in ('S0', 'K', 'zKp', 'S1p', 'C1b'):
        mean = {k: np.mean([single[(r, rule)][k] for r in BT]) for k in KEYS + tuple(MACH)}
        rows.append(['mean', ds, rule] + [f'{mean[k]:.4f}' for k in KEYS + tuple(MACH)])
        p(f'single {rule:4s} mean: AUC {mean["AUC"]:.2f} pAUC {mean["pAUC"]:.2f} mAUC {mean["mAUC"]:.2f} '
          f'Amean {mean["Amean"]:.2f} | per seed ' + ' '.join(f'{single[(r, rule)]["Amean"]:.2f}' for r in BT)
          + ' | ' + ' '.join(f'{m} {mean[m]:.1f}' for m in MACH))
    write_rows(os.path.join(SCORES, f'single_{ds}.csv'), rows, ['run', 'dataset', 'rule'] + list(KEYS) + MACH)

    # fused systems
    rows, fused = [], {}
    for name, sc in S.items():
        fused[name] = flat(ev(sc))
        rows.append([name, ds] + [f'{fused[name][k]:.4f}' for k in KEYS + tuple(MACH)])
        p(f'system {name:16s} AUC {fused[name]["AUC"]:.2f} pAUC {fused[name]["pAUC"]:.2f} mAUC {fused[name]["mAUC"]:.2f} '
          f'Amean {fused[name]["Amean"]:.2f} (Δ vs A8 {fused[name]["Amean"] - a8["Amean"]:+.2f})')
    write_rows(os.path.join(SCORES, f'systems_{ds}.csv'), rows, ['system', 'dataset'] + list(KEYS) + MACH)

    sel_path = os.path.join(SCORES, 'selection.json')
    if ds == 'dev':
        rule = max(CANDS, key=lambda c: np.mean([single[(r, c)]['Amean'] for r in BT]))
        fusion = max(FUSIONS, key=lambda f: fused[f]['Amean'])
        sel = dict(rule=rule, fusion=fusion, runs=BT,
                   dev_rule_Amean={c: float(np.mean([single[(r, c)]['Amean'] for r in BT])) for c in CANDS},
                   dev_fusion_Amean={f: fused[f]['Amean'] for f in FUSIONS}, dev_A8_Amean=a8['Amean'])
        json.dump(sel, open(sel_path, 'w'), indent=2)
        p(f'SELECTED (dev): single rule {rule}, fusion {fusion}')
    else:
        sel = json.load(open(sel_path))
        assert sel['runs'] == BT
        rule, fusion = sel['rule'], sel['fusion']
        dev_fused = sel['dev_fusion_Amean'][fusion] - sel['dev_A8_Amean']
        # D1: single BEATs model vs our single-model rule C1
        bt = [single[(r, rule)]['Amean'] for r in BT]
        c1 = c1_seed('eval')
        d1 = dict(rule=rule, beats_eval_Amean=float(np.mean(bt)), per_seed=bt, c1_eval8=C1_EVAL8,
                  delta=float(np.mean(bt) - C1_EVAL8), seeds_above=int(sum(x > C1_EVAL8 for x in bt)),
                  paired_c1=[c1.get(int(r[-4:])) for r in BT])
        d1['pass'] = bool(d1['delta'] >= 0.5 and d1['seeds_above'] >= 2)
        # D2: chosen fusion vs E2′[A8]
        ci = boot(ev, S[fusion], S['A8'])
        seeds = [fused[f'{fusion}[{r[-5:]}]']['Amean'] - a8['Amean'] for r in BT]
        d2 = dict(fusion=fusion, eval_delta=fused[fusion]['Amean'] - a8['Amean'], eval_ci95=ci,
                  dev_delta=dev_fused, per_seed_eval_delta=seeds)
        d2['pass'] = bool(d2['eval_delta'] >= 0.2 and ci[0] > 0 and dev_fused >= -0.1
                          and sum(x > 0 for x in seeds) >= 2)
        json.dump(dict(D1=d1, D2=d2), open(os.path.join(SCORES, 'decisions.json'), 'w'), indent=2)
        p(f'D1 single {rule}: eval Amean {d1["beats_eval_Amean"]:.2f} vs C1 {C1_EVAL8} -> Δ {d1["delta"]:+.2f}, '
          f'{d1["seeds_above"]}/{len(BT)} seeds above | paired C1 {d1["paired_c1"]} | PASS {d1["pass"]}')
        p(f'D2 {fusion}: eval Δ {d2["eval_delta"]:+.2f} CI {ci[0]:+.2f},{ci[1]:+.2f} | dev Δ {dev_fused:+.2f} | '
          f'one-seed Δ {" ".join(f"{x:+.2f}" for x in seeds)} | PASS {d2["pass"]}')
    txt = '\n'.join(L)
    print(txt)
    open(os.path.join(SCORES, f'scorecard_{ds}.txt'), 'w').write(txt + '\n')


if __name__ == '__main__':
    st, emb = sys.argv[1], sys.argv[2]
    assert st in ('dev', 'eval')
    os.makedirs(SCORES, exist_ok=True)
    stage(st, emb)
