#!/usr/bin/env python
"""timepool-head: tp vs ao_nogain, seed-paired, under R0, S1′ and C1, plus the ToyCar roll diagnostic.

    python ablation/timepool-head/score.py main <tp emb> <ao emb>   # scores/{metrics,perid}.csv, summary.json, scorecard.txt
    python ablation/timepool-head/score.py diag <tp emb> <ao emb>   # scores/diag_roll.csv, diag.json, scorecard_diag.txt

<tp emb> = this ablation's extract.py dumps; <ao emb> = ../pooled-knn's dumps of the ao_nogain
checkpoints (same extractor, same format). Seeds: env TP_SEEDS (default "2040 2041 2042 2043").

Every rule is computed the same way for both arms, with ../pooled-knn/score.py's Bank (imported
unchanged): per machine ID, bank = the ID's training-part clips (the replayed 95%), 1-NN cosine
distances on L2-normalised uncentred vectors, z-statistics over the bank scored leave-one-out.

  R0  = the CE of the ArcFace logit (eval.py's score; checked against the committed CSVs)
  S1′ = zR0′ + zK′                    (K: 128-d pre-head embedding)
  C1  = S1′ + zGmean                   (Gmean: time-and-frequency mean of mobilefacenet.conv2's map)

For tp, conv2's map is the input to the time mean that feeds the tied linear7, so Gmean is computed
from the same tensor as for ao_nogain. Amean = mean(AUC, pAUC), each averaged over the 6 machines.
Paired Δ = tp − ao_nogain at the same seed; se = sd of the per-seed Δ / sqrt(n).
"""
import csv
import importlib.util
import json
import os
import sys

import numpy as np
from sklearn import metrics as skm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from scores_path import scores_csv  # noqa: E402

_spec = importlib.util.spec_from_file_location('pk_score', os.path.join(HERE, '..', 'pooled-knn', 'score.py'))
pk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pk)
eb = pk.eb

SCORES = os.environ.get('TP_OUT', os.path.join(HERE, 'scores'))     # TP_OUT: tests only
PK_METRICS = os.path.join(HERE, '..', 'pooled-knn', 'scores', 'confirm_metrics.csv')
SEEDS = tuple(int(s) for s in os.environ.get('TP_SEEDS', '2040 2041 2042 2043').split())
STAGE1, STAGE2 = (2040, 2041, 2042, 2043), (2044, 2045, 2046, 2047)
ARMS = {'tp': 'SCLTFSTgramMFN_tp_s{}'.format, 'ao': 'SCLTFSTgramMFN_ao_nogain_s{}'.format}
MACH = eb.MACH
NONTOY = ('fan', 'pump', 'slider', 'valve')
RULES = ('R0', 'S1p', 'C1')                       # the three reported rules; C1 is primary
STAND = ('zR0p', 'zKp', 'Gmean')                  # standalone terms, descriptive
KEYS = ('AUC', 'pAUC', 'mAUC', 'Amean')
DS = ('dev', 'eval')
PK_KEYS = ('ce', 'pre', 'gmean', 'gmax', 'bmean', 'bmax')


def load_meta(tp_emb, ao_emb):
    a = dict(np.load(os.path.join(tp_emb, 'meta.npz')))
    b = dict(np.load(os.path.join(ao_emb, 'meta.npz')))
    for k in a:
        x, y = a[k], b[k]
        if k.endswith('paths'):
            x, y = pk.below_data(x), pk.below_data(y)
        assert np.array_equal(x, y), f'meta.npz disagree on {k}'
    return a


def run_scores(emb, run, meta):
    """{rule: (n_test,)} over every dev and eval test clip, for one checkpoint."""
    tr = pk.load_npz(os.path.join(emb, 'train', f'{run}.npz'))
    nte = len(meta['te_label'])
    te = {k: np.zeros((nte,) + tr[k].shape[1:], tr[k].dtype) for k in PK_KEYS}
    have = np.zeros(nte, bool)
    for ds in DS:
        a = pk.load_npz(os.path.join(emb, ds, f'{run}.npz'))
        for k in te:
            te[k][a['idx']] = a[k]
        have[a['idx']] = True
    assert have.all(), (run, int((~have).sum()))
    o = pk.Bank(tr, meta['tr_label']).score(te, meta['te_label'])
    S = {'R0': te['ce'].astype(np.float64)}
    for k in ('S1p', 'C1') + STAND:
        S[k] = o[k]
        assert np.isfinite(S[k]).all(), (run, k)
    return S


def committed_id_auc(run, ds):
    out = {}
    with open(scores_csv(run, ds), encoding='UTF-8') as f:
        for row in csv.DictReader(f):
            if row['scope'] == 'id':
                out[(row['machine'], row['id'])] = float(row['AUC'])
    return out


def flat(res, m=None):
    if m is None:
        o = res['overall']
        return dict(AUC=o['AUC'], pAUC=o['pAUC'], mAUC=o['mAUC'], Amean=o['mean_AUC_pAUC'])
    A, P = 100 * res[m]['AUC'], 100 * res[m]['pAUC']
    return dict(AUC=A, pAUC=P, mAUC=100 * res[m]['mAUC'], Amean=(A + P) / 2)


def mean_se(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float('nan')


# ------------------------------------------------------------------------------------------ main
def main_stage(tp_emb, ao_emb):
    meta = load_meta(tp_emb, ao_emb)
    ev = {ds: eb.Evaluator(meta, ds) for ds in DS}
    pk_ref = {}
    with open(PK_METRICS) as f:
        for r in csv.DictReader(f):
            if r['machine'] == 'overall':
                pk_ref[(r['system'], r['dataset'], r['rule'])] = float(r['AUC'])
    res, p0, rows, perid = {}, {}, [], []
    seeds = [s for s in SEEDS if os.path.exists(os.path.join(tp_emb, 'eval', f'{ARMS["tp"](s)}.npz'))]
    assert seeds, 'no tp dumps yet'
    for s in seeds:
        for arm, name in ARMS.items():
            run = name(s)
            S = run_scores(tp_emb if arm == 'tp' else ao_emb, run, meta)
            chk = {}
            for ds in DS:
                for rule in RULES + STAND:
                    res[(arm, s, ds, rule)] = r = ev[ds](S[rule])
                    for m in MACH + [None]:
                        v = flat(r, m)
                        rows.append([arm, run, s, ds, rule, m or 'overall'] + [f'{v[k]:.4f}' for k in KEYS])
                    if rule in RULES + ('Gmean',):
                        perid += [[arm, run, s, ds, rule, m, i, f'{100 * a:.4f}'] for m in MACH for i, a in r[m]['ids'].items()]
                com = committed_id_auc(run, ds)
                r0 = res[(arm, s, ds, 'R0')]
                chk[f'{ds}_R0_vs_evalpy_max_abs_id_auc'] = max(abs(r0[m]['ids'][i] - com[(m, i)]) for m in MACH for i in r0[m]['ids'])
                if arm == 'ao':
                    for rule, pkr in (('S1p', 'S1p'), ('C1', 'C1')):
                        chk[f'{ds}_{rule}_vs_pooled_knn_overall_AUC'] = abs(res[(arm, s, ds, rule)]['overall']['AUC'] - pk_ref[(run, ds, pkr)])
            p0[run] = chk
            print(run, json.dumps({k: round(v, 6) for k, v in chk.items()}), flush=True)
    worst_r0 = max(v for c in p0.values() for k, v in c.items() if 'R0_vs_evalpy' in k)
    worst_pk = max([v for c in p0.values() for k, v in c.items() if 'pooled_knn' in k] or [0])
    assert worst_r0 < 5e-4, f'R0 does not reproduce eval.py: {worst_r0:.2e}'
    assert worst_pk < 1e-3, f'ao_nogain S1′/C1 do not reproduce ../pooled-knn: {worst_pk:.2e}'

    with open(os.path.join(SCORES, 'metrics.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arm', 'run', 'seed', 'dataset', 'rule', 'machine'] + list(KEYS))
        w.writerows(rows)
    with open(os.path.join(SCORES, 'perid.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arm', 'run', 'seed', 'dataset', 'rule', 'machine', 'id', 'AUC'])
        w.writerows(perid)

    def val(arm, s, ds, rule, cell, k):
        """cell: 'overall', a machine, 'nontoy' (mean of the 4 machines' Amean) or 'machine id_xx' (AUC)."""
        r = res[(arm, s, ds, rule)]
        if cell == 'nontoy':
            return float(np.mean([flat(r, m)['Amean'] for m in NONTOY]))
        if ' ' in cell:
            m, i = cell.split()
            return 100 * r[m]['ids'][i]
        return flat(r, None if cell == 'overall' else cell)[k]

    out = {'seeds': seeds, 'p0': {'max_R0_vs_evalpy_id_auc': worst_r0, 'max_ao_vs_pooled_knn_overall_auc': worst_pk},
           'rules': {}}
    for rule in RULES + STAND:
        o = {}
        for ds in DS:
            r0 = res[('tp', seeds[0], ds, 'R0')]
            cells = ['overall'] + MACH + ['nontoy'] + [f'{m} {i}' for m in ('ToyCar', 'ToyConveyor') for i in sorted(r0[m]['ids'])]
            for cell in cells:
                ks = KEYS if cell == 'overall' else ('AUC', 'Amean') if cell in MACH else ('Amean',) if cell == 'nontoy' else ('AUC',)
                for k in ks:
                    tp = [val('tp', s, ds, rule, cell, k) for s in seeds]
                    ao = [val('ao', s, ds, rule, cell, k) for s in seeds]
                    d = [a - b for a, b in zip(tp, ao)]
                    m, se = mean_se(d)
                    o[f'{ds}|{cell}|{k}'] = {'tp': float(np.mean(tp)), 'ao': float(np.mean(ao)), 'd': m, 'se': se,
                                             'd_per_seed': [round(x, 4) for x in d], 'n_pos': int(sum(x > 0 for x in d))}
        out['rules'][rule] = o
    # how much the pooled kNN term adds within each arm (C1 − S1′)
    out['C1_minus_S1p'] = {f'{arm}|{ds}|{k}': mean_se([flat(res[(arm, s, ds, 'C1')])[k] - flat(res[(arm, s, ds, 'S1p')])[k] for s in seeds])
                           for arm in ARMS for ds in DS for k in ('AUC', 'pAUC', 'Amean')}

    C = out['rules']['C1']
    g = lambda key: C[key]['d']
    stage = 'stage1' if tuple(seeds) == STAGE1 else 'stage2' if tuple(seeds) == STAGE2 else None
    d1 = None
    if tuple(seeds) in (STAGE1, STAGE2):
        d1 = {
            '(a) dev Amean Δ ≥ +0.05, or (dev ≥ −0.10 and eval ≥ +0.30)':
                g('dev|overall|Amean') >= 0.05 or (g('dev|overall|Amean') >= -0.10 and g('eval|overall|Amean') >= 0.30),
            '(b) eval Amean Δ ≥ −0.20': g('eval|overall|Amean') >= -0.20,
            '(c) non-Toy Amean Δ ≥ −0.20 dev': g('dev|nontoy|Amean') >= -0.20,
            '(c) non-Toy Amean Δ ≥ −0.40 eval': g('eval|nontoy|Amean') >= -0.40,
            '(d) dev ToyConveyor AUC Δ ≥ −1.0': g('dev|ToyConveyor|AUC') >= -1.0,
        }
        d1['pass'] = all(d1.values())
    out['D1_C1'] = d1
    out['stage'] = stage
    json.dump(out, open(os.path.join(SCORES, f'summary_{"_".join(map(str, seeds))}.json'), 'w'), indent=2)

    L = [f'timepool-head: tp − ao_nogain, seeds {seeds}; P0 R0 vs eval.py ≤ {worst_r0:.1e} per-ID AUC; '
         f'ao S1′/C1 vs ../pooled-knn ≤ {worst_pk:.1e}']
    for rule in RULES + STAND:
        L.append(f'\n== {rule}: tp | ao_nogain | paired Δ (se) | per-seed Δ')
        for key, v in out['rules'][rule].items():
            if rule in STAND and not key.split('|')[1] in ('overall', 'ToyCar', 'ToyConveyor'):
                continue
            L.append(f'  {key:28s} {v["tp"]:7.2f} | {v["ao"]:7.2f} | {v["d"]:+6.2f} ({v["se"]:.2f}) | '
                     + ' '.join(f'{x:+.2f}' for x in v['d_per_seed']))
    L.append('\nC1 − S1′ within each arm (mean, se):')
    for k, (m, se) in out['C1_minus_S1p'].items():
        L.append(f'  {k:18s} {m:+.3f} ({se:.3f})')
    if d1:
        L.append(f'\nD1 ({stage}, rule C1):')
        for k, v in d1.items():
            L.append(f'  {k:62s} {v}')
    txt = '\n'.join(L)
    print(txt)
    open(os.path.join(SCORES, f'scorecard_{"_".join(map(str, seeds))}.txt'), 'w').write(txt + '\n')


# ------------------------------------------------------------------------------------------ diag
def diag_stage(tp_emb, ao_emb):
    """ToyCar under circular rolls k/8 (../pooled-knn's diagnostic), both arms, same seeds.
    Bank and z-statistics stay unrolled. Penalty(k) = mean over normal clips of s(k) − s(0), in sd of
    s(0), per ID, averaged over IDs; headline = mean over k = 3, 4, 5, then over seeds."""
    meta = load_meta(tp_emb, ao_emb)
    names = ('S1p', 'R0', 'zR0p', 'zKp', 'Gmean', 'C1')
    lab_all, y_all = meta['te_label'], meta['te_y']
    rows = []
    seeds = [s for s in SEEDS if os.path.exists(os.path.join(tp_emb, 'eval_ToyCar_rolls', f'{ARMS["tp"](s)}.npz'))]
    for s in seeds:
        for arm, name in ARMS.items():
            run, emb = name(s), (tp_emb if arm == 'tp' else ao_emb)
            tr = pk.load_npz(os.path.join(emb, 'train', f'{run}.npz'))
            bank = pk.Bank(tr, meta['tr_label'])
            del tr
            for ds in DS:
                base = pk.load_npz(os.path.join(emb, ds, f'{run}.npz'))
                rol = pk.load_npz(os.path.join(emb, f'{ds}_ToyCar_rolls', f'{run}.npz'))
                idx = rol['idx']
                pos = {v: j for j, v in enumerate(base['idx'])}
                bi = np.array([pos[v] for v in idx])
                lab, y = lab_all[idx], y_all[idx]
                ks = [0] + list(rol['rolls'])
                per_k = [(base['ce'][bi], bank.score({k: base[k][bi] for k in PK_KEYS}, lab))]
                for j in range(len(rol['rolls'])):
                    per_k.append((rol['ce'][:, j], bank.score({k: rol[k][:, j] for k in PK_KEYS}, lab)))
                for nm in names:
                    pick = lambda t: t[0].astype(np.float64) if nm == 'R0' else t[1][nm]
                    s0 = pick(per_k[0])
                    for k, t in zip(ks, per_k):
                        sc = pick(t)
                        pen = [np.mean(sc[(lab == c) & (y == 0)] - s0[(lab == c) & (y == 0)]) / s0[(lab == c) & (y == 0)].std()
                               for c in np.unique(lab)]
                        rows.append([arm, run, s, ds, nm, int(k), f'{np.mean(pen):.4f}', f'{100 * skm.roc_auc_score(y, sc):.4f}'])
            print('diag', run, flush=True)
    with open(os.path.join(SCORES, 'diag_roll.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arm', 'run', 'seed', 'dataset', 'score', 'k', 'penalty_sd', 'ToyCar_AUC'])
        w.writerows(rows)
    out = {'seeds': seeds}
    L = [f'ToyCar roll diagnostic, seeds {seeds}: headline penalty (sd, mean k=3..5) | penalty by k | AUC by k']
    for ds in DS:
        o = {}
        for arm in ARMS:
            for nm in names:
                sel = [r for r in rows if r[0] == arm and r[3] == ds and r[4] == nm]
                o[f'{arm}|{nm}'] = {
                    'headline_penalty_sd': float(np.mean([float(r[6]) for r in sel if r[5] in (3, 4, 5)])),
                    'penalty_by_k': [float(np.mean([float(r[6]) for r in sel if r[5] == k])) for k in range(8)],
                    'AUC_by_k': [float(np.mean([float(r[7]) for r in sel if r[5] == k])) for k in range(8)]}
                v = o[f'{arm}|{nm}']
                L.append(f'{ds:4s} {arm} {nm:5s} {v["headline_penalty_sd"]:7.2f} | ' + ' '.join(f'{x:6.2f}' for x in v['penalty_by_k'])
                         + ' | ' + ' '.join(f'{x:5.1f}' for x in v['AUC_by_k']))
        t, a = o['tp|S1p']['headline_penalty_sd'], o['ao|S1p']['headline_penalty_sd']
        o['verdict_tp_S1p'] = 'removes' if t <= 1 else 'reduces' if t <= 0.25 * a else 'does not'
        L.append(f'{ds}: verdict, tp S1′ vs ao S1′: {o["verdict_tp_S1p"]} ({t:.2f} vs {a:.2f} sd)')
        out[ds] = o
    json.dump(out, open(os.path.join(SCORES, 'diag.json'), 'w'), indent=2)
    txt = '\n'.join(L)
    print(txt)
    open(os.path.join(SCORES, 'scorecard_diag.txt'), 'w').write(txt + '\n')


if __name__ == '__main__':
    stage, tp_emb, ao_emb = sys.argv[1:4]
    os.makedirs(SCORES, exist_ok=True)
    {'main': main_stage, 'diag': diag_stage}[stage](tp_emb, ao_emb)
