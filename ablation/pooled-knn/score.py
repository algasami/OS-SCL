#!/usr/bin/env python
"""Score the pooled pre-linear7 kNN candidates; write scores/.

    python ablation/pooled-knn/score.py select  ablation/pooled-knn/results/emb   # stage A
    python ablation/pooled-knn/score.py confirm ablation/pooled-knn/results/emb   # stage B
    python ablation/pooled-knn/score.py diag    ablation/pooled-knn/results/emb   # ToyCar rolls

Rules (README, *Scores* / *Candidate rules*), per checkpoint and machine ID, bank = the ID's
training-part clips (the replayed 95%), every distance a 1-NN cosine distance on
L2-normalised uncentred vectors, every z-statistic over the bank scored leave-one-out:

  S1p = zR0' + zK'                         ../ensemble-backend/score.py (checked equal here)
  Gmean, Gmax                              z of the 1-NN distance of the global pooled vector
  Bmean, Bmax                              z of mean_b z(d_b), d_b = per-band 1-NN distance
  Gmm = z(zGmean + zGmax);  All = z(zGmean + zGmax + zBmean + zBmax)
  C1..C6 = S1p + {Gmean, Gmax, Gmm, Bmean, Bmax, All}

select: only seeds 2040-2043 are scored under the new features (dev only; eval is not
extracted in stage A). confirm: all 8 seeds, dev and eval. diag: ToyCar under rolls k/8.
"""
import csv
import importlib.util
import json
import os
import sys

import numpy as np
from sklearn import metrics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from scores_path import scores_csv    # noqa: E402

_spec = importlib.util.spec_from_file_location('eb_score', os.path.join(HERE, '..', 'ensemble-backend', 'score.py'))
eb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eb)

SCORES = os.path.join(HERE, 'scores')
AUGOPS_EMB = '/home/f74134118/OS-SCL-augops/ablation/aug-ops/results/emb'
S1P_CSV = os.path.join(HERE, '..', 'aug-ops', 'scores', 's1p.csv')
SEL, CONF = (2040, 2041, 2042, 2043), (2044, 2045, 2046, 2047)
ALL8 = SEL + CONF
RUN = 'SCLTFSTgramMFN_ao_nogain_s{}'.format
MACH = eb.MACH
FEATS = ('Gmean', 'Gmax', 'Gmm', 'Bmean', 'Bmax', 'All')
CANDS = {f'C{i + 1}': f for i, f in enumerate(FEATS)}
STAND = ('zR0p', 'zKp') + FEATS
N_BOOT = 2000


def l2(x):
    x = x.astype(np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def nn1(q, ref, loo=False):
    """1-NN cosine distance; q (n, d) or (n, B, d) per band -> (n,) or (n, B)."""
    if q.ndim == 2:
        s = q @ ref.T
        if loo:
            np.fill_diagonal(s, -np.inf)
        return 1.0 - s.max(1)
    s = np.matmul(q.transpose(1, 0, 2), ref.transpose(1, 2, 0))     # (B, n, m)
    if loo:
        for b in range(s.shape[0]):
            np.fill_diagonal(s[b], -np.inf)
    return (1.0 - s.max(2)).T


def l2b(k, x):
    """Band vectors are fp16 on disk: normalise in fp64, search in fp32 (1-cos resolution ~1e-7)."""
    x = l2(x)
    return x.astype(np.float32) if k.startswith('b') else x


def load_npz(path):
    a = dict(np.load(path))
    for k in ('ce', 'pre', 'gmean', 'gmax', 'bmean', 'bmax'):   # faulty host RAM: range-check on load
        v = a[k].astype(np.float32)
        assert np.isfinite(v).all() and np.abs(v).max() < (1e3 if k == 'ce' else 1e4), (path, k)
    return a


class Bank:
    """One checkpoint's per-ID banks and z-statistics; scores any query set."""

    def __init__(self, tr, trl):
        self.ids = {}
        for c in np.unique(trl):
            ref = (trl == c) & ~tr['valid']
            b = {k: l2b(k, tr[k][ref]) for k in ('pre', 'gmean', 'gmax', 'bmean', 'bmax')}
            st = {}
            self.ids[c] = (b, st)
            self._z(self._raw(b, tr['ce'][ref], b, loo=True), st, fit=True)

    @staticmethod
    def _raw(q, ce, b, loo=False):
        return dict(R0=ce.astype(np.float64), K=nn1(q['pre'], b['pre'], loo),
                    Gmean=nn1(q['gmean'], b['gmean'], loo), Gmax=nn1(q['gmax'], b['gmax'], loo),
                    bmean=nn1(q['bmean'], b['bmean'], loo), bmax=nn1(q['bmax'], b['bmax'], loo))

    @staticmethod
    def _z(raw, st, fit=False):
        def z(name, x):
            if fit:
                st[name] = (x.mean(0), x.std(0))
            mu, sd = st[name]
            return (x - mu) / sd
        o = {'zR0p': z('R0', raw['R0']), 'zKp': z('K', raw['K']),
             'Gmean': z('Gmean', raw['Gmean']), 'Gmax': z('Gmax', raw['Gmax'])}
        o['Bmean'] = z('Bmean', z('bmean', raw['bmean']).mean(1))
        o['Bmax'] = z('Bmax', z('bmax', raw['bmax']).mean(1))
        o['Gmm'] = z('Gmm', o['Gmean'] + o['Gmax'])
        o['All'] = z('All', o['Gmean'] + o['Gmax'] + o['Bmean'] + o['Bmax'])
        o['S1p'] = o['zR0p'] + o['zKp']
        for c, f in CANDS.items():
            o[c] = o['S1p'] + o[f]
        return o

    def score(self, te, labels):
        """te: dict of query arrays (n, ...) ; labels (n,). Returns {rule: (n,)}."""
        out = None
        for c in np.unique(labels):
            m = labels == c
            b, st = self.ids[c]
            q = {k: l2b(k, te[k][m]) for k in ('pre', 'gmean', 'gmax', 'bmean', 'bmax')}
            o = self._z(self._raw(q, te['ce'][m], b), st)
            for k, v in o.items():
                assert np.isfinite(v).all() and np.abs(v).max() < 1e6, (c, k)
            if out is None:
                out = {k: np.full(len(labels), np.nan) for k in o}
            for k, v in o.items():
                out[k][m] = v
        return out


def below_data(paths):
    return np.array([p.split('/data/', 1)[1] for p in paths])


class Fast:
    """Overall AUC / pAUC / Amean on one dataset (eval.py's definitions), for the bootstrap."""

    def __init__(self, meta, ds):
        self.ev = eb.Evaluator(meta, ds)
        self.sel = self.ev.sel

    def __call__(self, score, idx):
        s, y, mach = score[self.sel][idx], self.ev.y[idx], self.ev.mach[idx]
        a = [metrics.roc_auc_score(y[mach == m], s[mach == m]) for m in MACH]
        p = [metrics.roc_auc_score(y[mach == m], s[mach == m], max_fpr=0.1) for m in MACH]
        A, P = 100 * np.mean(a), 100 * np.mean(p)
        return {'AUC': A, 'pAUC': P, 'Amean': (A + P) / 2}


def flat(res):
    o = res['overall']
    return dict(AUC=o['AUC'], pAUC=o['pAUC'], mAUC=o['mAUC'], Amean=o['mean_AUC_pAUC'])


class Ctx:
    def __init__(self, emb):
        self.emb = emb
        self.meta = dict(np.load(os.path.join(emb, 'meta.npz')))
        old = dict(np.load(os.path.join(AUGOPS_EMB, 'meta.npz')))
        for k in self.meta:
            a, b = self.meta[k], old[k]
            if k.endswith('paths'):
                a, b = below_data(a), below_data(b)
            assert np.array_equal(a, b), f'meta.npz disagrees with aug-ops on {k}'
        self.ev = {ds: eb.Evaluator(self.meta, ds) for ds in ('dev', 'eval')}
        self.s1p_ref = {}
        with open(S1P_CSV) as f:
            for row in csv.DictReader(f):
                self.s1p_ref[(row['run'], row['rule'], row['dataset'])] = float(row['AUC'])

    def run(self, seed, sets, new=True):
        """Scores of one checkpoint over all test clips (NaN outside `sets`), plus P0 checks."""
        r = RUN(seed)
        tr = load_npz(os.path.join(self.emb, 'train', f'{r}.npz'))
        nte = len(self.meta['te_label'])
        te = {k: np.zeros((nte,) + tr[k].shape[1:], tr[k].dtype) for k in ('ce', 'pre', 'gmean', 'gmax', 'bmean', 'bmax')}
        have = np.zeros(nte, bool)
        for ds in sets:
            a = load_npz(os.path.join(self.emb, ds, f'{r}.npz'))
            for k in te:
                te[k][a['idx']] = a[k]
            have[a['idx']] = True
        old = dict(np.load(os.path.join(AUGOPS_EMB, f'{r}.npz')))
        chk = {'train_max_abs_ce': float(np.abs(tr['ce'] - old['tr_ce']).max()),
               'train_max_abs_pre': float(np.abs(tr['pre'] - old['tr_pre']).max()),
               'test_max_abs_ce': float(np.abs(te['ce'][have] - old['te_ce'][have]).max()),
               'test_max_abs_pre': float(np.abs(te['pre'][have] - old['te_pre'][have]).max()),
               'valid_same': bool(np.array_equal(tr['valid'], old['valid']))}
        assert chk['valid_same'], r
        # S1' from ../ensemble-backend, on the new dump
        ebs = eb.per_run_scores(self.meta, dict(tr_ce=tr['ce'], tr_pre=tr['pre'], valid=tr['valid'],
                                                te_ce=te['ce'], te_pre=te['pre']))
        S = {'S0': np.where(have, ebs['S0'], np.nan), 'S1p_eb': np.where(have, ebs['S1p'], np.nan)}
        for ds in sets:
            res0 = self.ev[ds](S['S0'])
            committed = {}
            with open(scores_csv(r, ds), encoding='UTF-8') as f:
                for row in csv.DictReader(f):
                    if row['scope'] == 'id':
                        committed[(row['machine'], row['id'])] = float(row['AUC'])
            chk[f'{ds}_S0_max_abs_id_auc_diff'] = max(abs(res0[m]['ids'][i] - committed[(m, i)])
                                                      for m in MACH for i in res0[m]['ids'])
            chk[f'{ds}_S1p_overall_vs_s1p_csv'] = abs(self.ev[ds](S['S1p_eb'])['overall']['AUC']
                                                     - self.s1p_ref[(r, 'S1p', ds)])
        if not new:
            S['S1p'] = S.pop('S1p_eb')
            return S, chk, None, None, None
        bank = Bank(tr, self.meta['tr_label'])
        lab = self.meta['te_label']
        idx = np.where(have)[0]
        o = bank.score({k: v[idx] for k, v in te.items()}, lab[idx])
        for k, v in o.items():
            S[k] = np.full(nte, np.nan)
            S[k][idx] = v
        chk['S1p_vs_eb_max_abs'] = float(np.nanmax(np.abs(S['S1p'] - S.pop('S1p_eb'))))
        assert chk['S1p_vs_eb_max_abs'] < 1e-8, chk
        for k, v in S.items():
            assert np.isfinite(v[idx]).all(), (r, k)
        return S, chk, bank, te, tr


def write_metrics(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['system', 'dataset', 'rule', 'machine', 'AUC', 'pAUC', 'mAUC', 'Amean'])
        w.writerows(rows)


def metric_rows(system, ds, rule, res):
    out = []
    for m in MACH:
        A, P = 100 * res[m]['AUC'], 100 * res[m]['pAUC']
        out.append([system, ds, rule, m, f'{A:.4f}', f'{P:.4f}', f'{100 * res[m]["mAUC"]:.4f}', f'{(A + P) / 2:.4f}'])
    o = flat(res)
    out.append([system, ds, rule, 'overall'] + [f'{o[k]:.4f}' for k in ('AUC', 'pAUC', 'mAUC', 'Amean')])
    return out


# ---------------------------------------------------------------------------------------- select
def select(ctx):
    rules = ('S0', 'S1p') + STAND + tuple(CANDS)
    res, chk, rows, fp16 = {}, {}, [], {}
    for s in ALL8:
        new = s in SEL
        S, chk[RUN(s)], bank, te, tr = ctx.run(s, ['dev'], new=new)
        if not new:          # confirm seeds: only the already-seen S0 / S1' checks
            continue
        for rule in rules:
            res[(s, rule)] = ctx.ev['dev'](S[rule])
            rows += metric_rows(RUN(s), 'dev', rule, res[(s, rule)])
        # P0: fp16 band storage. Global vectors rebuilt from fp16 bands vs the fp32 ones,
        # compared as raw 1-NN distances (z is monotone within an ID, so per-ID AUC is too).
        d = {}
        sel = ctx.ev['dev'].sel
        lab = ctx.meta['te_label'][sel]
        for g, fn in (('gmean', lambda x: x.astype(np.float64).mean(1)), ('gmax', lambda x: x.astype(np.float64).max(1))):
            src = 'b' + g[1:]
            raw32, raw16 = np.zeros(len(sel)), np.zeros(len(sel))
            for c in np.unique(lab):
                m = lab == c
                b = bank.ids[c][0]
                raw32[m] = nn1(l2(te[g][sel][m]), b[g])
                ref = (ctx.meta['tr_label'] == c) & ~tr['valid']
                raw16[m] = nn1(l2(fn(te[src][sel][m])), l2(fn(tr[src][ref])))
            f32, f16 = np.full(len(ctx.meta['te_label']), np.nan), np.full(len(ctx.meta['te_label']), np.nan)
            f32[sel], f16[sel] = raw32, raw16
            r32, r16 = ctx.ev['dev'](f32), ctx.ev['dev'](f16)
            d[g] = max(abs(r32[m]['ids'][i] - r16[m]['ids'][i]) for m in MACH for i in r32[m]['ids'])
        fp16[RUN(s)] = d
    write_metrics(os.path.join(SCORES, 'select_metrics.csv'), rows)

    O = lambda s, rule, k: flat(res[(s, rule)])[k]
    summ = {}
    for c in CANDS:
        dd = {k: [O(s, c, k) - O(s, 'S1p', k) for s in SEL] for k in ('AUC', 'pAUC', 'mAUC', 'Amean')}
        summ[c] = {f'd_{k}': float(np.mean(v)) for k, v in dd.items()}
        summ[c]['d_Amean_per_seed'] = [round(x, 4) for x in dd['Amean']]
        summ[c]['feature'] = CANDS[c]
    chosen = max(CANDS, key=lambda c: summ[c]['d_Amean'])
    out = {'chosen': chosen, 'chosen_feature': CANDS[chosen], 'D0_nothing_positive': summ[chosen]['d_Amean'] <= 0,
           'selection_seeds': list(SEL), 'candidates': summ,
           'standalone_dev_mean': {r: {k: float(np.mean([O(s, r, k) for s in SEL])) for k in ('AUC', 'pAUC', 'Amean')}
                                   for r in ('S0', 'S1p') + STAND}}
    json.dump(out, open(os.path.join(SCORES, 'select.json'), 'w'), indent=2)
    p0 = {'runs': chk, 'fp16_vs_fp32_global_max_abs_id_auc_diff': fp16}
    json.dump(p0, open(os.path.join(SCORES, 'p0_select.json'), 'w'), indent=2)

    L = []
    p = L.append
    s0 = max(max(c['dev_S0_max_abs_id_auc_diff'] for c in chk.values()), 0)
    p(f'P0  dev S0 max |per-ID AUC - eval.py| {s0:.2e} (< 5e-4: {s0 < 5e-4}); '
      f'max |R0 - aug-ops| train {max(c["train_max_abs_ce"] for c in chk.values()):.1e} '
      f'test {max(c["test_max_abs_ce"] for c in chk.values()):.1e}; '
      f'max |S1p - s1p.csv| {max(c["dev_S1p_overall_vs_s1p_csv"] for c in chk.values()):.4f}; '
      f'fp16 max per-ID AUC diff {max(max(v.values()) for v in fp16.values()):.1e}')
    p('standalone, 4 selection seeds, dev mean AUC / pAUC / Amean:')
    for r, v in out['standalone_dev_mean'].items():
        p(f'    {r:6s} {v["AUC"]:6.2f} {v["pAUC"]:6.2f} {v["Amean"]:6.2f}')
    p('candidates, Δ vs S1′ (4-seed mean): AUC pAUC Amean mAUC | Amean per seed')
    for c, v in summ.items():
        p(f'    {c} {v["feature"]:6s} {v["d_AUC"]:+.3f} {v["d_pAUC"]:+.3f} {v["d_Amean"]:+.3f} {v["d_mAUC"]:+.3f} | '
          + ' '.join(f'{x:+.3f}' for x in v['d_Amean_per_seed']))
    p(f'CHOSEN {chosen} = S1′ + z{CANDS[chosen]} (Δ Amean {summ[chosen]["d_Amean"]:+.3f}); D0: {out["D0_nothing_positive"]}')
    txt = '\n'.join(L)
    print(txt)
    open(os.path.join(SCORES, 'scorecard_select.txt'), 'w').write(txt + '\n')


# --------------------------------------------------------------------------------------- confirm
def boot_delta(fast, pairs, rng_seed=0):
    """Paired stratified clip-bootstrap of mean over pairs of (a - b): pairs = [(a, b), ...]."""
    rng = np.random.default_rng(rng_seed)
    draws = []
    for _ in range(N_BOOT):
        idx = fast.ev.resample(rng)
        d = {k: [] for k in ('AUC', 'pAUC', 'Amean')}
        for a, b in pairs:
            ra, rb = fast(a, idx), fast(b, idx)
            for k in d:
                d[k].append(ra[k] - rb[k])
        draws.append({k: np.mean(v) for k, v in d.items()})
    return {k: [float(x) for x in np.percentile([d[k] for d in draws], [2.5, 97.5])] for k in ('AUC', 'pAUC', 'Amean')}


def confirm(ctx):
    sel = json.load(open(os.path.join(SCORES, 'select.json')))
    ch = sel['chosen']
    rules = ('S0', 'S1p') + STAND + tuple(CANDS)
    S, chk = {}, {}
    for s in ALL8:
        S[s], chk[RUN(s)], _, _, _ = ctx.run(s, ['dev', 'eval'])
    res, rows, perid = {}, [], []
    for ds in ('dev', 'eval'):
        for s in ALL8:
            for rule in rules:
                res[(s, rule, ds)] = r = ctx.ev[ds](S[s][rule])
                rows += metric_rows(RUN(s), ds, rule, r)
                if rule in ('S0', 'S1p', ch):
                    perid += [[RUN(s), ds, rule, m, i, f'{100 * a:.4f}'] for m in MACH for i, a in r[m]['ids'].items()]
        for name, rule in (('E2p', 'S1p'), (f'E_{ch}', ch)):
            res[('G8', name, ds)] = r = ctx.ev[ds](sum(S[s][rule] for s in ALL8))
            rows += metric_rows('G8', ds, name, r)
            perid += [['G8', ds, name, m, i, f'{100 * a:.4f}'] for m in MACH for i, a in r[m]['ids'].items()]
    write_metrics(os.path.join(SCORES, 'confirm_metrics.csv'), rows)
    with open(os.path.join(SCORES, 'confirm_perid.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['system', 'dataset', 'rule', 'machine', 'id', 'AUC'])
        w.writerows(perid)

    O = lambda s, rule, ds, k: flat(res[(s, rule, ds)])[k]
    M = lambda s, rule, ds, m: 100 * res[(s, rule, ds)][m]['AUC']
    KEYS = ('AUC', 'pAUC', 'mAUC', 'Amean')

    def mean_d(seeds, rule, ds):
        return {k: float(np.mean([O(s, rule, ds, k) - O(s, 'S1p', ds, k) for s in seeds])) for k in KEYS}

    fast = {ds: Fast(ctx.meta, ds) for ds in ('dev', 'eval')}
    pt = fast['dev'](S[SEL[0]]['S1p'], np.arange(len(fast['dev'].sel)))
    assert abs(pt['AUC'] - O(SEL[0], 'S1p', 'dev', 'AUC')) < 1e-9
    out = {'chosen': ch, 'feature': CANDS[ch]}
    out['confirm_dev'] = dict(mean_d(CONF, ch, 'dev'), ci=boot_delta(fast['dev'], [(S[s][ch], S[s]['S1p']) for s in CONF]),
                              per_seed_Amean=[O(s, ch, 'dev', 'Amean') - O(s, 'S1p', 'dev', 'Amean') for s in CONF])
    out['eval8'] = dict(mean_d(ALL8, ch, 'eval'), ci=boot_delta(fast['eval'], [(S[s][ch], S[s]['S1p']) for s in ALL8]),
                        per_seed_Amean=[O(s, ch, 'eval', 'Amean') - O(s, 'S1p', 'eval', 'Amean') for s in ALL8])
    out['dev8'] = mean_d(ALL8, ch, 'dev')
    out['selection_dev'] = mean_d(SEL, ch, 'dev')
    for ds in ('dev', 'eval'):
        e = {k: O('G8', f'E_{ch}', ds, k) - O('G8', 'E2p', ds, k) for k in KEYS}
        e['ci'] = boot_delta(fast[ds], [(sum(S[s][ch] for s in ALL8), sum(S[s]['S1p'] for s in ALL8))])
        e['E2p'] = {k: O('G8', 'E2p', ds, k) for k in KEYS}
        e['E_chosen'] = {k: O('G8', f'E_{ch}', ds, k) for k in KEYS}
        out[f'ensemble_{ds}'] = e
    out['per_machine_dAUC'] = {ds: {m: float(np.mean([M(s, ch, ds, m) - M(s, 'S1p', ds, m) for s in ALL8])) for m in MACH}
                               for ds in ('dev', 'eval')}
    out['eval_ToyCar_id_05_dAUC'] = float(np.mean([100 * (res[(s, ch, 'eval')]['ToyCar']['ids']['id_05']
                                                         - res[(s, 'S1p', 'eval')]['ToyCar']['ids']['id_05']) for s in ALL8]))
    out['other_candidates'] = {c: {'confirm_dev': mean_d(CONF, c, 'dev'), 'eval8': mean_d(ALL8, c, 'eval')} for c in CANDS}
    out['abs'] = {f'{rule}_{ds}{n}': {k: float(np.mean([O(s, rule, ds, k) for s in seeds])) for k in KEYS}
                  for rule in ('S0', 'S1p', ch) for ds in ('dev', 'eval') for n, seeds in (('8', ALL8), ('_conf', CONF))}
    cd, ev8 = out['confirm_dev'], out['eval8']
    out['D1'] = {'selection_dAmean_pos': sel['candidates'][ch]['d_Amean'] > 0,
                 'confirm_dev_AUC_lb_pos': cd['ci']['AUC'][0] > 0,
                 'confirm_dev_Amean_lb_pos': cd['ci']['Amean'][0] > 0,
                 'eval8_dAmean_ge_-0.05': ev8['Amean'] >= -0.05}
    out['D1']['adopt'] = all(out['D1'].values())
    json.dump(out, open(os.path.join(SCORES, 'confirm.json'), 'w'), indent=2)
    json.dump(chk, open(os.path.join(SCORES, 'p0_confirm.json'), 'w'), indent=2)

    L = []
    p = L.append
    for ds in ('dev', 'eval'):
        w = max(c[f'{ds}_S0_max_abs_id_auc_diff'] for c in chk.values())
        p(f'P0  {ds}: S0 max |per-ID AUC - eval.py| {w:.2e} (< 5e-4: {w < 5e-4}); max |S1p - s1p.csv| '
          f'{max(c[f"{ds}_S1p_overall_vs_s1p_csv"] for c in chk.values()):.4f}')
    p(f'chosen {ch} = S1′ + z{CANDS[ch]}')
    f = lambda d: ' '.join(f'{k} {d[k]:+.3f}' for k in KEYS)
    p(f'selection dev (2040-43)  Δ {f(out["selection_dev"])}')
    p(f'confirm dev (2044-47)    Δ {f(cd)} | 95% CI AUC [{cd["ci"]["AUC"][0]:+.3f}, {cd["ci"]["AUC"][1]:+.3f}] '
      f'pAUC [{cd["ci"]["pAUC"][0]:+.3f}, {cd["ci"]["pAUC"][1]:+.3f}] Amean [{cd["ci"]["Amean"][0]:+.3f}, {cd["ci"]["Amean"][1]:+.3f}]'
      f' | per seed Amean ' + ' '.join(f'{x:+.3f}' for x in cd['per_seed_Amean']))
    p(f'eval (8 seeds)           Δ {f(ev8)} | 95% CI Amean [{ev8["ci"]["Amean"][0]:+.3f}, {ev8["ci"]["Amean"][1]:+.3f}]'
      f' | per seed Amean ' + ' '.join(f'{x:+.3f}' for x in ev8['per_seed_Amean']))
    p(f'dev (8 seeds)            Δ {f(out["dev8"])}')
    for ds in ('dev', 'eval'):
        p(f'per machine Δ AUC {ds} (8 seeds): ' + ' '.join(f'{m} {v:+.2f}' for m, v in out['per_machine_dAUC'][ds].items()))
    p(f'eval ToyCar id_05 Δ AUC (8 seeds) {out["eval_ToyCar_id_05_dAUC"]:+.2f}')
    for ds in ('dev', 'eval'):
        e = out[f'ensemble_{ds}']
        p(f'ensemble G8 {ds}: E2′ {e["E2p"]["AUC"]:.2f}/{e["E2p"]["pAUC"]:.2f}/{e["E2p"]["Amean"]:.2f} -> '
          f'Σ{ch} {e["E_chosen"]["AUC"]:.2f}/{e["E_chosen"]["pAUC"]:.2f}/{e["E_chosen"]["Amean"]:.2f}; '
          f'Δ Amean {e["Amean"]:+.3f} CI [{e["ci"]["Amean"][0]:+.3f}, {e["ci"]["Amean"][1]:+.3f}]')
    p('absolute, seed means AUC / pAUC / mAUC / Amean:')
    for k, v in out['abs'].items():
        p(f'    {k:22s} ' + ' '.join(f'{v[x]:6.2f}' for x in KEYS))
    p('other candidates (descriptive): confirm-dev Δ Amean | eval8 Δ Amean')
    for c, v in out['other_candidates'].items():
        p(f'    {c} {CANDS[c]:6s} {v["confirm_dev"]["Amean"]:+.3f} | {v["eval8"]["Amean"]:+.3f}')
    p(f'D1: {json.dumps(out["D1"])}')
    txt = '\n'.join(L)
    print(txt)
    open(os.path.join(SCORES, 'scorecard_confirm.txt'), 'w').write(txt + '\n')


# ------------------------------------------------------------------------------------------ diag
def diag(ctx):
    ch = json.load(open(os.path.join(SCORES, 'select.json')))['chosen']
    names = ('S1p', 'Gmean', 'Gmax', ch, 'zR0p', 'zKp')
    rows = []
    lab_all, y_all = ctx.meta['te_label'], ctx.meta['te_y']
    for s in ALL8:
        r = RUN(s)
        tr = load_npz(os.path.join(ctx.emb, 'train', f'{r}.npz'))
        bank = Bank(tr, ctx.meta['tr_label'])
        del tr
        for ds in ('dev', 'eval'):
            base = load_npz(os.path.join(ctx.emb, ds, f'{r}.npz'))
            rol = load_npz(os.path.join(ctx.emb, f'{ds}_ToyCar_rolls', f'{r}.npz'))
            idx = rol['idx']
            pos = {v: j for j, v in enumerate(base['idx'])}
            bi = np.array([pos[v] for v in idx])
            lab, y = lab_all[idx], y_all[idx]
            ks = [0] + list(rol['rolls'])
            per_k = [bank.score({k: base[k][bi] for k in ('ce', 'pre', 'gmean', 'gmax', 'bmean', 'bmax')}, lab)]
            for j in range(len(rol['rolls'])):
                per_k.append(bank.score({k: rol[k][:, j] for k in ('ce', 'pre', 'gmean', 'gmax', 'bmean', 'bmax')}, lab))
            for nm in names:
                s0 = per_k[0][nm]
                for k, sc in zip(ks, per_k):
                    pen = []
                    for c in np.unique(lab):
                        n = (lab == c) & (y == 0)
                        pen.append(np.mean(sc[n] - s0[n]) / s0[n].std())
                    auc = 100 * metrics.roc_auc_score(y, sc)
                    rows.append([r, ds, nm, k, f'{np.mean(pen):.4f}', f'{auc:.4f}'])
    with open(os.path.join(SCORES, 'diag_roll.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'dataset', 'score', 'k', 'penalty_sd', 'ToyCar_AUC'])
        w.writerows(rows)
    # headline: mean over seeds of the mean penalty at k = 3, 4, 5
    out = {}
    for ds in ('dev', 'eval'):
        o = {}
        for nm in names:
            sel = [x for x in rows if x[1] == ds and x[2] == nm]
            o[nm] = {'headline_penalty_sd': float(np.mean([float(x[4]) for x in sel if x[3] in (3, 4, 5)])),
                     'penalty_by_k': [float(np.mean([float(x[4]) for x in sel if x[3] == k])) for k in range(8)],
                     'AUC_by_k': [float(np.mean([float(x[5]) for x in sel if x[3] == k])) for k in range(8)]}
        g, s1 = o['Gmax']['headline_penalty_sd'], o['S1p']['headline_penalty_sd']
        o['verdict_Gmax'] = 'removes' if g <= 1 else 'reduces' if g <= 0.25 * s1 else 'does not'
        gm = o['Gmean']['headline_penalty_sd']
        o['verdict_Gmean_same_rule'] = 'removes' if gm <= 1 else 'reduces' if gm <= 0.25 * s1 else 'does not'
        out[ds] = o
    json.dump(out, open(os.path.join(SCORES, 'diag.json'), 'w'), indent=2)
    L = [f'ToyCar, 8 seeds; penalty of normal clips (sd of the unrolled score) and AUC, by roll k/8; chosen = {ch}']
    for ds, o in out.items():
        for nm in names:
            v = o[nm]
            L.append(f'{ds:4s} {nm:5s} headline {v["headline_penalty_sd"]:7.2f} sd | penalty ' +
                     ' '.join(f'{x:6.2f}' for x in v['penalty_by_k']) + ' | AUC ' + ' '.join(f'{x:5.1f}' for x in v['AUC_by_k']))
        L.append(f'{ds}: verdict zGmax: {o["verdict_Gmax"]}; zGmean: {o["verdict_Gmean_same_rule"]}')
    txt = '\n'.join(L)
    print(txt)
    open(os.path.join(SCORES, 'scorecard_diag.txt'), 'w').write(txt + '\n')


if __name__ == '__main__':
    stage, emb = sys.argv[1], sys.argv[2]
    os.makedirs(SCORES, exist_ok=True)
    {'select': select, 'confirm': confirm, 'diag': diag}[stage](Ctx(emb))
