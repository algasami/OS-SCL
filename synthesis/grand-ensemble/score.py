#!/usr/bin/env python
"""grand-ensemble: which ensemble of the already-dumped member pool is best, picked on DEV only?

    python synthesis/grand-ensemble/score.py repro   # reproduce E2′[A8], E2′[G8], F1k (dev + eval)
    python synthesis/grand-ensemble/score.py dev     # candidates + selection + optimism, dev ONLY
    python synthesis/grand-ensemble/score.py eval    # score the committed picks on eval, once

Rule (unchanged from phase 1, imported from ../../ablation/ensemble-backend/score.py):
  every from-scratch member gets S1′ = zR0′ + zK′ (per-ID z on its own training-part clips);
  BEATs members get the CE-free zK′ only (their train CE sits at the fp32 floor, so z(R0′) explodes);
  an ensemble is Σ_member w_member · rule(member), w = 1 unless a candidate says otherwise.

Importable API for other studies (fresh-confirm):

    import importlib.util
    spec = importlib.util.spec_from_file_location('ge', 'synthesis/grand-ensemble/score.py')
    ge = importlib.util.module_from_spec(spec); spec.loader.exec_module(ge)
    s = ge.ensemble_from_dumps([('/path/emb/SCLTFSTgramMFN_ao_nogain_s2060.npz', 'G'), ...],
                               weights=None)          # or {type: per-member weight}
    ev = ge.evaluator('dev'); ev(s)['overall']['mean_AUC_pAUC']

A dump is either an ensemble-backend extract.py npz (tr_ce, tr_pre, valid, te_ce, te_pre) or, for
timepool-head / pooled-knn extracts, the run's name inside an emb dir with train/ dev/ eval/
subdirectories (pass the path `<emb>/<run>`; see load_dump).
"""
import csv
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
ABL = os.path.join(REPO, 'ablation')
_spec = importlib.util.spec_from_file_location('eb_score', os.path.join(ABL, 'ensemble-backend', 'score.py'))
eb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eb)
MACH = eb.MACH

SCORES = os.environ.get('GE_SCORES', os.path.join(HERE, 'scores'))
CACHE = os.path.join(HERE, 'results', 'vec_cache')       # scratch (gitignored *.npz)
ROOT = '/home/f74134118'
N = 'SCLTFSTgramMFN_'
REF_META = f'{ROOT}/OS-SCL-litarms/ablation/lit-arms/results/emb/meta.npz'

# ---------------------------------------------------------------------------------- member pool
# type -> (dump dir, format, [runs]).  'flat' = ensemble-backend extract.py npz in dir;
# 'split' = timepool-head extract (dir/{train,dev,eval}/<run>.npz).
EMB = {
    'G':  f'{ROOT}/OS-SCL-augops/ablation/aug-ops/results/emb',
    'LA': f'{ROOT}/OS-SCL-litarms/ablation/lit-arms/results/emb',
    'S':  f'{ROOT}/OS-SCL-spec/ablation/spectrum-branch/results/emb',
    'T':  f'{ROOT}/OS-SCL-tphead/ablation/timepool-head/results/emb',
    'TS': f'{ROOT}/OS-SCL-soup/ablation/trunk-soup/results/emb',
    'PM': f'{ROOT}/OS-SCL-permix/ablation/permix/results/emb',
    'O':  f'{ROOT}/OS-SCL-confirm-seeds/ablation/confirm-seeds/results/emb',
    'B':  f'{ROOT}/OS-SCL-beats/ablation/beats/results/emb',
}
S4 = range(2040, 2044)
POOL = {   # type: (emb key, format, runs)        all 300 epochs (BEATs: 10k steps), fussion 1, m 0.4
    'G':  ('G', 'flat', [f'{N}ao_nogain_s{s}' for s in range(2040, 2048)]),   # fast + noise,shift (C2)
    'A':  ('LA', 'flat', [f'{N}la_aug_s{s}' for s in range(2040, 2048)]),     # fast + full aug
    'P':  ('LA', 'flat', [f'{N}fr_f1_s{s}' for s in S4]),                     # fast recipe, plain
    'Ga': ('LA', 'flat', [f'{N}la_gate_s{s}' for s in S4]),                   # fast + TriStat gate
    'E':  ('LA', 'flat', [f'{N}la_eca_s{s}' for s in S4]),                    # fast + ECA
    'S':  ('S', 'flat', [f'{N}sb_spec_s{s}' for s in S4]),                    # ao_nogain + |FFT| branch
    'T':  ('T', 'split', [f'{N}tp_s{s}' for s in S4]),                        # ao_nogain + time-tied linear7
    'PM': ('PM', 'flat', [f'{N}pm_s{s}' for s in S4]),                        # PerMix loss
    'TS': ('TS', 'flat', [f'{N}ts_soup4_s{s}' for s in (2050, 2051)]),        # one soup per trunk (forks ~ copies)
    'O':  ('O', 'flat', [f'{N}tfl_relu_max_s{s}' for s in range(2024, 2028)]  # old recipe, relu8
           + [f'{N}cs_relu_max_s{s}' for s in range(2028, 2032)]),
    'B':  ('B', 'flat', [f'BEATsASD_bt_arc_s{s}' for s in (2040, 2041, 2042)]),  # EXTERNAL PRETRAINING
}
SCRATCH_TYPES = ['G', 'A', 'P', 'Ga', 'E', 'S', 'T', 'PM', 'TS', 'O']
EXTERNAL = {'B'}
RULE = {t: ('zKp' if t in EXTERNAL else 'S1p') for t in POOL}
KEYS = ('AUC', 'pAUC', 'mAUC', 'mean_AUC_pAUC')
N_BOOT, RNG_SEED = 2000, 0


def run_type(run):
    for t, (_, _, runs) in POOL.items():
        if run in runs:
            return t
    raise KeyError(run)


def dump_path(run):
    t = run_type(run)
    key, fmt, _ = POOL[t]
    return os.path.join(EMB[key], run if fmt == 'split' else f'{run}.npz')


# ---------------------------------------------------------------------------------- loading
def below(a):
    return [p.split('/data/', 1)[1] for p in a]


def check_meta(path, ref=None):
    """Assert a dump dir's meta.npz lists the same clips in the same order below data/ as the reference."""
    a = dict(np.load(path))
    ref = ref or dict(np.load(REF_META))
    for k in ref:
        same = below(a[k]) == below(ref[k]) if k.endswith('paths') else np.array_equal(a[k], ref[k])
        assert same, f'meta mismatch on {k}: {path}'
    return a


_META = None


def meta():
    global _META
    if _META is None:
        _META = dict(np.load(REF_META))
    return _META


def _rng_check(path, k, v):
    v32 = v.astype(np.float32)
    lim = 1e3 if k.endswith('ce') else 1e4      # faulty host RAM: finite and sane magnitude
    assert np.isfinite(v32).all() and np.abs(v32).max() < lim, (path, k, float(np.nanmax(np.abs(v32))))


def load_dump(path):
    """-> dict(tr_ce, tr_pre, valid, te_ce, te_pre) for one member, range-checked.

    `path` is an extract.py npz, or `<emb>/<run>` for a split (train/dev/eval) timepool-head dump.
    """
    if path.endswith('.npz'):
        a = dict(np.load(path))
        out = {k: a[k] for k in ('tr_ce', 'tr_pre', 'valid', 'te_ce', 'te_pre')}
    else:
        emb, run = os.path.dirname(path), os.path.basename(path)
        tr = dict(np.load(os.path.join(emb, 'train', f'{run}.npz')))
        assert np.array_equal(tr['idx'], np.arange(len(tr['idx']))), 'train dump not in meta order'
        assert len(tr['rolls']) == 1 and tr['rolls'][0] == 0, 'train dump is a rolled copy'
        nte = len(meta()['te_label'])
        te_ce, te_pre = np.full(nte, np.nan, np.float32), np.full((nte, tr['pre'].shape[1]), np.nan, np.float32)
        have = np.zeros(nte, bool)
        for ds in ('dev', 'eval'):
            d = dict(np.load(os.path.join(emb, ds, f'{run}.npz')))
            assert len(d['rolls']) == 1 and d['rolls'][0] == 0
            te_ce[d['idx']], te_pre[d['idx']] = d['ce'], d['pre']
            assert not have[d['idx']].any()
            have[d['idx']] = True
        assert have.all(), (path, int((~have).sum()))
        out = dict(tr_ce=tr['ce'], tr_pre=tr['pre'], valid=tr['valid'], te_ce=te_ce, te_pre=te_pre)
    for k in ('tr_ce', 'tr_pre', 'te_ce', 'te_pre'):
        _rng_check(path, k, out[k])
    m = meta()
    assert len(out['tr_ce']) == len(m['tr_label']) and len(out['te_ce']) == len(m['te_label']), path
    assert 0.03 < out['valid'].mean() < 0.07, (path, 'valid fraction')
    return out


def member_rules(path, use_cache=True, cache_key=None):
    """{'S1p', 'zKp', 'zR0p', 'S0'} score vectors over every test clip, for one dump (cached in results/)."""
    key = cache_key or os.path.basename(path).replace('.npz', '')
    cp = os.path.join(CACHE, f'{key}.npz')
    if use_cache and os.path.exists(cp):
        try:
            c = dict(np.load(cp))
            if c.get('src', np.array('')).item() == path and all(np.isfinite(c[k]).all() for k in ('S1p', 'zKp')):
                return {k: c[k] for k in ('S1p', 'zKp', 'zR0p', 'S0')}
        except Exception as e:      # corrupt cache (faulty RAM / disk): recompute
            print(f'cache {cp} unreadable ({e}); recomputing', file=sys.stderr)
    with np.errstate(divide='ignore', invalid='ignore'):     # hold-out zR0 of BEATs divides by ~0; unused
        r = eb.per_run_scores(meta(), load_dump(path))
    out = {k: r[k] for k in ('S1p', 'zKp', 'zR0p', 'S0')}
    for k in ('zKp', 'S0'):
        assert np.isfinite(out[k]).all(), (path, k)
    if use_cache:
        os.makedirs(CACHE, exist_ok=True)
        np.savez(cp, src=np.array(path), **out)
    return out


def member_vector(path, typ, **kw):
    return member_rules(path, **kw)[RULE.get(typ, 'S1p')]


def ensemble_from_dumps(members, weights=None, **kw):
    """THE ADOPTED RULE. members: [(dump path, type)], type in POOL (e.g. 'G', 'T', 'S', 'B').

    weights: None (every member weight 1) or {type: per-member weight}. Returns Σ w·rule(member)
    over every dev+eval test clip in meta order. BEATs ('B') members use zK′, all others S1′.
    """
    s = 0.0
    for path, typ in members:
        w = 1.0 if weights is None else float(weights.get(typ, 0.0))
        if w:
            s = s + w * member_vector(path, typ, **kw)
    return s


def evaluator(ds):
    return eb.Evaluator(meta(), ds)


def flat(res):
    o = res['overall']
    d = {k: o[k] for k in KEYS}
    for m in MACH:
        d[m] = 100 * (res[m]['AUC'] + res[m]['pAUC']) / 2
        d[f'{m}_AUC'] = 100 * res[m]['AUC']
    return d


# ---------------------------------------------------------------------------------- candidates
def all_vectors():
    """{type: [member vector]} for the whole pool, every dump dir's meta checked first."""
    ref = meta()
    for key, d in EMB.items():
        check_meta(os.path.join(d, 'meta.npz'), ref)
    V = {}
    for t, (_, _, runs) in POOL.items():
        V[t] = [member_vector(dump_path(r), t) for r in runs]
        for v in V[t]:
            assert v.shape == ref['te_label'].shape and np.isfinite(v).all(), t
    return V


def system(V, spec):
    """spec: {type: per-member weight} -> Σ_t w_t Σ_m v_m."""
    return sum(w * sum(V[t]) for t, w in spec.items() if w)


def block(V, t):
    """Type block for greedy selection: the type's MEAN member vector (equal weight per type-step)."""
    return sum(V[t]) / len(V[t])


def n_members(spec):
    return sum(len(POOL[t][2]) for t, w in spec.items() if w)


# ---------------------------------------------------------------------------------- io helpers
def write_csv(name, header, rows):
    os.makedirs(SCORES, exist_ok=True)
    with open(os.path.join(SCORES, name), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fmt(x):
    return f'{x:.4f}' if isinstance(x, (float, np.floating)) else x


# ---------------------------------------------------------------------------------- stage: repro
COMMITTED_SYSTEMS = {   # name: (spec, csv, row key)
    'A8':    ({'A': 1}, 'nogain-ensemble/scores/systems_{}.csv'),
    'G8':    ({'G': 1}, 'nogain-ensemble/scores/systems_{}.csv'),
    'GA16':  ({'G': 1, 'A': 1}, 'nogain-ensemble/scores/systems_{}.csv'),
    'F20g':  ({'G': 1, 'P': 1, 'Ga': 1, 'E': 1}, 'nogain-ensemble/scores/systems_{}.csv'),
    'F20':   ({'A': 1, 'P': 1, 'Ga': 1, 'E': 1}, 'nogain-ensemble/scores/systems_{}.csv'),
    'relu8': ({'O': 1}, 'nogain-ensemble/scores/systems_{}.csv'),
    'F1k':   ({'A': 1, 'B': 1}, 'beats/scores/systems_{}.csv'),
}


def committed_amean(name, rel, ds):
    with open(os.path.join(ABL, rel.format(ds))) as f:
        for r in csv.DictReader(f):
            if r['system'] == name and r.get('scope', 'overall') == 'overall':
                return float(r.get('mean_AUC_pAUC') or r['Amean'])
    raise KeyError(name)


def committed_member_s1p():
    """{(run, ds): (AUC, pAUC)} of single-member S1′ from the ablations that produced them."""
    out = {}
    with open(os.path.join(ABL, 'aug-ops/scores/s1p.csv')) as f:
        for r in csv.DictReader(f):
            if r['rule'] == 'S1p':
                out[(r['run'], r['dataset'])] = (float(r['AUC']), float(r['pAUC']))
    with open(os.path.join(ABL, 'spectrum-branch/scores/s1p.csv')) as f:
        for r in csv.DictReader(f):
            if r['rule'] == 'S1p' and r['machine'] == 'ALL':
                out[(r['run'], r['dataset'])] = (float(r['AUC']), float(r['pAUC']))
    with open(os.path.join(ABL, 'permix/scores/s1p.csv')) as f:
        for r in csv.DictReader(f):
            if r['rule'] == 'S1p' and r['machine'] == 'ALL' and r['id'] == 'ALL':
                out[(r['run'], r['dataset'])] = (float(r['AUC']), float(r['pAUC']))
    with open(os.path.join(ABL, 'timepool-head/scores/metrics.csv')) as f:
        for r in csv.DictReader(f):
            if r['rule'] == 'S1p' and r['machine'] == 'overall' and r['arm'] == 'tp':
                out[(r['run'], r['dataset'])] = (float(r['AUC']), float(r['pAUC']))
    with open(os.path.join(ABL, 'trunk-soup/scores/systems.csv')) as f:
        for r in csv.DictReader(f):
            if r['rule'] == 'S1p':
                out[(r['system'], r['dataset'])] = (float(r['AUC']), float(r['pAUC']))
    return out


KNOWN_BAD = {   # committed single-member records that the CRC-verified dumps do not reproduce (see README)
    ('SCLTFSTgramMFN_tp_s2041', 'eval'):
        'only eval ToyConveyor S1′ differs (R0 matches; dump zip CRCs pass; timepool-head\'s own run_scores '
        'on the same dump gives this study\'s value) -> a faulty-RAM event during timepool-head scoring',
}


def stage_repro():
    V = all_vectors()
    ev = {ds: evaluator(ds) for ds in ('dev', 'eval')}
    rows, L = [], ['P0 reproduction: committed phase-1 numbers recomputed from the dumps (Amean = mean(AUC, pAUC))']
    worst = 0.0
    for name, (spec, rel) in COMMITTED_SYSTEMS.items():
        for ds in ev:
            got = flat(ev[ds](system(V, spec)))['mean_AUC_pAUC']
            com = committed_amean(name, rel, ds)
            worst = max(worst, abs(got - com))
            rows.append([name, ds, f'{got:.4f}', f'{com:.4f}', f'{got - com:+.1e}', rel.format(ds)])
            L.append(f'  {name:6s} {ds:4s} {got:.4f} vs committed {com:.4f} (d {got - com:+.1e})  [{rel.format(ds)}]')
    com_m = committed_member_s1p()
    mrows, mworst = [], 0.0
    for t in POOL:
        for r, v in zip(POOL[t][2], V[t]):
            for ds in ev:
                f_ = flat(ev[ds](v))
                c = com_m.get((r, ds))
                d = max(abs(f_['AUC'] - c[0]), abs(f_['pAUC'] - c[1])) if c else float('nan')
                if c and (r, ds) in KNOWN_BAD:
                    L.append(f'  KNOWN: {r} {ds} S1′ AUC {f_["AUC"]:.4f} pAUC {f_["pAUC"]:.4f} vs committed '
                             f'{c[0]:.4f} {c[1]:.4f} (|d| {d:.1e}); {KNOWN_BAD[(r, ds)]}')
                elif c:
                    mworst = max(mworst, d)
                mrows.append([t, r, RULE[t], ds] + [fmt(f_[k]) for k in KEYS] + [fmt(f_[f'{m}_AUC']) for m in MACH]
                             + ['' if c is None else f'{d:.1e}'])
    write_csv('members.csv', ['type', 'run', 'rule', 'dataset'] + list(KEYS) + [f'{m}_AUC' for m in MACH]
              + ['max_abs_diff_vs_committed'], mrows)
    crc_bad = []
    for t in POOL:      # zip CRC of every dump file: a failure is hardware (smartlab1 RAM), never trusted
        for r in POOL[t][2]:
            p = dump_path(r)
            files = [p] if p.endswith('.npz') else [os.path.join(os.path.dirname(p), d, f'{r}.npz')
                                                     for d in ('train', 'dev', 'eval')]
            for fpath in files:
                if zipfile.ZipFile(fpath).testzip() is not None:
                    crc_bad.append(fpath)
    L.append(f'  zip CRC over every dump file: {"all pass" if not crc_bad else "FAIL " + " ".join(crc_bad)}')
    write_csv('reproduction.csv', ['system', 'dataset', 'recomputed_Amean', 'committed_Amean', 'diff', 'source'], rows)
    L.append(f'  worst |d| systems {worst:.1e}; single-member S1′ (AUC, pAUC) vs committed, worst {mworst:.1e} '
             f'over {sum(1 for r in mrows if r[-1]) - len(KNOWN_BAD)} member-sets (KNOWN excluded)')
    L.append(f'  PASS: {worst < 5e-3 and mworst < 5e-3 and not crc_bad}')
    txt = '\n'.join(L)
    open(os.path.join(SCORES, 'reproduction.txt'), 'w').write(txt + '\n')
    print(txt)
    assert worst < 5e-3 and mworst < 5e-3 and not crc_bad, 'reproduction failed: stop'


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    {'repro': stage_repro}[stage]()
