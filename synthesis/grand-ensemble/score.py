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


# ---------------------------------------------------------------------------------- candidates (README, fixed)
FAST = ['G', 'A', 'P', 'Ga', 'E', 'S', 'T', 'PM', 'TS']
CANDS = {   # name: ('member', {type: per-member weight})  -- order = README table order (tie-break)
    'G8':     ('member', {'G': 1}),
    'GA16':   ('member', {'G': 1, 'A': 1}),
    'GS12':   ('member', {'G': 1, 'S': 1}),
    'GT12':   ('member', {'G': 1, 'T': 1}),
    'GST16':  ('member', {'G': 1, 'S': 1, 'T': 1}),
    'GAST24': ('member', {'G': 1, 'A': 1, 'S': 1, 'T': 1}),
    'NG18':   ('member', {'G': 1, 'S': 1, 'T': 1, 'TS': 1}),
    'FAST42': ('member', {t: 1 for t in FAST}),
    'ALL50':  ('member', {t: 1 for t in FAST + ['O']}),
}
ORDER = list(CANDS) + ['GREEDY']
SEEN_EVAL = {'G8': 'ablation/nogain-ensemble', 'GA16': 'ablation/nogain-ensemble'}
BARS = {'A8': ('member', {'A': 1}), 'G8': ('member', {'G': 1})}
R_GRID = (3 / 8, 3 / 4, 3 / 2)
GREEDY_MIN_GAIN, GREEDY_MAX_STEPS, TIE = 0.02, 10, 0.05
N_SPLITS = 20


def build(V, spec):
    """spec = ('member', {t: w per member}) or ('block', {t: count of mean-blocks}); missing types ignored."""
    kind, d = spec
    if kind == 'member':
        return sum(w * sum(V[t]) for t, w in d.items() if w and V.get(t))
    return sum(c * sum(V[t]) / len(V[t]) for t, c in d.items() if c and V.get(t))


def per_member_weights(V, spec):
    kind, d = spec
    return {t: (float(w) if kind == 'member' else w / len(V[t])) for t, w in d.items() if w and V.get(t)}


def total_weight(V, spec):
    return sum(w * len(V[t]) for t, w in per_member_weights(V, spec).items())


def n_used(V, spec):
    return sum(len(V[t]) for t in per_member_weights(V, spec))


def amean(ev, score, idx=None):
    return ev(score, idx)['overall']['mean_AUC_pAUC']


def greedy(V, ev, idx=None):
    """Caruana forward selection with replacement over type blocks (README). -> (spec, trace)."""
    blocks = {t: sum(V[t]) / len(V[t]) for t in SCRATCH_TYPES if V.get(t)}
    cur, counts, best, trace = 0.0, {}, -np.inf, []
    for step in range(GREEDY_MAX_STEPS):
        val, pick = -np.inf, None
        for t in SCRATCH_TYPES:          # strict > keeps the earlier type on ties
            if t in blocks:
                a = amean(ev, cur + blocks[t], idx)
                if a > val:
                    val, pick = a, t
        if step > 0 and val - best < GREEDY_MIN_GAIN:
            trace.append(dict(step=step + 1, type=pick, amean=val, gain=val - best, taken=False))
            break
        cur = cur + blocks[pick]
        counts[pick] = counts.get(pick, 0) + 1
        trace.append(dict(step=step + 1, type=pick, amean=val, gain=(val - best) if step else None, taken=True))
        best = val
    return ('block', counts), trace


def select(V, ev, idx=None):
    """Score every candidate (GREEDY re-run) on (a subset of) dev; apply the pick rule."""
    specs = dict(CANDS)
    specs['GREEDY'], trace = greedy(V, ev, idx)
    val = {n: amean(ev, build(V, specs[n]), idx) for n in ORDER}
    top = max(val.values())
    close = [n for n in ORDER if top - val[n] <= TIE]
    pick = min(close, key=lambda n: (n_used(V, specs[n]), ORDER.index(n)))
    return pick, specs, val, trace


def halves(V, h):
    return {t: (v[:len(v) // 2] if h == 0 else v[len(v) // 2:]) for t, v in V.items()}


def split_idx(ev, seed):
    rng = np.random.default_rng(seed)
    a, b = [], []
    for ix in ev.strata.values():
        p = rng.permutation(ix)
        a.append(p[:len(p) // 2])
        b.append(p[len(p) // 2:])
    return np.concatenate(a), np.concatenate(b)


def x1_specs(V, x0spec):
    """X0 pick + BEATs block at ratio r of the X0 total weight (per BEATs member r*W/3)."""
    W = total_weight(V, x0spec)
    out = {}
    for r in R_GRID:
        w = per_member_weights(V, x0spec)
        w['B'] = r * W / len(V['B'])
        out[f'X1[r={r:g}]'] = ('member', w)
    return out


# ---------------------------------------------------------------------------------- parallel bootstrap
_BOOT = {}


def _boot_chunk(chunk):
    ev, vecs, pairs, idxs = _BOOT['ev'], _BOOT['vecs'], _BOOT['pairs'], _BOOT['idxs']
    out = []
    for i in chunk:
        o = {n: ev(v, idxs[i])['overall'] for n, v in vecs.items()}
        out.append({f'{a}-{b}': {k: o[a][k] - o[b][k] for k in KEYS} for a, b in pairs})
    return out


def paired_boot(ev, vecs, pairs, n_boot=N_BOOT, seed=RNG_SEED, procs=4):
    import multiprocessing as mp
    rng = np.random.default_rng(seed)
    idxs = [ev.resample(rng) for _ in range(n_boot)]
    _BOOT.update(ev=ev, vecs=vecs, pairs=pairs, idxs=idxs)
    chunks = [list(range(i, n_boot, procs * 4)) for i in range(procs * 4)]
    with mp.get_context('fork').Pool(procs) as pool:
        res = [r for part in pool.map(_boot_chunk, chunks) for r in part]
    out = {}
    for a, b in pairs:
        key = f'{a}-{b}'
        out[key] = {k: [float(np.mean([r[key][k] for r in res])),
                        *map(float, np.percentile([r[key][k] for r in res], [2.5, 97.5]))] for k in KEYS}
    return out


def system_rows(name, n, ds, res):
    o = res['overall']
    rows = [[name, n, ds, 'overall', '', fmt(o['AUC']), fmt(o['pAUC']), fmt(o['mAUC']), fmt(o['mean_AUC_pAUC'])]]
    for m in MACH:
        A, P = 100 * res[m]['AUC'], 100 * res[m]['pAUC']
        rows.append([name, n, ds, m, '', fmt(A), fmt(P), fmt(100 * res[m]['mAUC']), fmt((A + P) / 2)])
        for i, a in sorted(res[m]['ids'].items()):
            rows.append([name, n, ds, m, i, fmt(100 * a), '', '', ''])
    return rows


SYS_HEADER = ['system', 'n', 'dataset', 'scope', 'id', 'AUC', 'pAUC', 'mAUC', 'Amean']


def type_corr(V, ev):
    """Mean over machine IDs of the Pearson correlation between type-mean scores on one set's clips."""
    types = [t for t in POOL if V.get(t)]
    M = {t: (sum(V[t]) / len(V[t]))[ev.sel] for t in types}
    C = np.zeros((len(types), len(types)))
    groups = [np.where((ev.mach == m) & (ev.id == i))[0] for m in MACH for i in np.unique(ev.id[ev.mach == m])]
    for g in groups:
        C += np.corrcoef(np.stack([M[t][g] for t in types]))
    return types, C / len(groups)


def stage_dev():
    V = all_vectors()
    ev = evaluator('dev')
    L = ['DEV ONLY (selection set). E2′ = Σ w·S1′ per member (BEATs: zK′). Amean = mean(AUC, pAUC).', '']
    pick, specs, val, trace = select(V, ev)
    res = {n: ev(build(V, specs[n])) for n in ORDER}
    bars = {'A8': ev(build(V, BARS['A8']))}
    rows = []
    for n in ORDER:
        rows += system_rows(n, n_used(V, specs[n]), 'dev', res[n])
    rows += system_rows('A8', 8, 'dev', bars['A8'])
    write_csv('systems_dev.csv', SYS_HEADER, rows)
    L.append(f'{"cand":8s} {"n":>3s}  {"AUC":>6s} {"pAUC":>6s} {"mAUC":>6s} {"Amean":>6s} | ' + ' '.join(f'{m[:7]:>7s}' for m in MACH)
             + '   (per machine: AUC)')
    for n, r in list(res.items()) + [('A8 (bar)', bars['A8'])]:
        o = r['overall']
        nn = n_used(V, specs[n]) if n in specs else 8
        L.append(f'{n:8s} {nn:3d}  ' + ' '.join(f'{o[k]:6.2f}' for k in KEYS) + ' | '
                 + ' '.join(f'{100 * r[m]["AUC"]:7.2f}' for m in MACH))
    L += ['', 'GREEDY trace (blocks = type-mean S1′; stop when best gain < +0.02):']
    for t in trace:
        L.append(f'  step {t["step"]}: +{t["type"]:3s} -> {t["amean"]:.4f}'
                 + (f' (gain {t["gain"]:+.4f})' if t['gain'] is not None else '') + ('' if t['taken'] else '  NOT TAKEN (stop)'))
    L.append(f'  GREEDY = {specs["GREEDY"][1]}  per-member weights {per_member_weights(V, specs["GREEDY"])}')
    L += ['', f'X0 PICK (highest dev Amean; tie within {TIE} -> smaller n): {pick}  dev Amean {val[pick]:.4f} '
          f'(G8 {val["G8"]:.4f}, Δ {val[pick] - val["G8"]:+.4f}; A8 {bars["A8"]["overall"]["mean_AUC_pAUC"]:.4f})']

    # ---- X1 (with external pretraining)
    x1 = x1_specs(V, specs[pick])
    x1res = {n: ev(build(V, s)) for n, s in x1.items()}
    f1k = ev(build(V, ('member', {'A': 1, 'B': 1})))
    x1val = {n: r['overall']['mean_AUC_pAUC'] for n, r in x1res.items()}
    top = max(x1val.values())
    x1pick = [n for n in x1 if top - x1val[n] <= TIE][0]      # grid is in increasing r: first = smallest r
    rows = []
    for n, r in x1res.items():
        rows += system_rows(n, n_used(V, x1[n]), 'dev', r)
    rows += system_rows('F1k', 11, 'dev', f1k)
    write_csv('x1_dev.csv', SYS_HEADER, rows)
    L += ['', 'X1 (WITH EXTERNAL PRETRAINING): X0 pick + 3 BEATs (zK′), BEATs block = r × X0 total weight']
    for n, v in x1val.items():
        L.append(f'  {n:12s} dev Amean {v:.4f}  per-BEATs-member w {x1[n][1]["B"]:.3f}')
    L.append(f'  F1k (A8 + B3, bar) dev Amean {f1k["overall"]["mean_AUC_pAUC"]:.4f}')
    L.append(f'  X1 PICK: {x1pick}')

    # ---- O1: clip split-half optimism
    o1 = []
    for sd in range(N_SPLITS):
        ia, ib = split_idx(ev, sd)
        pk, sp, va, _ = select(V, ev, ia)
        da = va[pk] - va['G8']
        db = amean(ev, build(V, sp[pk]), ib) - amean(ev, build(V, sp['G8']), ib)
        o1.append(dict(seed=sd, pick=pk, spec=sp[pk][1], delta_A=da, delta_B=db))
    opt = float(np.mean([r['delta_A'] - r['delta_B'] for r in o1]))
    freq = {n: sum(r['pick'] == n for r in o1) for n in ORDER}
    honest = (val[pick] - val['G8']) - opt
    L += ['', f'O1 clip split-half ({N_SPLITS} splits): mean Δ_A(pick−G8) {np.mean([r["delta_A"] for r in o1]):+.4f}, '
          f'mean Δ_B {np.mean([r["delta_B"] for r in o1]):+.4f} -> optimism {opt:+.4f}; '
          f'honest full-dev margin {val[pick] - val["G8"]:+.4f} − {opt:+.4f} = {honest:+.4f}',
          f'  half-A picks: {freq}   full-dev pick chosen in {freq[pick]}/{N_SPLITS}']

    # ---- O2: seed halves
    o2 = []
    for h in (0, 1):
        Vs, Vo = halves(V, h), halves(V, 1 - h)
        pk, sp, va, _ = select(Vs, ev)
        d_sel = va[pk] - va['G8']
        d_oth = amean(ev, build(Vo, sp[pk])) - amean(ev, build(Vo, sp['G8']))
        o2.append(dict(select_half=h + 1, pick=pk, spec=sp[pk][1], delta_select=d_sel, delta_other=d_oth))
        L.append(f'O2 seed half H{h + 1}: pick {pk} {sp[pk][1]}  Δ(pick−G4) on H{h + 1} {d_sel:+.4f}, '
                 f'rebuilt on H{2 - h} {d_oth:+.4f}')

    # ---- size-matched reduction for fresh-confirm
    W = {t: w * len(V[t]) for t, w in per_member_weights(V, specs[pick]).items()}
    others = {t: w for t, w in W.items() if t not in ('G', 'TS')}
    sm = dict(W_t=W)
    if 'G' not in W:
        sm.update(build='UNDEFINED: pick has no G', X=None)
    elif not others:
        sm.update(build='8G', X=None, per_member_weights={'G': 1.0})
    else:
        top_w = max(others.values())
        tied = [t for t, w in others.items() if abs(w - top_w) < 1e-9]
        if len(tied) > 1:
            full = amean(ev, build(V, specs[pick]))
            cost = {t: full - amean(ev, build(V, ('member', {k: v for k, v in per_member_weights(V, specs[pick]).items()
                                                               if k != t}))) for t in tied}
            X = max(tied, key=lambda t: cost[t])
            sm['tie_removal_cost'] = cost
        else:
            X = tied[0]
        wg, wx = W['G'] / 4, W[X] / 4
        sm.update(build=f'4G+4{X}', X=X, per_member_weights={'G': wg, X: wx})
        gsub = list(itertools.combinations(range(len(V['G'])), 4))
        xsub = list(itertools.combinations(range(len(V[X])), min(4, len(V[X]))))
        combos = [(g, x) for g in gsub for x in xsub]
        rng = np.random.default_rng(RNG_SEED)
        if len(combos) > 1000:
            combos = [combos[i] for i in sorted(rng.choice(len(combos), 1000, replace=False))]
        red = [amean(ev, wg * sum(V['G'][i] for i in g) + wx * sum(V[X][i] for i in x)) for g, x in combos]
        g4 = [amean(ev, sum(V['G'][i] for i in g)) for g in gsub]
        sm['dev_check'] = dict(n_combos=len(combos), reduced_mean=float(np.mean(red)),
                               reduced_p05=float(np.percentile(red, 5)), reduced_p95=float(np.percentile(red, 95)),
                               hom4_G_mean=float(np.mean(g4)), hom4_G_p05=float(np.percentile(g4, 5)),
                               hom4_G_p95=float(np.percentile(g4, 95)), G8=val['G8'],
                               reduced_minus_hom4G=float(np.mean(red) - np.mean(g4)))
        L.append(f'size-matched {sm["build"]} (w G {wg:.3f}, {X} {wx:.3f}) over {len(combos)} subsets: dev Amean '
                 f'{np.mean(red):.4f} [{np.percentile(red, 5):.4f}, {np.percentile(red, 95):.4f}] vs 4-G subsets '
                 f'{np.mean(g4):.4f} -> {np.mean(red) - np.mean(g4):+.4f}; G8 {val["G8"]:.4f}')

    # ---- diversity diagnostic
    types, C = type_corr(V, ev)
    write_csv('type_corr_dev.csv', ['type'] + types, [[t] + [f'{c:.4f}' for c in row] for t, row in zip(types, C)])

    # ---- dev bootstraps (descriptive)
    vecs = {n: build(V, specs[n]) for n in ORDER}
    vecs['A8'] = build(V, BARS['A8'])
    pairs = [(n, 'G8') for n in ORDER if n != 'G8'] + [(n, 'A8') for n in ORDER]
    for n in x1:
        vecs[n] = build(V, x1[n])
    vecs['F1k'] = build(V, ('member', {'A': 1, 'B': 1}))
    pairs += [(n, 'F1k') for n in x1] + [(n, pick) for n in x1]
    boot = paired_boot(ev, vecs, pairs)
    json.dump(boot, open(os.path.join(SCORES, 'bootstrap_dev.json'), 'w'), indent=2)
    L += ['', f'Paired clip-bootstrap ({N_BOOT}) on dev, Amean: mean [2.5%, 97.5%]']
    for k, v in boot.items():
        L.append(f'  {k:22s} {v["mean_AUC_pAUC"][0]:+.3f} [{v["mean_AUC_pAUC"][1]:+.3f}, {v["mean_AUC_pAUC"][2]:+.3f}]')

    sel = dict(x0_pick=pick, x0_spec=list(specs[pick]), x0_per_member_weights=per_member_weights(V, specs[pick]),
               x0_n=n_used(V, specs[pick]), dev_amean=val, dev_A8=bars['A8']['overall']['mean_AUC_pAUC'],
               greedy_spec=specs['GREEDY'][1], greedy_trace=trace,
               x1_pick=x1pick, x1_specs={n: s[1] for n, s in x1.items()}, x1_dev_amean=x1val,
               dev_F1k=f1k['overall']['mean_AUC_pAUC'],
               O1=dict(splits=o1, optimism=opt, honest_margin_vs_G8=honest, pick_freq=freq),
               O2=o2, size_matched=sm, seen_on_eval=SEEN_EVAL)
    json.dump(sel, open(os.path.join(SCORES, 'selection_dev.json'), 'w'), indent=2, default=float)
    txt = '\n'.join(L)
    open(os.path.join(SCORES, 'scorecard_dev.txt'), 'w').write(txt + '\n')
    print(txt)


def committed_clean(path):
    rel = os.path.relpath(path, REPO)
    tracked = subprocess.run(['git', '-C', REPO, 'ls-files', '--error-unmatch', rel], capture_output=True).returncode == 0
    clean = subprocess.run(['git', '-C', REPO, 'diff', '--quiet', 'HEAD', '--', rel]).returncode == 0
    return tracked and clean


def stage_eval():
    sel_path = os.path.join(SCORES, 'selection_dev.json')
    assert committed_clean(sel_path) and committed_clean(os.path.join(SCORES, 'dev_pick.json')), \
        'selection_dev.json / dev_pick.json must be committed and unmodified before eval is scored'
    sel = json.load(open(sel_path))
    V = all_vectors()
    ev = evaluator('eval')
    pick, x1pick = sel['x0_pick'], sel['x1_pick']
    specs = dict(CANDS)
    specs['GREEDY'] = ('block', sel['greedy_spec'])            # the committed dev result, not re-run
    assert list(specs[pick]) == sel['x0_spec'] or pick == 'GREEDY'
    x1 = {n: ('member', w) for n, w in sel['x1_specs'].items()}
    L = ['EVAL, SCORED ONCE with the committed dev picks. Amean = mean(AUC, pAUC).', '']
    res = {n: ev(build(V, specs[n])) for n in ORDER}
    res['A8'] = ev(build(V, BARS['A8']))
    rows = []
    for n in ORDER:
        rows += system_rows(n, n_used(V, specs[n]), 'eval', res[n])
    rows += system_rows('A8', 8, 'eval', res['A8'])
    # GREEDY prefixes (descriptive size curve)
    prefix, counts = [], {}
    for t in sel['greedy_trace']:
        if t['taken']:
            counts = dict(counts)
            counts[t['type']] = counts.get(t['type'], 0) + 1
            nm = 'GREEDY@' + str(t['step'])
            prefix.append(dict(step=t['step'], type=t['type'], dev=t['amean'],
                               eval=amean(ev, build(V, ('block', counts)))))
            rows += system_rows(nm, n_used(V, ('block', counts)), 'eval', ev(build(V, ('block', counts))))
    write_csv('systems_eval.csv', SYS_HEADER, rows)
    write_csv('greedy_trace.csv', ['step', 'type', 'dev_Amean', 'eval_Amean'],
              [[p_['step'], p_['type'], fmt(p_['dev']), fmt(p_['eval'])] for p_ in prefix])
    x1res = {n: ev(build(V, s_)) for n, s_ in x1.items()}
    x1res['F1k'] = ev(build(V, ('member', {'A': 1, 'B': 1})))
    rows = []
    for n, r in x1res.items():
        rows += system_rows(n, n_used(V, x1[n]) if n in x1 else 11, 'eval', r)
    write_csv('x1_eval.csv', SYS_HEADER, rows)

    L.append(f'{"system":12s} {"n":>3s}  {"AUC":>6s} {"pAUC":>6s} {"mAUC":>6s} {"Amean":>6s} | '
             + ' '.join(f'{m[:7]:>7s}' for m in MACH) + '   (per machine: AUC)   dev Amean')
    for n, r in list(res.items()):
        o = r['overall']
        nn = n_used(V, specs[n]) if n in specs else 8
        dv = sel['dev_amean'].get(n, sel['dev_A8'])
        L.append(f'{n:12s} {nn:3d}  ' + ' '.join(f'{o[k]:6.2f}' for k in KEYS) + ' | '
                 + ' '.join(f'{100 * r[m]["AUC"]:7.2f}' for m in MACH) + f'   {dv:.2f}'
                 + ('   <- X0 PICK' if n == pick else ''))
    L += ['', 'GREEDY prefixes (descriptive): ' + '  '.join(f'@{p_["step"]}+{p_["type"]} dev {p_["dev"]:.2f} eval {p_["eval"]:.2f}'
                                                        for p_ in prefix)]
    L += ['', 'WITH EXTERNAL PRETRAINING:']
    for n, r in x1res.items():
        o = r['overall']
        L.append(f'  {n:12s} ' + ' '.join(f'{o[k]:6.2f}' for k in KEYS) + ' | '
                 + ' '.join(f'{100 * r[m]["AUC"]:7.2f}' for m in MACH)
                 + f'   dev {sel["x1_dev_amean"].get(n, sel["dev_F1k"]):.2f}' + ('   <- X1 PICK' if n == x1pick else ''))

    # ---- bootstraps
    vecs = {n: build(V, specs[n]) for n in ORDER}
    vecs['A8'] = build(V, BARS['A8'])
    vecs.update({n: build(V, s_) for n, s_ in x1.items()})
    vecs['F1k'] = build(V, ('member', {'A': 1, 'B': 1}))
    pairs = [(n, 'A8') for n in ORDER] + [(n, 'G8') for n in ORDER if n != 'G8'] \
        + [(n, 'F1k') for n in x1] + [(x1pick, pick)]
    boot = paired_boot(ev, vecs, pairs)
    json.dump(boot, open(os.path.join(SCORES, 'bootstrap_eval.json'), 'w'), indent=2)
    L += ['', f'Paired clip-bootstrap ({N_BOOT}) on eval, Amean: point [2.5%, 97.5%]']
    pt = {n: r['overall']['mean_AUC_pAUC'] for n, r in list(res.items()) + list(x1res.items())}
    for k, v in boot.items():
        a, b = k.split('-', 1)
        L.append(f'  {k:22s} {pt[a] - pt[b]:+.3f} [{v["mean_AUC_pAUC"][1]:+.3f}, {v["mean_AUC_pAUC"][2]:+.3f}]')

    # ---- decisions
    d1 = dict(pick=pick, new=pick not in SEEN_EVAL, eval_delta_vs_A8=pt[pick] - pt['A8'],
              ci95_vs_A8=boot[f'{pick}-A8']['mean_AUC_pAUC'][1:],
              eval_delta_vs_G8=(pt[pick] - pt['G8']) if pick != 'G8' else 0.0,
              ci95_vs_G8=boot[f'{pick}-G8']['mean_AUC_pAUC'][1:] if pick != 'G8' else None,
              dev_delta_vs_A8=sel['dev_amean'][pick] - sel['dev_A8'],
              dev_delta_vs_G8=sel['dev_amean'][pick] - sel['dev_amean']['G8'])
    d1['replaces_A8'] = bool(d1['new'] and d1['eval_delta_vs_A8'] > 0 and d1['ci95_vs_A8'][0] > 0)
    d2 = dict(pick=x1pick, eval_delta_vs_F1k=pt[x1pick] - pt['F1k'], ci95_vs_F1k=boot[f'{x1pick}-F1k']['mean_AUC_pAUC'][1:],
              dev_delta_vs_F1k=sel['x1_dev_amean'][x1pick] - sel['dev_F1k'], label='WITH EXTERNAL PRETRAINING')
    d2['replaces_F1k'] = bool(d2['eval_delta_vs_F1k'] > 0 and d2['ci95_vs_F1k'][0] > 0)
    d4 = dict(honest_margin_vs_G8=sel['O1']['honest_margin_vs_G8'],
              flag_selection_noise=bool(sel['O1']['honest_margin_vs_G8'] <= 0))
    json.dump(dict(D1=d1, D2=d2, D4=d4), open(os.path.join(SCORES, 'decisions.json'), 'w'), indent=2, default=float)
    L += ['', f'D1 X0 {pick}: eval Δ vs A8 {d1["eval_delta_vs_A8"]:+.3f} CI [{d1["ci95_vs_A8"][0]:+.3f}, '
          f'{d1["ci95_vs_A8"][1]:+.3f}]; vs G8 {d1["eval_delta_vs_G8"]:+.3f} -> replaces A8: {d1["replaces_A8"]}',
          f'D2 X1 {x1pick} (WITH EXTERNAL PRETRAINING): eval Δ vs F1k {d2["eval_delta_vs_F1k"]:+.3f} CI '
          f'[{d2["ci95_vs_F1k"][0]:+.3f}, {d2["ci95_vs_F1k"][1]:+.3f}] -> replaces F1k: {d2["replaces_F1k"]}',
          f'D4 honest dev margin vs G8 {d4["honest_margin_vs_G8"]:+.3f} -> selection-noise flag: {d4["flag_selection_noise"]}']
    types, C = type_corr(V, ev)
    write_csv('type_corr_eval.csv', ['type'] + types, [[t] + [f'{c:.4f}' for c in row] for t, row in zip(types, C)])
    txt = '\n'.join(L)
    open(os.path.join(SCORES, 'scorecard_eval.txt'), 'w').write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    {'repro': stage_repro, 'dev': stage_dev, 'eval': stage_eval}[stage]()
