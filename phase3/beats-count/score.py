#!/usr/bin/env python
"""beats-count: X1f with 1 or 2 BEATs networks instead of 3 (README, pre-registered at 4b3613c).

    python phase3/beats-count/score.py dev  [out]   # DEV ONLY: dev_systems.csv, selection.json
    python phase3/beats-count/score.py eval [out]   # systems.csv (dev + eval), bootstrap.json, decisions.json;
                                                    # refuses unless scores/selection.json is committed, unmodified
Run both through run.sh (independent processes until two outputs are byte-identical; smartlab1's RAM).

Adds no scoring rule. Members, weights and every score vector come from ../../synthesis/ensemble-g8/score.py's
machinery (imported, which imports fresh-confirm -> grand-ensemble), exactly as ../beats-router does: S1' per
from-scratch member, zK' per BEATs member, each vector computed twice from separate reads (fc.ens_vector).
Metrics are fresh-confirm's `metrics` (fc.COLS); the paired bootstrap is grand-ensemble's `paired_boot`
(2000 draws, seed 0, stratified machine x ID x label).

Systems. fc.ens_vector(members, weights) is ensemble_from_dumps: s = 0; s = s + w[type] * vector(member), in
member order, with ONE weight per type. A BEATs subset therefore passes only the chosen BEATs members, with
type 'B' at the subset's per-net weight:
    block  (primary)    GS12f members at 1 + the k chosen BEATs nets at 4.5 / k   (block weight fixed at 4.5)
    member (secondary)  GS12f members at 1 + the k chosen BEATs nets at 1.5       (block weight 1.5 k)
Every subset vector is built directly with fc.ens_vector and must equal, bit for bit, GS12f's vector plus
w * each single-BEATs vector added in seed order (the same float operations). k = 3 under both weightings
(4.5 / 3 == 1.5 exactly) must equal X1f's vector bit for bit.
"""
import csv
import hashlib
import importlib.util
import itertools
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # paired_boot's pool forks; module names must resolve
    spec.loader.exec_module(mod)
    return mod


eg = _imp('eg_score', os.path.join(REPO, 'synthesis', 'ensemble-g8', 'score.py'))
fc, ge, sf = eg.fc, eg.ge, eg.sf
N = fc.N
MACH = list(ge.MACH)
TOL = eg.TOL                                 # 5e-4 Amean against ensemble-g8's committed record
COMMITTED = os.path.join(REPO, 'synthesis', 'ensemble-g8', 'scores', 'ensembles.csv')
EG_BOOT = os.path.join(REPO, 'synthesis', 'ensemble-g8', 'scores', 'bootstrap.json')
EMB = os.path.join(HERE, 'results', 'emb')   # symlink tree onto ../OS-SCL-syn-fresh (run.sh links)
SCORES = os.path.join(HERE, 'scores')
SEL_REL = 'phase3/beats-count/scores/selection.json'
DEVCSV_REL = 'phase3/beats-count/scores/dev_systems.csv'
G_SEEDS, S_SEEDS = (2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067), (2060, 2061, 2062, 2063)
B_SEEDS = (2040, 2041, 2042)                 # fc.BEATS order; B0, B1, B2
BLOCK_W, MEMBER_W = 4.5, 1.5                 # README: X1f's BEATs block weight; X1f's per-net weight
TIE = 0.05                                   # README selection: tie within 0.05 dev Amean -> lower seed numbers
D1A_MIN, D1B_MIN = -0.20, -0.10              # README D1_k (a), (b)
HEADER = ['system', 'subset', 'k', 'weighting', 'w_per_beats', 'dataset', 'machine', *fc.COLS]


# ----------------------------------------------------------------------------- helpers
def read(path):
    with open(path, encoding='UTF-8') as f:
        return list(csv.DictReader(f))


def f4(x):
    return f'{x:.4f}'


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def ci_of(boot, a, b):
    return boot[f'{a}-{b}']['mean_AUC_pAUC']          # [mean, 2.5 %, 97.5 %]


def inside(v, lo, hi):
    return bool(lo <= v <= hi)


def subsets():
    """[(subset name, (index, ...), k)] for k in 1, 2, in seed order."""
    return [('B' + ''.join(map(str, s)), s, k) for k in (1, 2) for s in itertools.combinations(range(3), k)]


def spec():
    """name -> dict(subset, k, weighting, w). GS12f, X1f, then every subset under both weightings."""
    S = {'GS12f': dict(subset='', idx=(), k=0, weighting='none', w=0.0),
         'X1f': dict(subset='B012', idx=(0, 1, 2), k=3, weighting='both', w=MEMBER_W)}
    for sub, idx, k in subsets():
        S[f'{sub}_block'] = dict(subset=sub, idx=idx, k=k, weighting='block', w=BLOCK_W / k)
        S[f'{sub}_member'] = dict(subset=sub, idx=idx, k=k, weighting='member', w=MEMBER_W)
    return S


# ----------------------------------------------------------------------------- vectors
def vectors():
    """{name: score vector} for every system in spec(), built with fc.ens_vector; construction asserted."""
    m = ge.meta()
    ge.check_meta(os.path.join(EMB, 'meta.npz'), m)
    sc = ([(fc.flat_path(EMB, f'{N}fc_nogain_s{s}', m), 'G') for s in G_SEEDS] +
          [(fc.flat_path(EMB, f'{N}fc_spec_s{s}', m), 'S') for s in S_SEEDS])
    assert fc.BEATS == [f'BEATsASD_bt_arc_s{s}' for s in B_SEEDS], fc.BEATS
    bm = [(ge.dump_path(r), 'B') for r in fc.BEATS]
    one = {'G': 1.0, 'S': 1.0}
    V = {'GS12f': fc.ens_vector(sc, one),
         'X1f': fc.ens_vector(sc + bm, {'G': 1.0, 'S': 1.0, 'B': MEMBER_W})}     # ensemble-g8's exact recipe
    single = [fc.ens_vector([b], {'B': 1.0}) for b in bm]
    chk = {}
    for name, s in spec().items():
        if name in V:
            continue
        V[name] = fc.ens_vector(sc + [bm[i] for i in s['idx']], {'G': 1.0, 'S': 1.0, 'B': s['w']})
        blk = V['GS12f']
        for i in s['idx']:
            blk = blk + s['w'] * single[i]
        assert np.array_equal(V[name], blk), f'ABORT: {name} direct != GS12f + w * singles'
        chk[name] = 'direct == GS12f + w * singles, bit for bit'
    for w in (BLOCK_W / 3, MEMBER_W):        # k = 3 under both weightings is X1f
        blk = V['GS12f']
        for i in range(3):
            blk = blk + w * single[i]
        assert np.array_equal(blk, V['X1f']), f'ABORT: k = 3 at w = {w} != X1f'
    chk['X1f'] = 'k = 3 at 4.5/3 and at 1.5, built as GS12f + w * singles, equals X1f bit for bit'
    members = dict(G=[f'{N}fc_nogain_s{s}' for s in G_SEEDS], S=[f'{N}fc_spec_s{s}' for s in S_SEEDS],
                   B=list(fc.BEATS))
    return V, chk, members


def score_set(V, ds):
    """{name: (row, per)} on one set; asserts X1f and GS12f against ensemble-g8's committed overall Amean."""
    ev = ge.evaluator(ds)
    R = {n: fc.metrics(ev, v) for n, v in V.items()}
    checks = []
    for x in read(COMMITTED):
        if x['machine'] == 'all' and x['dataset'] == ds and x['system'] in ('X1f', 'GS12f'):
            checks.append([x['system'], ds, R[x['system']][0]['Amean'], float(x['Amean'])])
    assert len(checks) == 2, checks
    for c in checks:
        print(f'check {c[0]} {c[1]}: {c[2]:.4f} vs ensemble-g8 {c[3]:.4f}', flush=True)
    worst = max(abs(g - w) for _, _, g, w in checks)
    assert worst < TOL, f'ABORT: {ds} systems differ from ensemble-g8/scores/ensembles.csv by {worst:.5f} Amean'
    return ev, R, worst


def rows_of(R, ds):
    S = spec()
    out = []
    for n in S:
        row, per = R[n]
        s = S[n]
        head = [n, s['subset'], s['k'], s['weighting'], f4(s['w'])]
        out.append(head + [ds, 'all'] + [f4(row[c]) for c in fc.COLS])
        out += [head + [ds, mm] + [f4(per[mm][c]) for c in fc.COLS] for mm in MACH]
    return out


def pick(Rdev):
    """README selection, per k: highest dev Amean under the primary (block) weighting; a tie within 0.05
    goes to the subset with the lower seed numbers (B0 < B1 < B2; B01 < B02 < B12)."""
    out = {}
    for k in (1, 2):
        cand = [(sub, idx) for sub, idx, kk in subsets() if kk == k]
        am = {sub: Rdev[f'{sub}_block'][0]['Amean'] for sub, _ in cand}
        best = max(am.values())
        tied = [(idx, sub) for sub, idx in cand if best - am[sub] <= TIE]
        chosen = min(tied)[1]
        out[f'X1f-{k}'] = dict(subset=chosen, system=f'{chosen}_block', w_per_beats=BLOCK_W / k,
                               beats_seeds=[B_SEEDS[i] for i in dict(cand)[chosen]],
                               dev_Amean_block=am, best_dev_Amean=best,
                               tied_within_0p05=[s for _, s in sorted(tied)],
                               decided_by_tie_rule=bool(len(tied) > 1 and chosen != max(am, key=am.get)))
    return out


# ----------------------------------------------------------------------------- stage: dev
def stage_dev(out):
    os.makedirs(out, exist_ok=True)
    V, chk, members = vectors()
    _, R, worst = score_set(V, 'dev')                  # dev only: no eval evaluator is built in this stage
    p = os.path.join(out, 'dev_systems.csv')
    fc.write_csv(p, HEADER, rows_of(R, 'dev'))
    sel = dict(picks=pick(R),
               rule='per k in {1, 2}: the subset with the highest dev Amean under the primary weighting '
                    '(BEATs block 4.5, per net 4.5 / k); a tie within 0.05 goes to the lower seed numbers',
               dev_systems_csv_sha256=sha256(p),
               checks=dict(dev_max_abs_diff_vs_ensemble_g8=float(worst), construction=chk),
               members=members, preregistration='4b3613c',
               note='computed on dev only; no eval metric exists at this point. Commit this file and '
                    'dev_systems.csv before `score.py eval`, which refuses to run otherwise.')
    with open(os.path.join(out, 'selection.json'), 'w') as f:
        json.dump(sel, f, indent=2)
    print(json.dumps(sel['picks'], indent=2))


# ----------------------------------------------------------------------------- stage: eval
def git(*a):
    return subprocess.run(['git', '-C', REPO, *a], capture_output=True, text=True)


def require_committed():
    for rel in (SEL_REL, DEVCSV_REL):
        if git('cat-file', '-e', f'HEAD:{rel}').returncode != 0:
            sys.exit(f'REFUSE: {rel} is not in HEAD; run `run.sh dev` and commit it first')
        if git('diff', '--quiet', 'HEAD', '--', rel).returncode != 0:
            sys.exit(f'REFUSE: {rel} differs from HEAD')
    with open(os.path.join(REPO, SEL_REL), encoding='UTF-8') as f:
        sel = json.load(f)
    if sha256(os.path.join(REPO, DEVCSV_REL)) != sel['dev_systems_csv_sha256']:
        sys.exit('REFUSE: dev_systems.csv does not match the sha256 in selection.json')
    return sel, git('rev-parse', 'HEAD').stdout.strip()


def stage_eval(out):
    sel, head = require_committed()
    os.makedirs(out, exist_ok=True)
    V, chk, members = vectors()
    S = spec()

    # ---- dev, recomputed: must match the committed dev record and the committed picks
    ev_d, Rd, worst_d = score_set(V, 'dev')
    dev_rows = rows_of(Rd, 'dev')
    committed = [[*r.values()] for r in read(os.path.join(REPO, DEVCSV_REL))]
    assert [[str(x) for x in r] for r in dev_rows] == committed, 'ABORT: dev metrics differ from dev_systems.csv'
    assert {k: v['subset'] for k, v in pick(Rd).items()} == {k: v['subset'] for k, v in sel['picks'].items()}
    P = {k: v['system'] for k, v in sel['picks'].items()}          # X1f-1, X1f-2 -> system names

    # ---- eval, once
    ev_e, Re, worst_e = score_set(V, 'eval')
    fc.write_csv(os.path.join(out, 'systems.csv'), HEADER, dev_rows + rows_of(Re, 'eval'))

    # ---- paired bootstrap (ge.paired_boot, same draws/seed as ensemble-g8)
    prim = [f'{sub}_block' for sub, _, _ in subsets()]
    vecs = {n: V[n] for n in ['X1f', 'GS12f'] + prim}
    pairs = [(P['X1f-1'], 'X1f'), (P['X1f-2'], 'X1f'), (P['X1f-1'], 'GS12f'), (P['X1f-2'], 'GS12f')]
    pairs += [(n, 'X1f') for n in prim if (n, 'X1f') not in pairs]
    pairs += [('X1f', 'GS12f')]                                   # cross-check against ensemble-g8
    boot = {'eval': ge.paired_boot(ev_e, vecs, pairs), 'dev': ge.paired_boot(ev_d, vecs, pairs)}
    with open(EG_BOOT, encoding='UTF-8') as f:
        egb = json.load(f)
    eg_diff = max(abs(g - w) for ds in ('dev', 'eval') for g, w in zip(ci_of(boot[ds], 'X1f', 'GS12f'),
                                                                     egb[ds]['X1f-GS12f']['mean_AUC_pAUC']))
    assert eg_diff < TOL, ('X1f-GS12f bootstrap differs from ensemble-g8', eg_diff)
    checks = dict(dev_max_abs_diff_vs_ensemble_g8=float(worst_d), eval_max_abs_diff_vs_ensemble_g8=float(worst_e),
                  X1f_minus_GS12f_bootstrap_vs_ensemble_g8=float(eg_diff), construction=chk,
                  dev_rows_equal_committed_dev_systems_csv=True, picks_rederived_from_dev=True)
    bout = dict(pairs=boot, picks=P, checks=checks,
                draws=dict(n_boot=ge.N_BOOT, seed=ge.RNG_SEED,
                           resampler='ensemble-backend Evaluator.resample (machine x ID x label strata)'),
                members=members)
    with open(os.path.join(out, 'bootstrap.json'), 'w') as f:
        json.dump(bout, f, indent=2)

    # ---- decisions
    A = {(n, ds): R[n][0]['Amean'] for ds, R in (('dev', Rd), ('eval', Re)) for n in S}
    D = {'selection': dict(picks=P, selection_commit_at_eval=head)}
    D['systems'] = {n: dict(subset=S[n]['subset'], k=S[n]['k'], weighting=S[n]['weighting'], w_per_beats=S[n]['w'],
                            dev=A[(n, 'dev')], eval=A[(n, 'eval')],
                            minus_X1f={ds: A[(n, ds)] - A[('X1f', ds)] for ds in ('dev', 'eval')},
                            minus_GS12f={ds: A[(n, ds)] - A[('GS12f', ds)] for ds in ('dev', 'eval')})
                    for n in S}
    D1 = {}
    for k in (1, 2):
        x = P[f'X1f-{k}']
        cx, cg = ci_of(boot['eval'], x, 'X1f'), ci_of(boot['eval'], x, 'GS12f')
        subs = [f'{sub}_block' for sub, _, kk in subsets() if kk == k]
        dl = {n: A[(n, 'eval')] - A[('X1f', 'eval')] for n in subs}
        a, b, c = bool(cx[1] >= D1A_MIN), bool(sum(v >= D1B_MIN for v in dl.values()) >= 2), bool(cg[1] > 0)
        D1[f'D1_{k}'] = dict(system=x,
                             a=dict(eval_ci_minus_X1f=cx, lower_bound=cx[1], threshold=D1A_MIN, passed=a),
                             b=dict(eval_delta_vs_X1f=dl, n_ge_threshold=int(sum(v >= D1B_MIN for v in dl.values())),
                                    threshold=D1B_MIN, passed=b),
                             c=dict(eval_ci_minus_GS12f=cg, lower_bound=cg[1], passed=c),
                             passed=bool(a and b and c),
                             dev_ci=dict(minus_X1f=ci_of(boot['dev'], x, 'X1f'),
                                         minus_GS12f=ci_of(boot['dev'], x, 'GS12f')))
    D.update(D1)
    lite = next((k for k in (1, 2) if D1[f'D1_{k}']['passed']), None)
    D['recommendation'] = (dict(X1f_lite=f'X1f-{lite}', system=P[f'X1f-{lite}'], k=lite,
                                note='X1f stays the reported accuracy number')
                           if lite else dict(X1f_lite=None, note='neither D1_1 nor D1_2 passes; X1f stays as it is'))
    # D2 (descriptive): mean Amean over subsets per k, both weightings
    D2 = {}
    for wt in ('block', 'member'):
        mk = {}
        for ds in ('dev', 'eval'):
            mk[ds] = {'0': A[('GS12f', ds)], '3': A[('X1f', ds)]}
            for k in (1, 2):
                mk[ds][str(k)] = float(np.mean([A[(f'{sub}_{wt}', ds)] for sub, _, kk in subsets() if kk == k]))
        g12, g23 = mk['eval']['2'] - mk['eval']['1'], mk['eval']['3'] - mk['eval']['2']
        D2[wt] = dict(mean_by_k=mk, eval_gain_1_to_2=g12, eval_gain_2_to_3=g23, diminishing_returns=bool(g12 > g23),
                      rule='diminishing returns if mean(k=2) - mean(k=1) > X1f - mean(k=2) (eval)')
    D['D2'] = D2

    # ---- predictions (README, 80 % intervals; hit = inside the interval / the stated outcome)
    q = {}
    m1, m2 = D2['block']['mean_by_k']['eval']['1'], D2['block']['mean_by_k']['eval']['2']
    q['Q1'] = dict(quantity='mean over k=1 subsets, eval Amean, primary', predicted='95.30 [95.15, 95.45]',
                   outcome=m1, hit=inside(m1, 95.15, 95.45))
    q['Q2'] = dict(quantity='mean over k=2 subsets, eval Amean, primary', predicted='95.44 [95.33, 95.53]',
                   outcome=m2, hit=inside(m2, 95.33, 95.53))
    l1, l2 = D1['D1_1']['a']['lower_bound'], D1['D1_2']['a']['lower_bound']
    q['Q3'] = dict(quantity='X1f-1 - X1f eval CI lower bound', predicted='-0.32 [-0.45, -0.18]; D1_1 fails (70 %)',
                   outcome=l1, hit=inside(l1, -0.45, -0.18), D1_1_failed=not D1['D1_1']['passed'],
                   outcome_hit=not D1['D1_1']['passed'])
    q['Q4'] = dict(quantity='X1f-2 - X1f eval CI lower bound', predicted='-0.17 [-0.25, -0.08]; D1_2 passes (55 %)',
                   outcome=l2, hit=inside(l2, -0.25, -0.08), D1_2_passed=D1['D1_2']['passed'],
                   outcome_hit=D1['D1_2']['passed'])
    q['Q5'] = dict(quantity='D2 diminishing returns (primary)', predicted='yes (70 %)',
                   outcome=D2['block']['diminishing_returns'], hit=D2['block']['diminishing_returns'])
    d6 = D2['member']['mean_by_k']['eval']['1'] - m1
    q['Q6'] = dict(quantity='secondary minus primary weighting, k=1 mean eval', predicted='-0.05 [-0.20, +0.10]',
                   outcome=d6, hit=inside(d6, -0.20, 0.10))
    k1 = [f'{sub}_block' for sub, _, kk in subsets() if kk == 1]
    best_eval = max(k1, key=lambda n: A[(n, 'eval')])
    same = best_eval == P['X1f-1']
    q['Q7'] = dict(quantity='the dev pick for k=1 is also the best k=1 subset on eval',
                   predicted='40 % (chance is 33 %)', outcome=same, dev_pick=P['X1f-1'], eval_best=best_eval,
                   hit=bool(same), note='hit = the stated event occurred; it was stated at 40 %, i.e. as less '
                                        'likely than not, so the Brier term is the fairer score',
                   brier=(0.40 - float(same)) ** 2)
    D['predictions'] = q
    D['checks'] = checks
    D['post_hoc'] = ('every network and every eval score of the members was seen before this study (phase-3 rule 2); '
                     'the subset systems themselves had not been scored')
    with open(os.path.join(out, 'decisions.json'), 'w') as f:
        json.dump(D, f, indent=2)
    print(json.dumps({k: D[k] for k in ('selection', 'D1_1', 'D1_2', 'recommendation', 'D2', 'predictions')},
                     indent=2))


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    dest = sys.argv[2] if len(sys.argv) > 2 else SCORES
    if stage == 'dev':
        stage_dev(dest)
    elif stage == 'eval':
        stage_eval(dest)
    else:
        sys.exit(__doc__)
