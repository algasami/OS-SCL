#!/usr/bin/env python
"""id-gate stage 1: the normal-only per-ID proxy against the dev per-ID BEATs effect (README, pre-registered at e637672).

    python phase3/id-gate/score.py stage1 [out]   # proxy_ids.csv, stage1.json -> out (default scores/)
Run through run.sh (independent processes until two outputs are byte-identical; smartlab1's RAM).

Adds no scoring rule. The proxy is ensemble-backend's zKp (eb.l2, eb.nn_dist, eb.z: 1-NN cosine distance to
ID i's training-part bank, z-scored with the bank's leave-one-out distances) applied to TRAINING clips:
  positives = every training clip of ID i (bank clips leave-one-out, held-out clips against the full bank),
  negatives = every training clip of the other IDs of the same machine type, against ID i's bank.
Members, weights, dumps and test vectors are ensemble-g8's (imported via beats-router's construction):
8 fc_nogain + 4 fc_spec (w 1), 3 BEATs (w 1.5); fc dumps flattened to shift 0 by fc.flat_path.

Before any proxy is used, every member's test-clip zKp is recomputed with the same bank / z code and asserted
equal to grand-ensemble's member_rules(...)['zKp'] (the adopted rule) on all dev + eval test clips.
Truth is computed on DEV ONLY (Amean per dev ID, X1f - GS12f); no eval-set metric of any system is computed.
"""
import csv
import importlib.util
import json
import os
import re
import sys
import warnings

import numpy as np
from scipy import stats
from sklearn import metrics as skm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


eg = _imp('eg_score', os.path.join(REPO, 'synthesis', 'ensemble-g8', 'score.py'))
fc, ge, sf = eg.fc, eg.ge, eg.sf
eb = ge.eb
ge.CACHE = os.path.join(HERE, 'results', 'vec_cache')   # never written (use_cache=False); kept inside this study
N = fc.N
MACH = list(ge.MACH)
TOL = eg.TOL                                  # 5e-4 Amean against ensemble-g8's committed record
SANITY_TOL = 1e-9
COMMITTED = os.path.join(REPO, 'synthesis', 'ensemble-g8', 'scores', 'ensembles.csv')
EMB = os.path.join(HERE, 'results', 'emb')    # symlink tree onto ensemble-g8's split dumps (run.sh links)
G_SEEDS, S_SEEDS = tuple(range(2060, 2068)), (2060, 2061, 2062, 2063)
W = {'G': 1.0, 'S': 1.0, 'B': 1.5}            # X1f's block weights
SYSTEMS = ('GS12f_kNN', 'X1f_kNN', 'scratch_block', 'beats_block')
N_BOOT, BOOT_SEED = 2000, 0                   # README D0(c)
SAT = 0.999                                   # README D0(a)
RHO_MIN = 0.40                                # README D0(b)


# ----------------------------------------------------------------------------- IDs
def id_table(m):
    """label -> dict(machine, id, set). Machine/ID from the training paths, cross-checked against the test
    list; set = the test set (dev / eval) in which the label appears."""
    tab = {}
    for c in np.unique(m['tr_label']):
        ps = m['tr_paths'][m['tr_label'] == c]
        mach = {p.split('/')[-3] for p in ps}
        ids = {re.search(r'(id_\d\d)', os.path.basename(p)).group(1) for p in ps}
        assert len(mach) == 1 and len(ids) == 1, (c, mach, ids)
        tab[int(c)] = dict(machine=mach.pop(), id=ids.pop(), set=None)
    for c in np.unique(m['te_label']):
        te = m['te_label'] == c
        trip = {(a, b, d) for a, b, d in zip(m['te_machine'][te], m['te_id'][te], m['te_dataset'][te])}
        assert len(trip) == 1, (c, trip)
        a, b, d = trip.pop()
        assert (tab[int(c)]['machine'], tab[int(c)]['id']) == (a, b), (c, tab[int(c)], a, b)
        tab[int(c)]['set'] = str(d)
    n = {s: sum(v['set'] == s for v in tab.values()) for s in ('dev', 'eval')}
    assert n == {'dev': 23, 'eval': 18} and len(tab) == 41, n
    return tab


def layout(m, tab):
    """Per label: (clip indices of positives, clip indices of negatives), both in meta order."""
    trl = m['tr_label']
    mach_of = np.array([tab[int(c)]['machine'] for c in trl])
    out = {}
    for c, v in tab.items():
        pos = np.where(trl == c)[0]
        neg = np.where((mach_of == v['machine']) & (trl != c))[0]
        out[c] = (pos, neg)
    return out


# ----------------------------------------------------------------------------- members
def members(m):
    """[(name, dump path, type)] in a fixed order: 8 fc_nogain, 4 fc_spec, 3 BEATs."""
    ge.check_meta(os.path.join(EMB, 'meta.npz'), m)
    out = [(f'{N}fc_nogain_s{s}', fc.flat_path(EMB, f'{N}fc_nogain_s{s}', m), 'G') for s in G_SEEDS]
    out += [(f'{N}fc_spec_s{s}', fc.flat_path(EMB, f'{N}fc_spec_s{s}', m), 'S') for s in S_SEEDS]
    out += [(r, ge.dump_path(r), 'B') for r in fc.BEATS]
    return out


def bank(trp, trl, valid, c):
    """ID c's reference bank (training part, ~valid) and its leave-one-out reference distances (k_ref)."""
    ref = (trl == c) & ~valid
    b = trp[ref]
    return ref, b, eb.nn_dist(b, b, loo=True)


def member_scores(path, m, lay):
    """One member: (flat proxy zKp over every (ID, clip) in `lay` order, test zKp over every test clip).
    Both come from the same bank / k_ref; the test half is the sanity check against the adopted rule."""
    a = ge.load_dump(path)
    trl, tel = m['tr_label'], m['te_label']
    valid = a['valid']
    trp, tep = eb.l2(a['tr_pre'].astype(np.float64)), eb.l2(a['te_pre'].astype(np.float64))
    parts = []
    for c, (pos, neg) in lay.items():
        ref, b, k_ref = bank(trp, trl, valid, c)
        d_pos = np.empty(len(pos))
        in_bank = ref[pos]
        rank_in_bank = np.cumsum(ref) - 1                 # row of a bank clip inside b
        d_pos[in_bank] = k_ref[rank_in_bank[pos[in_bank]]]   # leave-one-out: excludes itself
        d_pos[~in_bank] = eb.nn_dist(trp[pos[~in_bank]], b)
        d_neg = eb.nn_dist(trp[neg], b)
        parts.append(eb.z(np.concatenate([d_pos, d_neg]), k_ref))
    te_z = np.full(len(tel), np.nan)
    for c in np.unique(tel):
        _, b, k_ref = bank(trp, trl, valid, c)
        te = tel == c
        te_z[te] = eb.z(eb.nn_dist(tep[te], b), k_ref)
    return np.concatenate(parts), te_z


# ----------------------------------------------------------------------------- metrics
def auc_pauc(n_pos, s):
    """proxyAUC / proxy pAUC of one ID: negatives (other IDs) are the positive class, as anomalies in eval.py."""
    y = np.r_[np.zeros(n_pos), np.ones(len(s) - n_pos)]
    return skm.roc_auc_score(y, s), skm.roc_auc_score(y, s, max_fpr=0.1)


def dev_per_id(ev, v):
    """{(machine, id): (AUC, pAUC)} on dev test clips: AUC from ev's own per-ID entry, pAUC with the same
    sklearn call as eval.py / ev / fc.per_id restricted to the ID's clips."""
    res = ev(v)
    s = v[ev.sel]
    out = {}
    for mm in MACH:
        for i in np.unique(ev.id[ev.mach == mm]):
            k = (ev.mach == mm) & (ev.id == i)
            out[(mm, str(i))] = (float(res[mm]['ids'][i]), float(skm.roc_auc_score(ev.y[k], s[k], max_fpr=0.1)))
    return out, res


def spearman(x, y):
    return float(stats.spearmanr(x, y)[0])


def boot_rho(x, y, idx):
    """Spearman rho and its percentile CI over resampled IDs. A constant input (e.g. a saturated proxy whose
    differences are all exactly 0) leaves rho undefined: reported as None, never as a number."""
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', stats.ConstantInputWarning)
        r = np.array([stats.spearmanr(x[i], y[i])[0] for i in idx], dtype=float)
        rho = spearman(x, y)
    ok = np.isfinite(r)
    lo, hi = np.percentile(r[ok], [2.5, 97.5]) if ok.any() else (np.nan, np.nan)
    fin = lambda v: float(v) if np.isfinite(v) else None      # noqa: E731
    return dict(rho=fin(rho), ci_lo=fin(lo), ci_hi=fin(hi), boot_mean=fin(r[ok].mean()) if ok.any() else None,
                n_draws=int(len(r)), n_nan_draws=int((~ok).sum()), n_distinct_x=int(len(np.unique(x))))


def centred(x, groups):
    x = np.asarray(x, float).copy()
    for g in set(groups):
        k = np.array([gg == g for gg in groups])
        x[k] -= x[k].mean()
    return x


def f8(x):
    return f'{x:.8f}'


def inside(v, lo, hi):
    return None if v is None else bool(lo <= v <= hi)


def rd(x, n=10):
    return float(round(x, n))


# ----------------------------------------------------------------------------- stage 1
def stage1(out):
    os.makedirs(out, exist_ok=True)
    m = ge.meta()
    tab = id_table(m)
    lay = layout(m, tab)
    mem = members(m)

    # ---- 1. per-member proxy + sanity check against the adopted zKp (every member, dev + eval test clips)
    P, sanity = {}, {}
    for name, path, typ in mem:
        v, _n = sf.twice(lambda: np.concatenate(member_scores(path, m, lay)))
        n_prox = sum(len(p) + len(q) for p, q in lay.values())
        prox, te_z = v[:n_prox], v[n_prox:]
        with np.errstate(divide='ignore', invalid='ignore'):
            ref_z = ge.member_rules(path, use_cache=False)['zKp']
        diff = float(np.max(np.abs(te_z - ref_z)))
        sanity[name] = diff
        print(f'{name}: test zKp vs ge.member_rules max|diff| {diff:.2e}  proxy clips {n_prox}', flush=True)
        assert diff < SANITY_TOL, f'ABORT: {name} test zKp differs from the adopted rule by {diff:.2e}'
        assert np.isfinite(prox).all(), name
        P[name] = (prox, typ)

    # ---- 2. system proxy scores (Σ w · zKp) and per-ID proxyAUC / pAUC
    G = [n for n, _, t in mem if t in ('G', 'S')]
    B = [n for n, _, t in mem if t == 'B']
    scratch = sum(W[P[n][1]] * P[n][0] for n in G)
    beats = sum(W['B'] * P[n][0] for n in B)
    S = {'GS12f_kNN': scratch, 'X1f_kNN': scratch + beats, 'scratch_block': scratch, 'beats_block': beats}
    A, gap = {}, {}
    off = 0
    for c, (pos, neg) in lay.items():
        n = len(pos) + len(neg)
        for s, v in S.items():
            A[(c, s)] = auc_pauc(len(pos), v[off:off + n])
            # separation margin (descriptive): lowest negative score minus highest positive score
            gap[(c, s)] = float(v[off + len(pos):off + n].min() - v[off:off + len(pos)].max())
        off += n
    assert off == len(scratch)

    # ---- 3. truth, DEV ONLY: X1f and GS12f test vectors exactly as beats-router / ensemble-g8 build them
    bpath = lambda n: dict((x, p) for x, p, _ in mem)[n]        # noqa: E731
    gs = [(bpath(n), t) for n, _, t in mem if t in ('G', 'S')]
    bm = [(bpath(n), 'B') for n in B]
    V = {'GS12f': fc.ens_vector(gs, {'G': 1.0, 'S': 1.0}),
         'X1f': fc.ens_vector(gs + bm, {'G': 1.0, 'S': 1.0, 'B': 1.5})}
    ev = ge.evaluator('dev')
    committed = {(x['system'], x['dataset'], x['machine']): x for x in csv.DictReader(open(COMMITTED, encoding='UTF-8'))}
    repro, per_id = {}, {}
    for name, v in V.items():
        row, _ = fc.metrics(ev, v)
        pid, res = dev_per_id(ev, v)
        per_id[name] = pid
        # the per-ID numbers must rebuild the committed per-ID-averaged dev metrics, and AUC must match ev's
        a_pid = np.mean([np.mean([pid[k][0] for k in pid if k[0] == mm]) for mm in MACH]) * 100
        p_pid = np.mean([np.mean([pid[k][1] for k in pid if k[0] == mm]) for mm in MACH]) * 100
        cm = committed[(name, 'dev', 'all')]
        repro[name] = dict(Amean=row['Amean'], Amean_committed=float(cm['Amean']),
                           Amean_absdiff=abs(row['Amean'] - float(cm['Amean'])),
                           AUC_perID_rebuilt_absdiff_vs_fc=abs(a_pid - row['AUC_perID']),
                           pAUC_perID_rebuilt_absdiff_vs_fc=abs(p_pid - row['pAUC_perID']),
                           Amean_perID_absdiff_vs_committed=abs((a_pid + p_pid) / 2 - float(cm['Amean_perID'])))
        print(f'check {name} dev: Amean {row["Amean"]:.4f} vs ensemble-g8 {float(cm["Amean"]):.4f}; per-ID rebuild '
              f'{repro[name]["AUC_perID_rebuilt_absdiff_vs_fc"]:.1e} / {repro[name]["pAUC_perID_rebuilt_absdiff_vs_fc"]:.1e}',
              flush=True)
        assert repro[name]['Amean_absdiff'] < TOL, ('ABORT: dev Amean differs from ensemble-g8', repro[name])
        assert repro[name]['AUC_perID_rebuilt_absdiff_vs_fc'] < 1e-9 and \
            repro[name]['pAUC_perID_rebuilt_absdiff_vs_fc'] < 1e-9, ('ABORT: per-ID rebuild', repro[name])
        assert repro[name]['Amean_perID_absdiff_vs_committed'] < TOL, ('ABORT: per-ID vs committed', repro[name])
    amean_id = {n: {k: 100 * (a + p) / 2 for k, (a, p) in d.items()} for n, d in per_id.items()}

    # ---- 4. per-ID table
    dev = [c for c in tab if tab[c]['set'] == 'dev']
    evl = [c for c in tab if tab[c]['set'] == 'eval']
    order = sorted(tab, key=lambda c: (tab[c]['set'] != 'dev', MACH.index(tab[c]['machine']), tab[c]['id']))
    pi = {c: A[(c, 'X1f_kNN')][0] - A[(c, 'GS12f_kNN')][0] for c in tab}
    pib = {c: A[(c, 'beats_block')][0] - A[(c, 'scratch_block')][0] for c in tab}
    pip = {c: A[(c, 'X1f_kNN')][1] - A[(c, 'GS12f_kNN')][1] for c in tab}
    pibp = {c: A[(c, 'beats_block')][1] - A[(c, 'scratch_block')][1] for c in tab}
    delta = {c: amean_id['X1f'][(tab[c]['machine'], tab[c]['id'])] - amean_id['GS12f'][(tab[c]['machine'], tab[c]['id'])]
             for c in dev}
    header = (['set', 'machine', 'id', 'label', 'n_pos', 'n_neg'] + [f'proxyAUC_{s}' for s in SYSTEMS]
              + [f'proxypAUC_{s}' for s in SYSTEMS] + ['pi', 'pi_b', 'pi_pAUC', 'pi_b_pAUC',
                                                        'dev_Amean_GS12f', 'dev_Amean_X1f', 'delta'])
    rows = []
    for c in order:
        t = tab[c]
        key = (t['machine'], t['id'])
        rows.append([t['set'], t['machine'], t['id'], c, len(lay[c][0]), len(lay[c][1])]
                    + [f8(A[(c, s)][0]) for s in SYSTEMS] + [f8(A[(c, s)][1]) for s in SYSTEMS]
                    + [f8(pi[c]), f8(pib[c]), f8(pip[c]), f8(pibp[c])]
                    + ([f'{amean_id["GS12f"][key]:.4f}', f'{amean_id["X1f"][key]:.4f}', f'{delta[c]:.4f}']
                       if t['set'] == 'dev' else ['', '', '']))
    fc.write_csv(os.path.join(out, 'proxy_ids.csv'), header, rows)

    # ---- 5. D0 on the 23 dev IDs, descriptives, predictions
    dev = sorted(dev, key=lambda c: (MACH.index(tab[c]['machine']), tab[c]['id']))
    x = np.array([pi[c] for c in dev])
    y = np.array([delta[c] for c in dev])
    types = [tab[c]['machine'] for c in dev]
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(dev), size=(N_BOOT, len(dev)))
    sat = [c for c in dev if A[(c, 'GS12f_kNN')][0] >= SAT and A[(c, 'X1f_kNN')][0] >= SAT]
    sat_eval = [c for c in evl if A[(c, 'GS12f_kNN')][0] >= SAT and A[(c, 'X1f_kNN')][0] >= SAT]
    main = boot_rho(x, y, idx)
    d0a = len(sat) < len(dev) / 2
    d0b = main['rho'] is not None and main['rho'] >= RHO_MIN
    d0c = main['ci_lo'] is not None and main['ci_lo'] > 0
    within = boot_rho(centred(x, types), centred(y, types), idx)
    desc = dict(
        within_type=within,
        pi_b=boot_rho(np.array([pib[c] for c in dev]), y, idx),
        pi_pAUC=boot_rho(np.array([pip[c] for c in dev]), y, idx),
        pi_b_pAUC=boot_rho(np.array([pibp[c] for c in dev]), y, idx),
        pearson_pi_delta=(float(stats.pearsonr(x, y)[0]) if len(np.unique(x)) > 1 else None),
        sign_agreement=dict(n=int(np.sum(np.sign(x) == np.sign(y))), of=len(dev)),
        eval_ids_saturated=dict(n=len(sat_eval), of=len(evl),
                                ids=[f'{tab[c]["machine"]}-{tab[c]["id"][3:]}' for c in sat_eval]),
        median_proxyAUC={s: {st: float(np.median([A[(c, s)][0] for c in (dev if st == 'dev' else evl)]))
                             for st in ('dev', 'eval')} for s in SYSTEMS},
        min_proxyAUC={s: {st: float(min(A[(c, s)][0] for c in (dev if st == 'dev' else evl)))
                          for st in ('dev', 'eval')} for s in SYSTEMS},
        n_ids_proxyAUC_exactly_1={s: {st: int(sum(A[(c, s)][0] == 1.0 for c in (dev if st == 'dev' else evl)))
                                      for st in ('dev', 'eval')} for s in SYSTEMS},
        n_ids_proxypAUC_exactly_1={s: {st: int(sum(A[(c, s)][1] == 1.0 for c in (dev if st == 'dev' else evl)))
                                       for st in ('dev', 'eval')} for s in SYSTEMS},
        n_dev_ids_pi_exactly_0=int(np.sum(x == 0)),
        max_abs_pi={st: float(max(abs(pi[c]) for c in (dev if st == 'dev' else evl))) for st in ('dev', 'eval')},
        max_abs_pi_b={st: float(max(abs(pib[c]) for c in (dev if st == 'dev' else evl))) for st in ('dev', 'eval')},
        roundoff_note='every proxyAUC is 1.0 or 1 - 1.1e-16 (sklearn trapezoid roundoff); the nonzero pi / pi_b are '
                      'that roundoff, so rho(pi, Delta) and its CI are computed on float noise, not on a signal '
                      '(D0(a) is the part of the rule that says so)' if all(
                          abs(A[(c, s)][0] - 1) < 1e-12 for c in tab for s in SYSTEMS) else None,
        n_ids_perfectly_separated={s: {st: int(sum(gap[(c, s)] > 0 for c in (dev if st == 'dev' else evl)))
                                       for st in ('dev', 'eval')} for s in SYSTEMS},
        separation_margin_z={s: {st: dict(min=float(min(gap[(c, s)] for c in (dev if st == 'dev' else evl))),
                                          median=float(np.median([gap[(c, s)] for c in (dev if st == 'dev' else evl)])))
                                 for st in ('dev', 'eval')} for s in SYSTEMS},
        separation_margin_note='min over negatives minus max over positives of the system score (sum of w * zKp, '
                               'in reference-sd units of each member); > 0 means perfectly separated',
        frac_dev_ids_pi_pos=float(np.mean(x > 0)), frac_eval_ids_pi_pos=float(np.mean([pi[c] > 0 for c in evl])),
        note='scratch_block equals GS12f_kNN in the proxy (both are the 12 from-scratch zKp at weight 1), so '
             'pi_b = proxyAUC(beats_block) - proxyAUC(GS12f_kNN)')
    for c in tab:
        assert A[(c, 'scratch_block')] == A[(c, 'GS12f_kNN')]
    g1 = desc['median_proxyAUC']['GS12f_kNN']['dev']
    passed = bool(d0a and d0b and d0c)
    pred = dict(
        G1=dict(predicted='0.995 [0.97, 1.000]', outcome=g1, inside_80pct=inside(g1, 0.97, 1.0)),
        G2=dict(predicted='D0(a) saturated (fails): 45 %', outcome_saturated=not d0a, n_saturated=len(sat)),
        G3=dict(predicted='+0.10 [-0.30, +0.45]', outcome=main['rho'], inside_80pct=inside(main['rho'], -0.30, 0.45)),
        G4=dict(predicted='D0 fails (80 %)', outcome_D0_failed=not passed),
        G5=dict(predicted='+0.05 [-0.35, +0.40]', outcome=within['rho'], inside_80pct=inside(within['rho'], -0.35, 0.40)))
    J = dict(
        D0=dict(a=dict(rule='n dev IDs with proxyAUC >= 0.999 under BOTH GS12f-kNN and X1f-kNN < 23/2',
                       n_saturated=len(sat), of=len(dev), passed=bool(d0a),
                       ids=[f'{tab[c]["machine"]}-{tab[c]["id"][3:]}' for c in sat]),
                b=dict(rule='Spearman rho(pi, Delta) >= 0.40', rho=main['rho'], passed=bool(d0b)),
                c=dict(rule='95 % bootstrap CI over IDs (2000 draws, default_rng(0)) lower bound > 0',
                       ci=[main['ci_lo'], main['ci_hi']], passed=bool(d0c), n_nan_draws=main['n_nan_draws']),
                passed=passed, stage2='opened' if passed else 'not opened: the study ends at stage 1'),
        rho_pi_delta=main, descriptive=desc, predictions=pred,
        checks=dict(test_zKp_vs_member_rules_max_abs_diff={k: f'{v:.3e}' for k, v in sanity.items()},
                    dev_reproduction={n: {k: (rd(v, 6) if 'diff' not in k else f'{v:.3e}') for k, v in r.items()}
                                      for n, r in repro.items()},
                    note='every member (not only one fc and one BEATs) is checked; truth is dev only, no eval metric '
                         'of any system is computed'),
        setup=dict(members=[n for n, _, _ in mem], weights=W, n_dev_ids=len(dev), n_eval_ids=len(evl),
                   boot=dict(n=N_BOOT, seed=BOOT_SEED, resample='dev IDs with replacement, rng.integers(0, 23, (2000, 23))',
                             ci='percentile 2.5 / 97.5'),
                   pAUC='sklearn roc_auc_score(max_fpr=0.1), as eval.py'),
        post_hoc='every network and every eval score of every member was seen before this study (phase-3 rule 2)')
    with open(os.path.join(out, 'stage1.json'), 'w') as f:
        json.dump(J, f, indent=2)
    print(json.dumps({k: J[k] for k in ('D0', 'predictions')}, indent=2))
    print(json.dumps({k: (v['rho'], v['ci_lo'], v['ci_hi']) for k, v in desc.items()
                      if isinstance(v, dict) and 'rho' in v}, indent=2))


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    dest = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'scores')
    if stage == 'stage1':
        stage1(dest)
    else:
        sys.exit(__doc__)