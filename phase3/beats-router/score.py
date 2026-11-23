#!/usr/bin/env python
"""beats-router: score the 64 machine-type routers between X1f and GS12f (README, pre-registered at 8355114).

    python phase3/beats-router/score.py score  [out]   # everything; writes routers.csv, machine_delta.csv,
                                                       # crossbase.csv, bootstrap.json, decisions.json
    python phase3/beats-router/score.py decide [out]   # decisions.json only, from scores/{routers.csv,
                                                       # machine_delta.csv, crossbase.csv, bootstrap.json, cost.json}
Run both through run.sh (independent processes until two outputs are byte-identical; smartlab1's RAM).

Adds no scoring rule. Members, weights and every score vector come from ../../synthesis/ensemble-g8/score.py's
machinery (imported, which imports fresh-confirm -> grand-ensemble): S1' per from-scratch member, zK' per
BEATs member, each vector computed twice from separate reads. Metrics are fresh-confirm's `metrics` (fc.COLS),
the paired bootstrap is grand-ensemble's `paired_boot` (2000 draws, seed 0, stratified machine x ID x label).

Router R_M: per clip, X1f's score if the clip's machine type is in M, GS12f's otherwise (BEATs weight 1.5 on
M, 0 elsewhere). Every metric is computed within a machine type, and every member score is z-scored per
machine ID on that ID's own training clips, so R_M equals X1f on M and GS12f off M, machine by machine. That
is asserted, not assumed: all 64 routers are scored from their mixed vectors and their per-machine rows must
equal the base systems' rows; R* is also built directly (scratch block + 1.5 * type-masked BEATs block).

Per-machine bootstrap: the same draws as paired_boot (rng = default_rng(ge.RNG_SEED), ev.resample), each draw
scored with ev(v, idx), which already returns per-machine AUC / pAUC; no new metric helper is needed. The
identity resample is checked against the unresampled metrics.
"""
import csv
import importlib.util
import itertools
import json
import os
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
MACH = list(ge.MACH)                         # fan, pump, slider, ToyCar, ToyConveyor, valve -- bitmask order
DS = ('dev', 'eval')
TOL = eg.TOL                                 # 5e-4 Amean against ensemble-g8's committed record
README_TOL = 5e-3                            # README states R* to 3 decimals
README_RSTAR = {'dev': 95.264, 'eval': 95.397}
RSTAR_README = ('pump', 'ToyCar', 'valve')   # the README's stated R*; recomputed and asserted below
COMMITTED = os.path.join(REPO, 'synthesis', 'ensemble-g8', 'scores', 'ensembles.csv')
SRC_EMB = '/home/f74134118/OS-SCL-syn-fresh/synthesis/ensemble-g8/results/emb'   # read-only
EMB = os.path.join(HERE, 'results', 'emb')   # symlink tree onto SRC_EMB (run.sh links); flat/ cache lands here
G_SEEDS_OLD, G_SEEDS_NEW, S_SEEDS = (2060, 2061, 2062, 2063), (2064, 2065, 2066, 2067), (2060, 2061, 2062, 2063)
D1A_MAX, D1B_MIN = 0.70, -0.20               # README D1(a), D1(b)
D3_RATE = 0.90                               # README D3
D_CHUNKS = 64


# ----------------------------------------------------------------------------- small helpers
def mask_of(M):
    return ''.join('1' if m in M else '0' for m in MACH)


def types_of(mask):
    return [m for m, b in zip(MACH, mask) if b == '1']


def name_of(M):
    return '+'.join(m for m in MACH if m in M) or 'none'


def read(path):
    with open(path, encoding='UTF-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, header, rows):
    fc.write_csv(path, header, rows)


def f4(x):
    return f'{x:.4f}'


def amean_m(res, m):
    return 100 * (res[m]['AUC'] + res[m]['pAUC']) / 2


# ----------------------------------------------------------------------------- vectors
def systems():
    """Score vectors of X1f, GS12f and the cross-base systems, exactly as ensemble-g8 builds them."""
    m = ge.meta()
    ge.check_meta(os.path.join(EMB, 'meta.npz'), m)
    go = [(fc.flat_path(EMB, f'{N}fc_nogain_s{s}', m), 'G') for s in G_SEEDS_OLD]
    gn = [(fc.flat_path(EMB, f'{N}fc_nogain_s{s}', m), 'G') for s in G_SEEDS_NEW]
    sm = [(fc.flat_path(EMB, f'{N}fc_spec_s{s}', m), 'S') for s in S_SEEDS]
    bm = [(ge.dump_path(r), 'B') for r in fc.BEATS]
    one = {'G': 1.0, 'S': 1.0}
    V = {'GS12f': fc.ens_vector(go + gn + sm, one),
         'X1f': fc.ens_vector(go + gn + sm + bm, {'G': 1.0, 'S': 1.0, 'B': 1.5}),
         'G8f': fc.ens_vector(go + gn, one),
         'XG8': fc.ens_vector(go + gn + bm, {'G': 1.0, 'B': 1.0}),
         'GS8': fc.ens_vector(go + sm, fc.W_GS),
         'X1': fc.ens_vector(go + sm + bm, fc.W_X1)}
    B = fc.ens_vector(bm, {'B': 1.0})        # the BEATs block alone, for the direct R* construction
    members = dict(G=[os.path.basename(p) for p, _ in go + gn], S=[os.path.basename(p) for p, _ in sm], B=fc.BEATS)
    return V, B, members


def route(on, off, M, mach_all):
    """Per clip: `on`'s score where the clip's machine type is in M, `off`'s elsewhere."""
    sel = np.isin(mach_all, list(M))
    return np.where(sel, on, off)


# ----------------------------------------------------------------------------- per-machine bootstrap
_PB = {}


def _pm_chunk(chunk):
    ev, vecs, idxs = _PB['ev'], _PB['vecs'], _PB['idxs']
    out = []
    for i in chunk:
        res = {n: ev(v, idxs[i]) for n, v in vecs.items()}
        out.append({n: [[100 * r[mm]['AUC'], 100 * r[mm]['pAUC'], 100 * r[mm]['mAUC']] for mm in MACH]
                    for n, r in res.items()})
    return out


def per_machine_boot(ev, vecs, n_boot=ge.N_BOOT, seed=ge.RNG_SEED, procs=8):
    """{name: array (n_boot, 6 machines, [AUC, pAUC, mAUC])} on paired_boot's exact draws."""
    import multiprocessing as mp
    rng = np.random.default_rng(seed)
    idxs = [ev.resample(rng) for _ in range(n_boot)]
    _PB.update(ev=ev, vecs=vecs, idxs=idxs)
    chunks = [list(range(i, n_boot, D_CHUNKS)) for i in range(D_CHUNKS)]
    with mp.get_context('fork').Pool(procs) as pool:
        parts = pool.map(_pm_chunk, chunks)
    arr = {n: np.zeros((n_boot, len(MACH), 3)) for n in vecs}
    for chunk, out in zip(chunks, parts):
        for i, o in zip(chunk, out):
            for n in vecs:
                arr[n][i] = o[n]
    return arr


def router_overall(pm, M):
    """Overall Amean per draw of R_M, from the base systems' per-machine draws (separability)."""
    sel = np.array([m in M for m in MACH])
    a = np.where(sel[None, :, None], pm['X1f'], pm['GS12f'])
    return (a[:, :, 0].mean(1) + a[:, :, 1].mean(1)) / 2


def ci(x):
    return [float(np.mean(x)), *map(float, np.percentile(x, [2.5, 97.5]))]


# ----------------------------------------------------------------------------- stage: score
def score(out):
    os.makedirs(out, exist_ok=True)
    m = ge.meta()
    ev = {ds: ge.evaluator(ds) for ds in DS}
    mach_all = m['te_machine']
    V, B, members = systems()

    # ---- 1. reproduction of ensemble-g8's committed systems (abort on any miss)
    E, P = {}, {}
    for name, v in V.items():
        for ds in DS:
            E[(name, ds)], P[(name, ds)] = fc.metrics(ev[ds], v)
    checks = []
    for x in read(COMMITTED):
        if x['machine'] == 'all' and x['system'] in V:
            checks.append([x['system'], x['dataset'], E[(x['system'], x['dataset'])]['Amean'], float(x['Amean'])])
    assert len(checks) == 2 * len(V), checks
    for c in checks:
        print(f'check {c[0]} {c[1]}: {c[2]:.4f} vs ensemble-g8 {c[3]:.4f}', flush=True)
    worst = max(abs(g - w) for _, _, g, w in checks)
    assert worst < TOL, f'ABORT: systems differ from ensemble-g8/scores/ensembles.csv by {worst:.5f} Amean'

    # ---- 2. all 64 routers, scored from their mixed vectors; per-machine rows must equal the bases'
    R, RP, rows, sep = {}, {}, [], 0.0
    for k in range(len(MACH) + 1):
        for M in itertools.combinations(MACH, k):
            mask = mask_of(M)
            v = route(V['X1f'], V['GS12f'], M, mach_all)
            for ds in DS:
                row, per = fc.metrics(ev[ds], v)
                R[(mask, ds)], RP[(mask, ds)] = row, per
                for mm in MACH:
                    base = P[('X1f' if mm in M else 'GS12f', ds)][mm]
                    sep = max(sep, max(abs(per[mm][c] - base[c]) for c in fc.COLS))
                rows.append([mask, name_of(M), len(M), ds, 'all'] + [f4(row[c]) for c in fc.COLS])
                rows += [[mask, name_of(M), len(M), ds, mm] + [f4(per[mm][c]) for c in fc.COLS] for mm in MACH]
    assert sep < 1e-9, f'ABORT: a router changes an off-type machine by {sep:.2e} (separability broken)'
    assert R[('111111', 'dev')] == E[('X1f', 'dev')] and R[('000000', 'eval')] == E[('GS12f', 'eval')]
    write_csv(os.path.join(out, 'routers.csv'),
              ['router', 'on_types', 'n_on', 'dataset', 'machine', *fc.COLS], rows)

    # ---- 3. R* (dev), its direct construction, and the README's stated values
    dmach = {ds: {mm: P[('X1f', ds)][mm]['Amean'] - P[('GS12f', ds)][mm]['Amean'] for mm in MACH} for ds in DS}
    RS = tuple(mm for mm in MACH if dmach['dev'][mm] > 0)
    assert RS == RSTAR_README, f'ABORT: dev R* is {RS}, README states {RSTAR_README}'
    rs = mask_of(RS)
    scratch = V['GS12f']
    direct = scratch + 1.5 * np.where(np.isin(mach_all, list(RS)), B, 0.0)
    direct_chk = {}
    for ds in DS:
        row, per = fc.metrics(ev[ds], direct)
        direct_chk[ds] = dict(direct=row['Amean'], mixed=R[(rs, ds)]['Amean'], readme=README_RSTAR[ds],
                              max_abs_diff_all_cols=max(abs(row[c] - R[(rs, ds)][c]) for c in fc.COLS))
        assert direct_chk[ds]['max_abs_diff_all_cols'] < 1e-9, ('ABORT: direct R* != mixed R*', direct_chk)
        assert abs(row['Amean'] - README_RSTAR[ds]) < README_TOL, ('ABORT: R* != README', direct_chk)
        print(f'R* {ds}: direct {row["Amean"]:.4f}  mixed {R[(rs, ds)]["Amean"]:.4f}  README {README_RSTAR[ds]}', flush=True)
    # X1f's own vector = scratch + 1.5 * B (bit-for-bit is not required: summation order differs)
    x1f_rebuilt = max(abs(fc.metrics(ev[ds], scratch + 1.5 * B)[0]['Amean'] - E[('X1f', ds)]['Amean']) for ds in DS)
    assert x1f_rebuilt < 1e-9, x1f_rebuilt

    # ---- 4. per-machine bootstrap on paired_boot's draws -> machine CIs, R_ci, D3
    for ds in DS:   # identity resample == unresampled metrics
        res = ev[ds](V['X1f'], np.arange(len(ev[ds].sel)))
        idd = max(abs(amean_m(res, mm) - P[('X1f', ds)][mm]['Amean']) for mm in MACH)
        assert idd < 1e-9, ('identity resample', ds, idd)
    pm = {ds: per_machine_boot(ev[ds], {'X1f': V['X1f'], 'GS12f': V['GS12f']}) for ds in DS}
    dm_rows, mdelta, d3 = [], {}, {}
    for ds in DS:
        am = {n: (a[:, :, 0] + a[:, :, 1]) / 2 for n, a in pm[ds].items()}
        dlt = am['X1f'] - am['GS12f']                         # (n_boot, 6)
        mdelta[ds] = {}
        for j, mm in enumerate(MACH):
            c = ci(dlt[:, j])
            mdelta[ds][mm] = dict(delta=dmach[ds][mm], boot_mean=c[0], lo=c[1], hi=c[2],
                                  frac_pos=float(np.mean(dlt[:, j] > 0)))
            dm_rows.append([ds, mm, f4(dmach[ds][mm]), f4(c[0]), f4(c[1]), f4(c[2]), f4(mdelta[ds][mm]['frac_pos'])])
        on_b = dlt > 0
        obs_own = np.array([dmach[ds][mm] > 0 for mm in MACH])
        obs_dev = np.array([mm in RS for mm in MACH])
        d3[ds] = dict(frac_draws_set_equals_dev_Rstar=float(np.mean((on_b == obs_dev).all(1))),
                      frac_draws_set_equals_own_observed=float(np.mean((on_b == obs_own).all(1))),
                      own_observed_set=[mm for mm, b in zip(MACH, obs_own) if b])
    write_csv(os.path.join(out, 'machine_delta.csv'),
              ['dataset', 'machine', 'delta_X1f_minus_GS12f', 'boot_mean', 'ci_lo', 'ci_hi', 'frac_draws_pos'], dm_rows)
    RCI = tuple(mm for mm in MACH if mdelta['dev'][mm]['lo'] > 0)
    rci = mask_of(RCI)

    # ---- 5. paired bootstrap (ge.paired_boot, same draws/seed as ensemble-g8)
    vecs = {'X1f': V['X1f'], 'GS12f': V['GS12f'], 'Rstar': route(V['X1f'], V['GS12f'], RS, mach_all)}
    pairs = [('Rstar', 'X1f'), ('Rstar', 'GS12f'), ('X1f', 'GS12f')]
    rci_note = None
    if rci == rs:
        rci_note = 'R_ci equals R*; its pairs are the R* pairs'
    elif rci == '000000':
        rci_note = 'R_ci is empty (= GS12f); R_ci - GS12f is identically 0 and R_ci - X1f is GS12f - X1f'
    else:
        vecs['Rci'] = route(V['X1f'], V['GS12f'], RCI, mach_all)
        pairs += [('Rci', 'X1f'), ('Rci', 'GS12f')]
    boot = {ds: ge.paired_boot(ev[ds], vecs, pairs) for ds in DS}
    # cross-checks: the per-machine pass reproduces paired_boot (same draws), and X1f-GS12f equals ensemble-g8's
    for ds in DS:
        for a, b, Ma, Mb in (('Rstar', 'X1f', RS, MACH), ('Rstar', 'GS12f', RS, ()), ('X1f', 'GS12f', MACH, ())):
            d = router_overall(pm[ds], Ma) - router_overall(pm[ds], Mb)
            got, want = ci(d), boot[ds][f'{a}-{b}']['mean_AUC_pAUC']
            assert max(abs(g - w) for g, w in zip(got, want)) < 1e-9, ('per-machine pass != paired_boot', ds, a, b, got, want)
    with open(os.path.join(REPO, 'synthesis', 'ensemble-g8', 'scores', 'bootstrap.json'), encoding='UTF-8') as f:
        egb = json.load(f)
    eg_diff = max(abs(g - w) for ds in DS for g, w in zip(boot[ds]['X1f-GS12f']['mean_AUC_pAUC'],
                                                         egb[ds]['X1f-GS12f']['mean_AUC_pAUC']))
    assert eg_diff < TOL, ('X1f-GS12f bootstrap differs from ensemble-g8', eg_diff)

    # ---- 6. cross-base (descriptive): per-machine deltas and R* applied to each base pair
    xb_rows, xb = [], {}
    for ext, base in (('X1f', 'GS12f'), ('XG8', 'G8f'), ('X1', 'GS8')):
        key = f'{ext}-{base}'
        d = {ds: {mm: P[(ext, ds)][mm]['Amean'] - P[(base, ds)][mm]['Amean'] for mm in MACH} for ds in DS}
        agree = {mm: bool(np.sign(d['dev'][mm]) == np.sign(d['eval'][mm])) for mm in MACH}
        own = tuple(mm for mm in MACH if d['dev'][mm] > 0)
        val = {}
        for tag, M in (('Rstar', RS), ('own_dev_set', own)):
            v = route(V[ext], V[base], M, mach_all)
            val[tag] = {ds: fc.metrics(ev[ds], v)[0]['Amean'] for ds in DS}
        xb[key] = dict(delta=d, sign_agree=agree, n_agree=int(sum(agree.values())), own_dev_set=list(own),
                       Rstar_applied=val['Rstar'], own_dev_set_applied=val['own_dev_set'],
                       ext={ds: E[(ext, ds)]['Amean'] for ds in DS}, base={ds: E[(base, ds)]['Amean'] for ds in DS})
        for mm in MACH:
            xb_rows.append([key, mm, f4(d['dev'][mm]), f4(d['eval'][mm]), int(agree[mm])])
        xb_rows.append([key, 'all', f4(E[(ext, 'dev')]['Amean'] - E[(base, 'dev')]['Amean']),
                        f4(E[(ext, 'eval')]['Amean'] - E[(base, 'eval')]['Amean']), xb[key]['n_agree']])
    write_csv(os.path.join(out, 'crossbase.csv'), ['pair', 'machine', 'delta_dev', 'delta_eval', 'sign_agree'], xb_rows)

    # ---- 7. bootstrap.json: pairs + per-machine CIs + D3 + all inputs to decide()
    clips = {ds: {mm: int(np.sum((m['te_dataset'] == ds) & (mach_all == mm))) for mm in MACH} for ds in DS}
    boot_out = dict(pairs=boot, machine_delta=mdelta, d3=d3, Rstar=list(RS), Rstar_mask=rs, Rci=list(RCI),
                    Rci_mask=rci, Rci_note=rci_note, crossbase=xb, test_clips=clips,
                    checks=dict(max_abs_diff_vs_ensemble_g8=float(worst), separability_max_abs=float(sep),
                                Rstar_direct_vs_mixed=direct_chk, X1f_bootstrap_vs_ensemble_g8=float(eg_diff),
                                note='per-machine bootstrap pass reproduces paired_boot for R*-X1f, R*-GS12f, '
                                     'X1f-GS12f to 1e-9; identity resample = unresampled metrics'),
                    draws=dict(n_boot=ge.N_BOOT, seed=ge.RNG_SEED, resampler='ensemble-backend Evaluator.resample '
                               '(machine x ID x label strata)'),
                    members=members)
    with open(os.path.join(out, 'bootstrap.json'), 'w') as f:
        json.dump(boot_out, f, indent=2)
    decide(out, out)


# ----------------------------------------------------------------------------- stage: decide (no recomputation)
COST_KINDS = ('fc_nogain', 'fc_spec', 'beats')
COST_METRICS = ('flops', 'lat_b32_ms_per_clip', 'lat_b1_ms', 'params')   # flops = primary unit (README)


def cost_fractions(cost, rs_types, clips):
    """Type-weighted (and clip-weighted) cost of every router as a fraction of X1f's, per cost metric."""
    mc = cost.get('members', {})
    missing = [(k, x) for k in COST_KINDS for x in COST_METRICS if not isinstance(mc.get(k, {}).get(x), (int, float))]
    if missing:
        return None, f'cost.json lacks {missing}'
    fm = {k: {x: float(mc[k][x]) for x in COST_METRICS} for k in COST_KINDS}
    metrics_ = COST_METRICS
    out = {}
    for x in metrics_:
        cG = 8 * fm['fc_nogain'][x] + 4 * fm['fc_spec'][x]
        cB = 3 * fm['beats'][x]
        tot = cG + cB
        r = {}
        for k in range(len(MACH) + 1):
            for M in itertools.combinations(MACH, k):
                tw = (cG + cB * len(M) / 6) / tot
                cw = {ds: (cG + cB * sum(clips[ds][mm] for mm in M) / sum(clips[ds].values())) / tot for ds in DS}
                r[mask_of(M)] = dict(type_weighted=tw, clip_weighted=cw)
        out[x] = dict(per_member={k: fm[k][x] for k in fm}, X1f_total=tot, GS12f_total=cG,
                      beats_share_of_X1f=cB / tot, routers=r)
    return out, None


def inside(v, lo, hi):
    return bool(lo <= v <= hi)


def decide(src, out):
    rows = read(os.path.join(src, 'routers.csv'))
    with open(os.path.join(src, 'bootstrap.json'), encoding='UTF-8') as f:
        B = json.load(f)
    A = {(r['router'], r['dataset'], r['machine']): {c: float(r[c]) for c in fc.COLS} for r in rows}
    rs, rci, RS = B['Rstar_mask'], B['Rci_mask'], B['Rstar']
    pairs = B['pairs']
    am = lambda mask, ds: A[(mask, ds, 'all')]['Amean']                     # noqa: E731
    pci = lambda key, ds='eval': pairs[ds][key]['mean_AUC_pAUC']            # noqa: E731  [mean, lo, hi]
    D = {}
    D['systems'] = {name: {ds: A[(mask, ds, 'all')] for ds in DS}
                    for name, mask in (('X1f', '111111'), ('GS12f', '000000'), ('Rstar', rs), ('Rci', rci))}
    D['Rstar'] = dict(types=RS, mask=rs, rule='M = {m : dev Amean_m(X1f) - Amean_m(GS12f) > 0}',
                      minus_X1f={ds: am(rs, ds) - am('111111', ds) for ds in DS},
                      minus_GS12f={ds: am(rs, ds) - am('000000', ds) for ds in DS},
                      dev_rank_of_64=1 + sum(am(r['router'], 'dev') > am(rs, 'dev') for r in rows
                                             if r['machine'] == 'all' and r['dataset'] == 'dev'),
                      eval_rank_of_64=1 + sum(am(r['router'], 'eval') > am(rs, 'eval') for r in rows
                                              if r['machine'] == 'all' and r['dataset'] == 'eval'))
    D['Rci'] = dict(types=B['Rci'], mask=rci, note=B['Rci_note'],
                    dev_ci_by_machine={mm: [B['machine_delta']['dev'][mm][k] for k in ('delta', 'lo', 'hi')] for mm in MACH})

    # ---- cost (D1(a), P1-P3)
    cp = os.path.join(src, 'cost.json')
    if not os.path.exists(cp):
        cp = os.path.join(HERE, 'scores', 'cost.json')
    fr, why = (None, 'scores/cost.json not present')
    if os.path.exists(cp):
        with open(cp, encoding='UTF-8') as f:
            cost = json.load(f)
        if 'clips' in cost:
            assert cost['clips'] == B['test_clips'], ('cost.json clip counts != meta', cost['clips'], B['test_clips'])
        fr, why = cost_fractions(cost, RS, B['test_clips'])
        if fr and 'x1f' in cost:      # cost.py's own totals agree with the formula used here
            assert abs(fr['flops']['X1f_total'] - cost['x1f']['flops_total']) < 1, 'X1f FLOPs total'
            assert abs(fr['flops']['beats_share_of_X1f'] - cost['x1f']['beats_share_flops']) < 1e-9
            assert abs(fr['lat_b32_ms_per_clip']['beats_share_of_X1f'] - cost['x1f']['beats_share_lat_b32']) < 1e-9
    fl, lat32 = ('flops', 'lat_b32_ms_per_clip') if fr else (None, None)
    if fr:
        D['cost'] = {x: dict(per_member=v['per_member'], X1f_total=v['X1f_total'], GS12f_total=v['GS12f_total'],
                             beats_share_of_X1f=v['beats_share_of_X1f'], Rstar=v['routers'][rs],
                             Rci=v['routers'][rci]) for x, v in fr.items()}
        D['cost_source'] = os.path.relpath(cp, HERE)
    else:
        D['cost'] = dict(pending=why)

    # ---- D1
    lo_x1f, lo_gs = pci('Rstar-X1f')[1], pci('Rstar-GS12f')[1]
    d1b, d1c = bool(lo_x1f >= D1B_MIN), bool(lo_gs > 0)
    if fl:
        f_rs = fr[fl]['routers'][rs]['type_weighted']
        d1a = bool(f_rs <= D1A_MAX)
        d1 = dict(a=dict(metric=fl, Rstar_type_weighted_fraction=f_rs, threshold=D1A_MAX, passed=d1a,
                         Rstar_clip_weighted_fraction=fr[fl]['routers'][rs]['clip_weighted']))
    else:
        d1a = None
        d1 = dict(a=dict(pending=f'{why}; rerun `run.sh decide` once scores/cost.json exists (reads routers.csv + '
                                 'bootstrap.json + cost.json, no recomputation)', threshold=D1A_MAX))
    d1['b'] = dict(eval_ci_Rstar_minus_X1f=pci('Rstar-X1f'), lower_bound=lo_x1f, threshold=D1B_MIN, passed=d1b)
    d1['c'] = dict(eval_ci_Rstar_minus_GS12f=pci('Rstar-GS12f'), lower_bound=lo_gs, passed=d1c)
    d1['passed'] = (bool(d1a and d1b and d1c) if d1a is not None else
                    (False if not (d1b and d1c) else 'pending D1(a)'))
    d1['dev_ci'] = dict(Rstar_minus_X1f=pci('Rstar-X1f', 'dev'), Rstar_minus_GS12f=pci('Rstar-GS12f', 'dev'))
    D['D1'] = d1
    # ---- D2
    D['D2'] = dict(eval_ci_Rstar_minus_X1f=pci('Rstar-X1f'), passed=bool(pci('Rstar-X1f')[1] > 0))
    # ---- D3
    md = B['machine_delta']
    decided = {mm: bool(md['dev'][mm]['frac_pos'] >= D3_RATE or md['dev'][mm]['frac_pos'] <= 1 - D3_RATE) for mm in MACH}
    D['D3'] = dict(frac_dev_draws_delta_pos={mm: md['dev'][mm]['frac_pos'] for mm in MACH},
                   decided_ge_90pct=decided, passed=bool(all(decided.values())),
                   frac_dev_draws_set_equals_Rstar=B['d3']['dev']['frac_draws_set_equals_dev_Rstar'],
                   eval_descriptive=dict(frac_eval_draws_delta_pos={mm: md['eval'][mm]['frac_pos'] for mm in MACH},
                                         frac_eval_draws_set_equals_dev_Rstar=B['d3']['eval']['frac_draws_set_equals_dev_Rstar'],
                                         eval_observed_set=B['d3']['eval']['own_observed_set'],
                                         frac_eval_draws_set_equals_eval_observed=B['d3']['eval']['frac_draws_set_equals_own_observed']))
    D['machine_delta'] = md
    D['bootstrap_pairs'] = pairs
    D['crossbase'] = {k: dict(n_sign_agree=v['n_agree'], sign_agree=v['sign_agree'], own_dev_set=v['own_dev_set'],
                              Rstar_applied_eval=v['Rstar_applied']['eval'], Rstar_applied_dev=v['Rstar_applied']['dev'],
                              ext_eval=v['ext']['eval'], base_eval=v['base']['eval'],
                              Rstar_applied_minus_ext_eval=v['Rstar_applied']['eval'] - v['ext']['eval'],
                              own_dev_set_applied_eval=v['own_dev_set_applied']['eval'])
                      for k, v in B['crossbase'].items()}

    # ---- predictions
    Pr = {}
    if fl:
        s = fr[fl]['beats_share_of_X1f'] * 100
        Pr['P1'] = dict(predicted='95 % [88, 99]', outcome=s, inside_80pct=inside(s, 88, 99))
        f = fr[fl]['routers'][rs]['type_weighted']
        Pr['P2'] = dict(predicted='0.53 [0.50, 0.62]', outcome=f, inside_80pct=inside(f, 0.50, 0.62))
    else:
        Pr['P1'] = Pr['P2'] = dict(pending=why)
    if lat32:
        f = fr[lat32]['routers'][rs]['type_weighted']
        Pr['P3'] = dict(predicted='0.58 [0.50, 0.75]', metric=lat32, outcome=f, inside_80pct=inside(f, 0.50, 0.75))
    else:
        Pr['P3'] = dict(pending=why if not fr else 'no batch-32 latency metric found in cost.json')
    Pr['P4'] = dict(predicted='lower bound -0.26 [-0.36, -0.17]; D1(b) fails (60 %)', outcome=lo_x1f,
                    inside_80pct=inside(lo_x1f, -0.36, -0.17), D1b_failed=not d1b)
    Pr['P5'] = dict(predicted='eval CI of R* - GS12f excludes 0 from above: yes (85 %)', outcome=d1c,
                    ci=pci('Rstar-GS12f'))
    fp = D['D3']['frac_dev_draws_delta_pos']
    Pr['P6'] = dict(predicted='pump, ToyCar on in >= 95 % of dev draws; fan in 40-75 %; D3 fails (75 %)',
                    pump=fp['pump'], ToyCar=fp['ToyCar'], fan=fp['fan'],
                    pump_ToyCar_ge_95=bool(fp['pump'] >= 0.95 and fp['ToyCar'] >= 0.95),
                    fan_in_40_75=inside(fp['fan'], 0.40, 0.75), D3_failed=not D['D3']['passed'])
    Pr['P7'] = dict(predicted='{pump, ToyCar} (55 %); {pump} (25 %)', outcome=B['Rci'],
                    matches=('{pump, ToyCar}' if B['Rci'] == ['pump', 'ToyCar'] else
                             '{pump}' if B['Rci'] == ['pump'] else 'neither'))
    Pr['P8'] = dict(predicted='3 of 6 machines per base [2, 4]',
                    outcome={k: v['n_agree'] for k, v in B['crossbase'].items()},
                    inside_80pct={k: inside(v['n_agree'], 2, 4) for k, v in B['crossbase'].items()})
    D['predictions'] = Pr
    D['checks'] = B['checks']
    D['post_hoc'] = ('every network and every eval score here was seen before this study (phase-3 rule 2); '
                     'the point estimates are arithmetic from ensemble-g8 (README)')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'decisions.json'), 'w') as f:
        json.dump(D, f, indent=2)
    print(json.dumps({k: D[k] for k in ('Rstar', 'Rci', 'D1', 'D2', 'D3', 'predictions')}, indent=2))


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    dest = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'scores')
    if stage == 'score':
        score(dest)
    elif stage == 'decide':
        decide(os.path.join(HERE, 'scores'), dest)
    else:
        sys.exit(__doc__)
