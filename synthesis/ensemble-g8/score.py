#!/usr/bin/env python
"""Score the fresh ensembles of this study under the frozen rules; write scores/ (README, *Scoring*).

    python synthesis/ensemble-g8/score.py dryrun [out]   # all of score() on SEEN seeds 2040-47 (pseudo-fresh dumps)
    python synthesis/ensemble-g8/score.py score  [out]   # the fresh runs, ONCE, after the predictions are committed

Adds no scoring rule. Everything is ../fresh-confirm/score.py's machinery (imported): single networks
by ../single-final's score_member (R0, C1, C1ᵀ), ensembles by ../grand-ensemble's ensemble_from_dumps
(S1′ per member, BEATs zK′), each vector computed twice from separate reads. This file only names the
systems of this study, the paired bootstraps, the count-matched subset families and the E1-E7 table.

Systems (members; per-member weight):
  G4     4 old fc_nogain (2060-63); 1          already scored by fresh-confirm -- recomputed and asserted equal
  G4b    4 new fc_nogain (2064-67); 1
  G8f    all 8 fc_nogain; 1
  GS8    4 old fc_nogain; 2  + 4 fc_spec; 1    already scored by fresh-confirm -- recomputed and asserted equal
  GS12f  8 fc_nogain + 4 fc_spec; 1            GS12's exact recipe (grand-ensemble's dev pick)
  X1     GS8 + 3 BEATs; 1.5                    already scored -- recomputed and asserted equal   [external]
  X1f    GS12f + 3 BEATs; 1.5                  X1's exact recipe (r = 0.375 of the from-scratch weight) [external]
  XG8    G8f + 3 BEATs; 1.0                    same r = 0.375                                    [external]

The dry run uses ao_nogain_s2040-43 as "old", s2044-47 as "new" and sb_spec_s2040-43, and asserts that
G8f / GS12f / X1f reproduce grand-ensemble's committed G8 / GS12 / X1[r=0.375] and that the "new" single
networks reproduce single-final's committed C1ᵀ / C1.
"""
import csv
import importlib.util
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.dirname(HERE)


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fc = _imp('fc_score', os.path.join(SYN, 'fresh-confirm', 'score.py'))   # also makes np.load CRC-safe
sf, ge = fc.sf, fc.ge
N = fc.N
TOL = 5e-4          # Amean points; a recomputed system against its committed value
CANDIDATES = ('G8f', 'GS8', 'GS12f')        # E2: dev pick, in this order on an exact tie
EXTERNAL_OF = {'G8f': 'XG8', 'GS8': 'X1', 'GS12f': 'X1f'}
PAIRS = [('GS12f', 'G8f'), ('GS8', 'G8f'), ('GS12f', 'GS8'), ('G8f', 'G4'), ('G8f', 'G4b'), ('G4b', 'G4'),
         ('X1f', 'GS12f'), ('XG8', 'G8f'), ('X1f', 'XG8'), ('X1f', 'X1')]


def read(path):
    with open(path, encoding='UTF-8') as f:
        return list(csv.DictReader(f))


def build_pseudo(out, g_seeds, s_seeds):
    """Phase-1 data in extract.py's split format (fresh-confirm/score.py's build_pseudo, any seed list)."""
    tta = '/home/f74134118/OS-SCL-tta/ablation/tta/results/emb'
    pk = '/home/f74134118/OS-SCL-pknn/ablation/pooled-knn/results/emb'
    spec = '/home/f74134118/OS-SCL-spec/ablation/spectrum-branch/results/emb'
    m = ge.meta()
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, 'meta.npz'), **m)
    for s in g_seeds:
        r = f'{N}ao_nogain_s{s}'
        for split in ('train', 'dev', 'eval'):
            p = os.path.join(out, split, f'{r}.npz')
            if os.path.exists(p):
                continue
            t = dict(np.load(os.path.join(tta, split, f'{r}.npz')))
            q = dict(np.load(os.path.join(pk, split, f'{r}.npz')))
            assert np.array_equal(t['idx'], q['idx'])
            sel = [list(t['shifts']).index(f) for f in (0.0, 0.01, -0.01)]
            extra = dict(valid=q['valid']) if split == 'train' else {}
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.savez(p, ce=t['ce'][:, sel], pre=t['pre'][:, sel], gmean=q['gmean'], gmax=q['gmax'],
                     idx=t['idx'], shifts=np.array([0.0, 0.01, -0.01]), **extra)
    for s in s_seeds:
        r = f'{N}sb_spec_s{s}'
        a = dict(np.load(os.path.join(spec, f'{r}.npz')))
        z = lambda n: np.zeros((n, 512), np.float32)      # noqa: E731  gmean unused for S (ensemble only)
        for split in ('train', 'dev', 'eval'):
            p = os.path.join(out, split, f'{r}.npz')
            if os.path.exists(p):
                continue
            if split == 'train':
                d = dict(ce=a['tr_ce'][:, None], pre=a['tr_pre'][:, None], idx=np.arange(len(a['tr_ce'])),
                         valid=a['valid'], gmean=z(len(a['tr_ce'])))
            else:
                ix = np.where(m['te_dataset'] == split)[0]
                d = dict(ce=a['te_ce'][ix, None], pre=a['te_pre'][ix, None], idx=ix, gmean=z(len(ix)))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.savez(p, gmax=d['gmean'], shifts=np.array([0.0]), **d)


def score(emb, out, kinds, old, new, s_seeds, committed, extra_checks=None):
    """committed: directory holding fresh-confirm's single.csv / ensembles.csv for the old members."""
    m = ge.meta()
    ge.check_meta(os.path.join(emb, 'meta.npz'), m)
    ev = {ds: ge.evaluator(ds) for ds in ('dev', 'eval')}
    Go = [f'{N}{kinds[0]}_s{s}' for s in old]
    Gn = [f'{N}{kinds[0]}_s{s}' for s in new]
    S = [f'{N}{kinds[1]}_s{s}' for s in s_seeds]
    missing = [r for r in Go + Gn + S
               if not all(os.path.exists(os.path.join(emb, sp, f'{r}.npz')) for sp in ('train', 'dev', 'eval'))]
    assert not missing, f'no dumps for {missing} (README: no seed is dropped or replaced in this study)'
    print('old G:', Go, '\nnew G:', Gn, '\nS:', S, flush=True)

    # ---- single networks: the new runs are scored here; the old ones are read from their committed record
    rules = ('R0', 'C1', 'C1T')
    single_rows, mach_rows, R = [], [], {}
    for r in Gn:
        v = fc.single_vectors(emb, r, m, rules)
        for ds in ('dev', 'eval'):
            for rule in rules:
                row, per = fc.metrics(ev[ds], v[rule])
                R[(r, ds, rule)] = row
                single_rows.append([r, ds, rule] + [f'{row[k]:.4f}' for k in fc.COLS])
                mach_rows += [[r, ds, rule, mm] + [f'{per[mm][k]:.4f}' for k in fc.COLS] for mm in ge.MACH]
    for x in read(os.path.join(committed, 'single.csv')):
        if x['run'] in Go and x['rule'] in rules:
            R[(x['run'], x['dataset'], x['rule'])] = {k: float(x[k]) for k in fc.COLS}
    assert all((r, ds, rule) in R for r in Go + Gn for ds in ('dev', 'eval') for rule in rules)

    # ---- ensembles
    go = [(fc.flat_path(emb, r, m), 'G') for r in Go]
    gn = [(fc.flat_path(emb, r, m), 'G') for r in Gn]
    sm = [(fc.flat_path(emb, r, m), 'S') for r in S]
    bm = [(ge.dump_path(r), 'B') for r in fc.BEATS]
    one = {'G': 1.0, 'S': 1.0}
    V = {'G4': fc.ens_vector(go, one), 'G4b': fc.ens_vector(gn, one), 'G8f': fc.ens_vector(go + gn, one),
         'GS8': fc.ens_vector(go + sm, fc.W_GS), 'GS12f': fc.ens_vector(go + gn + sm, one),
         'X1': fc.ens_vector(go + sm + bm, fc.W_X1),
         'X1f': fc.ens_vector(go + gn + sm + bm, {'G': 1.0, 'S': 1.0, 'B': 1.5}),
         'XG8': fc.ens_vector(go + gn + bm, {'G': 1.0, 'B': 1.0})}
    NM = {'G4': 4, 'G4b': 4, 'G8f': 8, 'GS8': 8, 'GS12f': 12, 'X1': 11, 'X1f': 15, 'XG8': 11}
    ens_rows, E = [], {}
    for name, v in V.items():
        for ds in ('dev', 'eval'):
            row, per = fc.metrics(ev[ds], v)
            E[(name, ds)] = row
            ens_rows.append([name, NM[name], ds, 'all'] + [f'{row[k]:.4f}' for k in fc.COLS])
            ens_rows += [[name, NM[name], ds, mm] + [f'{per[mm][k]:.4f}' for k in fc.COLS] for mm in ge.MACH]

    # ---- consistency with the committed record: nothing is written if an old system does not reproduce
    checks = []
    for x in read(os.path.join(committed, 'ensembles.csv')):
        if x['machine'] == 'all' and x['system'] in ('G4', 'GS8', 'X1'):
            checks.append((f"{x['system']} {x['dataset']} vs {os.path.relpath(committed, SYN)}",
                           E[(x['system'], x['dataset'])]['Amean'], float(x['Amean'])))
    assert len(checks) == 6, checks
    for name, ds, want, src in (extra_checks or []):
        checks.append((f'{name} {ds} vs {src}', E[(name, ds)]['Amean'], want))
    for label, got, want in checks:
        print(f'check {label}: {got:.4f} vs committed {want:.4f}', flush=True)
    worst = max(abs(g - w) for _, g, w in checks)
    assert worst < TOL, f'recomputed systems differ from the committed record by {worst:.4f} Amean'

    os.makedirs(out, exist_ok=True)
    fc.write_csv(os.path.join(out, 'single.csv'), ['run', 'dataset', 'rule', *fc.COLS], single_rows)
    fc.write_csv(os.path.join(out, 'single_machines.csv'), ['run', 'dataset', 'rule', 'machine', *fc.COLS], mach_rows)
    fc.write_csv(os.path.join(out, 'ensembles.csv'), ['system', 'members', 'dataset', 'machine', *fc.COLS], ens_rows)

    boot = {ds: ge.paired_boot(ev[ds], V, PAIRS) for ds in ('dev', 'eval')}
    with open(os.path.join(out, 'bootstrap.json'), 'w') as f:
        json.dump(boot, f, indent=2)

    # ---- count-matched subset families: 8 members, equal weights, j of them sb_spec (j = 2, 4), against G8f.
    # Reported with a fixed reading rule (README, E3); the paired bootstrap resamples clips, these resample members.
    mv = {p: sf.twice(lambda p=p, t=t: ge.member_vector(p, t, use_cache=False))[0] for p, t in go + gn + sm}
    gall, names = go + gn, Go + Gn
    sub_rows, fam = [], {}
    for j in (2, 4):
        dl = {'dev': [], 'eval': []}
        for gp in itertools.combinations(range(len(gall)), 8 - j):
            for sp in itertools.combinations(range(len(sm)), j):
                v = sum(mv[gall[i][0]] for i in gp) + sum(mv[sm[i][0]] for i in sp)
                for ds in ('dev', 'eval'):
                    a = sf.twice_res(lambda: ev[ds](v))['overall']['mean_AUC_pAUC']
                    dl[ds].append(a - E[('G8f', ds)]['Amean'])
                    sub_rows.append([j, '+'.join(names[i][-4:] for i in gp), '+'.join(S[i][-4:] for i in sp), ds,
                                     f'{a:.4f}', f'{dl[ds][-1]:.4f}'])
        fam[f'S{j}'] = {ds: dict(mean_delta=float(np.mean(x)), min=float(np.min(x)), max=float(np.max(x)),
                                 n_pos=int(np.sum(np.array(x) > 0)), n_subsets=len(x)) for ds, x in dl.items()}
    fc.write_csv(os.path.join(out, 'subsets.csv'), ['n_S', 'G_seeds', 'S_seeds', 'dataset', 'Amean', 'delta_vs_G8f'], sub_rows)

    # ---- E1-E7 (README)
    A = lambda name, ds: E[(name, ds)]['Amean']                                   # noqa: E731
    ci = lambda a, b, ds='eval': boot[ds][f'{a}-{b}']['mean_AUC_pAUC'][1:]        # noqa: E731
    dlt = lambda a, b: {ds: A(a, ds) - A(b, ds) for ds in ('dev', 'eval')}        # noqa: E731
    mean = lambda runs, ds, rule, k='Amean': float(np.mean([R[(r, ds, rule)][k] for r in runs]))  # noqa: E731
    D = {}
    D['E1'] = {name: {ds: E[(name, ds)] for ds in ('dev', 'eval')} for name in ('G4b', 'G8f', 'GS12f')}
    pick = max(CANDIDATES, key=lambda c: (round(A(c, 'dev'), 4), -CANDIDATES.index(c)))
    D['E2'] = dict(dev={c: A(c, 'dev') for c in CANDIDATES}, pick=pick, pick_eval=E[(pick, 'eval')],
                   pick_members=NM[pick])
    conf = bool(ci('GS12f', 'G8f')[0] > 0)
    D['E3'] = dict(GS12f_minus_G8f=dlt('GS12f', 'G8f'), eval_ci=ci('GS12f', 'G8f'), dev_ci=ci('GS12f', 'G8f', 'dev'),
                   confirmed_on_eval=conf,
                   GS8_minus_G8f=dlt('GS8', 'G8f'), GS8_minus_G8f_eval_ci=ci('GS8', 'G8f'),
                   count_matched=fam,
                   type_effect={k: {ds: bool(f[ds]['mean_delta'] > 0 and f[ds]['n_pos'] >= 0.75 * f[ds]['n_subsets'])
                                    for ds in ('dev', 'eval')} for k, f in fam.items()})
    if pick == 'G8f':
        rec, why = 'G8f', 'G8f is the dev pick'
    elif ci(pick, 'G8f')[0] > 0:
        rec, why = pick, f'{pick} - G8f eval CI excludes 0'
    else:
        rec, why = 'G8f', f'{pick} - G8f eval CI includes 0: the simpler build is not shown to be worse'
    D['E4'] = dict(recommended=rec, reason=why, pick_minus_G8f=dlt(pick, 'G8f') if pick != 'G8f' else None,
                   pick_minus_G8f_eval_ci=ci(pick, 'G8f') if pick != 'G8f' else None)
    D['E5'] = dict(G8f_minus_G4=dlt('G8f', 'G4'), G8f_minus_G4_eval_ci=ci('G8f', 'G4'),
                   G8f_minus_G4b=dlt('G8f', 'G4b'), G8f_minus_G4b_eval_ci=ci('G8f', 'G4b'),
                   G4b_minus_G4=dlt('G4b', 'G4'), note='G4b - G4 is one draw of the seed-level noise of a 4-net ensemble')
    n = len(Gn)
    se = lambda sd: sd * np.sqrt(1 / n + 1 / 8)  # noqa: E731
    e6 = dict(new4={ds: mean(Gn, ds, 'C1T') for ds in ('dev', 'eval')},
              bar_dev=fc.P1['c1t_dev'] - 2 * se(fc.P1['sd_dev']), bar_eval=fc.P1['c1t_eval'] - 2 * se(fc.P1['sd_eval']),
              fresh8={rule: {ds: {k: mean(Go + Gn, ds, rule, k) for k in fc.COLS} for ds in ('dev', 'eval')} for rule in rules},
              fresh8_sd={rule: {ds: float(np.std([R[(r, ds, rule)]['Amean'] for r in Go + Gn], ddof=1))
                                for ds in ('dev', 'eval')} for rule in rules})
    e6['passed'] = bool(e6['new4']['dev'] >= e6['bar_dev'] and e6['new4']['eval'] >= e6['bar_eval'])
    for rule in ('C1T', 'R0'):
        for tag, runs in (('new4', Gn), ('fresh8', Go + Gn)):
            cnt = {k: int(sum(R[(r, 'dev', rule)][k] > fc.PAPER[k] for r in runs)) for k in ('AUC', 'pAUC', 'mAUC')}
            cnt['eval_mean'] = int(sum(R[(r, 'eval', rule)]['Amean'] > fc.PAPER['eval_mean'] for r in runs))
            cnt['all4'] = int(sum(all(R[(r, 'dev', rule)][k] > fc.PAPER[k] for k in ('AUC', 'pAUC', 'mAUC'))
                                  and R[(r, 'eval', rule)]['Amean'] > fc.PAPER['eval_mean'] for r in runs))
            e6[f'above_paper_{rule}_{tag}'] = dict(cnt, n=len(runs))
    D['E6'] = e6
    ext = EXTERNAL_OF[pick]
    D['E7'] = dict(label='WITH EXTERNAL PRETRAINING', system=ext, of_pick=pick,
                   level={x: {ds: E[(x, ds)] for ds in ('dev', 'eval')} for x in ('X1f', 'XG8')},
                   X1f_minus_GS12f=dlt('X1f', 'GS12f'), X1f_minus_GS12f_eval_ci=ci('X1f', 'GS12f'),
                   XG8_minus_G8f=dlt('XG8', 'G8f'), XG8_minus_G8f_eval_ci=ci('XG8', 'G8f'),
                   X1f_minus_XG8=dlt('X1f', 'XG8'), X1f_minus_XG8_eval_ci=ci('X1f', 'XG8'))
    D['checks'] = dict(max_abs_diff_Amean=float(worst), n=len(checks))
    D['members'] = dict(G_old=Go, G_new=Gn, S=S, B=fc.BEATS)
    with open(os.path.join(out, 'decisions.json'), 'w') as f:
        json.dump(D, f, indent=2, default=float)
    print(json.dumps(D, indent=2, default=float))


def committed_ge(system, ds, table):
    for x in read(os.path.join(SYN, 'grand-ensemble', 'scores', f'{table}_{ds}.csv')):
        if x['system'] == system and x['scope'] == 'overall':
            return float(x['Amean'])
    raise KeyError((system, ds, table))


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    if stage == 'score':
        score(emb=os.path.join(HERE, 'results', 'emb'), out=sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'scores'),
              kinds=('fc_nogain', 'fc_spec'), old=(2060, 2061, 2062, 2063), new=(2064, 2065, 2066, 2067),
              s_seeds=(2060, 2061, 2062, 2063), committed=os.path.join(SYN, 'fresh-confirm', 'scores'))
    elif stage == 'dryrun':
        emb = os.path.join(HERE, 'results', 'selftest_emb')
        build_pseudo(emb, range(2040, 2048), range(2040, 2044))
        out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'results', 'dryrun_scores')
        extra = [(mine, ds, committed_ge(theirs, ds, table), f'grand-ensemble {theirs}')
                 for mine, theirs, table in (('G8f', 'G8', 'systems'), ('GS12f', 'GS12', 'systems'), ('X1f', 'X1[r=0.375]', 'x1'))
                 for ds in ('dev', 'eval')]
        score(emb=emb, out=out, kinds=('ao_nogain', 'sb_spec'), old=(2040, 2041, 2042, 2043), new=(2044, 2045, 2046, 2047),
              s_seeds=(2040, 2041, 2042, 2043), committed=os.path.join(SYN, 'fresh-confirm', 'scores', 'dryrun_p1'),
              extra_checks=extra)
        # the "new" single networks against single-final's committed per-seed numbers
        ref = {}
        for f in ('metrics_stageA.csv', 'metrics_stageB.csv', 'metrics_eval.csv'):
            for row in read(os.path.join(SYN, 'single-final', 'scores', f)):
                if row['machine'] == 'overall':
                    ref[(row['system'], row['dataset'], row['rule'])] = float(row['Amean'])
        worst = max(abs(float(x['Amean']) - ref[(x['run'], x['dataset'], x['rule'])])
                    for x in read(os.path.join(out, 'single.csv')) if x['rule'] in ('C1', 'C1T'))
        print(f'dry-run singles vs single-final: max |diff| {worst:.2e}')
        assert worst < TOL, worst
    else:
        sys.exit(__doc__)
