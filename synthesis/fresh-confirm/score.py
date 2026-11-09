#!/usr/bin/env python
"""Score the fresh checkpoints under the frozen phase-2 rules (README, *Scoring*); write scores/.

    python synthesis/fresh-confirm/score.py selftest   # pseudo-fresh dumps built from phase-1 dumps
    python synthesis/fresh-confirm/score.py score      # the fresh runs (once, after the freeze)

Adds no rule. Single network: ../single-final/score.py score_member (C1ᵀ, C1); R0 = shift-0 CE.
Ensembles: ../grand-ensemble/score.py ensemble_from_dumps (S1′ per member, BEATs zK′). This file
only converts extract.py's split dumps to the inputs those take, and computes the F1–F6 table.

selftest builds the same split-format dumps from phase-1 data (ao_nogain_s2040–2043 from the tta +
pooled-knn dumps; sb_spec_s2040–2043 from spectrum-branch's flat dumps, shift 0 only) and asserts that
the committed per-seed C1ᵀ / C1 numbers (single-final) and the 4 G (w 2) + 4 S (w 1) ensemble value
(recomputed from the phase-1 flat dumps with grand-ensemble's own loader) are reproduced.
Every score vector is computed twice and must agree bit for bit (smartlab1's host RAM).
"""
import csv
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.dirname(HERE)


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_np_load = np.load


def robust_load(path, *a, **k):
    """np.load for .npz that never trusts the page cache: smartlab1's RAM flipped bits in cached pages
    of files that are clean on disk (2026-11-09: two CRC failures, both clean after eviction). Evict,
    read every array (which checks the zip CRC), retry on failure; returns a plain dict."""
    import zipfile
    if not (isinstance(path, str) and path.endswith('.npz')):
        return _np_load(path, *a, **k)
    for t in range(4):
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        try:
            with _np_load(path, *a, **k) as z:
                return {key: z[key] for key in z.files}
        except (zipfile.BadZipFile, ValueError, OSError) as e:
            print(f'reread {path} after {type(e).__name__} (try {t}): page-cache bit flip', file=sys.stderr, flush=True)
    raise IOError(f'{path}: CRC failed 4 times after eviction')


np.load = robust_load          # every loader below (single-final, grand-ensemble) goes through this
sf = _imp('sf_score', os.path.join(SYN, 'single-final', 'score.py'))
ge = _imp('ge_score', os.path.join(SYN, 'grand-ensemble', 'score.py'))
eb = ge.eb

SCORES = os.path.join(HERE, 'scores')
N = 'SCLTFSTgramMFN_'
SEEDS = (2060, 2061, 2062, 2063)
BEATS = [f'BEATsASD_bt_arc_s{s}' for s in (2040, 2041, 2042)]
PAPER = dict(AUC=95.71, pAUC=90.23, mAUC=91.23, eval_mean=93.22)
P1 = dict(c1t_dev=94.26, c1t_eval=94.10, sd_dev=0.31, sd_eval=0.62)
W_GS = {'G': 2.0, 'S': 1.0}
W_X1 = {'G': 2.0, 'S': 1.0, 'B': 1.5}
LIM = {'ce': 1e3, 'pre': 1e4, 'gmean': 1e4}


# ----------------------------------------------------------------------------- dumps -> inputs
def load_split(emb, run, split):
    keys = ('ce', 'pre', 'gmean', 'idx', 'shifts') + (('valid',) if split == 'train' else ())
    return sf.load(os.path.join(emb, split, f'{run}.npz'), keys, LIM)


def single_inputs(emb, run, m):
    """tr, te (full test list, meta order), valid -- in score_member's format."""
    tr = load_split(emb, run, 'train')
    assert np.array_equal(tr['idx'], np.arange(len(m['tr_label']))), (run, 'train not in meta order')
    nte = len(m['te_label'])
    K = len(tr['shifts'])
    te = dict(ce=np.full((nte, K), np.nan, np.float32), pre=np.full((nte, K, tr['pre'].shape[-1]), np.nan, np.float32),
              gmean=np.full((nte, tr['gmean'].shape[1]), np.nan, np.float32), shifts=tr['shifts'])
    have = np.zeros(nte, bool)
    for ds in ('dev', 'eval'):
        d = load_split(emb, run, ds)
        assert np.array_equal(d['shifts'], tr['shifts'])
        for k in ('ce', 'pre', 'gmean'):
            te[k][d['idx']] = d[k]
        assert not have[d['idx']].any()
        have[d['idx']] = True
    assert have.all(), (run, int((~have).sum()))
    return tr, te, tr['valid']


def flat_path(emb, run, m):
    """Write (once) the ensemble-backend flat npz grand-ensemble's loader reads: shift-0 R0 and embedding."""
    od = os.path.join(emb, 'flat')
    p = os.path.join(od, f'{run}.npz')
    if not os.path.exists(p):
        tr, te, valid = single_inputs(emb, run, m)
        i0 = list(tr['shifts']).index(0.0)
        os.makedirs(od, exist_ok=True)
        sf_save = _imp('pk_extract_save', os.path.join(SYN, '..', 'ablation', 'pooled-knn', 'extract.py')).save_verified
        sf_save(p, tr_ce=tr['ce'][:, i0], tr_pre=tr['pre'][:, i0], valid=valid,
                te_ce=te['ce'][:, i0], te_pre=te['pre'][:, i0])
    return p


# ----------------------------------------------------------------------------- scoring
def single_vectors(emb, run, m, rules):
    tr, te, valid = single_inputs(emb, run, m)
    out = {}
    for r in rules:
        if r == 'R0':
            i0 = list(tr['shifts']).index(0.0)
            out[r] = te['ce'][:, i0].astype(np.float64)
        else:
            out[r], _ = sf.twice(lambda r=r: sf.score_member(tr, te, m['tr_label'], m['te_label'], valid, r))
    return out


def ens_vector(members, weights):
    v, _ = sf.twice(lambda: ge.ensemble_from_dumps(members, weights=weights, use_cache=False))
    return v


def metrics(ev, v):
    res = sf.twice_res(lambda: ev(v))
    o = res['overall']
    row = dict(AUC=o['AUC'], pAUC=o['pAUC'], mAUC=o['mAUC'], Amean=o['mean_AUC_pAUC'])
    per = {mm: dict(AUC=100 * res[mm]['AUC'], pAUC=100 * res[mm]['pAUC'], mAUC=100 * res[mm]['mAUC'],
                    Amean=100 * (res[mm]['AUC'] + res[mm]['pAUC']) / 2) for mm in ge.MACH}
    return row, per


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ----------------------------------------------------------------------------- selftest
def build_pseudo(out):
    """Phase-1 data in extract.py's split format (ao_nogain from tta + pooled-knn, sb_spec flat)."""
    tta = '/home/f74134118/OS-SCL-tta/ablation/tta/results/emb'
    pk = '/home/f74134118/OS-SCL-pknn/ablation/pooled-knn/results/emb'
    spec = '/home/f74134118/OS-SCL-spec/ablation/spectrum-branch/results/emb'
    m = ge.meta()
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, 'meta.npz'), **m)
    for s in (2040, 2041, 2042, 2043):
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
            np.savez(p, gmax=d['gmean'], shifts=np.array([0.0]), **d)


def selftest():
    emb = os.path.join(HERE, 'results', 'selftest_emb')
    build_pseudo(emb)
    m = ge.meta()
    ev = {ds: ge.evaluator(ds) for ds in ('dev', 'eval')}
    ref = {}
    for f in ('metrics_stageA.csv', 'metrics_eval.csv'):
        with open(os.path.join(SYN, 'single-final', 'scores', f)) as fh:
            for row in csv.DictReader(fh):
                if row['machine'] == 'overall':
                    ref[(row['system'], row['dataset'], row['rule'])] = float(row['Amean'])
    worst = 0.0
    for s in (2040, 2041, 2042, 2043):
        r = f'{N}ao_nogain_s{s}'
        v = single_vectors(emb, r, m, ('C1T', 'C1'))
        for ds in ('dev', 'eval'):
            for rule in ('C1T', 'C1'):
                got = metrics(ev[ds], v[rule])[0]['Amean']
                want = ref[(r, ds, rule)]
                worst = max(worst, abs(got - want))
                print(f'single {r[-5:]} {ds} {rule}: {got:.4f} vs committed {want:.4f}', flush=True)
    G = [(flat_path(emb, f'{N}ao_nogain_s{s}', m), 'G') for s in (2040, 2041, 2042, 2043)]
    S = [(flat_path(emb, f'{N}sb_spec_s{s}', m), 'S') for s in (2040, 2041, 2042, 2043)]
    G1 = [(ge.dump_path(f'{N}ao_nogain_s{s}'), 'G') for s in (2040, 2041, 2042, 2043)]
    S1 = [(ge.dump_path(f'{N}sb_spec_s{s}'), 'S') for s in (2040, 2041, 2042, 2043)]
    a, b = ens_vector(G + S, W_GS), ens_vector(G1 + S1, W_GS)
    for ds in ('dev', 'eval'):
        x, y = metrics(ev[ds], a)[0]['Amean'], metrics(ev[ds], b)[0]['Amean']
        worst = max(worst, abs(x - y))
        print(f'GS8 (4G w2 + 4S w1, s2040-43) {ds}: pseudo-fresh {x:.4f} vs phase-1 flat {y:.4f}', flush=True)
    os.makedirs(SCORES, exist_ok=True)
    ok = bool(worst < 5e-4)
    with open(os.path.join(SCORES, 'selftest.json'), 'w') as f:
        json.dump(dict(max_abs_diff_Amean=float(worst), passed=ok), f, indent=2)
    print('SELFTEST', 'PASSED' if ok else 'FAILED', f'max |diff| {worst:.2e}')
    return ok


# ----------------------------------------------------------------------------- the fresh scoring
def score():
    emb = os.path.join(HERE, 'results', 'emb')
    m = ge.meta()
    ge.check_meta(os.path.join(emb, 'meta.npz'), m)
    ev = {ds: ge.evaluator(ds) for ds in ('dev', 'eval')}
    G = [f'{N}fc_nogain_s{s}' for s in SEEDS]
    S = [f'{N}fc_spec_s{s}' for s in SEEDS]
    have = lambda r: all(os.path.exists(os.path.join(emb, sp, f'{r}.npz')) for sp in ('train', 'dev', 'eval'))  # noqa: E731
    G, S = [r for r in G if have(r)], [r for r in S if have(r)]
    print('fresh G:', G, '\nfresh S:', S, flush=True)
    assert len(G) >= 3, 'need >= 3 fresh ao_nogain seeds (README)'

    single_rows, mach_rows, R = [], [], {}
    for r, rules in [(r, ('R0', 'C1', 'C1T')) for r in G] + [(r, ('R0', 'S1p', 'C1')) for r in S]:
        v = single_vectors(emb, r, m, [x for x in rules if x != 'S1p'])
        if 'S1p' in rules:
            v['S1p'] = ge.member_vector(flat_path(emb, r, m), 'S', use_cache=False)
        for ds in ('dev', 'eval'):
            for rule in rules:
                row, per = metrics(ev[ds], v[rule])
                R[(r, ds, rule)] = row
                single_rows.append([r, ds, rule] + [f'{row[k]:.4f}' for k in ('AUC', 'pAUC', 'mAUC', 'Amean')])
                mach_rows += [[r, ds, rule, mm] + [f'{per[mm][k]:.4f}' for k in ('AUC', 'pAUC', 'mAUC', 'Amean')]
                              for mm in ge.MACH]
    os.makedirs(SCORES, exist_ok=True)
    write_csv(os.path.join(SCORES, 'single.csv'), ['run', 'dataset', 'rule', 'AUC', 'pAUC', 'mAUC', 'Amean'], single_rows)

    gm = [(flat_path(emb, r, m), 'G') for r in G]
    sm = [(flat_path(emb, r, m), 'S') for r in S]
    bm = [(ge.dump_path(r), 'B') for r in BEATS]
    V = {'G4': ens_vector(gm, {'G': 1.0}), 'GS8': ens_vector(gm + sm, W_GS) if sm else None,
         'X1': ens_vector(gm + sm + bm, W_X1) if sm else None}
    V = {k: v for k, v in V.items() if v is not None}
    ens_rows, E = [], {}
    for name, v in V.items():
        for ds in ('dev', 'eval'):
            row, per = metrics(ev[ds], v)
            E[(name, ds)] = row
            ens_rows.append([name, ds, 'all'] + [f'{row[k]:.4f}' for k in ('AUC', 'pAUC', 'mAUC', 'Amean')])
            ens_rows += [[name, ds, mm] + [f'{per[mm][k]:.4f}' for k in ('AUC', 'pAUC', 'mAUC', 'Amean')]
                         for mm in ge.MACH]
    write_csv(os.path.join(SCORES, 'ensembles.csv'), ['system', 'dataset', 'machine', 'AUC', 'pAUC', 'mAUC', 'Amean'], ens_rows)
    write_csv(os.path.join(SCORES, 'single_machines.csv'), ['run', 'dataset', 'rule', 'machine', 'AUC', 'pAUC', 'mAUC', 'Amean'], mach_rows)
    boot = {}
    pairs = [p for p in (('GS8', 'G4'), ('X1', 'GS8')) if all(x in V for x in p)]
    for ds in ('dev', 'eval'):
        if pairs:
            boot[ds] = ge.paired_boot(ev[ds], V, pairs)
    with open(os.path.join(SCORES, 'bootstrap.json'), 'w') as f:
        json.dump(boot, f, indent=2)

    # ---- F1-F6
    mean = lambda ds, rule, k='Amean': float(np.mean([R[(r, ds, rule)][k] for r in G]))  # noqa: E731
    se = lambda sd, n: sd * np.sqrt(1 / n + 1 / 8)  # noqa: E731
    n = len(G)
    D = {}
    D['F1'] = dict(dev=mean('dev', 'C1T'), eval=mean('eval', 'C1T'),
                   bar_dev=P1['c1t_dev'] - 2 * se(P1['sd_dev'], n), bar_eval=P1['c1t_eval'] - 2 * se(P1['sd_eval'], n))
    D['F1']['passed'] = bool(D['F1']['dev'] >= D['F1']['bar_dev'] and D['F1']['eval'] >= D['F1']['bar_eval'])
    for rule in ('C1T', 'R0'):
        cnt = {k: int(sum(R[(r, 'dev', rule)][k] > PAPER[k] for r in G)) for k in ('AUC', 'pAUC', 'mAUC')}
        cnt['eval_mean'] = int(sum(R[(r, 'eval', rule)]['Amean'] > PAPER['eval_mean'] for r in G))
        D[f'F2_{rule}'] = dict(seeds_above=cnt, n=n, passed=bool(all(c >= n - 1 for c in cnt.values())))
    D['F3'] = {ds: dict(delta=mean(ds, 'C1T') - mean(ds, 'C1'),
                        seeds_pos=int(sum(R[(r, ds, 'C1T')]['Amean'] > R[(r, ds, 'C1')]['Amean'] for r in G)))
               for ds in ('dev', 'eval')}
    D['F3']['replicates_on_dev'] = bool(D['F3']['dev']['delta'] > 0)
    if 'GS8' in V:
        d = {ds: E[('GS8', ds)]['Amean'] - E[('G4', ds)]['Amean'] for ds in ('dev', 'eval')}
        ci = boot['eval']['GS8-G4']['mean_AUC_pAUC']
        D['F4'] = dict(delta=d, eval_ci=ci[1:], n_S=len(S), passed=bool(d['dev'] > 0 and d['eval'] > 0),
                       confirmed=bool(d['dev'] > 0 and ci[1] > 0))
        D['F5'] = {ds: E[('GS8', ds)] for ds in ('dev', 'eval')}
        d = {ds: E[('X1', ds)]['Amean'] - E[('GS8', ds)]['Amean'] for ds in ('dev', 'eval')}
        D['F6'] = dict(delta=d, eval_ci=boot['eval']['X1-GS8']['mean_AUC_pAUC'][1:], passed=bool(d['eval'] > 0),
                       label='WITH EXTERNAL PRETRAINING')
    D['seeds'] = dict(G=G, S=S)
    with open(os.path.join(SCORES, 'decisions.json'), 'w') as f:
        json.dump(D, f, indent=2, default=float)
    print(json.dumps(D, indent=2, default=float))


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    if stage == 'selftest':
        sys.exit(0 if selftest() else 1)
    elif stage == 'score':
        score()
    else:
        sys.exit(__doc__)
