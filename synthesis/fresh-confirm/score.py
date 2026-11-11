#!/usr/bin/env python
"""Score the fresh checkpoints under the frozen phase-2 rules (README, *Scoring*); write scores/.

    python synthesis/fresh-confirm/score.py selftest   # pseudo-fresh dumps built from phase-1 dumps
    python synthesis/fresh-confirm/score.py dryrun     # all of score() on those pseudo-fresh dumps -> scores/dryrun_p1 via run.sh
    python synthesis/fresh-confirm/score.py score      # the fresh runs (once, after the freeze)

Adds no rule. Single network: ../single-final/score.py score_member (C1ᵀ, C1); R0 = shift-0 CE.
Ensembles: ../grand-ensemble/score.py ensemble_from_dumps (S1′ per member, BEATs zK′). This file
only converts extract.py's split dumps to the inputs those take, and computes the F1–F6 table.

selftest builds the same split-format dumps from phase-1 data (ao_nogain_s2040–2043 from the tta +
pooled-knn dumps; sb_spec_s2040–2043 from spectrum-branch's flat dumps, shift 0 only) and asserts that
the committed per-seed C1ᵀ / C1 numbers (single-final) and the 4 G (w 2) + 4 S (w 1) ensemble value
(recomputed from the phase-1 flat dumps with grand-ensemble's own loader) are reproduced.
Every score vector is computed twice, each time from its own read of the dumps, and must agree bit for
bit (smartlab1's host RAM). run.sh adds the outer guard: `score` and `dryrun` are run as independent
processes until two of them write byte-identical output, and only that output is kept.
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
    sys.modules[name] = mod    # ge.paired_boot pickles its worker by module name (found by `dryrun`)
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


def agree(f, tries=6):
    """f() -> dict of arrays, recomputed until two results agree bit for bit (as sf.twice, for dicts)."""
    seen = []
    for _ in range(tries):
        d = f()
        for u in seen:
            if u.keys() == d.keys() and all(np.array_equal(u[k], d[k], equal_nan=True) for k in d):
                return d
        seen.append(d)
    sys.exit(f'ABORT: {tries} reads never agreed')


_FLAT, _SAVE = set(), []


def flat_path(emb, run, m):
    """The ensemble-backend flat npz grand-ensemble's loader reads: shift-0 R0 and embedding. Rebuilt once
    per process (never trusted from an earlier run: the split dumps may have been re-extracted since), from
    two reads of the split dumps that agree."""
    od = os.path.join(emb, 'flat')
    p = os.path.join(od, f'{run}.npz')
    if p in _FLAT:
        return p

    def once():
        tr, te, valid = single_inputs(emb, run, m)
        i0 = list(tr['shifts']).index(0.0)
        return dict(tr_ce=tr['ce'][:, i0], tr_pre=tr['pre'][:, i0], valid=valid,
                    te_ce=te['ce'][:, i0], te_pre=te['pre'][:, i0])
    os.makedirs(od, exist_ok=True)
    if not _SAVE:
        _SAVE.append(_imp('pk_extract_save', os.path.join(SYN, '..', 'ablation', 'pooled-knn', 'extract.py')).save_verified)
    _SAVE[0](p, **agree(once))
    _FLAT.add(p)
    return p


# ----------------------------------------------------------------------------- scoring
def single_vectors(emb, run, m, rules):
    """{rule: score vector}. Each of the two agreeing computations starts from its own read of the dumps,
    so a bit flip in the loaded arrays is caught as well as one in the arithmetic."""
    def once():
        tr, te, valid = single_inputs(emb, run, m)
        i0 = list(tr['shifts']).index(0.0)
        return np.stack([te['ce'][:, i0].astype(np.float64) if r == 'R0' else
                         sf.score_member(tr, te, m['tr_label'], m['te_label'], valid, r) for r in rules])
    v, _ = sf.twice(once)
    return dict(zip(rules, v))


def ens_vector(members, weights):
    v, _ = sf.twice(lambda: ge.ensemble_from_dumps(members, weights=weights, use_cache=False))
    return v


COLS = ('AUC', 'pAUC', 'mAUC', 'Amean', 'AUC_perID', 'pAUC_perID', 'Amean_perID')


def per_id(ev, v):
    """The literature's convention (the official DCASE rule): AUC and pAUC per machine ID, averaged over a
    type's IDs, then over the six types. Reported beside eval.py's pooled-ID numbers; decides nothing."""
    from sklearn import metrics as skm
    s, out = v[ev.sel], {}
    for mm in ge.MACH:
        ks = [(ev.mach == mm) & (ev.id == i) for i in np.unique(ev.id[ev.mach == mm])]
        out[mm] = [100 * float(np.mean([skm.roc_auc_score(ev.y[k], s[k], max_fpr=f) for k in ks])) for f in (None, 0.1)]
    return out


def metrics(ev, v):
    res = sf.twice_res(lambda: ev(v))
    pid = sf.twice_res(lambda: per_id(ev, v))
    o = res['overall']
    A, P = (float(np.mean([pid[mm][j] for mm in ge.MACH])) for j in (0, 1))
    row = dict(AUC=o['AUC'], pAUC=o['pAUC'], mAUC=o['mAUC'], Amean=o['mean_AUC_pAUC'],
               AUC_perID=A, pAUC_perID=P, Amean_perID=(A + P) / 2)
    per = {mm: dict(AUC=100 * res[mm]['AUC'], pAUC=100 * res[mm]['pAUC'], mAUC=100 * res[mm]['mAUC'],
                    Amean=100 * (res[mm]['AUC'] + res[mm]['pAUC']) / 2, AUC_perID=pid[mm][0], pAUC_perID=pid[mm][1],
                    Amean_perID=(pid[mm][0] + pid[mm][1]) / 2) for mm in ge.MACH}
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
def score(emb=None, out=None, kinds=('fc_nogain', 'fc_spec'), seeds=SEEDS, s_rules=('R0', 'S1p', 'C1')):
    """The fresh scoring (defaults). `dryrun` calls the same code on pseudo-fresh phase-1 dumps, writing to
    its own directory (scores/dryrun_p1), so every line below has run once before the fresh checkpoints are scored."""
    emb = emb or os.path.join(HERE, 'results', 'emb')
    SCORES = out or globals()['SCORES']
    m = ge.meta()
    ge.check_meta(os.path.join(emb, 'meta.npz'), m)
    ev = {ds: ge.evaluator(ds) for ds in ('dev', 'eval')}
    G = [f'{N}{kinds[0]}_s{s}' for s in seeds]
    S = [f'{N}{kinds[1]}_s{s}' for s in seeds]
    have = lambda r: all(os.path.exists(os.path.join(emb, sp, f'{r}.npz')) for sp in ('train', 'dev', 'eval'))  # noqa: E731
    # No silent drop: every run must have its three dumps. FC_DROP="fc_spec_s2061 ..." removes a run, only
    # for the README's reasons (training died, or a dump failed its CRC three times); it is recorded.
    drop = set(os.environ.get('FC_DROP', '').split())
    G, S = [[r for r in X if r[len(N):] not in drop] for X in (G, S)]
    missing = [r for r in G + S if not have(r)]
    assert not missing, f'no dumps for {missing}: extract them, or name them in FC_DROP (README, Scoring)'
    print('fresh G:', G, '\nfresh S:', S, '\ndropped:', sorted(drop), flush=True)
    assert len(G) >= 3, 'need >= 3 fresh ao_nogain seeds (README)'

    single_rows, mach_rows, R = [], [], {}
    for r, rules in [(r, ('R0', 'C1', 'C1T')) for r in G] + [(r, tuple(s_rules)) for r in S]:
        v = single_vectors(emb, r, m, [x for x in rules if x != 'S1p'])
        if 'S1p' in rules:
            fp = flat_path(emb, r, m)
            v['S1p'] = sf.twice(lambda: ge.member_vector(fp, 'S', use_cache=False))[0]
        for ds in ('dev', 'eval'):
            for rule in rules:
                row, per = metrics(ev[ds], v[rule])
                R[(r, ds, rule)] = row
                single_rows.append([r, ds, rule] + [f'{row[k]:.4f}' for k in COLS])
                mach_rows += [[r, ds, rule, mm] + [f'{per[mm][k]:.4f}' for k in COLS]
                              for mm in ge.MACH]
    os.makedirs(SCORES, exist_ok=True)
    write_csv(os.path.join(SCORES, 'single.csv'), ['run', 'dataset', 'rule', *COLS], single_rows)

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
            ens_rows.append([name, ds, 'all'] + [f'{row[k]:.4f}' for k in COLS])
            ens_rows += [[name, ds, mm] + [f'{per[mm][k]:.4f}' for k in COLS]
                         for mm in ge.MACH]
    write_csv(os.path.join(SCORES, 'ensembles.csv'), ['system', 'dataset', 'machine', *COLS], ens_rows)
    write_csv(os.path.join(SCORES, 'single_machines.csv'), ['run', 'dataset', 'rule', 'machine', *COLS], mach_rows)
    boot = {}
    pairs = [p for p in (('GS8', 'G4'), ('X1', 'GS8')) if all(x in V for x in p)]
    for ds in ('dev', 'eval'):
        if pairs:
            boot[ds] = ge.paired_boot(ev[ds], V, pairs)
    with open(os.path.join(SCORES, 'bootstrap.json'), 'w') as f:
        json.dump(boot, f, indent=2)

    # ---- F4b (README addendum 2026-11-10, written before any fresh score): member-count-matched swap.
    # F4 compares 8 nets with 4, so it cannot tell the sb_spec type from "more members". F4b holds the
    # count at 4: every 2 G (w 2) + 2 S (w 1) subset against the 4-G ensemble. Reported, no decision.
    f4b = None
    if len(gm) >= 2 and len(sm) >= 2:
        mv = {p: sf.twice(lambda p=p, t=t: ge.member_vector(p, t, use_cache=False))[0] for p, t in gm + sm}
        sub_rows, dl = [], {'dev': [], 'eval': []}
        for gp in itertools.combinations(range(len(gm)), 2):
            for sp in itertools.combinations(range(len(sm)), 2):
                v = W_GS['G'] * sum(mv[gm[i][0]] for i in gp) + W_GS['S'] * sum(mv[sm[i][0]] for i in sp)
                for ds in ('dev', 'eval'):
                    a = metrics(ev[ds], v)[0]['Amean']
                    dl[ds].append(a - E[('G4', ds)]['Amean'])
                    sub_rows.append(['+'.join(G[i][-5:] for i in gp), '+'.join(S[i][-5:] for i in sp), ds,
                                     f'{a:.4f}', f'{dl[ds][-1]:.4f}'])
        write_csv(os.path.join(SCORES, 'f4b_subsets.csv'), ['G_pair', 'S_pair', 'dataset', 'Amean', 'delta_vs_G4'], sub_rows)
        f4b = {ds: dict(mean_delta=float(np.mean(x)), min=float(np.min(x)), max=float(np.max(x)),
                        n_pos=int(np.sum(np.array(x) > 0)), n_subsets=len(x)) for ds, x in dl.items()}

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
        joint = int(sum(all(R[(r, 'dev', rule)][k] > PAPER[k] for k in ('AUC', 'pAUC', 'mAUC'))
                        and R[(r, 'eval', rule)]['Amean'] > PAPER['eval_mean'] for r in G))
        # `passed` is the per-number count frozen at 5c41150 (the aug-ops precedent); the joint count
        # (one seed beating all four numbers at once) is the stricter reading, reported beside it.
        D[f'F2_{rule}'] = dict(seeds_above=cnt, seeds_above_all4=joint, n=n,
                               passed=bool(all(c >= n - 1 for c in cnt.values())))
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
    if f4b:
        D['F4b'] = dict(f4b, type_effect={ds: bool(f4b[ds]['mean_delta'] > 0 and f4b[ds]['n_pos'] >= 0.75 * f4b[ds]['n_subsets'])
                                          for ds in ('dev', 'eval')},
                        note='count-matched: 2 G (w 2) + 2 S (w 1) subsets minus the 4-G ensemble; reported only')
    D['seeds'] = dict(G=G, S=S, dropped=sorted(drop))
    with open(os.path.join(SCORES, 'decisions.json'), 'w') as f:
        json.dump(D, f, indent=2, default=float)
    print(json.dumps(D, indent=2, default=float))


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    if stage == 'selftest':
        sys.exit(0 if selftest() else 1)
    elif stage == 'score':
        score(out=sys.argv[2] if len(sys.argv) > 2 else None)
    elif stage == 'dryrun':
        # the whole of score() on SEEN phase-1 members in fresh format (selftest's pseudo-fresh dumps); the
        # pseudo sb_spec dumps carry no gmean, so C1 is skipped for them. Output goes to its own directory, never to scores/*.csv.
        emb = os.path.join(HERE, 'results', 'selftest_emb')
        build_pseudo(emb)
        score(emb=emb, out=sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'results', 'dryrun_scores'),
              kinds=('ao_nogain', 'sb_spec'), seeds=(2040, 2041, 2042, 2043), s_rules=('R0', 'S1p'))
    else:
        sys.exit(__doc__)
