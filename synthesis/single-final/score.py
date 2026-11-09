#!/usr/bin/env python
"""single-final: does C1ᵀ (C1 with `small` TTA on R0 and K) beat C1 for one `ao_nogain` net?

    python synthesis/single-final/score.py check    # reproduction only: C1, S1′, S1′ᵀ (all already seen)
    python synthesis/single-final/score.py stageA   # dev, seeds 2040-2043: C1ᵀ vs C1 (gate)
    python synthesis/single-final/score.py stageB   # dev, seeds 2044-2047 + 8-seed dev (needs stageA.json committed)
    python synthesis/single-final/score.py eval     # eval, 8 seeds, scored once (needs stageB.json committed)

Rules (README, *Design*), per checkpoint and machine ID; bank = the ID's training-part clips
(the replayed 95%, `valid` == False); every distance a 1-NN cosine distance on L2-normalised
uncentred vectors; every z-statistic = mean / sd over the bank, K and G scored leave-one-out:

  C1   = zR0′ + zK′ + zGmean                    ../../ablation/pooled-knn (adopted 2026-11-05)
  S1′ᵀ = zR0′ᵀ + zK′ᵀ                           ../../ablation/tta (adopted 2026-11-03)
  C1ᵀ  = zR0′ᵀ + zK′ᵀ + zGmean                  this study's candidate

  ᵀ (`small` TTA): copies at circular shifts 0, +1 %, -1 % of the clip.
     R0ᵀ  = mean over copies of the per-copy CE;
     preᵀ = mean over copies of the L2-normalised pre-head embedding (then L2-normalised again
            by the kNN), exactly ../../ablation/tta/score.py `versions()`.
     The training clips (bank, z-statistics) get the same treatment.
  zGmean is NOT shifted: time-mean pre-linear7 feature, shown roll-invariant by pooled-knn's
  diagnostic (0.13 / 0.20 sd at half-clip rolls); no shifted gmean dump exists.

Importable API for later studies (fresh-confirm): `score_member(...)` and `FINAL_RULE`.
"""
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
ABL = os.path.join(REPO, 'ablation')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pk = _load('pk_score', os.path.join(ABL, 'pooled-knn', 'score.py'))   # l2, nn1, Fast, boot_delta, eb
eb = pk.eb                                                             # Evaluator, MACH, PAPER

SCORES = os.path.join(HERE, 'scores')
CACHE = os.path.join(HERE, 'results', 'cache')    # verified per-member score vectors (*.npz: gitignored)
TTA_EMB = os.environ.get('SF_TTA_EMB', '/home/f74134118/OS-SCL-tta/ablation/tta/results/emb')
PK_EMB = os.environ.get('SF_PK_EMB', '/home/f74134118/OS-SCL-pknn/ablation/pooled-knn/results/emb')
SEL, CONF = (2040, 2041, 2042, 2043), (2044, 2045, 2046, 2047)
ALL8 = SEL + CONF
RUN = 'SCLTFSTgramMFN_ao_nogain_s{}'.format
MACH, PAPER = eb.MACH, eb.PAPER
SMALL = (0.0, 0.01, -0.01)
KEYS = ('AUC', 'pAUC', 'mAUC', 'Amean')

# The adopted single-model rule. Written by `score.py eval` into scores/final_rule.json; this
# constant is what score_member() uses when no rule is passed. It is set from the committed
# decision (see README, *Decision*) -- see final_rule.json for the provenance.
_FR = os.path.join(SCORES, 'final_rule.json')
FINAL_RULE = json.load(open(_FR))['rule'] if os.path.exists(_FR) else None


# ------------------------------------------------------------------------------- scoring core
def tta_reduce(ce, pre, shifts=None, keep=SMALL):
    """K-copy arrays -> one vector per clip, as ../../ablation/tta/score.py does.

    ce (n,) or (n, K); pre (n, 128) or (n, K, 128); shifts (K,) fractions of the clip.
    keep=(0.0,) returns the unshifted copy; keep=SMALL averages R0 and the L2-normalised
    embedding over shifts 0, +1 %, -1 %. Returns float32 (ce (n,), pre (n, 128)).
    """
    if ce.ndim == 1:
        assert tuple(keep) == (0.0,), 'single-copy input cannot be TTA-reduced'
        return ce, pre
    sh = [round(float(x), 4) for x in shifts]
    cols = [sh.index(round(x, 4)) for x in keep]      # KeyError-like ValueError if a shift is missing
    if len(cols) == 1:
        return ce[:, cols[0]], pre[:, cols[0]]
    p = pre[:, cols] / np.linalg.norm(pre[:, cols], axis=2, keepdims=True)
    return ce[:, cols].mean(1), p.mean(1)


def member_terms(tr_ce, tr_pre, te_ce, te_pre, tr_label, te_label, valid, tr_gmean=None, te_gmean=None):
    """Per-clip z-terms of one checkpoint: {'zR0p', 'zKp'[, 'zGmean']} over the te clips.

    tr_*: training clips (all 36,283, in meta order; the bank is tr_label == id & ~valid);
    te_*: query clips with labels te_label (machine-ID class 0..40). ce (n,), pre (n, 128),
    gmean (n, 512). Pass already TTA-reduced ce / pre (tta_reduce) for the ᵀ rules.
    """
    trp, tep = pk.l2(tr_pre), pk.l2(te_pre)
    trc, tec = tr_ce.astype(np.float64), te_ce.astype(np.float64)
    g = tr_gmean is not None
    if g:
        trg, teg = pk.l2(tr_gmean), pk.l2(te_gmean)
    out = {k: np.full(len(te_label), np.nan) for k in ('zR0p', 'zKp') + (('zGmean',) if g else ())}

    def z(x, ref):
        return (x - ref.mean()) / ref.std()
    for c in np.unique(te_label):
        ref, m = (tr_label == c) & ~valid, te_label == c
        assert ref.sum() > 100, (c, ref.sum())
        out['zR0p'][m] = z(tec[m], trc[ref])
        out['zKp'][m] = z(pk.nn1(tep[m], trp[ref]), pk.nn1(trp[ref], trp[ref], loo=True))
        if g:
            out['zGmean'][m] = z(pk.nn1(teg[m], trg[ref]), pk.nn1(trg[ref], trg[ref], loo=True))
    for k, v in out.items():
        assert np.isfinite(v).all() and np.abs(v).max() < 1e6, k
    return out


RULES = {   # rule -> (TTA shifts for R0 and K, uses zGmean)
    'C1': ((0.0,), True),
    'S1p': ((0.0,), False),
    'S1pT': (SMALL, False),
    'C1T': (SMALL, True),
}


def score_member(tr, te, tr_label, te_label, valid, rule=None):
    """Final single-model anomaly score of one network (higher = more anomalous).

    tr, te: dicts for the training clips (all of them, meta order) and the test clips:
      'ce'    (n, K) float  per-copy ArcFace CE of the clip's own ID (R0)
      'pre'   (n, K, 128)   per-copy pre-head embedding
      'shifts'(K,)          circular shift of each copy, fraction of the clip; must contain
                            0.0 (and +-0.01 for a ᵀ rule). K = 1 with shifts [0.] is fine for C1.
      'gmean' (n, 512)      time-and-frequency mean of the pre-linear7 map, UNshifted
                            (pooled-knn extract.py `gmean`)
    tr_label (n_tr,), te_label (n_te,): machine-ID class (0..40, dataloader order);
    valid (n_tr,) bool: the checkpoint's replayed 5 % held-out split (not in the bank).
    rule: 'C1', 'C1T', 'S1pT' or 'S1p'; default FINAL_RULE (scores/final_rule.json).
    """
    rule = rule or FINAL_RULE
    assert rule in RULES, rule
    keep, g = RULES[rule]
    trc, trp = tta_reduce(tr['ce'], tr['pre'], tr.get('shifts'), keep)
    tec, tep = tta_reduce(te['ce'], te['pre'], te.get('shifts'), keep)
    t = member_terms(trc, trp, tec, tep, tr_label, te_label, valid,
                     tr['gmean'] if g else None, te['gmean'] if g else None)
    return t['zR0p'] + t['zKp'] + (t['zGmean'] if g else 0.0)


# ------------------------------------------------------------------------------- data loading
def load(path, keys, lim):
    """np.load only `keys`, range-checked (faulty host RAM; np.load also checks the zip CRC)."""
    z = np.load(path)
    a = {k: z[k] for k in keys}
    for k, bound in lim.items():
        v = a[k]
        assert np.isfinite(v).all() and np.abs(v).max() < bound, (path, k, float(np.abs(v).max()))
    return a


LIM = {'ce': 1e3, 'pre': 1e4}
LIMG = dict(LIM, gmean=1e4)


def below_data(paths):
    return np.array([p.split('/data/', 1)[1] for p in paths])


def digest(*objs):
    h = hashlib.blake2b(digest_size=16)

    def walk(o):
        if isinstance(o, dict):
            for k in sorted(o):
                walk(o[k])
        else:
            h.update(np.ascontiguousarray(o).tobytes())
    for o in objs:
        walk(o)
    return h.hexdigest()


def twice(f, tries=6):
    """Run f until two runs agree bit for bit (smartlab1's host RAM flips bits mid-computation:
    the first check run lost 0.17 AUC on one machine of one seed that way). Returns (value, n_runs)."""
    seen = []
    for n in range(1, tries + 1):
        v = f()
        for u in seen:
            if np.array_equal(u, v, equal_nan=True):
                if n > 2:
                    print(f'    note: recomputation needed ({n} runs) -- bit flip in host RAM', flush=True)
                return v, n
        seen.append(v)
    sys.exit(f'ABORT: {tries} computations of one score vector never agreed')


def twice_res(f, tries=4):
    """Evaluator results computed until two runs agree exactly (same RAM caveat as twice())."""
    seen = []
    for _ in range(tries):
        r = f()
        key = json.dumps(r, sort_keys=True, default=float)
        if key in seen:
            return r
        seen.append(key)
    sys.exit('ABORT: metric computation never agreed with itself')


class Ctx:
    def __init__(self):
        self.meta = dict(np.load(os.path.join(TTA_EMB, 'meta.npz')))
        pm = dict(np.load(os.path.join(PK_EMB, 'meta.npz')))
        self.align = {}
        for k in self.meta:
            a, b = self.meta[k], pm[k]
            if k.endswith('paths'):
                a, b = below_data(a), below_data(b)
            ok = bool(np.array_equal(a, b))
            self.align[f'meta_{k}_equal'] = ok
            if not ok:
                sys.exit(f'ABORT: tta and pooled-knn meta.npz disagree on {k}: the dump families do not '
                         f'index the same clips in the same order')
        self.ev = {ds: eb.Evaluator(self.meta, ds) for ds in ('dev', 'eval')}

    def member(self, seed, ds):
        """Both dump families of one checkpoint on train + one test split, aligned and cross-checked."""
        r = RUN(seed)
        ttr = load(os.path.join(TTA_EMB, 'train', f'{r}.npz'), ('ce', 'pre', 'idx', 'shifts'), LIM)
        tte = load(os.path.join(TTA_EMB, ds, f'{r}.npz'), ('ce', 'pre', 'idx', 'shifts'), LIM)
        ptr = load(os.path.join(PK_EMB, 'train', f'{r}.npz'), ('ce', 'pre', 'gmean', 'idx', 'valid'), LIMG)
        pte = load(os.path.join(PK_EMB, ds, f'{r}.npz'), ('ce', 'pre', 'gmean', 'idx'), LIMG)
        ntr, nte = len(self.meta['tr_label']), len(self.meta['te_label'])
        sel = self.ev[ds].sel
        chk = {}
        for nm, a, n, want in (('tta_train', ttr, ntr, np.arange(ntr)), ('pk_train', ptr, ntr, np.arange(ntr)),
                               ('tta_' + ds, tte, nte, sel), ('pk_' + ds, pte, nte, sel)):
            ok = bool(np.array_equal(np.sort(a['idx']), want))
            chk[f'{nm}_idx_covers_split'] = ok
            if not ok:
                sys.exit(f'ABORT: {r} {nm}: idx does not cover the {ds}/train clip list')
        if not np.array_equal(tte['idx'], pte['idx']) or not np.array_equal(ttr['idx'], ptr['idx']):
            sys.exit(f'ABORT: {r}: tta and pooled-knn dumps list clips in different orders')
        chk['idx_same_order'] = True
        for a in (ttr, tte):
            assert [round(float(x), 4) for x in a['shifts'][:3]] == [0.0, 0.01, -0.01], a['shifts']
        # cross-family check on the unshifted copy: training bank must be bit-identical (both
        # reproduced the aug-ops dump); test clips differ only in the last, differently sized batch
        chk['train_max_abs_ce_tta0_vs_pk'] = float(np.abs(ttr['ce'][:, 0] - ptr['ce']).max())
        chk['train_max_abs_pre_tta0_vs_pk'] = float(np.abs(ttr['pre'][:, 0] - ptr['pre']).max())
        chk[f'{ds}_max_abs_ce_tta0_vs_pk'] = float(np.abs(tte['ce'][:, 0] - pte['ce']).max())
        chk[f'{ds}_max_abs_pre_tta0_vs_pk'] = float(np.abs(tte['pre'][:, 0] - pte['pre']).max())
        chk[f'{ds}_n_clips_ce_differ'] = int((tte['ce'][:, 0] != pte['ce']).sum())
        assert chk['train_max_abs_ce_tta0_vs_pk'] == 0 and chk['train_max_abs_pre_tta0_vs_pk'] == 0, (r, chk)
        assert chk[f'{ds}_max_abs_ce_tta0_vs_pk'] < 1e-3 and chk[f'{ds}_max_abs_pre_tta0_vs_pk'] < 1e-2, (r, chk)
        tr = dict(ce=ttr['ce'], pre=ttr['pre'], shifts=ttr['shifts'], gmean=ptr['gmean'])
        te = dict(ce=tte['ce'], pre=tte['pre'], shifts=tte['shifts'], gmean=pte['gmean'])
        pkd = dict(tr=dict(ce=ptr['ce'], pre=ptr['pre'], gmean=ptr['gmean']),
                   te=dict(ce=pte['ce'], pre=pte['pre'], gmean=pte['gmean']))
        return tr, te, pkd, ptr['valid'], self.meta['te_label'][tte['idx']], tte['idx'], chk

    def scores(self, seed, ds, rules):
        """{rule: full-length score vector (NaN off ds)}, plus checks. 'C1' and 'S1p' come from the
        pooled-knn dumps (the incumbent exactly); 'C1_0' is C1 rebuilt from the TTA dump's
        shift-0 copy (cross-family check); ᵀ rules use the TTA dump."""
        cache = os.path.join(CACHE, f'{ds}_{RUN(seed)}_{"-".join(rules)}.npz')
        if os.path.exists(cache):            # a verified result of an earlier (crashed / aborted) run
            c = np.load(cache, allow_pickle=False)
            out = {r: c[r] for r in rules}
            if digest(out) == str(c['digest']):
                return out, json.loads(str(c['chk']))
            print(f'    note: cache {cache} fails its digest; recomputing', flush=True)
        ballast = []
        for attempt in range(8):
            tr, te, pkd, valid, lab, idx, chk = self.member(seed, ds)
            h0 = digest(tr, te, pkd, valid)
            out, chk = self._score(tr, te, pkd, valid, lab, idx, chk, rules)
            # inputs unchanged while in RAM? A mismatch can also be a transient misread (the faulty
            # line flips on read too), so re-read up to 3 times before reloading the member
            if any(digest(tr, te, pkd, valid) == h0 for _ in range(3)):
                chk['input_reloads'] = attempt
                os.makedirs(CACHE, exist_ok=True)
                np.savez(cache, digest=digest(out), chk=json.dumps(chk), **out)
                return out, chk
            print(f'    note: input arrays changed in RAM while scoring {RUN(seed)} {ds}; reloading', flush=True)
            # keep a random-size block alive so the reload lands on other physical pages
            ballast.append(np.ones(int(np.random.default_rng().integers(1, 64)) << 20, np.uint8))
        sys.exit('ABORT: input arrays kept changing in RAM')

    def _score(self, tr, te, pkd, valid, lab, idx, chk, rules):
        trl, nte = self.meta['tr_label'], len(self.meta['te_label'])
        out = {}
        chk['recomputed_after_disagreement'] = []
        for rule in rules:
            if rule in ('C1', 'S1p'):
                f = lambda: score_member(pkd['tr'], pkd['te'], trl, lab, valid, rule)   # noqa: E731
            elif rule == 'C1_0':
                f = lambda: score_member(tr, te, trl, lab, valid, 'C1')                 # noqa: E731
            else:
                f = lambda: score_member(tr, te, trl, lab, valid, rule)                 # noqa: E731
            s, n = twice(f)
            if n > 2:
                chk['recomputed_after_disagreement'].append(rule)
            v = np.full(nte, np.nan)
            v[idx] = s
            out[rule] = v
        return out, chk


# ------------------------------------------------------------------------------- bookkeeping
def flat(res):
    o = res['overall']
    return dict(AUC=o['AUC'], pAUC=o['pAUC'], mAUC=o['mAUC'], Amean=o['mean_AUC_pAUC'])


def metric_rows(system, ds, rule, res):
    out = []
    for m in MACH:
        A, P = 100 * res[m]['AUC'], 100 * res[m]['pAUC']
        out.append([system, ds, rule, m, f'{A:.4f}', f'{P:.4f}', f'{100 * res[m]["mAUC"]:.4f}', f'{(A + P) / 2:.4f}'])
    o = flat(res)
    out.append([system, ds, rule, 'overall'] + [f'{o[k]:.4f}' for k in KEYS])
    return out


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def committed(path):
    """True if `path` is tracked and unmodified in git."""
    rel = os.path.relpath(path, REPO)
    r1 = subprocess.run(['git', '-C', REPO, 'ls-files', '--error-unmatch', rel], capture_output=True)
    r2 = subprocess.run(['git', '-C', REPO, 'diff', '--quiet', 'HEAD', '--', rel])
    return r1.returncode == 0 and r2.returncode == 0


def evaluate(ctx, seeds, ds, rules):
    """Score `rules` for `seeds` on `ds`; returns S[(seed, rule)], R[(seed, rule)], checks."""
    S, R, chk = {}, {}, {}
    for s in seeds:
        sc, chk[RUN(s)] = ctx.scores(s, ds, rules)
        for rule, v in sc.items():
            S[(s, rule)] = v
            R[(s, rule)] = twice_res(lambda: ctx.ev[ds](v))
        print(f'  scored {RUN(s)} {ds}: ' + ' '.join(f'{r} {flat(R[(s, r)])["Amean"]:.3f}' for r in rules), flush=True)
    return S, R, chk


def mean_abs(R, seeds, rule):
    return {k: float(np.mean([flat(R[(s, rule)])[k] for s in seeds])) for k in KEYS}


def delta(R, seeds, a, b):
    d = {k: [flat(R[(s, a)])[k] - flat(R[(s, b)])[k] for s in seeds] for k in KEYS}
    o = {f'd_{k}': float(np.mean(v)) for k, v in d.items()}
    o['d_Amean_per_seed'] = [round(x, 4) for x in d['Amean']]
    o['n_pos_Amean'] = int(sum(x > 0 for x in d['Amean']))
    o['n'] = len(seeds)
    o['per_machine_dAUC'] = {m: float(np.mean([100 * (R[(s, a)][m]['AUC'] - R[(s, b)][m]['AUC']) for s in seeds]))
                             for m in MACH}
    o['per_machine_dAmean'] = {m: float(np.mean([50 * (R[(s, a)][m]['AUC'] + R[(s, a)][m]['pAUC']
                                                       - R[(s, b)][m]['AUC'] - R[(s, b)][m]['pAUC']) for s in seeds]))
                               for m in MACH}
    return o


def perid_rows(R, seeds, ds, rules):
    return [[RUN(s), ds, rule, m, i, f'{100 * a:.4f}'] for s in seeds for rule in rules
            for m in MACH for i, a in sorted(R[(s, rule)][m]['ids'].items())]


# ------------------------------------------------------------------------------- stages
def check(ctx):
    """Reproduce the committed C1 (pooled-knn) and S1′ᵀ (tta) numbers; no C1ᵀ is computed."""
    ref = {}
    with open(os.path.join(ABL, 'pooled-knn', 'scores', 'confirm_metrics.csv')) as f:
        for row in csv.DictReader(f):
            if row['rule'] in ('C1', 'S1p'):
                for k in KEYS:
                    ref[(row['system'], row['dataset'], row['rule'], row['machine'], k)] = float(row[k])
    for ds in ('dev', 'eval'):
        with open(os.path.join(ABL, 'tta', 'scores', f'{ds}_runs.csv')) as f:
            for row in csv.DictReader(f):
                if row['version'] == 'Ts' and row['rule'] == 'S1p' and '_ao_nogain_' in row['run']:
                    for k, col in (('AUC', 'AUC'), ('pAUC', 'pAUC'), ('mAUC', 'mAUC'), ('Amean', 'mean_AUC_pAUC')):
                        if row[col]:
                            ref[(row['run'], ds, 'S1pT', row['scope'], k)] = float(row[col])
    rules = ('C1', 'S1p', 'S1pT', 'C1_0')
    rows, out, chks = [], {'max_abs_diff_vs_committed': {}, 'n_cells_compared': {}}, {}
    absm = {}
    for ds in ('dev', 'eval'):
        S, R, chk = evaluate(ctx, ALL8, ds, rules)
        chks[ds] = chk
        for (s, rule), res in R.items():
            rows += metric_rows(RUN(s), ds, rule, res)
        for rule in ('C1', 'S1p', 'S1pT'):
            diffs = []
            for s in ALL8:
                for row in metric_rows(RUN(s), ds, rule, R[(s, rule)]):
                    for k, v in zip(KEYS, row[4:]):
                        key = (RUN(s), ds, rule, row[3], k)
                        if key in ref:
                            diffs.append(abs(float(v) - ref[key]))
            out['max_abs_diff_vs_committed'][f'{rule}_{ds}'] = max(diffs)
            out['n_cells_compared'][f'{rule}_{ds}'] = len(diffs)
        out[f'C1_0_vs_C1_{ds}_max_abs_overall'] = {k: max(abs(flat(R[(s, 'C1_0')])[k] - flat(R[(s, 'C1')])[k])
                                                         for s in ALL8) for k in KEYS}
        for rule in rules:
            absm[f'{rule}_{ds}8'] = mean_abs(R, ALL8, rule)
    out['source'] = {'C1, S1p': 'ablation/pooled-knn/scores/confirm_metrics.csv (4 dp)',
                     'S1pT': 'ablation/tta/scores/{dev,eval}_runs.csv, version Ts, rule S1p (4 dp)'}
    out['tolerance'] = 1e-3
    out['pass'] = all(v <= 1e-3 for v in out['max_abs_diff_vs_committed'].values())
    out['abs_8seed'] = absm
    out['alignment'] = dict(ctx.align, per_member=chks)
    write_csv(os.path.join(SCORES, 'metrics_check.csv'),
              ['system', 'dataset', 'rule', 'machine', 'AUC', 'pAUC', 'mAUC', 'Amean'], rows)
    json.dump(out, open(os.path.join(SCORES, 'repro_check.json'), 'w'), indent=2)
    L = ['Reproduction check (no C1ᵀ number computed). max |recomputed − committed| over every per-seed '
         'per-machine and overall cell (AUC, pAUC, mAUC, Amean; committed files are 4 dp):']
    for k, v in out['max_abs_diff_vs_committed'].items():
        L.append(f'    {k:10s} {v:.2e}  ({out["n_cells_compared"][k]} cells)')
    for ds in ('dev', 'eval'):
        L.append(f'    C1 from TTA shift-0 copy vs C1 from pooled dump, {ds}, max |Δ overall| over seeds: '
                 + ' '.join(f'{k} {v:.1e}' for k, v in out[f'C1_0_vs_C1_{ds}_max_abs_overall'].items()))
    L.append(f'PASS (≤ 1e-3): {out["pass"]}')
    L.append('8-seed means AUC / pAUC / mAUC / Amean:')
    for k, v in absm.items():
        L.append(f'    {k:12s} ' + ' '.join(f'{v[x]:6.2f}' for x in KEYS))
    txt = '\n'.join(L)
    print(txt)
    open(os.path.join(SCORES, 'scorecard_check.txt'), 'w').write(txt + '\n')
    if not out['pass']:
        sys.exit('ABORT: reproduction check failed; nothing new is scored')


def _stage(ctx, name, seeds, ds, boot_sets):
    rules = ('C1', 'S1pT', 'C1T')
    S, R, chk = evaluate(ctx, seeds, ds, rules)
    rows = [r for (s, rule), res in R.items() for r in metric_rows(RUN(s), ds, rule, res)]
    write_csv(os.path.join(SCORES, f'metrics_{name}.csv'),
              ['system', 'dataset', 'rule', 'machine', 'AUC', 'pAUC', 'mAUC', 'Amean'], rows)
    write_csv(os.path.join(SCORES, f'perid_{name}.csv'), ['system', 'dataset', 'rule', 'machine', 'id', 'AUC'],
              perid_rows(R, seeds, ds, rules))
    fast = pk.Fast(ctx.meta, ds)
    pt = fast(S[(seeds[0], 'C1')], np.arange(len(fast.sel)))
    assert abs(pt['AUC'] - flat(R[(seeds[0], 'C1')])['AUC']) < 1e-9
    out = {'stage': name, 'dataset': ds, 'seeds': list(seeds), 'abs': {r: mean_abs(R, seeds, r) for r in rules}}
    for tag, sub in boot_sets.items():
        d = delta(R, sub, 'C1T', 'C1')
        print(f'  bootstrap {tag} ({len(sub)} seeds) ...', flush=True)
        d['ci95'] = pk.boot_delta(fast, [(S[(s, 'C1T')], S[(s, 'C1')]) for s in sub])
        d['S1pT_minus_C1'] = {k: v for k, v in delta(R, sub, 'S1pT', 'C1').items() if k.startswith('d_') or k == 'n_pos_Amean'}
        out[tag] = d
    out['eval_ToyCar_id_05'] = None
    if ds == 'eval':
        out['eval_ToyCar_id_05'] = {r: float(np.mean([100 * R[(s, r)]['ToyCar']['ids']['id_05'] for s in seeds])) for r in rules}
        out['eval_valve_id_05'] = {r: float(np.mean([100 * R[(s, r)]['valve']['ids']['id_05'] for s in seeds])) for r in rules}
    return out, R, chk


def fmt_delta(tag, d):
    ci = d['ci95']
    return (f'{tag}: C1ᵀ − C1  ΔAUC {d["d_AUC"]:+.3f} ΔpAUC {d["d_pAUC"]:+.3f} ΔmAUC {d["d_mAUC"]:+.3f} '
            f'ΔAmean {d["d_Amean"]:+.3f} [{ci["Amean"][0]:+.3f}, {ci["Amean"][1]:+.3f}]  seeds>0 {d["n_pos_Amean"]}/{d["n"]}'
            f'  | per seed ' + ' '.join(f'{x:+.3f}' for x in d['d_Amean_per_seed'])
            + '\n      ΔAUC by machine: ' + ' '.join(f'{m} {v:+.2f}' for m, v in d['per_machine_dAUC'].items())
            + '\n      (S1′ᵀ − C1 ΔAmean ' + f'{d["S1pT_minus_C1"]["d_Amean"]:+.3f})')


def fmt_abs(out):
    return '\n'.join(f'    {r:5s} ' + ' '.join(f'{v[k]:6.2f}' for k in KEYS) for r, v in out['abs'].items())


def stageA(ctx):
    rc = os.path.join(SCORES, 'repro_check.json')
    if not committed(rc) or not json.load(open(rc))['pass']:
        sys.exit('ABORT: repro_check.json is not committed or did not pass')
    out, _, chk = _stage(ctx, 'stageA', SEL, 'dev', {'sel_dev': SEL})
    out['gate_A_pass'] = out['sel_dev']['d_Amean'] > 0
    json.dump(out, open(os.path.join(SCORES, 'stageA.json'), 'w'), indent=2)
    txt = (f'Stage A (dev, seeds 2040-2043). 4-seed means AUC / pAUC / mAUC / Amean:\n{fmt_abs(out)}\n'
           + fmt_delta('sel dev', out['sel_dev']) + f'\nGate A (Δ Amean > 0): {out["gate_A_pass"]}')
    print(txt)
    open(os.path.join(SCORES, 'scorecard_stageA.txt'), 'w').write(txt + '\n')


def stageB(ctx):
    if not committed(os.path.join(SCORES, 'stageA.json')):
        sys.exit('ABORT: stageA.json not committed: confirm seeds stay unscored')
    out, R, chk = _stage(ctx, 'stageB', ALL8, 'dev', {'conf_dev': CONF, 'dev8': ALL8})
    out['seeds_scored_new'] = list(CONF)
    A = json.load(open(os.path.join(SCORES, 'stageA.json')))
    out['D1_dev'] = {'gate_A': A['gate_A_pass'],
                     'dev8_dAmean_pos': out['dev8']['d_Amean'] > 0,
                     'dev8_ge6_of_8_pos': out['dev8']['n_pos_Amean'] >= 6,
                     'conf_dev_dAmean_pos': out['conf_dev']['d_Amean'] > 0}
    out['D1_dev']['all'] = all(out['D1_dev'].values())
    json.dump(out, open(os.path.join(SCORES, 'stageB.json'), 'w'), indent=2)
    txt = (f'Stage B (dev). 8-seed means AUC / pAUC / mAUC / Amean (paper dev {PAPER["dev"]}):\n{fmt_abs(out)}\n'
           + fmt_delta('confirm dev (2044-47)', out['conf_dev']) + '\n' + fmt_delta('dev 8 seeds', out['dev8'])
           + f'\nD1 dev conditions: {json.dumps(out["D1_dev"])}')
    print(txt)
    open(os.path.join(SCORES, 'scorecard_stageB.txt'), 'w').write(txt + '\n')


def stage_eval(ctx):
    if not committed(os.path.join(SCORES, 'stageB.json')):
        sys.exit('ABORT: stageB.json not committed: eval stays unscored')
    if os.path.exists(os.path.join(SCORES, 'eval.json')) and committed(os.path.join(SCORES, 'eval.json')):
        sys.exit('ABORT: eval.json already committed: eval is scored once')
    out, R, chk = _stage(ctx, 'eval', ALL8, 'eval', {'eval8': ALL8})
    B = json.load(open(os.path.join(SCORES, 'stageB.json')))
    D1 = dict(B['D1_dev'])
    D1.pop('all')
    D1['eval8_dAmean_ge_0'] = out['eval8']['d_Amean'] >= 0
    D1['adopt_C1T'] = all(D1.values())
    out['D1'] = D1
    json.dump(out, open(os.path.join(SCORES, 'eval.json'), 'w'), indent=2)
    rule = 'C1T' if D1['adopt_C1T'] else 'C1'
    dev_abs, ev_abs = B['abs'][rule], out['abs'][rule]
    fr = {'rule': rule,
          'name': 'C1ᵀ = zR0′ᵀ + zK′ᵀ + zGmean' if rule == 'C1T' else 'C1 = zR0′ + zK′ + zGmean',
          'network': 'ao_nogain recipe (ablation/aug-ops: --wave_aug_ops noise,shift), EMA weights, eval mode',
          'terms': {
              'zR0p': 'z-score of R0 = ArcFace CE of the clip\'s own machine-ID logit'
                      + (' averaged over circular shifts 0, +1%, -1% of the 160,000-sample clip' if rule == 'C1T' else ', unshifted'),
              'zKp': '1-NN cosine distance of the pre-head 128-d embedding'
                     + (' (mean over the 3 shifted copies of the L2-normalised embedding, re-normalised)' if rule == 'C1T' else ', unshifted')
                     + ' to the ID\'s training-part clips; bank z-stats leave-one-out',
              'zGmean': '1-NN cosine distance of gmean (512-d mean over the 8x20 grid of mobilefacenet.conv2 output, '
                        'the input to linear7), UNshifted, to the ID\'s training-part clips; bank z-stats leave-one-out'},
          'bank': 'per machine ID: training clips with valid == False (the replayed 95% the checkpoint trained on); '
                  + ('the bank gets the same TTA as the test clips' if rule == 'C1T' else 'unshifted'),
          'z_statistics': 'per ID, mean and population sd over the bank (K and G: each bank clip scored leave-one-out)',
          'vectors': 'L2-normalised, uncentred, float64',
          'shifts': list(SMALL) if rule == 'C1T' else [0.0],
          'shift_op': 'torch.roll of the wave by round(f*160000) samples; mel recomputed from the rolled wave '
                      '(ablation/tta/extract_tta.py); shift 0 uses the CPU mel',
          'code': 'synthesis/single-final/score.py: score_member(tr, te, tr_label, te_label, valid, rule)',
          'decision': D1,
          'eight_seed_means': {'dev': dev_abs, 'eval': ev_abs}}
    json.dump(fr, open(_FR, 'w'), indent=2, ensure_ascii=False)
    txt = (f'Eval (8 seeds, scored once). 8-seed means AUC / pAUC / mAUC / Amean (paper eval mean {PAPER["eval_mean"]}):\n'
           f'{fmt_abs(out)}\n' + fmt_delta('eval 8 seeds', out['eval8'])
           + f'\neval ToyCar id_05: {json.dumps(out["eval_ToyCar_id_05"])}\neval valve id_05: {json.dumps(out["eval_valve_id_05"])}'
           + f'\nD1: {json.dumps(D1)}\nADOPTED single-model rule: {fr["name"]}')
    print(txt)
    open(os.path.join(SCORES, 'scorecard_eval.txt'), 'w').write(txt + '\n')


def summary(_ctx=None):
    """From scores/ only: 8-seed means per rule x set x machine, and per-ID C1ᵀ − C1 (summary.csv,
    perid_delta.csv). Dev rows come from stage B (8 seeds), eval rows from the eval stage."""
    M, P = {}, {}
    for name, ds in (('stageB', 'dev'), ('eval', 'eval')):
        for r in csv.DictReader(open(os.path.join(SCORES, f'metrics_{name}.csv'))):
            M.setdefault((ds, r['rule'], r['machine']), []).append({k: float(r[k]) for k in KEYS})
        for r in csv.DictReader(open(os.path.join(SCORES, f'perid_{name}.csv'))):
            P.setdefault((ds, r['machine'], r['id'], r['rule']), {})[r['system']] = float(r['AUC'])
    rows = []
    for (ds, rule, m), v in sorted(M.items()):
        assert len(v) == 8, (ds, rule, m, len(v))
        mu = {k: np.mean([x[k] for x in v]) for k in KEYS}
        base = {k: np.mean([x[k] for x in M[(ds, 'C1', m)]]) for k in KEYS}
        rows.append([ds, rule, m] + [f'{mu[k]:.4f}' for k in KEYS] + [f'{mu[k] - base[k]:+.4f}' for k in KEYS])
    write_csv(os.path.join(SCORES, 'summary.csv'),
              ['dataset', 'rule', 'machine'] + [f'{k}_mean8' for k in KEYS] + [f'd_{k}_vs_C1' for k in KEYS], rows)
    rows = []
    for (ds, m, i, rule), v in sorted(P.items()):
        if rule != 'C1T':
            continue
        b = P[(ds, m, i, 'C1')]
        d = [v[k] - b[k] for k in sorted(v)]
        rows.append([ds, m, i, f'{np.mean(list(b.values())):.3f}', f'{np.mean(list(v.values())):.3f}',
                     f'{np.mean(d):+.3f}', sum(x > 0 for x in d), len(d)])
    write_csv(os.path.join(SCORES, 'perid_delta.csv'),
              ['dataset', 'machine', 'id', 'C1_AUC_mean8', 'C1T_AUC_mean8', 'dAUC', 'n_pos', 'n'], rows)
    for r in rows:
        print(*r)


if __name__ == '__main__':
    os.makedirs(SCORES, exist_ok=True)
    stage = sys.argv[1]
    if stage == 'summary':
        sys.exit(summary())
    {'check': check, 'stageA': stageA, 'stageB': stageB, 'eval': stage_eval}[stage](Ctx())
