#!/usr/bin/env python
"""E2′ ensembles of the 8 ao_nogain models against E2′[A8] (see README).

    python ablation/nogain-ensemble/score.py dev     # stage A: dev only; eval is never touched
    python ablation/nogain-ensemble/score.py eval    # stage B: after the eval predictions are committed

The rule (S1′ per member, E2′ = sum of members' S1′), the metrics and the clip bootstrap are
../ensemble-backend/score.py's, imported unchanged. Dumps come from ../ensemble-backend/extract.py
runs in three ablations' results/emb (scratch); their clip order is checked to agree below data/.

Writes, per stage <ds>:
  scores/systems_<ds>.csv      fixed systems: overall + per-machine AUC/pAUC/mAUC
  scores/perid_<ds>.csv        fixed systems: per-ID AUC
  scores/combos_<ds>.csv       4-member families: mean and 5/95% over seed subsets, incl. paired Δ
  scores/bootstrap_<ds>.json   paired clip-bootstrap, every candidate (and F20) − A8
  scores/scorecard_<ds>.txt    everything above, readable
"""
import csv
import importlib.util
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('eb_score', os.path.join(HERE, '..', 'ensemble-backend', 'score.py'))
eb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eb)
MACH, PAPER = eb.MACH, eb.PAPER

ROOT = '/home/f74134118'
EMB = [f'{ROOT}/OS-SCL-augops/ablation/aug-ops/results/emb',              # ao_nogain
       f'{ROOT}/OS-SCL-litarms/ablation/lit-arms/results/emb',            # fr_f1, la_aug, la_gate, la_eca
       f'{ROOT}/OS-SCL-confirm-seeds/ablation/confirm-seeds/results/emb']  # old recipe (relu8)
N = 'SCLTFSTgramMFN_'
SEEDS = list(range(2040, 2048))
TYPES = {   # all 300 epochs, fussion 1, m 0.4, ht basic
    'nogain': [f'{N}ao_nogain_s{s}' for s in SEEDS],                       # fast + --wave_aug_ops noise,shift
    'aug':    [f'{N}la_aug_s{s}' for s in SEEDS],                          # fast + full aug
    'plain':  [f'{N}fr_f1_s{s}' for s in range(2040, 2044)],
    'gate':   [f'{N}la_gate_s{s}' for s in range(2040, 2044)],
    'eca':    [f'{N}la_eca_s{s}' for s in range(2040, 2044)],
    'old':    [f'{N}cs_relu_max_s{s}' for s in range(2028, 2032)]
              + [f'{N}tfl_relu_max_s{s}' for s in range(2024, 2028)],
}
OTHERS = TYPES['plain'] + TYPES['gate'] + TYPES['eca']
SYSTEMS = {  # fixed before any ensemble score existed
    'G8':    TYPES['nogain'],
    'GA16':  TYPES['nogain'] + TYPES['aug'],
    'F20g':  OTHERS + TYPES['nogain'],        # ../mixed-ensemble's F20 with aug -> nogain
    'A8':    TYPES['aug'],                    # the bar: the reported best ensemble
    'F20':   OTHERS + TYPES['aug'],           # reference (mixed-ensemble's best on eval, not adopted)
    'relu8': TYPES['old'],                    # reference (the previous bar)
}
CANDIDATES = ['G8', 'GA16', 'F20g']   # stage A picks one of these on dev
BAR = 'A8'
RNG_SEED, N_BOOT = 0, 2000
KEYS = ('AUC', 'pAUC', 'mAUC', 'mean_AUC_pAUC')


def load():
    metas = [dict(np.load(os.path.join(d, 'meta.npz'))) for d in EMB]
    below = lambda a: [p.split('/data/', 1)[1] for p in a]   # noqa: E731
    for m in metas[1:]:
        for k in metas[0]:
            same = below(m[k]) == below(metas[0][k]) if k.endswith('paths') else np.array_equal(m[k], metas[0][k])
            assert same, f'meta mismatch on {k}'
    where = {}
    for d in EMB:
        for f in os.listdir(d):
            if f.endswith('.npz') and f != 'meta.npz':
                where.setdefault(f[:-4], os.path.join(d, f))
    need = sorted(set(r for v in TYPES.values() for r in v))
    missing = [r for r in need if r not in where]
    assert not missing, f'no dump for {missing}'
    return metas[0], {r: where[r] for r in need}


def fmt_row(name, n, o):
    return f'{name:7s} {n:3d}  ' + ' '.join(f'{o[k]:6.2f}' for k in KEYS)


def main():
    ds = sys.argv[1]
    assert ds in ('dev', 'eval')
    meta, paths = load()
    ev = eb.Evaluator(meta, ds)
    S = {r: eb.per_run_scores(meta, dict(np.load(p)))['S1p'] for r, p in paths.items()}
    out = os.path.join(HERE, 'scores')
    os.makedirs(out, exist_ok=True)
    lines = [f'{ds.upper()} ONLY. E2′ = sum of members\' S1′. AUC/pAUC/mAUC x100; mean = mean(AUC, pAUC). '
             f'Paper (one model): dev AUC {PAPER["dev"]["AUC"]} pAUC {PAPER["dev"]["pAUC"]} '
             f'mAUC {PAPER["dev"]["mAUC"]}; eval mean {PAPER["eval_mean"]}.', '']

    # ---- fixed systems -------------------------------------------------------------------
    sysres = {}
    with open(os.path.join(out, f'systems_{ds}.csv'), 'w', newline='') as f, \
            open(os.path.join(out, f'perid_{ds}.csv'), 'w', newline='') as g:
        w, wi = csv.writer(f), csv.writer(g)
        w.writerow(['system', 'n', 'dataset', 'scope', 'AUC', 'pAUC', 'mAUC', 'mean_AUC_pAUC'])
        wi.writerow(['system', 'n', 'dataset', 'machine', 'id', 'AUC'])
        lines.append(f'{"system":7s} {"n":>3s}  {"AUC":>6s} {"pAUC":>6s} {"mAUC":>6s} {"mean":>6s} | '
                     + ' '.join(f'{m[:7]:>7s}' for m in MACH))
        for name, mem in SYSTEMS.items():
            res = ev(sum(S[r] for r in mem))
            sysres[name] = res
            o = res['overall']
            w.writerow([name, len(mem), ds, 'overall'] + [f'{o[k]:.4f}' for k in KEYS])
            for m in MACH:
                w.writerow([name, len(mem), ds, m] + [f'{100 * res[m][k]:.4f}' for k in ('AUC', 'pAUC', 'mAUC')] + [''])
                for i, a in sorted(res[m]['ids'].items()):
                    wi.writerow([name, len(mem), ds, m, i, f'{100 * a:.4f}'])
            lines.append(fmt_row(name, len(mem), o) + ' | ' + ' '.join(f'{100 * res[m]["AUC"]:7.2f}' for m in MACH))
    lines += ['', 'ToyCar per ID (AUC):']
    for name, res in sysres.items():
        lines.append(f'{name:7s} ' + '  '.join(f'{i} {100 * a:.2f}' for i, a in sorted(res['ToyCar']['ids'].items())))

    # ---- 4-member families over seed subsets (70 each), and the seed-paired difference -------
    lines += ['', 'Families over all 70 four-seed subsets of 2040-2047: mean [5%, 95%]. '
              'paired = nogain(subset) − aug(same subset).']
    fam = {k: [] for k in ('hom4_nogain', 'hom4_aug', 'paired4')}
    fam_m = {k: {m: [] for m in MACH} for k in fam}
    for sub in itertools.combinations(range(len(SEEDS)), 4):
        g = ev(sum(S[TYPES['nogain'][i]] for i in sub))
        a = ev(sum(S[TYPES['aug'][i]] for i in sub))
        fam['hom4_nogain'].append(g['overall'])
        fam['hom4_aug'].append(a['overall'])
        fam['paired4'].append({k: g['overall'][k] - a['overall'][k] for k in KEYS})
        for m in MACH:
            fam_m['hom4_nogain'][m].append(100 * g[m]['AUC'])
            fam_m['hom4_aug'][m].append(100 * a[m]['AUC'])
            fam_m['paired4'][m].append(100 * (g[m]['AUC'] - a[m]['AUC']))
    with open(os.path.join(out, f'combos_{ds}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['family', 'n_members', 'n_combos', 'dataset', 'scope', 'metric', 'mean', 'p05', 'p95',
                    'frac_pos'])
        for k, rows in fam.items():
            cells = [('overall', key, [r[key] for r in rows]) for key in KEYS] + \
                    [(m, 'AUC', fam_m[k][m]) for m in MACH]
            for scope, key, v in cells:
                w.writerow([k, 4, len(rows), ds, scope, key, f'{np.mean(v):.4f}', f'{np.percentile(v, 5):.4f}',
                            f'{np.percentile(v, 95):.4f}', f'{np.mean(np.array(v) > 0):.4f}' if k == 'paired4' else ''])
            lines.append(f'{k:11s} combos={len(rows)}  '
                         + '  '.join(f'{key[:4]} {np.mean([r[key] for r in rows]):+6.2f} '
                                     f'[{np.percentile([r[key] for r in rows], 5):+.2f}, '
                                     f'{np.percentile([r[key] for r in rows], 95):+.2f}]' for key in KEYS)
                         + ' | ' + ' '.join(f'{m[:7]} {np.mean(fam_m[k][m]):+.2f}' for m in MACH))
        pos = np.mean([r['mean_AUC_pAUC'] > 0 for r in fam['paired4']])
        lines.append(f'paired4: nogain ahead on mean(AUC, pAUC) in {100 * pos:.0f}% of subsets')

    # ---- paired clip-bootstrap: every candidate and F20 against A8 ---------------------------
    rng = np.random.default_rng(RNG_SEED)
    versus = CANDIDATES + ['F20']
    sc = {n: sum(S[r] for r in SYSTEMS[n]) for n in versus + [BAR]}
    diffs = {n: {k: [] for k in KEYS} for n in versus}
    for _ in range(N_BOOT):
        idx = ev.resample(rng)
        bar = ev(sc[BAR], idx)['overall']
        for n in versus:
            o = ev(sc[n], idx)['overall']
            for k in KEYS:
                diffs[n][k].append(o[k] - bar[k])
    boot = {n: {k: [float(np.mean(v)), *map(float, np.percentile(v, [2.5, 97.5]))] for k, v in d.items()}
            for n, d in diffs.items()}
    json.dump(boot, open(os.path.join(out, f'bootstrap_{ds}.json'), 'w'), indent=2)
    lines += ['', f'Paired clip-bootstrap ({N_BOOT}), system − E2′[{BAR}]: mean [2.5%, 97.5%]']
    for n, d in boot.items():
        lines.append(f'{n:6s} ' + '  '.join(f'{k[:4]} {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]' for k, v in d.items()))

    if ds == 'dev':
        best = max(sysres[n]['overall']['mean_AUC_pAUC'] for n in CANDIDATES)
        close = [n for n in CANDIDATES if best - sysres[n]['overall']['mean_AUC_pAUC'] <= 0.05]
        pick = min(close, key=lambda n: len(SYSTEMS[n]))   # tie within 0.05 -> smaller n
        lines += ['', f'Dev pick (highest dev mean; a tie within 0.05 goes to the smaller n): {pick}']

    txt = '\n'.join(lines)
    open(os.path.join(out, f'scorecard_{ds}.txt'), 'w').write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
