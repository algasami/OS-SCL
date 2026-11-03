#!/usr/bin/env python
"""Mixed-type E2′ ensembles over every checkpoint with an extract.py dump (see README).

    python ablation/mixed-ensemble/score.py dev     # stage A: dev only; eval is never touched
    python ablation/mixed-ensemble/score.py eval    # stage B: after the eval predictions are committed

The rule (S1′ per member, E2′ = sum of members' S1′), the metrics and the bootstrap are
../ensemble-backend/score.py's, imported unchanged. Dumps come from three ablations'
results/emb; their clip order is checked to agree below data/ before anything is scored.

Writes, per stage <ds>:
  scores/systems_<ds>.csv      fixed systems: overall + per-machine AUC/pAUC/mAUC
  scores/combos_<ds>.csv       size-matched combination families: mean and 5/95% over combos
  scores/bootstrap_<ds>.json   paired clip-bootstrap of the pre-registered contrasts
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
EMB = [f'{ROOT}/OS-SCL-litarms/ablation/lit-arms/results/emb',
       f'{ROOT}/OS-SCL-confirm-seeds/ablation/confirm-seeds/results/emb',
       f'{ROOT}/OS-SCL-scoring-probe/ablation/ensemble-backend/results/emb']
N = 'SCLTFSTgramMFN_'
TYPES = {   # model type -> members (all 300 epochs, fussion 1, m 0.4, ht basic)
    'plain': [f'{N}fr_f1_s{s}' for s in range(2040, 2044)],             # fast recipe
    'aug':   [f'{N}la_aug_s{s}' for s in range(2040, 2048)],
    'gate':  [f'{N}la_gate_s{s}' for s in range(2040, 2044)],
    'eca':   [f'{N}la_eca_s{s}' for s in range(2040, 2044)],
    'old':   [f'{N}cs_relu_max_s{s}' for s in range(2028, 2032)]          # old recipe, = confirm-seeds N4
             + [f'{N}tfl_relu_max_s{s}' for s in range(2024, 2028)],     # O4
    'lin':   [f'{N}tfl_linear_avg_s{s}' for s in range(2024, 2028)],
}
FAST = TYPES['plain'] + TYPES['aug'] + TYPES['gate'] + TYPES['eca']
SYSTEMS = {  # fixed systems, named before any ensemble score existed
    'P4':    TYPES['plain'],
    'A8':    TYPES['aug'],
    'PA12':  TYPES['plain'] + TYPES['aug'],
    'F20':   FAST,
    'ALL32': FAST + TYPES['old'] + TYPES['lin'],
    'N4':    TYPES['old'][:4],          # confirm-seeds' adopted E2′[N4]: the king
    'relu8': TYPES['old'],              # E2′[relu8]
}
CANDIDATES = ['P4', 'A8', 'PA12', 'F20', 'ALL32']   # stage A picks one of these on dev
KING = 'relu8'
COMBO_CAP, RNG_SEED, N_BOOT = 1000, 0, 2000
KEYS = ('AUC', 'pAUC', 'mAUC', 'mean_AUC_pAUC')


def families():
    """Size-matched families: name -> list of member tuples (all combos, or COMBO_CAP sampled)."""
    t = TYPES
    fam = {
        'hom4_plain': [tuple(t['plain'])],
        'hom4_aug':   list(itertools.combinations(t['aug'], 4)),
        'hom4_old':   list(itertools.combinations(t['old'], 4)),
        'mix4_PAGE':  list(itertools.product(t['plain'], t['aug'], t['gate'], t['eca'])),
        'mix4_PGE+P': [(p1, p2, g, e) for p1, p2 in itertools.combinations(t['plain'], 2)
                       for g in t['gate'] for e in t['eca']],         # mix without aug
        'hom8_aug':   [tuple(t['aug'])],
        'hom8_old':   [tuple(t['old'])],
        'mix8_PAGE':  [a + b + c + d for a in itertools.combinations(t['plain'], 2)
                       for b in itertools.combinations(t['aug'], 2)
                       for c in itertools.combinations(t['gate'], 2)
                       for d in itertools.combinations(t['eca'], 2)],
    }
    rng = np.random.default_rng(RNG_SEED)
    for k, v in fam.items():
        if len(v) > COMBO_CAP:
            fam[k] = [v[i] for i in sorted(rng.choice(len(v), COMBO_CAP, replace=False))]
    return fam


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


def main():
    ds = sys.argv[1]
    assert ds in ('dev', 'eval')
    meta, paths = load()
    ev = eb.Evaluator(meta, ds)
    S = {}
    for r, p in paths.items():
        S[r] = eb.per_run_scores(meta, dict(np.load(p)))['S1p']
    out = os.path.join(HERE, 'scores')
    lines = [f'{ds.upper()} ONLY. E2′ = sum of members\' S1′. AUC/pAUC/mAUC x100; '
             f'mean = mean(AUC, pAUC). Paper (one model, dev): AUC {PAPER["dev"]["AUC"]} pAUC '
             f'{PAPER["dev"]["pAUC"]} mAUC {PAPER["dev"]["mAUC"]}; eval mean {PAPER["eval_mean"]}.', '']

    # ---- fixed systems -------------------------------------------------------------------
    sysres = {}
    with open(os.path.join(out, f'systems_{ds}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['system', 'n', 'dataset', 'scope', 'AUC', 'pAUC', 'mAUC', 'mean_AUC_pAUC'])
        lines.append(f'{"system":8s} {"n":>3s}  {"AUC":>6s} {"pAUC":>6s} {"mAUC":>6s} {"mean":>6s} | '
                     + ' '.join(f'{m[:7]:>7s}' for m in MACH))
        for name, mem in SYSTEMS.items():
            res = ev(sum(S[r] for r in mem))
            sysres[name] = res
            o = res['overall']
            w.writerow([name, len(mem), ds, 'overall'] + [f'{o[k]:.4f}' for k in KEYS])
            for m in MACH:
                w.writerow([name, len(mem), ds, m] + [f'{100 * res[m][k]:.4f}' for k in ('AUC', 'pAUC', 'mAUC')] + [''])
            lines.append(f'{name:8s} {len(mem):3d}  ' + ' '.join(f'{o[k]:6.2f}' for k in KEYS) + ' | '
                         + ' '.join(f'{100 * res[m]["AUC"]:7.2f}' for m in MACH))

    # ---- size-matched families -------------------------------------------------------------
    lines += ['', 'Families: mean over combos [5%, 95%]; per machine = mean AUC over combos']
    with open(os.path.join(out, f'combos_{ds}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['family', 'n_members', 'n_combos', 'dataset', 'scope', 'metric', 'mean', 'p05', 'p95'])
        for fam, combos in families().items():
            vals = {('overall', k): [] for k in KEYS}
            vals.update({(m, 'AUC'): [] for m in MACH})
            for c in combos:
                res = ev(sum(S[r] for r in c))
                for k in KEYS:
                    vals[('overall', k)].append(res['overall'][k])
                for m in MACH:
                    vals[(m, 'AUC')].append(100 * res[m]['AUC'])
            for (scope, k), v in vals.items():
                w.writerow([fam, len(combos[0]), len(combos), ds, scope, k, f'{np.mean(v):.4f}',
                            f'{np.percentile(v, 5):.4f}', f'{np.percentile(v, 95):.4f}'])
            o = {k: vals[('overall', k)] for k in KEYS}
            lines.append(f'{fam:11s} n={len(combos[0]):<2d} combos={len(combos):<4d} '
                         + '  '.join(f'{k[:4]} {np.mean(o[k]):6.2f} [{np.percentile(o[k], 5):.2f}, '
                                     f'{np.percentile(o[k], 95):.2f}]' for k in ('AUC', 'mean_AUC_pAUC'))
                         + ' | ' + ' '.join(f'{m[:7]} {np.mean(vals[(m, "AUC")]):.2f}' for m in MACH))

    # ---- paired clip-bootstrap: every candidate vs the king ---------------------------------
    rng = np.random.default_rng(RNG_SEED)
    sc = {n: sum(S[r] for r in SYSTEMS[n]) for n in CANDIDATES + [KING]}
    diffs = {n: {k: [] for k in KEYS} for n in CANDIDATES}
    for _ in range(N_BOOT):
        idx = ev.resample(rng)
        king = ev(sc[KING], idx)['overall']
        for n in CANDIDATES:
            o = ev(sc[n], idx)['overall']
            for k in KEYS:
                diffs[n][k].append(o[k] - king[k])
    boot = {n: {k: [float(np.mean(v)), *map(float, np.percentile(v, [2.5, 97.5]))] for k, v in d.items()}
            for n, d in diffs.items()}
    json.dump(boot, open(os.path.join(out, f'bootstrap_{ds}.json'), 'w'), indent=2)
    lines += ['', f'Paired clip-bootstrap ({N_BOOT}), candidate − E2′[{KING}]: mean [2.5%, 97.5%]']
    for n, d in boot.items():
        lines.append(f'{n:6s} ' + '  '.join(f'{k[:4]} {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]' for k, v in d.items()))

    txt = '\n'.join(lines)
    open(os.path.join(out, f'scorecard_{ds}.txt'), 'w').write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
