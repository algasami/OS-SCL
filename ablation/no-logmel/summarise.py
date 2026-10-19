#!/usr/bin/env python
"""Cell means and the README's contrasts C1-C4, D1/D2 -> scores/summary.csv.

    /home/f74134118/anaconda3/bin/python ablation/no-logmel/summarise.py   # base: imports make_plots

Δ = 4-seed mean difference of R0 AUC (points). se is the unpaired (Welch) se;
the 90% interval is a seed bootstrap (resample each cell's seeds, 10k draws).
Reads scores/ only (this ablation's and ../fast-recipe's via scores_path).
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_plots import CELLS, MACH, values  # noqa: E402

CONTRASTS = [('C1', 'nl_f5', 'fr_f1'), ('C2a', 'nl_f5', 'fr_f4'), ('C2b', 'nl_f6', 'nl_f2'),
             ('C3', 'nl_f5', 'nl_f6'), ('C4', 'nl_f7', 'nl_f6')]
MARGIN = -0.20


def boot(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.choice(a, (n, len(a))).mean(1) - rng.choice(b, (n, len(b))).mean(1)
    return np.percentile(d, [5, 95])


def main():
    out = []
    print(f'{"cell":7s} {"ds":4s} {"n":>2s} {"AUC":>6s} {"sd":>5s} {"pAUC":>6s} {"mAUC":>6s}')
    for p, label, *_ in CELLS:
        for ds in ('dev', 'eval'):
            a = values(p, ds)
            if not len(a):
                continue
            pa, ma = values(p, ds, key='pAUC'), values(p, ds, key='mAUC')
            sd = a.std(ddof=1) if len(a) > 1 else float('nan')
            print(f'{p:7s} {ds:4s} {len(a):2d} {a.mean():6.2f} {sd:5.2f} {pa.mean():6.2f} {ma.mean():6.2f}')
            out.append({'kind': 'cell', 'name': p, 'dataset': ds, 'machine': 'overall', 'n': len(a),
                        'mean': f'{a.mean():.3f}', 'sd': f'{sd:.3f}', 'pAUC': f'{pa.mean():.3f}',
                        'mAUC': f'{ma.mean():.3f}'})
    print()
    res = {}
    for c, x, y in CONTRASTS:
        for ds in ('dev', 'eval'):
            for m in [None] + MACH:
                a, b = values(x, ds, m), values(y, ds, m)
                if len(a) < 2 or len(b) < 2:
                    continue
                d = a.mean() - b.mean()
                se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
                lo, hi = boot(a, b)
                out.append({'kind': 'contrast', 'name': f'{c}:{x}-{y}', 'dataset': ds,
                            'machine': m or 'overall', 'n': f'{len(a)}v{len(b)}', 'delta': f'{d:.3f}',
                            'se': f'{se:.3f}', 'boot90_lo': f'{lo:.3f}', 'boot90_hi': f'{hi:.3f}'})
                if m is None:
                    res[(c, ds)] = d
                    print(f'{c:4s} {x}-{y} {ds:4s} n={len(a)}v{len(b)}  Δ {d:+.2f}  se {se:.2f}  '
                          f'90% [{lo:+.2f}, {hi:+.2f}]')
    print()
    ok = lambda c: all((c, ds) in res and res[(c, ds)] >= MARGIN for ds in ('dev', 'eval'))  # noqa: E731
    have = lambda *cs: all((c, ds) in res for c in cs for ds in ('dev', 'eval'))  # noqa: E731
    if have('C1'):
        print(f'D1 (drop Log-Mel from the paper model): {"PASS" if ok("C1") else "FAIL"}')
        out.append({'kind': 'decision', 'name': 'D1', 'mean': 'PASS' if ok('C1') else 'FAIL'})
    if have('C2a', 'C2b'):
        p = ok('C2a') and ok('C2b')
        print(f'D2 (TFgram functionally replaces Log-Mel): {"PASS" if p else "FAIL"}')
        out.append({'kind': 'decision', 'name': 'D2', 'mean': 'PASS' if p else 'FAIL'})
    fields = ['kind', 'name', 'dataset', 'machine', 'n', 'mean', 'sd', 'pAUC', 'mAUC', 'delta', 'se',
              'boot90_lo', 'boot90_hi']
    with open(os.path.join(HERE, 'scores', 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)


if __name__ == '__main__':
    main()
