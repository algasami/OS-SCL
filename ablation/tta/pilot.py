#!/usr/bin/env python
"""Pilot: how position-dependent is R0 on normal held-out clips, per shift scheme? (README, Pilot)

    python ablation/tta/pilot.py ablation/tta/results/emb

Reads <emb>/valid/<run>.npz from extract_tta.py --split valid: each run's own replayed 5%
held-out training clips (normal, never trained on), 16 shifts. No test clip is read.

Per run and scheme S (a subset of the 16 shifts, always containing 0), over the 41 IDs:
  rho      mean over clips of var_k∈S R0(clip, k), divided by var over clips of R0(clip, 0):
           the share of a normal clip's score spread that comes from where events sit in the clip.
           Averaging S's copies removes part of it.
  penalty  (mean R0 over S's copies − mean R0 at shift 0) / sd of R0 at shift 0: does shifting
           make normal clips look anomalous (the splice, or positions the model never saw)?
  shrink   sd over clips of the S-averaged R0, divided by sd at shift 0.
  cos      mean cosine of a shifted pre-head embedding to the shift-0 one.
All are pooled as means over IDs (per-ID first, so large IDs don't dominate).
Writes scores/pilot.csv and prints the table and the scheme pick (README rule).
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMES = {'small': [0, 0.01, -0.01],
           'match': [0, 0.03, -0.03, 0.055, -0.055, 0.08, -0.08],
           'wide': [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]}
PENALTY_MAX = 0.25


def main():
    emb = sys.argv[1]
    meta = np.load(os.path.join(emb, 'meta.npz'))
    trl = meta['tr_label']
    rows = []
    for f in sorted(os.listdir(os.path.join(emb, 'valid'))):
        a = np.load(os.path.join(emb, 'valid', f))
        run = f[:-4].replace('SCLTFSTgramMFN_', '')
        ce, pre, idx, sh = a['ce'].astype(np.float64), a['pre'].astype(np.float64), a['idx'], list(a['shifts'])
        lab = trl[idx]
        pre = pre / np.linalg.norm(pre, axis=2, keepdims=True)
        for name, S in SCHEMES.items():
            cols = [sh.index(s) for s in S]
            stats = {k: [] for k in ('rho', 'penalty', 'shrink', 'cos')}
            for c in np.unique(lab):
                x = ce[lab == c][:, cols]
                if len(x) < 5:
                    continue
                v0, sd0 = x[:, 0].var(), x[:, 0].std()
                stats['rho'].append(x.var(1).mean() / v0)
                stats['penalty'].append((x.mean() - x[:, 0].mean()) / sd0)
                stats['shrink'].append(x.mean(1).std() / sd0)
                p = pre[lab == c][:, cols]
                stats['cos'].append((p[:, 1:] * p[:, :1]).sum(2).mean())
            rows.append([run, name, len(S), len(idx)] + [f'{np.mean(v):.4f}' for v in stats.values()])
    out = os.path.join(HERE, 'scores')
    os.makedirs(out, exist_ok=True)
    # per single shift and machine: the penalty of that one copy vs shift 0 (figure 1)
    mach = np.array([p.split('/')[-3] for p in meta['tr_paths']])
    with open(os.path.join(out, 'pilot_shift.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'shift', 'machine', 'penalty', 'cos'])
        for fn in sorted(os.listdir(os.path.join(emb, 'valid'))):
            a = np.load(os.path.join(emb, 'valid', fn))
            run = fn[:-4].replace('SCLTFSTgramMFN_', '')
            ce, pre, idx = a['ce'].astype(np.float64), a['pre'].astype(np.float64), a['idx']
            pre = pre / np.linalg.norm(pre, axis=2, keepdims=True)
            lab, mm = trl[idx], mach[idx]
            for j, s in enumerate(a['shifts']):
                for m in ['all'] + sorted(set(mm)):
                    pen, cs = [], []
                    for c in np.unique(lab[(mm == m) | (m == 'all')]):
                        x = ce[lab == c]
                        pen.append((x[:, j].mean() - x[:, 0].mean()) / x[:, 0].std())
                        cs.append((pre[lab == c][:, j] * pre[lab == c][:, 0]).sum(1).mean())
                    w.writerow([run, f'{s:.3f}', m, f'{np.mean(pen):.4f}', f'{np.mean(cs):.4f}'])
    with open(os.path.join(out, 'pilot.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'scheme', 'K', 'n_clips', 'rho', 'penalty', 'shrink', 'cos'])
        w.writerows(rows)
    print(f'{"run":22s} {"scheme":6s} {"K":>2s} {"rho":>6s} {"penalty":>8s} {"shrink":>7s} {"cos":>6s}')
    for r in rows:
        print(f'{r[0]:22s} {r[1]:6s} {r[2]:2d} {r[4]:>6s} {r[5]:>8s} {r[6]:>7s} {r[7]:>6s}')
    # README pilot rule: on ao_nogain, among {match, wide}, the larger rho whose penalty ≤ PENALTY_MAX
    ng = {s: [r for r in rows if r[0].startswith('ao_nogain') and r[1] == s] for s in ('match', 'wide')}
    m = {s: (np.mean([float(r[4]) for r in v]), np.mean([float(r[5]) for r in v])) for s, v in ng.items()}
    ok = [s for s in m if m[s][1] <= PENALTY_MAX]
    pick = max(ok, key=lambda s: m[s][0]) if ok else 'match'
    print('\nao_nogain means: ' + '; '.join(f'{s} rho {v[0]:.3f} penalty {v[1]:+.3f}' for s, v in m.items()))
    print(f'pilot pick: {pick}')


if __name__ == '__main__':
    main()
