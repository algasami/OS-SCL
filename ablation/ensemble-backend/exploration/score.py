"""Exploratory dev-only comparison of embedding back-ends vs eval.py's rule (R0)."""
import sys
import numpy as np
from sklearn import metrics
from sklearn.covariance import LedoitWolf

D = sys.argv[1]
RUNS = sys.argv[2:]
MACH = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def knn(tr, te, loo=False):
    s = te @ tr.T
    if loo:
        np.fill_diagonal(s, -np.inf)
    return 1 - s.max(1)


def maha(tr, te):
    lw = LedoitWolf().fit(tr)
    d = te - lw.location_
    return np.einsum('ij,jk,ik->i', d, lw.precision_, d)


def maha_loo_proxy(tr):
    lw = LedoitWolf().fit(tr)
    d = tr - lw.location_
    return np.einsum('ij,jk,ik->i', d, lw.precision_, d)


def scores_for(tr, te):
    out = {}
    for c in ('pre', 'post'):
        a, b = l2(tr[c]), l2(te[c])
        out[c] = (a, b)
    S = {}
    ids = sorted(set(te['label']))
    for k in ['R0', 'R0z', 'knn_post', 'knn_pre', 'knn_pre_z', 'maha_post', 'maha_pre', 'maha_pre_z',
              'ens_R0z+knn_pre_z']:
        S[k] = np.zeros(len(te['label']))
    for lab in ids:
        mt, me = tr['label'] == lab, te['label'] == lab
        z = lambda s, ref: (s - ref.mean()) / ref.std()
        S['R0'][me] = te['ce'][me]
        S['R0z'][me] = z(te['ce'][me], tr['ce'][mt])
        S['knn_post'][me] = knn(out['post'][0][mt], out['post'][1][me])
        kp = knn(out['pre'][0][mt], out['pre'][1][me])
        S['knn_pre'][me] = kp
        S['knn_pre_z'][me] = z(kp, knn(out['pre'][0][mt], out['pre'][0][mt], loo=True))
        S['maha_post'][me] = maha(tr['post'][mt], te['post'][me])
        mp = maha(tr['pre'][mt], te['pre'][me])
        S['maha_pre'][me] = mp
        S['maha_pre_z'][me] = z(mp, maha_loo_proxy(tr['pre'][mt]))
    S['ens_R0z+knn_pre_z'] = S['R0z'] + S['knn_pre_z']
    return S


def evaluate(te, s):
    rows = {}
    for m in MACH:
        mm = te['machine'] == m
        y = te['y'][mm]
        auc = metrics.roc_auc_score(y, s[mm])
        pauc = metrics.roc_auc_score(y, s[mm], max_fpr=0.1)
        ida = [metrics.roc_auc_score(te['y'][mm & (te['id'] == i)], s[mm & (te['id'] == i)])
               for i in sorted(set(te['id'][mm]))]
        rows[m] = (auc, pauc, min(ida), np.mean(ida))
    return rows


allres = {}
for r in RUNS:
    tr = dict(np.load(f'{D}/{r}_train.npz'))
    te = dict(np.load(f'{D}/{r}_test.npz'))
    S = scores_for(tr, te)
    allres[r] = {k: evaluate(te, v) for k, v in S.items()}

rules = list(next(iter(allres.values())).keys())
print('DEV ONLY. overall = mean over 6 machines. AUC / pAUC / mAUC / idAUC, x100')
for r in RUNS:
    print('\n' + r)
    for k in rules:
        v = np.array([allres[r][k][m] for m in MACH]) * 100
        per = ' '.join(f'{x:6.2f}' for x in v[:, 0])
        print(f'  {k:20s} {v[:,0].mean():6.2f} {v[:,1].mean():6.2f} {v[:,2].mean():6.2f} {v[:,3].mean():6.2f} | {per}')
print('\nmean delta vs R0 over runs (AUC, pAUC, mAUC) and #runs with AUC delta > 0')
for k in rules[1:]:
    d = np.array([[np.array([allres[r][k][m][j] - allres[r]['R0'][m][j] for m in MACH]).mean() for j in range(3)]
                  for r in RUNS]) * 100
    pm = np.array([[allres[r][k][m][0] - allres[r]['R0'][m][0] for m in MACH] for r in RUNS]).mean(0) * 100
    print(f'  {k:20s} {d[:,0].mean():+6.2f} {d[:,1].mean():+6.2f} {d[:,2].mean():+6.2f}  wins {int((d[:,0]>0).sum())}/{len(RUNS)}'
          f'  | per machine ' + ' '.join(f'{x:+5.2f}' for x in pm))
print('machines:', MACH)

# ---- ensembles across the given runs (same clip order in every npz) ----
print('\nENSEMBLES over all runs given (sum of per-ID z-scores)')
Ss = {}
for r in RUNS:
    tr = dict(np.load(f'{D}/{r}_train.npz')); te = dict(np.load(f'{D}/{r}_test.npz'))
    Ss[r] = scores_for(tr, te)
te0 = dict(np.load(f'{D}/{RUNS[0]}_test.npz'))
for k in ['R0z', 'knn_pre_z', 'ens_R0z+knn_pre_z']:
    s = sum(Ss[r][k] for r in RUNS)
    v = np.array([evaluate(te0, s)[m] for m in MACH]) * 100
    single = np.mean([np.array([allres[r]['R0'][m] for m in MACH])[:, 0].mean() * 100 for r in RUNS])
    print(f'  ens[{k:18s}] x{len(RUNS)}: {v[:,0].mean():6.2f} {v[:,1].mean():6.2f} {v[:,2].mean():6.2f}  '
          f'(mean single-run R0 AUC {single:.2f}) | ' + ' '.join(f'{x:6.2f}' for x in v[:, 0]))
