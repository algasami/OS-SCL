import sys, itertools, numpy as np
sys.argv = ['x', 'emb', 'SCLTFSTgramMFN_tfl_relu_max_s2024']
exec(open('score.py').read().split("allres = {}")[0])
R = ['SCLTFSTgramMFN_tfl_relu_max_s2024','SCLTFSTgramMFN_tfl_relu_max_s2025','SCLTFSTgramMFN_tfl_linear_avg_s2024','SCLTFSTgramMFN_tfl_linear_avg_s2025']
S = {r: scores_for(dict(np.load(f'emb/{r}_train.npz')), dict(np.load(f'emb/{r}_test.npz'))) for r in R}
te = dict(np.load(f'emb/{R[0]}_test.npz'))
for k in ['R0z', 'ens_R0z+knn_pre_z']:
  for n in (1, 2, 3, 4):
    vals = []
    for c in itertools.combinations(R, n):
        v = np.array([evaluate(te, sum(S[r][k] for r in c))[m] for m in MACH]) * 100
        vals.append(v[:, :3].mean(0))
    vals = np.array(vals)
    print(f'{k:18s} n={n} combos={len(vals)}  AUC {vals[:,0].mean():.2f}  pAUC {vals[:,1].mean():.2f}  mAUC {vals[:,2].mean():.2f}')
