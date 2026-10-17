import sys, numpy as np
from scipy.stats import rankdata
sys.argv = ['x', 'emb', 'x']
exec(open('score.py').read().split("allres = {}")[0])
R = ['SCLTFSTgramMFN_tfl_relu_max_s2024','SCLTFSTgramMFN_tfl_relu_max_s2025','SCLTFSTgramMFN_tfl_linear_avg_s2024','SCLTFSTgramMFN_tfl_linear_avg_s2025']
te = {r: dict(np.load(f'emb/{r}_test.npz')) for r in R}
tr = {r: dict(np.load(f'emb/{r}_train.npz')) for r in R}
t0 = te[R[0]]
def show(name, s):
    v = np.array([evaluate(t0, s)[m] for m in MACH]) * 100
    print(f'{name:34s} AUC {v[:,0].mean():.2f}  pAUC {v[:,1].mean():.2f}  mAUC {v[:,2].mean():.2f}')
show('single R0, mean of 4', None) if False else None
print('single R0 (mean of 4 models): AUC %.2f' % np.mean([np.mean([evaluate(te[r], te[r]['ce'])[m][0] for m in MACH])*100 for r in R]))
show('raw sum of R0 (no normalisation)', sum(te[r]['ce'] for r in R))
show('sum of log R0', sum(np.log(te[r]['ce']) for r in R))
show('sum of per-ID z R0 (train stats)', sum(scores_for(tr[r], te[r])['R0z'] for r in R))
# held-out valid split is not reconstructed here; check how biased train-clip CE is vs unseen normal test clips
for r in R[:1]:
    for m in MACH:
        mt = tr[r]['machine'] == m; mn = (te[r]['machine'] == m) & (te[r]['y'] == 0)
        print(f'  {m:12s} median CE train-normal {np.median(tr[r]["ce"][mt]):.3f}  test-normal {np.median(te[r]["ce"][mn]):.3f}')
