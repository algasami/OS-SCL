"""Does the TFgram stack compute an energy envelope *at init*, and why?

No training. Builds the TFgram branch at random init with the activation's
negative slope swept from 0 (ReLU) to 1 (identity, i.e. a fully linear stack),
under both pooling modes, and measures CKA against Log-Mel on the same 656
clips as cka_probe.py.

Hypothesis under test: LeakyReLU(a) = (1+a)/2 * x + (1-a)/2 * |x|. Pooling over
~500 samples averages the zero-mean linear term away and keeps the rectified
|x| term, which is an energy envelope, i.e. a spectrogram. If so, CKA with
Log-Mel should stay high for every a < 1 under avg pooling and collapse only at
a = 1.

    python ablation/tfgram-redesign/init_probe.py <gpu> <out.json>
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
import model.waveq as waveq
from dataloader import train_dataset

DEV = torch.device(f'cuda:{int(sys.argv[1])}')
OUT = sys.argv[2]
SLOPES = [0.0, 0.2, 0.5, 0.8, 0.95, 1.0]
SEEDS = [2024, 2025, 2026]
NAME_LIST = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']


def cka(X, Y):
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    return ((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm())).item()


ds = train_dataset('data/dataset', NAME_LIST, {})
labels = ds.labels.numpy()
rng = np.random.default_rng(0)  # same subsample as cka_probe.py
idx = np.concatenate([rng.choice(np.where(labels == c)[0], 16, replace=False)
                      for c in np.unique(labels)])
wavs = torch.stack([ds[i][0] for i in idx])
mel = torch.stack([ds[i][1] for i in idx]).squeeze(1).double()
mel_clip = mel.reshape(len(idx), -1)
mel_frame = mel.permute(0, 2, 1).reshape(-1, 128)

results = []
for slope in SLOPES:
    # the blocks call waveq.tf_act at forward time, so patching it here
    # changes every activation site in the branch
    waveq.tf_act = lambda x, act, a=slope: F.leaky_relu(x, a)
    for pool in ('max', 'avg'):
        for seed in SEEDS:
            torch.manual_seed(seed)
            net = waveq.TFgram(41, act='leaky', pool=pool).to(DEV).eval()
            with torch.no_grad():
                tf = torch.cat([net(wavs[s:s + 32].to(DEV), True).cpu()
                                for s in range(0, len(idx), 32)]).double()
            row = {'slope': slope, 'pool': pool, 'seed': seed,
                   'cka_clip': cka(mel_clip, tf.reshape(len(idx), -1)),
                   'cka_frame': cka(mel_frame, tf.permute(0, 2, 1).reshape(-1, 128))}
            results.append(row)
            print(f"slope {slope:4.2f} {pool}  seed {seed}  clip {row['cka_clip']:.3f}  frame {row['cka_frame']:.3f}",
                  flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump({'note': 'TFgram at random init (eval mode, BN at init stats = identity); '
                       'no training. CKA vs Log-Mel on 656 clips, 16 per machine ID.',
               'results': results}, f, indent=2)
