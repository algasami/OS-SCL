"""Would a decorrelated TFgram just be a second Tgram? (init only, no training)

init_probe.py found that only a fully linear stack with linear pooling stops
TFgram looking like Log-Mel. A linear stack of linear decimations is a learned
filterbank on the waveform, which is also what Tgram's front end
(`conv_extrctor`, a 1024-tap stride-512 conv) is. This probe checks how close
the two are.

Two metrics, because they answer different questions:
  CKA  -- linear CKA. Invariant to rotation and isotropic scale, but NOT to a
          general linear map, so two different random filterbanks of the same
          signal can score low CKA while carrying the same information.
  R2   -- frame-level ridge-free linear predictability, both directions: how
          much of B's variance is a linear function of A's frame. This is the
          "is the content already there" question.

Control: two independently seeded Tgram front ends compared with each other,
which calibrates what "same information, different random filterbank" scores.

    python ablation/tfgram-redesign/tgram_probe.py <gpu> <out.json>
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
import model.waveq as waveq
from model.net import TgramNet
from dataloader import train_dataset

DEV = torch.device(f'cuda:{int(sys.argv[1])}')
OUT = sys.argv[2]
SEEDS = [2024, 2025, 2026]
NAME_LIST = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
# (label, negative slope, pool, post-hoc frame-rate nonlinearity)
TF_VARIANTS = [('relu_max', 0.0, 'max', False),
               ('leaky_avg', 0.2, 'avg', False),
               ('linear_max', 1.0, 'max', False),
               ('linear_avg', 1.0, 'avg', False),
               ('linear_first', 1.0, 'avg', True)]   # linear stack, LeakyReLU(0.2) after 313 frames


def cka(X, Y):
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    return ((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm())).item()


def lin_r2(X, Y):
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    X1 = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype)], 1)
    W = torch.linalg.lstsq(X1, Y).solution
    return (1 - ((Y - X1 @ W) ** 2).sum() / (Y ** 2).sum()).item()


def run(fn):
    with torch.no_grad():
        return torch.cat([fn(wavs[s:s + 32].to(DEV)).cpu() for s in range(0, len(idx), 32)]).double()


def compare(a, b):
    fa, fb = (v.permute(0, 2, 1).reshape(-1, 128) for v in (a, b))
    return {'cka_clip': cka(a.reshape(len(idx), -1), b.reshape(len(idx), -1)),
            'cka_frame': cka(fa, fb), 'r2_a_to_b': lin_r2(fa, fb), 'r2_b_to_a': lin_r2(fb, fa)}


ds = train_dataset('data/dataset', NAME_LIST, {})
labels = ds.labels.numpy()
rng = np.random.default_rng(0)  # same 656-clip subsample as cka_probe.py / init_probe.py
idx = np.concatenate([rng.choice(np.where(labels == c)[0], 16, replace=False)
                      for c in np.unique(labels)])
wavs = torch.stack([ds[i][0] for i in idx])
mel = torch.stack([ds[i][1] for i in idx]).squeeze(1).double()

results = []
for seed in SEEDS:
    torch.manual_seed(seed)
    tg = TgramNet(mel_bins=128, win_len=1024, hop_len=512).to(DEV).eval()
    torch.manual_seed(seed + 1000)
    tg_other = TgramNet(mel_bins=128, win_len=1024, hop_len=512).to(DEV).eval()
    reps = {'Tgram': run(tg), 'Tgram_front': run(tg.conv_extrctor),
            'Tgram_front_other_seed': run(tg_other.conv_extrctor)}

    row = {'seed': seed, 'pairs': {}}
    row['pairs']['control: Tgram_front ~ Tgram_front_other_seed'] = compare(
        reps['Tgram_front'], reps['Tgram_front_other_seed'])
    row['pairs']['Log-Mel ~ Tgram'] = compare(mel, reps['Tgram'])

    for label, slope, pool, post in TF_VARIANTS:
        waveq.tf_act = lambda x, act, a=slope: F.leaky_relu(x, a)
        torch.manual_seed(seed)
        net = waveq.TFgram(41, act='leaky', pool=pool).to(DEV).eval()
        tf = run(lambda w: F.leaky_relu(net(w, True), 0.2) if post else net(w, True))
        for other in ('Tgram', 'Tgram_front'):
            row['pairs'][f'{other} ~ TF:{label}'] = compare(reps[other], tf)
        row['pairs'][f'Log-Mel ~ TF:{label}'] = compare(mel, tf)
    results.append(row)
    for k, v in row['pairs'].items():
        print(f"seed {seed}  {k:48s} cka_clip {v['cka_clip']:.3f}  cka_frame {v['cka_frame']:.3f}  "
              f"R2 a->b {v['r2_a_to_b']:.3f}  b->a {v['r2_b_to_a']:.3f}", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump({'note': 'Random init, eval mode, no training; 656 clips, 16 per machine ID. '
                       'Pair "A ~ B": r2_a_to_b is the share of B predictable linearly from A, frame-wise.',
               'results': results}, f, indent=2)
