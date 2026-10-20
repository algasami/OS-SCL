"""Margin-trap diagnostic, copied from ../no-logmel/diag_margin.py and extended to read
state snapshots (student or EMA weights) as well as model.pth.

usage (repo root):  python ablation/margin-warmup/diag_margin.py <target> [...]
  <target> = <ckpt_dir>                         the best-EMA model.pth
           | <ckpt_dir>/state_e60.pth:net       the student at the end of epoch 60
           | <ckpt_dir>/state_e60.pth:ema_net   the EMA net at the end of epoch 60
Env DEV (default cuda:0; cpu works, ~1 min for f7).

Same 1024 training clips (rng 0) as ../no-logmel's diag, so rows are comparable.
Always measured at the run's final margin cfg['m'] (0.4), whatever the student's
warm-up margin was at that epoch: the trap is defined against the scoring margin.
acc(no margin) = argmax of cos; acc(margin) = cos(theta_tgt + m) > max other cos;
cos tgt = mean cos to own class weight; emb cos = mean pairwise embedding cosine.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from dataloader import machine_ids  # noqa: E402
from model.net import SCLTFSTgramMFN  # noqa: E402

NAMES = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
off, lab = 0, {}
for n in NAMES:
    for j, i in enumerate(machine_ids(n)):
        lab[(n, i)] = off + j
    off += len(machine_ids(n))

files = open('data/cache/train_files.txt').read().split()
y_all = np.array([lab[(f.split('/')[-3], 'id_' + f.split('_id_')[1][:2])] for f in files])
wav = np.load('data/cache/train_wav.npy', mmap_mode='r')
mel = np.load('data/cache/train_mel.npy', mmap_mode='r')
idx = np.sort(np.random.default_rng(0).choice(len(files), 1024, replace=False))
xw = torch.from_numpy(np.ascontiguousarray(wav[idx])).float()
xm = torch.from_numpy(np.ascontiguousarray(mel[idx])).float()
y = torch.from_numpy(y_all[idx])
dev = torch.device(os.environ.get('DEV', 'cuda:0'))


def load(target):
    """(label, ckpt dir, state_dict) for one target."""
    if os.path.isdir(target):
        d = target
        return os.path.basename(d.rstrip('/')), d, torch.load(os.path.join(d, 'model.pth'), map_location='cpu')
    path, key = (target.rsplit(':', 1) + ['ema_net'])[:2] if ':' in target else (target, 'ema_net')
    d = os.path.dirname(path)
    st = torch.load(path, map_location='cpu', weights_only=False)   # our own trainer file
    tag = os.path.basename(path).replace('state_', '').replace('.pth', '')
    return f'{os.path.basename(d)}@{tag}:{key}', d, st[key]


for target in sys.argv[1:]:
    label, d, sd = load(target)
    cfg = json.load(open(os.path.join(d, 'config.json')))
    net = SCLTFSTgramMFN(m=cfg['m'], cfg=cfg).to(dev)
    net.load_state_dict(sd)
    net.eval()
    feats = []
    with torch.no_grad():
        for b in range(0, len(y), 64):
            _, f = net(xw[b:b + 64].to(dev), xm[b:b + 64].to(dev), y[b:b + 64].to(dev), train=True)
            feats.append(f.cpu())
    f = torch.cat(feats)
    W = F.normalize(net.arcface.weight.detach().cpu())
    cos = f @ W.T
    tgt = cos.gather(1, y[:, None])[:, 0]
    oth = cos.scatter(1, y[:, None], -2).max(1).values
    acc = (cos.argmax(1) == y).float().mean()
    m = cfg['m']
    marg = (torch.cos(torch.acos(tgt.clamp(-1, 1)) + m) > oth).float().mean()
    g = f @ f.T
    same = (y[:, None] == y[None]) & ~torch.eye(len(y), dtype=bool)
    diff = y[:, None] != y[None]
    print(f'{label:44s} acc(no margin) {acc:.3f}  acc(margin {m}) {marg:.3f}  '
          f'cos tgt {tgt.mean():+.3f} max-other {oth.mean():+.3f}  '
          f'emb cos same-ID {g[same].mean():+.3f} diff-ID {g[diff].mean():+.3f}', flush=True)
