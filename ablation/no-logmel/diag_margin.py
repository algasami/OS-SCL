"""Is a stalled run separating classes but failing the ArcFace margin, or collapsed?
usage: python ablation/no-logmel/diag_margin.py <ckpt_dir> [...]   (run from the repo root)"""
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
dev = torch.device(os.environ.get('DEV', 'cuda:2'))

for d in sys.argv[1:]:
    cfg = json.load(open(os.path.join(d, 'config.json')))
    net = SCLTFSTgramMFN(m=cfg['m'], cfg=cfg).to(dev)
    net.load_state_dict(torch.load(os.path.join(d, 'model.pth'), map_location=dev))
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
    print(f'{os.path.basename(d):32s} acc(no margin) {acc:.3f}  acc(margin {m}) {marg:.3f}  '
          f'cos tgt {tgt.mean():+.3f} max-other {oth.mean():+.3f}  '
          f'emb cos same-ID {g[same].mean():+.3f} diff-ID {g[diff].mean():+.3f}')
