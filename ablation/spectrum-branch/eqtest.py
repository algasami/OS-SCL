"""Flag-off equality: a few real training steps through Trainer.train in two repos.

    python ablation/spectrum-branch/eqtest.py <repo> <out.pt> [--device cuda:1] [--spec 1]

Run once with <repo> = this worktree and once with main's checkout, then
`eqtest.py --compare a.pt b.pt`. Each run seeds as train.py does, builds net +
EMA, and trains 1 epoch of 3 batches from data/cache (wave_aug noise,shift,
mixup, SupCon; fp16 autocast on CUDA), then validates once. It saves the
student's and the EMA's state_dicts. With --spec_branch off, every tensor must
be torch.equal across the two repos.
"""
import argparse
import os
import shutil
import sys

import numpy as np
import torch


def run(repo, out, device, spec, batch, steps):
    os.chdir(repo)
    sys.path.insert(0, repo)
    from torch.utils.data import DataLoader, TensorDataset
    from model.net import SCLTFSTgramMFN
    from train import random_seed
    from trainer import Trainer

    cuda = device.startswith('cuda')
    desc = f'eqtest_tmp_{os.getpid()}'
    cfg = dict(m=0.4, gpu_num=int(device.split(':')[1]) if cuda else 0, epoch=1, batch_size=batch,
               fussion=1, ht='basic', desc=desc, seed=2040, lean=1, amp='fp16' if cuda else 'none',
               wave_aug=1, wave_aug_ops='noise,shift')
    if spec:
        cfg['spec_branch'] = 1
    random_seed(cfg['seed'])
    wav = np.load('data/cache/train_wav.npy', mmap_mode='r')
    mel = np.load('data/cache/train_mel.npy', mmap_mode='r')
    n = batch * (steps + 1)
    w, m = torch.from_numpy(np.array(wav[:n])), torch.from_numpy(np.array(mel[:n]))
    assert torch.isfinite(m).all() and m.abs().max() <= 200 and w.abs().max() <= 1   # host-RAM check
    y = torch.arange(n) % 41
    tr = DataLoader(TensorDataset(w[:-batch], m[:-batch], y[:-batch]), batch_size=batch)
    va = DataLoader(TensorDataset(w[-batch:], m[-batch:], y[-batch:]), batch_size=batch)

    dev = torch.device(device)
    net = SCLTFSTgramMFN(num_classes=41, m=cfg['m'], cfg=cfg).to(dev)
    ema = SCLTFSTgramMFN(num_classes=41, m=cfg['m'], cfg=cfg).to(dev)
    for p in ema.parameters():
        p.detach_()
    trainer = Trainer(device=dev, net=net, ema_net=ema, epochs=1, cfg=cfg)
    trainer.train(tr, va)
    vl, _ = trainer.valid(va)
    torch.save({'net': {k: v.cpu() for k, v in net.state_dict().items()},
                'ema': {k: v.cpu() for k, v in ema.state_dict().items()}, 'valid_loss': vl}, out)
    shutil.rmtree(f'check_points/SCLTFSTgramMFN_{desc}', ignore_errors=True)
    print(f'{repo}: valid loss {vl:.6f}, {len(net.state_dict())} tensors -> {out}')


def compare(a, b):
    A, B = torch.load(a), torch.load(b)
    bad = 0
    for part in ('net', 'ema'):
        assert A[part].keys() == B[part].keys(), f'{part}: key sets differ'
        for k in A[part]:
            if not torch.equal(A[part][k], B[part][k]):
                bad += 1
                print(f'DIFF {part}.{k}: max |d| {(A[part][k].float() - B[part][k].float()).abs().max():.3e}')
    n = len(A['net']) + len(A['ema'])
    print(f'{n - bad}/{n} tensors torch.equal; valid loss {A["valid_loss"]:.6f} vs {B["valid_loss"]:.6f}')
    return bad == 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('repo', nargs='?')
    ap.add_argument('out', nargs='?')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--spec', type=int, default=0)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--compare', nargs=2)
    a = ap.parse_args()
    if a.compare:
        sys.exit(0 if compare(*a.compare) else 1)
    run(os.path.abspath(a.repo), os.path.abspath(a.out), a.device, a.spec, a.batch, a.steps)
