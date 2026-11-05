"""How much does the trained head use the spectrum embedding? (diagnostic, README P7)

    python ablation/spectrum-branch/head_share.py --gpu 0 SCLTFSTgramMFN_sb_spec_s2040 ...

For each run's EMA net (check_points/<run>/model.pth), on the first 512 dev-test
clips of data/cache: z_mel = W[:, :128] e_mfn and z_spec = W[:, 128:] e_spec, with W
the head's first Linear (64 x 256). Reports, averaged over the 64 units,
var(z_spec) / (var(z_mel) + var(z_spec)), plus the two embeddings' mean norms.
Appends to scores/head_share.csv (one row per run; reruns replace the row).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, ROOT)
from dataloader import cached_item_ok  # noqa: E402
from model.net import SCLTFSTgramMFN  # noqa: E402

OUT = os.path.join(HERE, 'scores', 'head_share.csv')


@torch.no_grad()
def share(run, dev, n=512):
    ck = os.path.join(ROOT, 'check_points', run)
    cfg = json.load(open(os.path.join(ck, 'config.json')))
    net = SCLTFSTgramMFN(m=cfg['m'], cfg=cfg).to(dev)
    net.load_state_dict(torch.load(os.path.join(ck, 'model.pth'), map_location=dev), strict=True)
    net.eval()
    wav = np.load(os.path.join(ROOT, 'data/cache/devtest_wav.npy'), mmap_mode='r')
    mel = np.load(os.path.join(ROOT, 'data/cache/devtest_mel.npy'), mmap_mode='r')
    idx = np.linspace(0, len(wav) - 1, n).astype(int)        # spread over all machines
    embs = []
    for chunk in np.array_split(idx, n // 64):
        w, m = np.array(wav[chunk]), np.array(mel[chunk])
        assert all(cached_item_ok(a, b) for a, b in zip(w, m)), 'host-RAM bit flip in cache; rerun'
        embs.append(net.embed(torch.from_numpy(w).to(dev), torch.from_numpy(m).to(dev), train=True).float())
    e = torch.cat(embs)
    W = net.head.head[0].weight.float()
    z_mel, z_spec = e[:, :128] @ W[:, :128].T, e[:, 128:] @ W[:, 128:].T
    v_mel, v_spec = z_mel.var(0), z_spec.var(0)
    return {'run': run, 'spec_var_share': f'{(v_spec / (v_mel + v_spec)).mean().item():.4f}',
            'norm_mfn': f'{e[:, :128].norm(dim=1).mean().item():.3f}',
            'norm_spec': f'{e[:, 128:].norm(dim=1).mean().item():.3f}', 'clips': n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(f'cuda:{a.gpu}')
    rows = {}
    if os.path.exists(OUT):
        rows = {r['run']: r for r in csv.DictReader(open(OUT))}
    for run in a.runs:
        rows[run] = share(run, dev)
        print(rows[run])
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'spec_var_share', 'norm_mfn', 'norm_spec', 'clips'])
        w.writeheader()
        w.writerows(rows[k] for k in sorted(rows))


if __name__ == '__main__':
    main()
