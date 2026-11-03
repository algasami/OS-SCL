#!/usr/bin/env python
"""Dump R0 and the pre-head embedding of every clip under K circular time shifts.

    python ablation/tta/extract_tta.py --gpu 1 --split train|dev|eval|valid \
        --shifts 0,0.03,-0.03 --out ablation/tta/results/emb --runs SCLTFSTgramMFN_ao_nogain_s2040 ...

Same forward pass as ../ensemble-backend/extract.py (imported, not copied): EMA weights,
eval mode, architecture from the committed config.json. Each clip is decoded once on the
CPU (with its CPU Log-Mel), moved to the GPU once, and rolled there by every shift.
Shift 0 uses the CPU mel, so it reproduces extract.py; other shifts recompute the mel on
the GPU from the rolled wave (as trainer.augment does; ~1e-5 dB from the CPU mel).

A shift f is a fraction of the 160,000-sample clip: k = round(f * N) samples, torch.roll
(circular, so the clip's end is spliced onto its start, as in training's shift op).

Splits: train (all 36,283 training clips, the S1′ bank), dev / eval (test clips),
valid (the pilot: each run's own replayed 5% held-out split only).

Writes <out>/<split>/<run>.npz: ce (n, K) float32, pre (n, K, 128) float32, idx (n,)
into meta's train or test list, shifts (K,) fractions. <out>/meta.npz is extract.py's.
Resumable: an existing <run>.npz is skipped.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('eb_extract', os.path.join(HERE, '..', 'ensemble-backend', 'extract.py'))
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)          # also puts the repo root and ablation/ on sys.path
from dataloader import Wave2Mel, train_dataset     # noqa: E402
from losses import ASDLoss                         # noqa: E402
from model.net import SCLTFSTgramMFN               # noqa: E402
from scores_path import scores_dir                 # noqa: E402

N = 160000
CKPT_ROOTS = ['/home/f74134118/OS-SCL-augops/check_points', '/home/f74134118/OS-SCL-litarms/check_points']


def ckpt(run):
    for r in CKPT_ROOTS:
        p = os.path.join(r, run, 'model.pth')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(run)


def run_pass(nets, paths, labels, idx, device, ce, shifts, bs, desc):
    gmel = Wave2Mel()
    gmel.mel_transform.to(device)
    gmel.amplitude_to_db.to(device)
    ks = [int(round(f * N)) for f in shifts]
    acc = {r: {'ce': [], 'pre': []} for r in nets}
    loader = DataLoader(ex.Clips([paths[i] for i in idx], labels[idx]), batch_size=bs, num_workers=8,
                        pin_memory=True)
    with torch.no_grad():
        for wav, mel, lab in tqdm(loader, desc=desc, dynamic_ncols=True, mininterval=60):
            wav, mel, lab = wav.to(device), mel.to(device), lab.to(device)
            assert wav.shape[-1] == N, wav.shape
            b = {r: ([], []) for r in nets}
            for k in ks:
                if k == 0:
                    w, m = wav, mel
                else:
                    w = torch.roll(wav, k, dims=-1)
                    m = gmel(w)
                for r, net in nets.items():
                    logits, pre = ex.forward(net, w, m, lab)
                    b[r][0].append(ce(logits, lab))
                    b[r][1].append(pre)
            for r in nets:
                acc[r]['ce'].append(torch.stack(b[r][0], 1).cpu().numpy())
                acc[r]['pre'].append(torch.stack(b[r][1], 1).cpu().numpy())
    out = {r: {k: np.concatenate(v) for k, v in a.items()} for r, a in acc.items()}
    for r, a in out.items():   # smartlab1's RAM flips bits: never keep a value we cannot trust
        assert np.isfinite(a['ce']).all() and np.isfinite(a['pre']).all(), r
        assert a['ce'].min() >= 0 and a['ce'].max() < 1e3 and np.abs(a['pre']).max() < 1e4, r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--split', required=True, choices=['train', 'dev', 'eval', 'valid'])
    ap.add_argument('--shifts', required=True, help='comma-separated fractions of the clip')
    ap.add_argument('--out', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--group', type=int, default=4, help='checkpoints per data pass (GPU memory)')
    args = ap.parse_args()
    shifts = [float(s) for s in args.shifts.split(',')]
    data = '/home/f74134118/OS-SCL/data'
    device = torch.device(f'cuda:{args.gpu}')
    od = os.path.join(args.out, args.split)
    os.makedirs(od, exist_ok=True)

    ref = train_dataset(os.path.join(data, 'dataset'), ex.NAME_LIST, {})
    tr_paths, tr_labels = list(ref.unrolled_files_list), ref.labels.numpy()
    te = ex.test_table(data)
    mp = os.path.join(args.out, 'meta.npz')
    if not os.path.exists(mp):
        np.savez(mp, tr_paths=np.array(tr_paths), tr_label=tr_labels,
                 te_paths=np.array([t[0] for t in te]), te_label=np.array([t[1] for t in te]),
                 te_y=np.array([t[2] for t in te]), te_machine=np.array([t[3] for t in te]),
                 te_id=np.array([t[4] for t in te]), te_dataset=np.array([t[5] for t in te]))
    te_paths, te_labels = [t[0] for t in te], np.array([t[1] for t in te])
    te_ds = np.array([t[5] for t in te])

    todo = [r for r in args.runs if not os.path.exists(os.path.join(od, f'{r}.npz'))]
    ce = ASDLoss(reduction=False).to(device)

    def load(r):
        with open(os.path.join(scores_dir(r), 'config.json'), encoding='UTF-8') as f:
            cfg = json.load(f)
        net = SCLTFSTgramMFN(m=cfg['m'], cfg=cfg).to(device)
        net.load_state_dict(torch.load(ckpt(r), map_location=device), strict=True)
        return cfg, net.eval()

    def save(r, a, idx):
        np.savez(os.path.join(od, f'{r}.npz'), ce=a['ce'].astype(np.float32), pre=a['pre'].astype(np.float32),
                 idx=idx, shifts=np.array(shifts))
        print('wrote', args.split, r, a['ce'].shape, flush=True)

    if args.split == 'valid':          # each run's own held-out clips, one run at a time
        for r in todo:
            cfg, net = load(r)
            _, vidx = ex.replay_valid(data, cfg['seed'], len(tr_paths))
            vidx = np.sort(vidx)
            out = run_pass({r: net}, tr_paths, torch.as_tensor(tr_labels), vidx, device, ce, shifts,
                           args.bs, f'valid {r[-5:]}')
            save(r, out[r], vidx)
        return

    if args.split == 'train':
        paths, labels, idx = tr_paths, torch.as_tensor(tr_labels), np.arange(len(tr_paths))
    else:
        paths, labels, idx = te_paths, torch.as_tensor(te_labels), np.where(te_ds == args.split)[0]
    for g in range(0, len(todo), args.group):
        grp = todo[g:g + args.group]
        nets = {r: load(r)[1] for r in grp}
        out = run_pass(nets, paths, labels, idx, device, ce, shifts, args.bs, f'{args.split} {len(grp)} runs')
        for r in grp:
            save(r, out[r], idx)
        del nets
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
