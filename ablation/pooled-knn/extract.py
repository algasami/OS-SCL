#!/usr/bin/env python
"""Dump pooled pre-linear7 features, R0 and the pre-head embedding, per clip.

    python ablation/pooled-knn/extract.py --gpu 1 --split train|dev|eval --out ablation/pooled-knn/results/emb \
        --runs SCLTFSTgramMFN_ao_nogain_s2040 ... [--machine ToyCar --rolls 1,2,3,4,5,6,7]

Forward pass = ../ensemble-backend/extract.py's (imported, not copied): EMA weights, eval
mode, architecture from the committed config.json, batch 32 in the same clip order. A
forward hook on mobilefacenet.conv2 (the input to linear7, 512 x 8 x 20 = channels x
frequency x time) keeps only pooled views of the map:

  bmean, bmax  (n, 8, 512) fp16   per frequency band, mean / max over the 20 time steps
  gmean, gmax  (n, 512)    fp32   mean / max over all 8 x 20 positions
  ce (n,), pre (n, 128)    fp32   R0 and the pre-head embedding, as extract.py

--rolls k1,k2,..: the diagnostic. Each clip is also circularly rolled by k/8 of the clip on
the GPU and its mel recomputed from the rolled wave (../tta's extract_tta.py); arrays gain
a roll axis after n. Roll 0 is not repeated: it is the plain dump.

Writes <out>/<split>[_<machine>_rolls]/<run>.npz (+ idx into meta's train or test list;
train also gets `valid` / `valid_order`, the replayed 5% split). <out>/meta.npz is
extract.py's. Every array is range-checked before it is written (smartlab1's RAM flips
bits). Resumable: an existing <run>.npz is skipped.
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
DATA = '/home/f74134118/OS-SCL/data'
CKPT_ROOT = '/home/f74134118/OS-SCL-augops/check_points'
KEYS = ('ce', 'pre', 'gmean', 'gmax', 'bmean', 'bmax')


def load(r, device):
    with open(os.path.join(scores_dir(r), 'config.json'), encoding='UTF-8') as f:
        cfg = json.load(f)
    net = SCLTFSTgramMFN(m=cfg['m'], cfg=cfg).to(device)
    net.load_state_dict(torch.load(os.path.join(CKPT_ROOT, r, 'model.pth'), map_location=device), strict=True)
    net.eval()
    grab = {}
    net.mobilefacenet.conv2.register_forward_hook(lambda mod, inp, out: grab.__setitem__('map', out))
    return cfg, net, grab


def pooled(net, grab, w, m, lab, ce):
    logits, pre = ex.forward(net, w, m, lab)
    fm = grab.pop('map').float()
    assert fm.shape[1:] == (512, 8, 20), fm.shape
    return dict(ce=ce(logits, lab), pre=pre.float(),
                gmean=fm.mean((2, 3)), gmax=fm.amax((2, 3)),
                bmean=fm.mean(3).transpose(1, 2).half(), bmax=fm.amax(3).transpose(1, 2).half())


def check(a, r):
    """Range check: never keep a value we cannot trust."""
    for k, v in a.items():
        v32 = v.astype(np.float32)
        assert np.isfinite(v32).all(), (r, k)
        lim = 1e3 if k == 'ce' else 1e4
        assert np.abs(v32).max() < lim, (r, k, float(np.abs(v32).max()))
    assert a['ce'].min() >= 0, r


def run_pass(nets, paths, labels, idx, device, ce, rolls, bs, desc):
    gmel = Wave2Mel()
    gmel.mel_transform.to(device)
    gmel.amplitude_to_db.to(device)
    ks = [int(round(k / 8 * N)) for k in rolls] if rolls else [0]
    acc = {r: {k: [] for k in KEYS} for r in nets}
    loader = DataLoader(ex.Clips([paths[i] for i in idx], labels[idx]), batch_size=bs, num_workers=8,
                        pin_memory=True)
    with torch.no_grad():
        for wav, mel, lab in tqdm(loader, desc=desc, dynamic_ncols=True, mininterval=60):
            wav, mel, lab = wav.to(device), mel.to(device), lab.to(device)
            assert wav.shape[-1] == N, wav.shape
            per = {r: {k: [] for k in KEYS} for r in nets}
            for k in ks:
                if k == 0:
                    w, m = wav, mel
                else:
                    w = torch.roll(wav, k, dims=-1)
                    m = gmel(w)
                for r, (net, grab) in nets.items():
                    for key, v in pooled(net, grab, w, m, lab, ce).items():
                        per[r][key].append(v)
            for r in nets:
                for key in KEYS:
                    v = torch.stack(per[r][key], 1) if rolls else per[r][key][0]
                    acc[r][key].append(v.cpu().numpy())
    out = {r: {k: np.concatenate(v) for k, v in a.items()} for r, a in acc.items()}
    for r, a in out.items():
        check(a, r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--split', required=True, choices=['train', 'dev', 'eval'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--machine', default=None, help='restrict test clips to one machine type')
    ap.add_argument('--rolls', default=None, help='comma-separated k: roll by k/8 of the clip')
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--group', type=int, default=8, help='checkpoints per data pass')
    args = ap.parse_args()
    rolls = [int(k) for k in args.rolls.split(',')] if args.rolls else None
    assert not rolls or (args.split != 'train' and 0 not in rolls)
    device = torch.device(f'cuda:{args.gpu}')
    sub = args.split + (f'_{args.machine}_rolls' if rolls else '')
    od = os.path.join(args.out, sub)
    os.makedirs(od, exist_ok=True)

    ref = train_dataset(os.path.join(DATA, 'dataset'), ex.NAME_LIST, {})
    tr_paths, tr_labels = list(ref.unrolled_files_list), ref.labels.numpy()
    te = ex.test_table(DATA)
    mp = os.path.join(args.out, 'meta.npz')
    if not os.path.exists(mp):
        np.savez(mp, tr_paths=np.array(tr_paths), tr_label=tr_labels,
                 te_paths=np.array([t[0] for t in te]), te_label=np.array([t[1] for t in te]),
                 te_y=np.array([t[2] for t in te]), te_machine=np.array([t[3] for t in te]),
                 te_id=np.array([t[4] for t in te]), te_dataset=np.array([t[5] for t in te]))
    if args.split == 'train':
        paths, labels, idx = tr_paths, torch.as_tensor(tr_labels), np.arange(len(tr_paths))
    else:
        sel = np.array([t[5] == args.split and (args.machine is None or t[3] == args.machine) for t in te])
        paths, labels, idx = [t[0] for t in te], torch.as_tensor([t[1] for t in te]), np.where(sel)[0]

    todo = [r for r in args.runs if not os.path.exists(os.path.join(od, f'{r}.npz'))]
    ce = ASDLoss(reduction=False).to(device)
    for g in range(0, len(todo), args.group):
        grp = todo[g:g + args.group]
        loaded = {r: load(r, device) for r in grp}
        nets = {r: (v[1], v[2]) for r, v in loaded.items()}
        out = run_pass(nets, paths, labels, idx, device, ce, rolls, args.bs, f'{sub} {len(grp)} runs')
        for r in grp:
            extra = {}
            if args.split == 'train':
                mask, vidx = ex.replay_valid(DATA, loaded[r][0]['seed'], len(tr_paths))
                extra = dict(valid=mask, valid_order=vidx)
            np.savez(os.path.join(od, f'{r}.npz'), idx=idx, rolls=np.array(rolls or [0]), **out[r], **extra)
            print('wrote', sub, r, {k: v.shape for k, v in out[r].items()}, flush=True)
        del nets, loaded, out
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
