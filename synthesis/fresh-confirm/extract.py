#!/usr/bin/env python
"""Dump everything any phase-2 rule needs for fresh checkpoints, in one pass per split.

    python synthesis/fresh-confirm/extract.py --gpu 0 --split train|dev|eval \
        --out synthesis/fresh-confirm/results/emb --runs SCLTFSTgramMFN_fc_nogain_s2060 ...

Forward pass = ablation/ensemble-backend/extract.py's (imported): EMA weights, eval mode,
architecture from the committed config.json (synthesis/fresh-confirm/scores/<run>/), batch 32
in the same clip order. Per clip, under circular shifts 0, +1 %, -1 % of the clip
(ablation/tta's `small` scheme; shift 0 uses the CPU mel, the others recompute it on the GPU
from the rolled wave, as extract_tta.py does):

  ce   (n, 3)        float32   R0 per shift       (ablation/tta format; [:, 0] = plain R0)
  pre  (n, 3, 128)   float32   pre-head embedding per shift
  gmean, gmax (n, 512) float32 mean / max of mobilefacenet.conv2's map, shift 0 only
                                (ablation/pooled-knn format)

Writes <out>/<split>/<run>.npz (+ idx into meta's train or test list, shifts; train also gets
`valid` / `valid_order`, the replayed 5 % split) and <out>/meta.npz (extract.py's). Each array
is range-checked, then written, fsynced, evicted and CRC-tested from disk (smartlab1's RAM).
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
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
ABL = os.path.join(REPO, 'ablation')


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex = _imp('eb_extract', os.path.join(ABL, 'ensemble-backend', 'extract.py'))  # puts repo + ablation/ on sys.path
pk = _imp('pk_extract', os.path.join(ABL, 'pooled-knn', 'extract.py'))         # save_verified
from dataloader import Wave2Mel, train_dataset     # noqa: E402
from losses import ASDLoss                         # noqa: E402
from model.net import SCLTFSTgramMFN               # noqa: E402
from scores_path import scores_dir                 # noqa: E402

N = 160000
SHIFTS = (0.0, 0.01, -0.01)
DATA = '/home/f74134118/OS-SCL/data'
CKPT_ROOT = os.path.join(REPO, 'check_points')


def load(r, device, root=CKPT_ROOT):
    with open(os.path.join(scores_dir(r), 'config.json'), encoding='UTF-8') as f:
        cfg = json.load(f)
    net = SCLTFSTgramMFN(m=cfg['m'], cfg=cfg).to(device)
    net.load_state_dict(torch.load(os.path.join(root, r, 'model.pth'), map_location=device), strict=True)
    net.eval()
    grab = {}
    net.mobilefacenet.conv2.register_forward_hook(lambda mod, inp, out: grab.__setitem__('map', out))
    return cfg, net, grab


def check(a, r):
    for k, v in a.items():
        v32 = v.astype(np.float32)
        assert np.isfinite(v32).all(), (r, k)
        lim = 1e3 if k == 'ce' else 1e4
        assert np.abs(v32).max() < lim, (r, k, float(np.abs(v32).max()))
    assert a['ce'].min() >= 0, r


def run_pass(nets, paths, labels, idx, device, ce, bs, desc):
    gmel = Wave2Mel()
    gmel.mel_transform.to(device)
    gmel.amplitude_to_db.to(device)
    ks = [int(round(f * N)) for f in SHIFTS]
    acc = {r: {k: [] for k in ('ce', 'pre', 'gmean', 'gmax')} for r in nets}
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
                for r, (net, grab) in nets.items():
                    logits, pre = ex.forward(net, w, m, lab)
                    fm = grab.pop('map').float()
                    b[r][0].append(ce(logits, lab))
                    b[r][1].append(pre.float())
                    if k == 0:
                        assert fm.shape[1:] == (512, 8, 20), fm.shape
                        acc[r]['gmean'].append(fm.mean((2, 3)).cpu().numpy())
                        acc[r]['gmax'].append(fm.amax((2, 3)).cpu().numpy())
            for r in nets:
                acc[r]['ce'].append(torch.stack(b[r][0], 1).cpu().numpy())
                acc[r]['pre'].append(torch.stack(b[r][1], 1).cpu().numpy())
    out = {r: {k: np.concatenate(v).astype(np.float32) for k, v in a.items()} for r, a in acc.items()}
    for r, a in out.items():
        check(a, r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--split', required=True, choices=['train', 'dev', 'eval'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--group', type=int, default=4, help='checkpoints per data pass')
    ap.add_argument('--ckpt_root', default=CKPT_ROOT, help='check_points dir (reproduction test only)')
    ap.add_argument('--limit', type=int, default=0, help='first n clips only (reproduction test only)')
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}')
    od = os.path.join(args.out, args.split)
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
        sel = np.array([t[5] == args.split for t in te])
        paths, labels, idx = [t[0] for t in te], torch.as_tensor([t[1] for t in te]), np.where(sel)[0]

    if args.limit:
        idx = idx[:args.limit]
    todo = [r for r in args.runs if not os.path.exists(os.path.join(od, f'{r}.npz'))]
    ce = ASDLoss(reduction=False).to(device)
    for g in range(0, len(todo), args.group):
        grp = todo[g:g + args.group]
        loaded = {r: load(r, device, args.ckpt_root) for r in grp}
        nets = {r: (v[1], v[2]) for r, v in loaded.items()}
        out = run_pass(nets, paths, labels, idx, device, ce, args.bs, f'{args.split} {len(grp)} runs')
        for r in grp:
            extra = {}
            if args.split == 'train':
                mask, vidx = ex.replay_valid(DATA, loaded[r][0]['seed'], len(tr_paths))
                extra = dict(valid=mask, valid_order=vidx)
            pk.save_verified(os.path.join(od, f'{r}.npz'), idx=idx, shifts=np.array(SHIFTS), **out[r], **extra)
            print('wrote', args.split, r, {k: v.shape for k, v in out[r].items()}, flush=True)
        del nets, loaded, out
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
