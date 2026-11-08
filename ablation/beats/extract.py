#!/usr/bin/env python
"""Dump what the scorers need for BEATsASD runs, in ../ensemble-backend/extract.py's format.

    python ablation/beats/extract.py --gpu 0 --out ablation/beats/results/emb --runs BEATsASD_bt_arc_s2040 ...

Per run, for all 36,283 training clips and every dev and eval test clip (meta.npz order):
  ce     R0: CE of the margin ArcFace logits vs the clip's own ID (eval.py's score)
  pre    the 128-d embedding before ArcFace (the kNN back-end's K; AnoPatch's x_i)
  gmean  the mean of the 496 final BEATs tokens, 768-d (pooled-knn's Gmean analogue, C1b)
plus `valid` / `valid_order` (the replayed 5% split, checked against the run's config.json)
and `seed`. Keys: tr_ce tr_pre tr_gmean valid valid_order te_ce te_pre te_gmean seed -- a
superset of extract.py's, so ../ensemble-backend/score.py's per_run_scores reads it unchanged.

Eval mode, final weights, batch 32; --amp none (fp32) by default. Training clips come from
data/cache (range-checked, re-decoded if bad), test clips from the wav files. Every array is
range-checked and every file written with pooled-knn's save_verified (fsync, drop from the
page cache, CRC re-read). Resumable: an existing <run>.npz is skipped.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex = _load('eb_extract', os.path.join(HERE, '..', 'ensemble-backend', 'extract.py'))   # puts ROOT on sys.path
pk = _load('pk_extract', os.path.join(HERE, '..', 'pooled-knn', 'extract.py'))
tb = _load('bt_train', os.path.join(HERE, 'train_beats.py'))
from dataloader import load_wav, train_dataset     # noqa: E402
from model.beats_asd import build                  # noqa: E402

N = 160000


class TestClips(Dataset):
    def __init__(self, paths, labels):
        self.paths, self.labels = paths, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        y, _ = load_wav(self.paths[i])
        y = y.reshape(-1)
        assert y.shape[0] == N and torch.isfinite(y).all() and y.abs().max() <= 1.0, self.paths[i]
        return y, self.labels[i]


def check(a, r):
    for k, v in a.items():
        v = v.astype(np.float32)
        assert np.isfinite(v).all(), (r, k)
        assert np.abs(v).max() < (1e3 if k == 'ce' else 1e4), (r, k, float(np.abs(v).max()))
    assert a['ce'].min() >= 0, r


def run_pass(nets, loader, device, amp, desc):
    acc = {r: {k: [] for k in ('ce', 'pre', 'gmean')} for r in nets}
    with torch.no_grad():
        for w, lab in tqdm(loader, desc=desc, dynamic_ncols=True, mininterval=60):
            w, lab = w.to(device, non_blocking=True), lab.to(device)
            for r, net in nets.items():
                with torch.autocast('cuda', dtype=amp, enabled=amp is not None):
                    logits, pre, gm = net(w, lab)
                acc[r]['ce'].append(torch.nn.functional.cross_entropy(logits.float(), lab, reduction='none').cpu().numpy())
                acc[r]['pre'].append(pre.float().cpu().numpy())
                acc[r]['gmean'].append(gm.float().cpu().numpy())
    out = {r: {k: np.concatenate(v) for k, v in a.items()} for r, a in acc.items()}
    for r, a in out.items():
        check(a, r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--ckpt_root', default=os.path.join(ROOT, 'check_points'))
    ap.add_argument('--cache_dir', default=os.path.join(ROOT, 'data', 'cache'))
    ap.add_argument('--amp', choices=['none', 'fp16', 'bf16'], default='none')
    ap.add_argument('--bs', type=int, default=32)
    args = ap.parse_args()
    data = os.path.join(ROOT, 'data')
    device = torch.device(f'cuda:{args.gpu}')
    amp = {'none': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.amp]
    os.makedirs(args.out, exist_ok=True)
    todo = [r for r in args.runs if not os.path.exists(os.path.join(args.out, f'{r}.npz'))]
    if not todo:
        print('all runs extracted')
        return

    ref = train_dataset(os.path.join(data, 'dataset'), ex.NAME_LIST, {})
    tr_paths, tr_labels = list(ref.unrolled_files_list), ref.labels.numpy()
    te = ex.test_table(data)
    mp = os.path.join(args.out, 'meta.npz')
    if not os.path.exists(mp):
        pk.save_verified(mp, tr_paths=np.array(tr_paths), tr_label=tr_labels,
                         te_paths=np.array([t[0] for t in te]), te_label=np.array([t[1] for t in te]),
                         te_y=np.array([t[2] for t in te]), te_machine=np.array([t[3] for t in te]),
                         te_id=np.array([t[4] for t in te]), te_dataset=np.array([t[5] for t in te]))

    nets, cfgs, splits = {}, {}, {}
    for r in todo:
        with open(os.path.join(args.ckpt_root, r, 'config.json'), encoding='UTF-8') as f:
            cfgs[r] = json.load(f)
        mask, vidx = ex.replay_valid(data, cfgs[r]['seed'], len(tr_paths))
        assert int(vidx.sum()) == cfgs[r]['valid_idx_sum'] and len(vidx) == cfgs[r]['n_valid'], r
        splits[r] = (mask, vidx)
        net, _ = build(cfgs[r], pretrained=False)
        net.load_state_dict(torch.load(os.path.join(args.ckpt_root, r, 'model.pth'), map_location='cpu',
                                       weights_only=True), strict=True)
        nets[r] = net.to(device).eval()

    trl = DataLoader(tb.Waves(tr_paths, torch.as_tensor(tr_labels), args.cache_dir), batch_size=args.bs,
                     num_workers=8, pin_memory=True)
    tel = DataLoader(TestClips([t[0] for t in te], torch.as_tensor([t[1] for t in te])), batch_size=args.bs,
                     num_workers=8, pin_memory=True)
    tr = run_pass(nets, trl, device, amp, f'train {len(nets)} runs')
    te_out = run_pass(nets, tel, device, amp, f'test {len(nets)} runs')
    for r in todo:
        mask, vidx = splits[r]
        pk.save_verified(os.path.join(args.out, f'{r}.npz'),
                         tr_ce=tr[r]['ce'], tr_pre=tr[r]['pre'], tr_gmean=tr[r]['gmean'], valid=mask, valid_order=vidx,
                         te_ce=te_out[r]['ce'], te_pre=te_out[r]['pre'], te_gmean=te_out[r]['gmean'],
                         seed=cfgs[r]['seed'])
        print('wrote', r, flush=True)


if __name__ == '__main__':
    main()
