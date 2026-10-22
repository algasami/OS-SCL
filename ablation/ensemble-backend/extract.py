#!/usr/bin/env python
"""Dump every per-clip quantity score.py needs, once, for every checkpoint.

    python ablation/ensemble-backend/extract.py --gpu 0 --repo <checkout with data/, check_points/> \
        --out ablation/ensemble-backend/results/emb --runs SCLTFSTgramMFN_tfl_relu_max_s2024 ...

Per checkpoint, for all 36,283 training clips and every dev and eval test clip:
  ce   eval.py's anomaly score R0: CE of the margin ArcFace logits vs the clip's own ID
  pre  the pre-head MobileFaceNet embedding (128-d), input to the kNN back-end
plus, for training clips, `valid`: membership of the 5% held-out split, replayed
from the run's seed in train.py's order (random_seed -> train_dataset ->
dataset_split; random_split is the first torch RNG draw).

The forward pass is eval.py's (EMA weights, eval mode, architecture flags from
the committed config.json) but batched: in eval mode BatchNorm uses running
stats, so a clip's output does not depend on its batch. TFgram's reshape needs
train=True for batch > 1 and False for batch 1, as in eval.py/trainer.valid().

Every clip is decoded once and pushed through all checkpoints. Resumable: a
run whose <run>.npz exists is skipped.
"""
import argparse
import glob
import json
import os
import random
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..'))
from dataloader import Wave2Mel, train_dataset     # noqa: E402
from losses import ASDLoss                         # noqa: E402
from model.net import SCLTFSTgramMFN  # noqa: E402
from scores_path import scores_dir                 # noqa: E402
from utils import dataset_split                    # noqa: E402

NAME_LIST = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
TEST_ROOTS = {'dev': 'dataset', 'eval': 'eval_dataset'}


def label_list(name):
    if name == 'ToyConveyor':
        return [f'id_0{i}' for i in range(1, 7)]
    if name == 'ToyCar':
        return [f'id_0{i}' for i in range(1, 8)]
    return [f'id_0{i}' for i in range(0, 7)]


OFFSET = {m: sum(len(label_list(n)) for n in NAME_LIST[:i]) for i, m in enumerate(NAME_LIST)}


def random_seed(seed):  # same as train.py
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def replay_valid(data, seed, n):
    random_seed(seed)
    ds = train_dataset(os.path.join(data, 'dataset'), NAME_LIST, {})
    _, valid = dataset_split(ds)
    mask = np.zeros(n, dtype=bool)
    mask[np.array(valid.indices)] = True
    return mask, np.array(valid.indices)


class Clips(Dataset):
    def __init__(self, paths, labels):
        self.paths, self.labels, self.mel = paths, labels, Wave2Mel()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        wav, sr = torchaudio.load(self.paths[i])
        wav = wav[..., :sr * 10]
        return wav, self.mel(wav), self.labels[i]


def test_table(data):
    rows = []
    for ds, root in TEST_ROOTS.items():
        for m in NAME_LIST:
            for p in sorted(glob.glob(os.path.join(data, root, m, 'test', '*.wav'))):
                b = os.path.basename(p)
                idl = re.search(r'id_\d\d', b).group()
                rows.append((p, OFFSET[m] + label_list(m).index(idl),
                             int(b.startswith('anomaly')), m, idl, ds))
    return rows


def forward(net, wav, mel, label):
    # the model's own branch stacking, so new input channels (--mel_gate) score correctly
    x = net.branch_input(wav, mel, train=wav.shape[0] > 1)
    _, pre = net.mobilefacenet(x)
    logits = net.arcface(F.normalize(net.head(pre), dim=1), label)
    return logits, pre


def run_pass(nets, paths, labels, device, ce, bs, desc):
    acc = {r: {'ce': [], 'pre': []} for r in nets}
    loader = DataLoader(Clips(paths, labels), batch_size=bs, num_workers=8, pin_memory=True)
    with torch.no_grad():
        for wav, mel, lab in tqdm(loader, desc=desc, dynamic_ncols=True, mininterval=30):
            wav, mel, lab = wav.to(device), mel.to(device), lab.to(device)
            for r, net in nets.items():
                logits, pre = forward(net, wav, mel, lab)
                acc[r]['ce'].append(ce(logits, lab).cpu().numpy())
                acc[r]['pre'].append(pre.cpu().numpy())
    return {r: {k: np.concatenate(v) for k, v in a.items()} for r, a in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--bs', type=int, default=32)
    args = ap.parse_args()
    data = os.path.join(args.repo, 'data')
    device = torch.device(f'cuda:{args.gpu}')
    os.makedirs(args.out, exist_ok=True)

    todo = [r for r in args.runs if not os.path.exists(os.path.join(args.out, f'{r}.npz'))]
    if not todo:
        print('all runs extracted')
        return

    # the training set in train_dataset's own order -- the order random_split indexes
    ref = train_dataset(os.path.join(data, 'dataset'), NAME_LIST, {})
    tr_paths, tr_labels = list(ref.unrolled_files_list), ref.labels.numpy()
    te = test_table(data)
    np.savez(os.path.join(args.out, 'meta.npz'),
             tr_paths=np.array(tr_paths), tr_label=tr_labels,
             te_paths=np.array([t[0] for t in te]), te_label=np.array([t[1] for t in te]),
             te_y=np.array([t[2] for t in te]), te_machine=np.array([t[3] for t in te]),
             te_id=np.array([t[4] for t in te]), te_dataset=np.array([t[5] for t in te]))

    cfgs, nets = {}, {}
    for r in todo:
        with open(os.path.join(scores_dir(r), 'config.json'), encoding='UTF-8') as f:
            cfgs[r] = json.load(f)
    splits = {r: replay_valid(data, cfgs[r]['seed'], len(tr_paths)) for r in todo}
    for r in todo:
        net = SCLTFSTgramMFN(m=cfgs[r]['m'], cfg=cfgs[r]).to(device)
        state = torch.load(os.path.join(args.repo, 'check_points', r, 'model.pth'), map_location=device)
        net.load_state_dict(state, strict=True)
        nets[r] = net.eval()
    ce = ASDLoss(reduction=False).to(device)

    tr = run_pass(nets, tr_paths, torch.as_tensor(tr_labels), device, ce, args.bs, 'train')
    te_out = run_pass(nets, [t[0] for t in te], torch.as_tensor([t[1] for t in te]), device, ce,
                      args.bs, 'test')
    for r in todo:
        mask, vidx = splits[r]
        np.savez(os.path.join(args.out, f'{r}.npz'),
                 tr_ce=tr[r]['ce'], tr_pre=tr[r]['pre'], valid=mask, valid_order=vidx,
                 te_ce=te_out[r]['ce'], te_pre=te_out[r]['pre'], seed=cfgs[r]['seed'])
        print('wrote', r)


if __name__ == '__main__':
    main()
