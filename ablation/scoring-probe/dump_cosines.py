#!/usr/bin/env python
"""Dump the 41 ArcFace cosines of every test clip, once per trained run.

    python ablation/scoring-probe/dump_cosines.py --gpu 0 --data_root <repo>/data \
        --ckpt_root <repo>/check_points --out ablation/scoring-probe/results/cosines \
        --datasets dev eval --runs SCLTFSTgramMFN_tfl_relu_max_s2024 ...

Every scoring rule in score_rules.py is a function of these cosines (plus the
clip's own class index), so the GPU pass runs once per run and dataset and the
rules are compared offline in float64.

The forward pass is eval.py's: batch size 1, `train=False`, EMA weights, the
architecture flags taken from the run's committed config.json. Alongside the
cosines it stores eval.py's own score (float32 CE of the margin logits), so
R0 can be checked against both that and the committed eval CSVs.

Each test set is loaded (and its Log-Mel computed) once per machine and pushed
through every run, rather than once per run as eval.py does. Resumable: a
(run, dataset) whose .npz exists is skipped.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..'))
from dataloader import test_dataset           # noqa: E402
from losses import ASDLoss                    # noqa: E402
from model.net import SCLTFSTgramMFN          # noqa: E402
from scores_path import scores_dir            # noqa: E402

NAME_LIST = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
DATASETS = {'dev': 'dataset', 'eval': 'eval_dataset'}


def load_run(name, ckpt_root, device):
    with open(os.path.join(scores_dir(name), 'config.json'), encoding='UTF-8') as f:
        cfg = json.load(f)
    net = SCLTFSTgramMFN(m=cfg['m'], cfg=cfg).to(device)
    state = torch.load(os.path.join(ckpt_root, name, 'model.pth'), map_location=device)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--ckpt_root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--datasets', nargs='+', default=['dev', 'eval'], choices=list(DATASETS))
    ap.add_argument('--runs', nargs='+', required=True)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    os.makedirs(args.out, exist_ok=True)
    out_path = lambda run, ds: os.path.join(args.out, f'{run}_{ds}.npz')
    criterion = ASDLoss(reduction=False).to(device)

    for ds in args.datasets:
        todo = [r for r in args.runs if not os.path.exists(out_path(r, ds))]
        if not todo:
            print(f'[{ds}] all runs dumped, skipping')
            continue
        nets = {r: load_run(r, args.ckpt_root, device) for r in todo}
        acc = {r: {k: [] for k in ('cos', 'label', 'y', 'machine', 'id', 'ce_eval')} for r in todo}

        for machine in NAME_LIST:
            # cfg is only stored by test_dataset, never read
            test_ds = test_dataset(os.path.join(args.data_root, DATASETS[ds]), machine, NAME_LIST, {})
            loader = DataLoader(test_ds, batch_size=1, pin_memory=True, num_workers=4)
            with torch.no_grad():
                for x_wav, x_mel, label, y, id_label in tqdm(loader, desc=f'{ds} {machine}',
                                                             leave=False, dynamic_ncols=True):
                    x_wav, x_mel, label = x_wav.to(device), x_mel.to(device), label.to(device)
                    for r, net in nets.items():
                        logits, feature = net(x_wav, x_mel, label, train=False)
                        cos = F.linear(feature, F.normalize(net.arcface.weight))
                        a = acc[r]
                        a['cos'].append(cos.squeeze(0).cpu().numpy())
                        a['ce_eval'].append(criterion(logits, label).item())
                        a['label'].append(label.item())
                        a['y'].append(y.item())
                        a['machine'].append(machine)
                        a['id'].append(''.join(id_label))
            del test_ds, loader
            print(f'[{ds}] {machine} done')

        for r in todo:
            a = acc[r]
            np.savez_compressed(out_path(r, ds),
                                cos=np.stack(a['cos']).astype(np.float32),
                                label=np.array(a['label'], dtype=np.int64),
                                y=np.array(a['y'], dtype=np.int64),
                                machine=np.array(a['machine']),
                                id=np.array(a['id']),
                                ce_eval=np.array(a['ce_eval'], dtype=np.float32))
            print('wrote', out_path(r, ds))
        del nets
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
