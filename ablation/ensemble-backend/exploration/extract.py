"""Exploratory: dump embeddings of dev-ID train clips and dev test clips.

Per clip: pre-head MobileFaceNet feature (128), post-head L2-normed feature (128),
the 41 ArcFace cosines, eval.py's CE score. Dev set only -- eval is held out for a
pre-registered confirmation.

    python extract.py --repo /home/f74134118/OS-SCL --gpu 0 --out <dir> --runs ...
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

NAME_LIST = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']


def label_list(name):
    if name == 'ToyConveyor':
        return [f'id_0{i}' for i in range(1, 7)]
    if name == 'ToyCar':
        return [f'id_0{i}' for i in range(1, 8)]
    return [f'id_0{i}' for i in range(0, 7)]


def offsets():
    off, o = {}, 0
    for n in NAME_LIST:
        off[n] = o
        o += len(label_list(n))
    return off


class Clips(Dataset):
    def __init__(self, items):
        from dataloader import Wave2Mel
        self.items = items
        self.mel = Wave2Mel()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, y = self.items[i]
        wav, sr = torchaudio.load(path)
        wav = wav[..., :sr * 10]
        return wav, self.mel(wav), label, y


def build_items(data, split):
    off = offsets()
    items, meta = [], []
    for m in NAME_LIST:
        test_ids = sorted({re.search(r'id_\d\d', os.path.basename(p)).group()
                           for p in glob.glob(f'{data}/dataset/{m}/test/*.wav')})
        pat = f'{data}/dataset/{m}/{split}/*.wav'
        for p in sorted(glob.glob(pat)):
            b = os.path.basename(p)
            idl = re.search(r'id_\d\d', b).group()
            if idl not in test_ids:
                continue  # eval-set IDs: not needed for a dev-only probe
            y = 1 if b.startswith('anomaly') else 0
            items.append((p, off[m] + label_list(m).index(idl), y))
            meta.append((m, idl))
    return items, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--bs', type=int, default=32)
    args = ap.parse_args()
    sys.path.insert(0, args.repo)
    from model.net import SCLTFSTgramMFN
    from losses import ASDLoss

    dev = torch.device(f'cuda:{args.gpu}')
    nets = {}
    for r in args.runs:
        cdir = os.path.join(args.repo, 'check_points', r)
        cfg = json.load(open(os.path.join(cdir, 'config.json')))
        net = SCLTFSTgramMFN(m=cfg['m'], cfg=cfg).to(dev)
        net.load_state_dict(torch.load(os.path.join(cdir, 'model.pth'), map_location=dev), strict=True)
        nets[r] = net.eval()
    ce = ASDLoss(reduction=False).to(dev)
    os.makedirs(args.out, exist_ok=True)

    for split in ('train', 'test'):
        items, meta = build_items(os.path.join(args.repo, 'data'), split)
        print(split, len(items), 'clips')
        acc = {r: {k: [] for k in ('pre', 'post', 'cos', 'ce')} for r in nets}
        loader = DataLoader(Clips(items), batch_size=args.bs, num_workers=8, pin_memory=True)
        with torch.no_grad():
            for wav, mel, label, _ in tqdm(loader, dynamic_ncols=True, mininterval=30):
                wav, mel, label = wav.to(dev), mel.to(dev), label.to(dev)
                for r, net in nets.items():
                    # BN is in eval mode, so batching is per-sample identical to eval.py;
                    # TFgram's reshape needs train=True for batch>1, False for batch 1
                    tr = wav.shape[0] > 1
                    br = []
                    for name in __import__('model.net', fromlist=['x']).FUSSION_BRANCHES[net.cfg['fussion']]:
                        if name == 'mel':
                            br.append(mel)
                        elif name == 'tgram':
                            br.append(net.tgramnet(wav).unsqueeze(1))
                        else:
                            br.append(net.TFgramNet(wav, tr).unsqueeze(1))
                    _, pre = net.mobilefacenet(torch.cat(br, 1))
                    post = F.normalize(net.head(pre), dim=1)
                    logits = net.arcface(post, label)
                    a = acc[r]
                    a['pre'].append(pre.cpu().numpy())
                    a['post'].append(post.cpu().numpy())
                    a['cos'].append(F.linear(post, F.normalize(net.arcface.weight)).cpu().numpy())
                    a['ce'].append(ce(logits, label).cpu().numpy())
        lab = np.array([i[1] for i in items]); yy = np.array([i[2] for i in items])
        for r in nets:
            a = acc[r]
            np.savez(os.path.join(args.out, f'{r}_{split}.npz'),
                     **{k: np.concatenate(v) for k, v in a.items()},
                     label=lab, y=yy, machine=np.array([m for m, _ in meta]),
                     id=np.array([i for _, i in meta]))
        print('done', split)


if __name__ == '__main__':
    main()
