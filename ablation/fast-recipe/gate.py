"""G1 cache identity and G2 probe = eval.py (see README).

    python ablation/fast-recipe/gate.py --gpu 1 [--ckpt_dir ...]
Writes ablation/fast-recipe/scores/gate.txt.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dataloader import Wave2Mel, load_wav, train_dataset  # noqa: E402
from model.net import SCLTFSTgramMFN  # noqa: E402
from scores_path import scores_csv  # noqa: E402
from probe import devtest_metrics, load_devtest  # noqa: E402

NAMES = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--cache', default='data/cache')
    ap.add_argument('--ckpt_dir', default='/home/f74134118/OS-SCL-confirm-seeds/check_points')
    ap.add_argument('--run', default='SCLTFSTgramMFN_cs_relu_max_s2028')
    a = ap.parse_args()
    lines = []
    rng = np.random.default_rng(0)

    # G1: cached item == live decode + mel, bit for bit. DataLoader workers run torch
    # single-threaded, and so does cache_data.py; mirror that (the mel's float sums
    # depend on the thread count at the 1e-5 dB level).
    torch.set_num_threads(1)
    live = train_dataset('data/dataset', NAMES, {})
    cached = train_dataset('data/dataset', NAMES, {'cache_dir': a.cache})
    worst = 0.0
    for i in rng.choice(len(live), 64, replace=False):
        for x, y in zip(live[int(i)], cached[int(i)]):
            worst = max(worst, float((x.float() - y.float()).abs().max()))
    wav, mel, meta = load_devtest(a.cache)
    m = Wave2Mel()
    for i in rng.choice(len(meta), 64, replace=False):
        y, _ = load_wav(meta[i]['file'])
        worst = max(worst, float((y - torch.from_numpy(np.array(wav[i]))).abs().max()),
                    float((m(y) - torch.from_numpy(np.array(mel[i]))).abs().max()))
    lines.append(f'G1 cache identity: max |live - cached| over 64 train + 64 devtest items = {worst:.3g} '
                 f'-> {"ok" if worst == 0 else "FAIL"}')

    # G2: batched fp32 probe reproduces eval.py's committed per-machine AUC
    run_dir = os.path.join(a.ckpt_dir, a.run)
    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    dev = torch.device(f'cuda:{a.gpu}')
    net = SCLTFSTgramMFN(num_classes=41, m=cfg['m'], cfg=cfg).to(dev)
    net.load_state_dict(torch.load(os.path.join(run_dir, 'model.pth'), map_location=dev))
    got = devtest_metrics(net, (wav, mel, meta), dev)
    got16 = devtest_metrics(net, (wav, mel, meta), dev, torch.float16)
    ref = {}
    with open(scores_csv(a.run, 'dev')) as f:
        for r in csv.DictReader(f):
            if r['scope'] == 'machine':
                ref[r['machine']] = float(r['AUC'])
    d32 = max(abs(got[f'{k}_AUC'] - v) for k, v in ref.items())
    d16 = max(abs(got16[f'{k}_AUC'] - v) for k, v in ref.items())
    lines.append(f'G2 probe vs eval.py ({a.run}): max |machine AUC diff| fp32 {d32:.2e} '
                 f'-> {"ok" if d32 <= 5e-4 else "FAIL"} | fp16 probe {d16:.2e} (reported)')
    lines.append(f'   probe fp32 overall AUC {got["AUC"]:.5f} pAUC {got["pAUC"]:.5f} mAUC {got["mAUC"]:.5f}')
    txt = '\n'.join(lines)
    print(txt)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scores', 'gate.txt')
    open(out, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
