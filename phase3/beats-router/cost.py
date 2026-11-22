#!/usr/bin/env python
"""Per-member inference cost for the beats-router study -> scores/cost.json.

    python phase3/beats-router/cost.py [--gpu 0] [--out phase3/beats-router/scores/cost.json]

Each member is measured in its own subprocess (`--member`), because the from-scratch nets and
BEATsASD live in different worktrees whose `model` packages clash. Forward = what extraction does:
  fc_*  : ensemble-backend/extract.forward = net.embed -> head -> L2norm -> ArcFace, train flag = (batch > 1),
          so batch 1 runs TFgram train=False and batch 32 train=True (the squeeze trap), exactly as extract.py.
  beats : BEATsASD.tokens -> ASP -> fc -> ArcFace on a precomputed fbank (fbank timed separately).
FLOPs: torch.utils.flop_counter on CPU, 2*MACs convention of that counter.
"""
import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FRESH = '/home/f74134118/OS-SCL-syn-fresh'
BEATS = '/home/f74134118/OS-SCL-beats'
DATA = '/home/f74134118/OS-SCL/data'
NAME_LIST = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
CK = {'fc_nogain': f'{FRESH}/check_points/SCLTFSTgramMFN_fc_nogain_s2060',
      'fc_spec': f'{FRESH}/check_points/SCLTFSTgramMFN_fc_spec_s2060',
      'beats': f'{BEATS}/check_points/BEATsASD_bt_arc_s2040'}
N = 160000


def stats(ts):
    import numpy as np
    q = np.percentile(ts, [25, 50, 75])
    return float(q[1]), float(q[2] - q[0])


def timeit(fn, dev, runs=50, warm=10):
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(dev)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize(dev)
        ts.append((time.perf_counter() - t) * 1e3)
    return stats(ts)


def flops_of(fn):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        fn()
    return int(fc.get_total_flops())


def member(name, gpu):
    import torch
    import torch.nn.functional as F
    cfg = json.load(open(os.path.join(CK[name], 'config.json'), encoding='UTF-8'))
    out = {}
    if name.startswith('fc_'):
        sys.path.insert(0, FRESH)
        from model.net import SCLTFSTgramMFN
        from dataloader import Wave2Mel
        net = SCLTFSTgramMFN(m=cfg['m'], cfg=cfg)
        net.load_state_dict(torch.load(os.path.join(CK[name], 'model.pth'), map_location='cpu'), strict=True)
        net.eval()
        out['lean'] = bool(cfg.get('lean', False))

        def mk(b, dev):
            return (torch.randn(b, 1, N, device=dev).clamp(-1, 1) * 0.1,
                    torch.randn(b, 1, 128, 313, device=dev), torch.zeros(b, dtype=torch.long, device=dev))

        def fwd(n, wav, mel, lab):
            pre = n.embed(wav, mel, train=wav.shape[0] > 1)
            return n.arcface(F.normalize(n.head(pre), dim=1), lab)
        # front end: Log-Mel on CPU per clip (excluded from primary cost)
        mel = Wave2Mel()
        w = torch.randn(1, N) * 0.1
        for _ in range(3):
            mel(w)
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            mel(w)
            ts.append((time.perf_counter() - t) * 1e3)
        out['frontend'] = {'logmel_cpu_ms_per_clip': stats(ts)[0], 'logmel_cpu_iqr': stats(ts)[1],
                           'cpu_threads': torch.get_num_threads()}
    else:
        sys.path.insert(0, BEATS)
        from model.beats_asd import build
        net, _ = build(cfg, pretrained=False)
        net.load_state_dict(torch.load(os.path.join(CK[name], 'model.pth'), map_location='cpu', weights_only=True),
                            strict=True)
        net.eval()

        def mk(b, dev):
            wav = torch.randn(b, N, device=dev).clamp(-1, 1) * 0.1
            return net.fbank(wav), None, torch.zeros(b, dtype=torch.long, device=dev)

        def fwd(n, fb, _m, lab):
            tok = n.tokens(fb)
            pre = n.fc(n.pool(tok.transpose(1, 2)))
            return n.arcface(pre.float(), lab)
        w = torch.randn(1, N) * 0.1
        net.fbank(w)
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            net.fbank(w)
            ts.append((time.perf_counter() - t) * 1e3)
        out['frontend'] = {'fbank_cpu_ms_per_clip': stats(ts)[0], 'fbank_cpu_iqr': stats(ts)[1],
                           'cpu_threads': torch.get_num_threads()}
    out['params'] = sum(p.numel() for p in net.parameters())
    # FLOPs on CPU
    a = mk(1, 'cpu')
    out['flops'] = flops_of(lambda: fwd(net, *a))
    a4 = mk(4, 'cpu')
    out['flops_b4_per_clip'] = flops_of(lambda: fwd(net, *a4)) / 4
    # GPU latency
    dev = torch.device(f'cuda:{gpu}')
    net = net.to(dev)
    with torch.no_grad():
        for b, tag in ((1, 'b1'), (32, 'b32')):
            a = mk(b, dev)
            m, iqr = timeit(lambda: fwd(net, *a), dev)
            if b == 32:
                m, iqr = m / b, iqr / b
                out['lat_b32_ms_per_clip'], out['lat_b32_iqr'] = m, iqr
            else:
                out['lat_b1_ms'], out['lat_b1_iqr'] = m, iqr
        if name.startswith('fc_'):
            # train=False at batch>1 is the squeeze trap; record that extraction does not use it
            out['note'] = 'b1 uses TFgram train=False (as eval.py); b32 uses train=True (as extract.py)'
        else:
            # BEATs fbank per clip on GPU is not what extraction does (it runs ta_kaldi.fbank on the device)
            w = torch.randn(32, N, device=dev).clamp(-1, 1) * 0.1
            m, iqr = timeit(lambda: net.fbank(w), dev, runs=20, warm=3)
            out['frontend']['fbank_gpu_ms_per_clip_b32'] = m / 32
    out['gpu_name'] = torch.cuda.get_device_name(dev)
    return out


def clip_counts():
    sys.path.insert(0, os.path.join(FRESH, 'ablation', 'ensemble-backend'))
    roots = {'dev': 'dataset', 'eval': 'eval_dataset'}
    return {ds: {m: len(glob.glob(os.path.join(DATA, r, m, 'test', '*.wav'))) for m in NAME_LIST}
            for ds, r in roots.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--member')
    ap.add_argument('--out', default=os.path.join(HERE, 'scores', 'cost.json'))
    args = ap.parse_args()
    assert args.gpu != 1, 'GPU 1 is slow; never use it'
    if args.member:
        print(json.dumps(member(args.member, args.gpu)))
        return
    import torch
    busy = subprocess.run(['nvidia-smi', f'--id={args.gpu}', '--query-gpu=utilization.gpu,memory.used',
                           '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    print('gpu state (util %, MiB):', busy.strip())
    res = {}
    for nm in CK:
        for attempt in range(3):   # smartlab1's faulty host RAM segfaults long-lived processes: retry
            p = subprocess.run([sys.executable, __file__, '--member', nm, '--gpu', str(args.gpu)],
                               capture_output=True, text=True)
            if p.returncode == 0:
                break
            print(nm, 'attempt', attempt, 'rc', p.returncode, p.stderr[-300:], flush=True)
        assert p.returncode == 0, p.stderr[-3000:]
        res[nm] = json.loads(p.stdout.strip().splitlines()[-1])
        print(nm, res[nm], flush=True)
    gpu_name = res['beats'].pop('gpu_name')
    for r in res.values():
        r.pop('gpu_name', None)
    fe = {nm: res[nm].pop('frontend') for nm in res}
    comp = {'fc_nogain': 8, 'fc_spec': 4, 'beats': 3}
    tot = sum(comp[k] * res[k]['flops'] for k in comp)
    totl = sum(comp[k] * res[k]['lat_b32_ms_per_clip'] for k in comp)
    cost = {
        '_doc': ('members: per-clip network forward only, 10 s @ 16 kHz. flops = torch.utils.flop_counter total on CPU at '
                 'batch 1 (counter convention: 2 per MAC), flops_b4_per_clip = batch 4 / 4 (check). params = all '
                 'parameters of the net as loaded (lean nets have no dead modules). lat_*_ms = median ms of 50 timed runs '
                 'after 10 warm-up, GPU 0, fp32, eval, no_grad, synchronize; *_iqr = 75th-25th percentile; b32 values are '
                 'per clip (batch time / 32). fc_* b1 uses TFgram train=False (eval.py), b32 train=True (extract.py); '
                 'beats timed on a precomputed fbank. frontends are EXCLUDED from the primary cost: Log-Mel CPU ms/clip '
                 '(torchaudio Wave2Mel), BEATs Kaldi fbank CPU ms/clip and GPU ms/clip at b32. clips: test wav counts '
                 'per machine type for dev and eval. x1f = 8 fc_nogain + 4 fc_spec + 3 BEATs: flops_total and '
                 'lat_b32_ms_per_clip_total sum over members; beats_share_* = BEATs part of that total. Latency is never '
                 'compared across GPUs.'),
        'members': res, 'frontends': fe, 'clips': clip_counts(),
        'x1f': {'composition': comp, 'flops_total': tot, 'beats_share_flops': 3 * res['beats']['flops'] / tot,
                'lat_b32_ms_per_clip_total': totl,
                'beats_share_lat_b32': 3 * res['beats']['lat_b32_ms_per_clip'] / totl},
        'host': {'gpu_name': gpu_name, 'gpu_index': args.gpu, 'torch': torch.__version__,
                 'date': datetime.date.today().isoformat()},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(cost, open(args.out, 'w'), indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
