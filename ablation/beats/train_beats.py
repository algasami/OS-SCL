#!/usr/bin/env python
"""Fine-tune BEATs + ASP + ArcFace on the 41 DCASE 2020 machine IDs (ablation/beats).

    python ablation/beats/train_beats.py --gpu_num 0 --seed 2040 --desc bt_arc_s2040 \
        --cache_dir data/cache [--steps 10000 --micro_bs 32 --accum 8 --amp bf16 --grad_ckpt 1]

Run from the repo root. Writes check_points/BEATsASD_<desc>/{model.pth, config.json} and,
every --state_every optimizer steps, state_last.pth for --crash_resume.

Same data protocol as train.py: random_seed(seed) -> train_dataset -> dataset_split (the
first torch RNG draw), so the 5% validation split of seed s is ao_nogain_s<s>'s and
ensemble-backend/extract.py's replay_valid reproduces it. Labels are dataloader.py's 41 IDs.

Recipe (README *Recipe*; AnoPatch): AdamW lr 1e-4 (wd 0.01), linear warm-up 960 steps then
cosine to 0, 10k optimizer steps of 32 x 8 accumulated clips (~74 epochs), SpecAugment
freq 40 / time 80, plain ArcFace CE (s 30, m 0.4), no mixup, no EMA; the deliverable is the
final weights. Validation (ArcFace CE with margin, as trainer.valid) every --valid_every
steps, printed as an `EPOCH: <step>` line that ablation/distill_curves.py parses.

Data order is a pure function of (seed, step): epoch e is torch.randperm(n, generator seeded
seed*1000+e), dropping the last partial micro-batch, so a crash-resume continues it exactly.
Cached waves are range-checked (smartlab1's RAM flips bits) and re-decoded if bad.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from dataloader import load_wav, train_dataset      # noqa: E402
from model.beats_asd import build                   # noqa: E402
from utils import dataset_split                     # noqa: E402

NAME_LIST = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
N = 160000


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu_num', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--desc', required=True)
    ap.add_argument('--cache_dir', default='data/cache')
    ap.add_argument('--beats_ckpt', default='data/pretrained/hf/BEATs_iter3.pt')
    ap.add_argument('--steps', type=int, default=10000, help='optimizer steps')
    ap.add_argument('--warmup', type=int, default=960)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--micro_bs', type=int, default=32)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--m', type=float, default=0.4)
    ap.add_argument('--arc_s', type=float, default=30.0)
    ap.add_argument('--emb_dim', type=int, default=128)
    ap.add_argument('--freq_mask', type=int, default=40)
    ap.add_argument('--time_mask', type=int, default=80)
    ap.add_argument('--amp', choices=['none', 'fp16', 'bf16'], default='bf16')
    ap.add_argument('--grad_ckpt', type=int, choices=[0, 1], default=0)
    ap.add_argument('--sdpa', type=int, choices=[0, 1], default=0, help='fused attention (model/beats edit)')
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--valid_every', type=int, default=250)
    ap.add_argument('--state_every', type=int, default=500)
    ap.add_argument('--crash_resume', default=None)
    ap.add_argument('--bench', type=int, default=0,
                    help='>0: time this many optimizer steps after 2 warm-up steps, print, exit (no files)')
    return ap.parse_args()


def random_seed(seed):   # train.py's
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Waves(Dataset):
    """Training clips as (wave (160000,), label) from cache_data.py's train_wav.npy."""

    def __init__(self, files, labels, cache_dir):
        self.files, self.labels = files, labels
        self.path = os.path.join(cache_dir, 'train_wav.npy') if cache_dir else None
        self.mm = None
        if self.path:
            with open(os.path.join(cache_dir, 'train_files.txt')) as f:
                cached = f.read().split()
            tail = lambda fs: [x.split(os.sep + 'dataset' + os.sep, 1)[-1] for x in fs]  # noqa: E731
            if tail(cached) != tail(files):
                raise SystemExit(f'{cache_dir}/train_files.txt does not match the training file list')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        if self.path:
            if self.mm is None:
                self.mm = np.load(self.path, mmap_mode='r')
            w = np.array(self.mm[i]).reshape(-1)
            if w.shape[0] == N and np.isfinite(w).all() and np.abs(w).max() <= 1.0:
                return torch.from_numpy(w), self.labels[i]
            print(f'cache: train item {i} out of range; re-decoded', flush=True)
        y, _ = load_wav(self.files[i])
        return y.reshape(-1), self.labels[i]


class StepSampler(Sampler):
    """Micro-batches from `start` on; epoch e = randperm(idx) seeded seed*1000+e, last partial dropped."""

    def __init__(self, idx, bs, seed, start, total):
        self.idx, self.bs, self.seed, self.start, self.total = np.asarray(idx), bs, seed, start, total
        self.per_epoch = len(self.idx) // bs

    def __len__(self):
        return self.total - self.start

    def __iter__(self):
        k = self.start
        while k < self.total:
            e, j = divmod(k, self.per_epoch)
            g = torch.Generator().manual_seed(self.seed * 1000 + e)
            perm = self.idx[torch.randperm(len(self.idx), generator=g).numpy()]
            while j < self.per_epoch and k < self.total:
                yield perm[j * self.bs:(j + 1) * self.bs].tolist()
                j += 1
                k += 1


def lr_at(step, cfg):
    if step < cfg['warmup']:
        return cfg['lr'] * (step + 1) / cfg['warmup']
    p = (step - cfg['warmup']) / max(1, cfg['steps'] - cfg['warmup'])
    return cfg['lr'] * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def valid(net, loader, device, amp):
    net.eval()
    tot, correct, n, losses = 0.0, 0, 0, []
    with torch.no_grad():
        for w, y in loader:
            w, y = w.to(device, non_blocking=True), y.to(device)
            with torch.autocast('cuda', dtype=amp, enabled=amp is not None):
                logits, _, _ = net(w, y)
            loss = F.cross_entropy(logits.float(), y)
            losses.append(loss.item())
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    net.train()
    return float(np.mean(losses)), correct / n


def save_atomic(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    args = parse()
    cfg = vars(args).copy()
    name = f'BEATsASD_{args.desc}'
    cfg['model_name'] = name
    ck = os.path.join('check_points', name)
    device = torch.device(f'cuda:{args.gpu_num}')
    amp = {'none': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.amp]

    random_seed(args.seed)
    ds = train_dataset('data/dataset', NAME_LIST, {})   # the file list and labels only
    tr_split, va_split = dataset_split(ds)               # first torch RNG draw, as train.py
    files, labels = list(ds.unrolled_files_list), ds.labels
    data = Waves(files, labels, args.cache_dir)
    tr_idx, va_idx = list(tr_split.indices), list(va_split.indices)
    cfg['n_train'], cfg['n_valid'] = len(tr_idx), len(va_idx)
    cfg['valid_idx_sum'] = int(np.sum(va_idx))          # extract.py checks its replay against this

    net, bcfg = build(cfg, pretrained=True)
    cfg['beats_cfg'] = bcfg
    net = net.to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler('cuda', enabled=amp == torch.float16)

    step = 0
    if args.crash_resume:
        st = torch.load(args.crash_resume, map_location='cpu', weights_only=False)
        net.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        scaler.load_state_dict(st['scaler'])
        step = st['step']
        random.setstate(st['py_rng'])
        np.random.set_state(st['np_rng'])
        torch.set_rng_state(st['torch_rng'])
        torch.cuda.set_rng_state(st['cuda_rng'], device)
        print(f'crash-resumed {args.crash_resume} at step {step}', flush=True)
    if not args.bench:
        os.makedirs(ck, exist_ok=True)
        with open(os.path.join(ck, 'config.json'), 'w', encoding='UTF-8') as f:
            json.dump(cfg, f, indent=2)

    total = args.steps * args.accum
    if args.bench:
        total = step * args.accum + (args.bench + 2) * args.accum
    sampler = StepSampler(tr_idx, args.micro_bs, args.seed, step * args.accum, total)
    nw = args.num_workers
    loader = DataLoader(data, batch_sampler=sampler, num_workers=nw, pin_memory=True,
                        persistent_workers=False, prefetch_factor=4 if nw else None)
    vloader = DataLoader(torch.utils.data.Subset(data, va_idx), batch_size=64, num_workers=nw, pin_memory=True)
    print(f'{name} | train {len(tr_idx)} valid {len(va_idx)} | {sampler.per_epoch} micro-batches/epoch | '
          f'{args.steps} steps x {args.micro_bs}x{args.accum} = {args.steps * args.micro_bs * args.accum / len(tr_idx):.1f} '
          f'epochs | params {sum(p.numel() for p in net.parameters())}', flush=True)

    run_loss, run_acc, run_n, skipped = 0.0, 0.0, 0, 0
    t0, tb = time.time(), None
    micro = 0
    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    for w, y in loader:
        w, y = w.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=amp, enabled=amp is not None):
            logits, _, _ = net(w, y, augment=True)
        loss = F.cross_entropy(logits.float(), y)
        scaler.scale(loss / args.accum).backward()
        run_loss += loss.item()
        run_acc += (logits.argmax(1) == y).float().mean().item()
        run_n += 1
        micro += 1
        if micro % args.accum:
            continue
        for g in opt.param_groups:
            g['lr'] = lr_at(step, cfg)
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), float('inf'))
        if torch.isfinite(gn):
            scaler.step(opt)
        else:
            skipped += 1
            print(f'skip | step {step} | non-finite grad norm, {skipped} skipped so far', flush=True)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        if args.bench:
            if step == sampler.start // args.accum + 2:
                torch.cuda.synchronize(device)
                tb = time.time()
            continue
        if step % 50 == 0:
            el = time.time() - t0
            print(f'step {step} | loss {run_loss / run_n:.4f} | acc {run_acc / run_n:.4f} | gn {gn.item():.2f} | '
                  f'lr {lr_at(step - 1, cfg):.2e} | {el / 3600:.2f} h', flush=True)
        if step % args.valid_every == 0 or step == args.steps:
            vl, va = valid(net, vloader, device, amp)
            print(f'lr: {lr_at(step - 1, cfg):.7f} | EPOCH: {step} | Train_loss: {run_loss / run_n:.5f} | '
                  f'Train_accuracy: {run_acc / run_n:.5f} | Valid_loss: {vl:.5f} | Valid_accuracy: {va:.5f}',
                  flush=True)
            run_loss, run_acc, run_n = 0.0, 0.0, 0
        if step % args.state_every == 0 and step < args.steps:
            save_atomic(dict(model=net.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step,
                             py_rng=random.getstate(), np_rng=np.random.get_state(),
                             torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device)),
                        os.path.join(ck, 'state_last.pth'))

    if args.bench:
        torch.cuda.synchronize(device)
        dt = (time.time() - tb) / args.bench
        peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
        print(f'BENCH gpu {args.gpu_num} | micro_bs {args.micro_bs} accum {args.accum} amp {args.amp} '
              f'grad_ckpt {args.grad_ckpt} sdpa {args.sdpa} | {dt:.2f} s/step | {args.micro_bs * args.accum / dt:.1f} clips/s | '
              f'peak {peak:.2f} GiB | projected {dt * args.steps / 3600:.2f} h for {args.steps} steps', flush=True)
        return
    assert step == args.steps, step
    save_atomic(net.state_dict(), os.path.join(ck, 'model.pth'))
    print(f'done | {step} steps | {(time.time() - t0) / 3600:.2f} h | skipped {skipped} | '
          f'peak {torch.cuda.max_memory_allocated(device) / 2 ** 30:.2f} GiB', flush=True)


if __name__ == '__main__':
    main()
