#!/usr/bin/env python
"""Patch-level memory banks (PatchCore-style): per-patch nearest-neighbour distances.

    python ablation/memory-bank/bank.py --gpu 0 --repo <checkout with data/, check_points/> \
        --emb <ensemble-backend results/emb> --out ablation/memory-bank/results \
        --runs SCLTFSTgramMFN_tfl_relu_max_s2024 ... [--imagenet-mel] [--stats-only --ids 8 9]

For every checkpoint, arm and machine ID (README, *Arms*):
  bank   the patches of the ID's training-part clips (the seed's 95% split, replayed by
         ../ensemble-backend and cached in its <run>.npz `valid` mask)
  test   every dev/eval test clip of the ID: per-patch L2 distance to the nearest bank patch
  stats  200 training-part clips of the ID (np.random.default_rng(0)), scored
         leave-one-clip-out: every patch of the clip itself is masked out of the bank
Per-patch distances are written, not clip scores, so score.py can take the max (primary)
or the top-10 mean (sensitivity) without another GPU pass.

Arms, each patch vector averaged over its 3x3 (A2: 3-frame) neighbourhood first:
  mfn        MobileFaceNet blocks[3] (+) blocks[5] upsampled to 16x40     256-d, 640 patches
  mfn_l2     blocks[3] alone                                             128-d, 640  (sensitivity)
  mfn_l3     blocks[5] alone, native 8x20                                128-d, 160  (sensitivity)
  mfn_cs10   mfn on a 10% greedy k-centre coreset (128-d random proj.)              (sensitivity)
  tfgram     TFgram pre_block3 (+) pre_block2 avg-pooled x2 to 313       256-d, 313
  imagenet   WideResNet-50 layer2 (+) layer3 upsampled, on the 3-ch stack 1536-d, 640
  imagenet_mel  as imagenet on Log-Mel x3; no OS-SCL weights, run once (--imagenet-mel)
ImageNet inputs are standardised per channel with mean/sd over the checkpoint's
training-part clips (imagenet_mel: over all training clips -- it has no split).

Work is ID-major: an ID's clips are decoded once and pushed through every checkpoint, and
only one bank is on the GPU at a time. L2 distances use |q|^2 + |b|^2 - 2 q.b with the
dot product in fp16 on features centred on the bank mean (translation leaves L2
unchanged and shrinks the fp16 rounding); norms are fp32. Resumable per (arm, run, ID).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'ensemble-backend'))
from extract import Clips, forward               # noqa: E402
from losses import ASDLoss                       # noqa: E402
from model.net import SCLTFSTgramMFN             # noqa: E402
from scores_path import scores_dir               # noqa: E402

MODEL_ARMS = ['mfn', 'mfn_l2', 'mfn_l3', 'mfn_cs10', 'tfgram', 'imagenet']
N_STATS, CORESET, PROJ_DIM, BS = 200, 0.10, 128, 32
Q_CHUNK, B_CHUNK = 4096, 65536


# ---- nearest-neighbour search --------------------------------------------------------

def nn_dist(q, b, q_owner=None, b_owner=None, exact=False):
    """Per-row L2 distance from q (M, D) to its nearest row of b (N, D), both fp16 and
    centred. With owners, b rows sharing the query's owner are excluded (leave-one-out).
    exact=True does the dot product in fp32 -- the reference the fp16 path is checked against."""
    bn = torch.cat([b[j:j + B_CHUNK].float().pow(2).sum(1) for j in range(0, len(b), B_CHUNK)])
    out = torch.empty(len(q), device=q.device)
    for i in range(0, len(q), Q_CHUNK):
        qc = q[i:i + Q_CHUNK]
        qn = qc.float().pow(2).sum(1, keepdim=True)
        if exact:
            qc = qc.float()
        best = torch.full((len(qc),), float('inf'), device=q.device)
        for j in range(0, len(b), B_CHUNK):
            bc = b[j:j + B_CHUNK].float() if exact else b[j:j + B_CHUNK]
            d = (qc @ bc.T).float().mul_(-2).add_(qn).add_(bn[None, j:j + B_CHUNK])
            if q_owner is not None:
                d.masked_fill_(q_owner[i:i + Q_CHUNK, None] == b_owner[None, j:j + B_CHUNK], float('inf'))
            best = torch.minimum(best, d.min(1).values)
            del d
        out[i:i + Q_CHUNK] = best
    return out.clamp_min_(0).sqrt_()


def kcenter(x, frac, seed=0):
    """PatchCore's greedy coreset: k-centre on a 128-d random projection of x."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    p = x.float()
    if p.shape[1] > PROJ_DIM:
        p = p @ torch.randn(p.shape[1], PROJ_DIM, device=x.device, generator=g) / PROJ_DIM ** 0.5
    k = max(1, int(round(frac * len(p))))
    # the loop is sequential and bandwidth-bound: one fp16 mat-vec per centre, fp32 norms
    pn, p = p.pow(2).sum(1), p.half()
    sel = torch.empty(k, dtype=torch.long, device=x.device)
    sel[0] = torch.randint(len(p), (1,), device=x.device, generator=g)
    mind = (pn + pn[sel[0]] - 2 * (p @ p[sel[0]]).float()).clamp_min_(0)
    for t in range(1, k):
        sel[t] = mind.argmax()
        mind = torch.minimum(mind, pn + pn[sel[t]] - 2 * (p @ p[sel[t]]).float())
    return sel


# ---- features -------------------------------------------------------------------------

def pool2(x):
    return F.avg_pool2d(x, 3, 1, 1)


def to_patches(x):  # (B, C, H, W) or (B, C, T) -> (B, patches, C)
    return x.flatten(2).transpose(1, 2)


class WRN:
    """WideResNet-50 up to layer3; returns PatchCore's layer2 (+) layer3 patches."""

    def __init__(self, device):
        self.m = torchvision.models.wide_resnet50_2(weights='IMAGENET1K_V1').to(device).eval()

    def __call__(self, x):
        m = self.m
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        l2 = m.layer2(m.layer1(x))
        l3 = m.layer3(l2)
        l2, l3 = pool2(l2), pool2(l3)
        l3 = F.interpolate(l3, size=l2.shape[-2:], mode='bilinear', align_corners=False)
        return to_patches(torch.cat([l2, l3], 1))


class Hooked:
    """A checkpoint with hooks on the layers the arms read."""

    def __init__(self, run, repo, device):
        with open(os.path.join(scores_dir(run), 'config.json'), encoding='UTF-8') as f:
            cfg = json.load(f)
        self.net = SCLTFSTgramMFN(m=cfg['m'], cfg=cfg).to(device)
        self.net.load_state_dict(torch.load(os.path.join(repo, 'check_points', run, 'model.pth'),
                                            map_location=device), strict=True)
        self.net.eval()
        self.h = {}
        mf, tf = self.net.mobilefacenet, self.net.TFgramNet
        mf.register_forward_pre_hook(lambda m, i: self.h.__setitem__('stack', i[0]))
        for k, mod in (('b3', mf.blocks[3]), ('b5', mf.blocks[5]), ('t2', tf.pre_block2), ('t3', tf.pre_block3)):
            mod.register_forward_hook(lambda m, i, o, k=k: self.h.__setitem__(k, o))

    def __call__(self, wav, mel, lab, ce):
        logits, _ = forward(self.net, wav, mel, lab)
        h = self.h
        b3, b5 = pool2(h['b3']), pool2(h['b5'])
        b5up = F.interpolate(b5, size=b3.shape[-2:], mode='bilinear', align_corners=False)
        t2 = F.avg_pool1d(F.avg_pool1d(h['t2'], 2), 3, 1, 1, count_include_pad=False)
        t3 = F.avg_pool1d(h['t3'], 3, 1, 1, count_include_pad=False)
        return {'ce': ce(logits, lab),
                'mfn': to_patches(torch.cat([b3, b5up], 1)),
                'mfn_l2': to_patches(b3), 'mfn_l3': to_patches(b5),
                'tfgram': to_patches(torch.cat([t2, t3], 1)),
                'stack': h['stack']}


def standardise(x, st):
    m = torch.tensor(st['mean'], device=x.device).view(1, -1, 1, 1)
    s = torch.tensor(st['sd'], device=x.device).view(1, -1, 1, 1)
    return (x - m) / s


# ---- channel statistics for the ImageNet arms ----------------------------------------

def chanstats(models, meta, emb, out, device, want_mel):
    """Mean/sd per input channel over training-part clips (one pass, all runs at once)."""
    todo = [r for r in models if not os.path.exists(os.path.join(out, 'chanstats', f'{r}.json'))]
    mel_path = os.path.join(out, 'chanstats', 'imagenet_mel.json')
    want_mel = want_mel and not os.path.exists(mel_path)
    if not todo and not want_mel:
        return
    os.makedirs(os.path.join(out, 'chanstats'), exist_ok=True)
    keep = {r: ~np.load(os.path.join(emb, f'{r}.npz'))['valid'] for r in todo}
    acc = {r: torch.zeros(3, 3, dtype=torch.float64) for r in todo}   # n, sum, sumsq per channel
    mel_acc = torch.zeros(3, dtype=torch.float64)
    paths, labels = list(meta['tr_paths']), torch.as_tensor(meta['tr_label'])
    loader = DataLoader(Clips(paths, labels), batch_size=BS, num_workers=8, pin_memory=True)
    ce = ASDLoss(reduction=False).to(device)
    i0 = 0
    with torch.no_grad():
        for wav, mel, lab in loader:
            ix = np.arange(i0, i0 + len(lab))
            i0 += len(lab)
            wav, mel, lab = wav.to(device), mel.to(device), lab.to(device)
            if want_mel:
                v = mel.double()
                mel_acc += torch.stack([torch.tensor(float(v.numel())), v.sum().cpu(), v.pow(2).sum().cpu()])
            for r in todo:
                sel = torch.as_tensor(keep[r][ix], device=device)
                if not sel.any():
                    continue
                models[r](wav, mel, lab, ce)
                x = models[r].h['stack'][sel].double()
                n = x[:, 0].numel()
                acc[r] += torch.stack([torch.full((3,), float(n), dtype=torch.float64),
                                       x.sum((0, 2, 3)).cpu(), x.pow(2).sum((0, 2, 3)).cpu()])
    for r in todo:
        n, s, ss = acc[r]
        mu = s / n
        json.dump({'mean': mu.tolist(), 'sd': (ss / n - mu ** 2).sqrt().tolist()},
                  open(os.path.join(out, 'chanstats', f'{r}.json'), 'w'))
    if want_mel:
        n, s, ss = mel_acc
        mu = float(s / n)
        sd = float((ss / n - mu ** 2).sqrt())
        json.dump({'mean': [mu] * 3, 'sd': [sd] * 3}, open(mel_path, 'w'))


# ---- one ID ---------------------------------------------------------------------------

def centred(x, mu):
    """x (n, D) fp16 minus mu, in chunks so the bank never exists in fp32 on the GPU."""
    out = torch.empty_like(x)
    for j in range(0, len(x), B_CHUNK):
        out[j:j + B_CHUNK] = (x[j:j + B_CHUNK].float() - mu).half()
    return out


def search(feats, bank_ix, stat_ix, test_ix, cs_frac=None, check_loo=False):
    """feats (n_clips, P, D) fp16 on GPU. Returns per-patch NN distances for stats (LOO)
    and test clips, plus, with check_loo, the leave-one-out and fp16 checks."""
    n, P, D = feats.shape
    b = feats[bank_ix].reshape(-1, D)
    mu = sum(b[j:j + B_CHUNK].float().sum(0) for j in range(0, len(b), B_CHUNK)) / len(b)
    b = centred(b, mu)
    owner = torch.as_tensor(np.repeat(bank_ix, P), device=feats.device)
    if cs_frac:
        keep = kcenter(b, cs_frac)
        b, owner = b[keep], owner[keep]
    res = {'bank_patches': len(b)}
    q = centred(feats[stat_ix].reshape(-1, D), mu)
    q_owner = torch.as_tensor(np.repeat(stat_ix, P), device=feats.device)
    res['stats'] = nn_dist(q, b, q_owner, owner).view(len(stat_ix), P)
    if check_loo:
        # (1) without the mask a stats clip finds itself: distance ~0
        res['unmasked_self_max'] = float(nn_dist(q[:2 * P], b).max())
        # (2) the mask equals physically deleting the clip from the bank (fp32 reference),
        #     which also measures the fp16 dot-product error
        errs = []
        for k in range(2):
            ref = nn_dist(q[k * P:(k + 1) * P], b[owner != stat_ix[k]], exact=True)
            errs.append((ref - res['stats'][k]).abs() / ref.clamp_min(1e-6))
            assert ref.min() > 0, 'a stats clip still matches itself after deletion'
        res['loo_vs_deleted_max_rel_err'] = float(torch.cat(errs).max())
        res['loo_vs_deleted_mean_rel_err'] = float(torch.cat(errs).mean())
        res['stats_min'] = float(res['stats'].min())
    if test_ix is not None and len(test_ix):
        res['test'] = nn_dist(centred(feats[test_ix].reshape(-1, D), mu), b).view(len(test_ix), P)
    return res


def load_clips(paths, labels):
    loader = DataLoader(Clips(paths, torch.as_tensor(labels)), batch_size=BS, num_workers=8)
    W, M, L = [], [], []
    for wav, mel, lab in loader:
        W.append(wav)
        M.append(mel)
        L.append(lab)
    return torch.cat(W), torch.cat(M), torch.cat(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--emb', required=True, help="../ensemble-backend's results/emb")
    ap.add_argument('--out', required=True)
    ap.add_argument('--runs', nargs='*', default=[])
    ap.add_argument('--arms', nargs='+', default=MODEL_ARMS)
    ap.add_argument('--imagenet-mel', action='store_true', help='also run the seed-free arm')
    ap.add_argument('--ids', nargs='+', type=int, help='class indices to run (default all 41)')
    ap.add_argument('--stats-only', action='store_true',
                    help='debug: training-clip LOO statistics only; no test clip is touched')
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}')
    torch.backends.cuda.matmul.allow_tf32 = False
    meta = dict(np.load(os.path.join(args.emb, 'meta.npz')))
    for p in (meta['tr_paths'][0], meta['te_paths'][0]):
        assert os.path.exists(p), f'{p} missing: meta.npz was built against another checkout'
    tag = 'debug' if args.stats_only else 'full'
    out = os.path.join(args.out, tag)
    os.makedirs(out, exist_ok=True)

    cached = {r: dict(np.load(os.path.join(args.emb, f'{r}.npz'))) for r in args.runs}
    models = {r: Hooked(r, args.repo, device) for r in args.runs}
    wrn = WRN(device) if ('imagenet' in args.arms or args.imagenet_mel) else None
    if wrn is not None:
        chanstats(models if 'imagenet' in args.arms else {}, meta, args.emb, args.out, device,
                  args.imagenet_mel)
        cst = {r: json.load(open(os.path.join(args.out, 'chanstats', f'{r}.json')))
               for r in args.runs if 'imagenet' in args.arms}
        mst = json.load(open(os.path.join(args.out, 'chanstats', 'imagenet_mel.json'))) \
            if args.imagenet_mel else None
    ce = ASDLoss(reduction=False).to(device)
    jobs = [(a, r) for r in args.runs for a in args.arms] + ([('imagenet_mel', 'none')] if args.imagenet_mel else [])
    log = {}

    for c in (args.ids if args.ids is not None else range(int(meta['tr_label'].max()) + 1)):
        fn = lambda a, r: os.path.join(out, a, r, f'id{c:02d}.npz')
        todo = [(a, r) for a, r in jobs if not os.path.exists(fn(a, r))]
        if not todo:
            continue
        t0 = time.time()
        tr_ix = np.where(meta['tr_label'] == c)[0]
        te_ix = np.array([], dtype=int) if args.stats_only else np.where(meta['te_label'] == c)[0]
        paths = list(meta['tr_paths'][tr_ix]) + list(meta['te_paths'][te_ix])
        wav, mel, lab = load_clips(paths, np.concatenate([meta['tr_label'][tr_ix], meta['te_label'][te_ix]]))
        n_tr = len(tr_ix)
        t_load = time.time() - t0

        def split(valid):
            """Local indices: bank = training part, stats = 200 of them, test = after n_tr."""
            bank = np.where(~valid[tr_ix])[0]
            rng = np.random.default_rng(0)
            stat = np.sort(rng.choice(bank, min(N_STATS, len(bank)), replace=False))
            return bank, stat, np.arange(n_tr, n_tr + len(te_ix))

        def run_arms(feat_fn, arms, valid, rkey):
            stored = [a for a in arms if a != 'mfn_cs10'] + (['mfn'] if 'mfn_cs10' in arms and 'mfn' not in arms else [])
            feats = {a: [] for a in stored}
            ce_all = []
            with torch.no_grad():
                for i in range(0, len(lab), BS):
                    f = feat_fn(wav[i:i + BS].to(device), mel[i:i + BS].to(device), lab[i:i + BS].to(device))
                    for a in stored:
                        feats[a].append(f[a].half())
                    if 'ce' in f:
                        ce_all.append(f['ce'].float().cpu())
            bank, stat, test = split(valid)
            feats = {a: torch.cat(v) for a, v in feats.items()}
            for a in sorted(arms, key=lambda a: a == 'mfn'):   # mfn last: mfn_cs10 reads it
                fa = feats['mfn'] if a == 'mfn_cs10' else feats.pop(a)
                res = search(fa, bank, stat, None if args.stats_only else test,
                             cs_frac=CORESET if a == 'mfn_cs10' else None, check_loo=args.stats_only)
                del fa
                os.makedirs(os.path.dirname(fn(a, rkey)), exist_ok=True)
                rec = {'stat_tr_index': tr_ix[stat], 'stats': res['stats'].half().cpu().numpy(),
                       'n_bank_clips': len(bank)}
                if 'test' in res:
                    rec.update(test_te_index=te_ix, test=res['test'].half().cpu().numpy())
                np.savez(fn(a, rkey), **rec)
                log[(a, rkey, c)] = {k: v for k, v in res.items() if not torch.is_tensor(v)}
                torch.cuda.empty_cache()
            return torch.cat(ce_all) if ce_all else None

        for r in args.runs:
            arms = [a for a in args.arms if (a, r) in todo]
            if not arms:
                continue
            m = models[r]

            def feat(w, x, y):
                f = m(w, x, y, ce)
                if 'imagenet' in arms:
                    f['imagenet'] = wrn(standardise(f['stack'], cst[r]))
                return f
            ce_now = run_arms(feat, arms, cached[r]['valid'], r)
            # P0 for this extraction: R0 must match ../ensemble-backend's cached arrays
            ref = np.concatenate([cached[r]['tr_ce'][tr_ix], cached[r]['te_ce'][te_ix]])
            err = float(np.abs(ce_now.numpy() - ref).max() / max(1e-6, np.abs(ref).max()))
            log[('ce_check', r, c)] = {'max_rel_err': err}
            assert err < 1e-3, f'{r} id{c}: R0 does not match the cached extraction ({err:.2e})'
        if args.imagenet_mel and ('imagenet_mel', 'none') in todo:
            run_arms(lambda w, x, y: {'imagenet_mel': wrn(standardise(x.expand(-1, 3, -1, -1), mst))},
                     ['imagenet_mel'], np.zeros(len(meta['tr_label']), dtype=bool), 'none')
        print(f'id{c:02d}: {n_tr} train + {len(te_ix)} test clips, load {t_load:.0f}s, '
              f'total {time.time() - t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB',
              flush=True)
        for k, v in log.items():
            if k[2] == c:
                print(f'    {k[0]:12s} {k[1][-22:]:22s} {v}', flush=True)
    json.dump({f'{a}|{r}|{c}': v for (a, r, c), v in log.items()},
              open(os.path.join(out, f'log_{os.getpid()}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
