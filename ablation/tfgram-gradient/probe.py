"""Step 0 of the 'make TFgram learn' plan: does the TFgram branch receive any
usable learning signal, and did it actually move during training?

Two independent measurements:

  drift   -- compare the trained checkpoint against the EXACT random init it
             started from (reproducible: train.py seeds before constructing the
             model). Scale-invariant cosine similarity separates "shrunk by
             WeightEMA's weight decay" from "actually rotated by learning".
             Built-in control: TFgram's DEAD submodules (conv_block1..6, fc1,
             fc_audioset, bn0, pre_block4 -- allocated but never called by
             TFgram.forward) receive zero gradient, so their drift is the pure
             weight-decay baseline. Live TFgram ~ dead TFgram => learned nothing.

  grad    -- run real training steps and record, per branch:
               |g|         raw gradient norm
               |g|/|W|     relative gradient norm
               SNR         |mean g| / mean |g| over the window, in [0,1].
                           This is what AdamW actually responds to: Adam divides
                           by sqrt(v), so a consistent-but-tiny gradient still
                           yields a ~lr-sized step, while an inconsistent one
                           cancels in the momentum buffer and yields nothing.
               |dW|/|W|    the effective AdamW step actually taken.

Usage: python ablation/tfgram-gradient/probe.py --gpu 2 [--steps 200] [--seed 2024]
"""
import argparse, json, os, random, sys, warnings
import numpy as np
import torch

warnings.filterwarnings('ignore', category=UserWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from torch.utils.data import DataLoader
from dataloader import train_dataset
from utils import dataset_split, mixup_data, os_scl
from losses import ASDLoss, SupConLoss
from model.net import SCLTFSTgramMFN
import yaml

# TFgram submodules actually reached by TFgram.forward (model/waveq.py)
TF_LIVE = ('pre_conv0', 'pre_bn0', 'pre_block1', 'pre_block2', 'pre_block3')


def group_of(name):
    """Map a parameter name to a branch group."""
    if name.startswith('TFgramNet.'):
        sub = name[len('TFgramNet.'):].split('.')[0]
        return 'TFgram_live' if sub in TF_LIVE else 'TFgram_dead'
    if name.startswith('tgramnet.'):
        return 'Tgram'
    if name.startswith('mobilefacenet.'):
        return 'MobileFaceNet'
    if name.startswith('head.'):
        return 'head'
    if name.startswith('arcface.'):
        return 'arcface'
    return 'other'


def random_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build(cfg, device, seed):
    """Replicate train.py's construction order exactly, so the init is reproduced.

    train.py: random_seed -> train_dataset -> dataset_split -> model -> model_ema.
    dataset_split calls random_split, which consumes the torch RNG, so the
    dataset MUST be built here even though only the loader is returned.
    WeightEMA.__init__ then copies ema_net's weights INTO the student, so
    model_ema's init is the init the whole system actually starts from.
    """
    random_seed(seed)
    ds = train_dataset('data/dataset',
                       ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve'], cfg)
    train_ds, _ = dataset_split(ds)
    model = SCLTFSTgramMFN(num_classes=41, m=cfg['m'], cfg=cfg).to(device)
    model_ema = SCLTFSTgramMFN(num_classes=41, m=cfg['m'], cfg=cfg).to(device)
    init_state = {k: v.detach().clone() for k, v in model_ema.state_dict().items()}
    model.load_state_dict(init_state)          # what WeightEMA.__init__ does
    loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=cfg.get('num_workers', 8), persistent_workers=True)
    return model, init_state, loader


def measure_drift(trained, init):
    """Per-group cosine similarity, norm ratio and relative drift vs init."""
    acc = {}
    for k, w_t in trained.items():
        if k not in init or not w_t.dtype.is_floating_point:
            continue
        if k.endswith('running_mean') or k.endswith('running_var'):
            continue                            # buffers: not weights, handled separately
        g = group_of(k)
        w_i = init[k].float().flatten()
        w_t = w_t.float().flatten().to(w_i.device)
        a = acc.setdefault(g, dict(dot=0., ni=0., nt=0., nd=0., n=0))
        a['dot'] += torch.dot(w_t, w_i).item()
        a['ni'] += torch.dot(w_i, w_i).item()
        a['nt'] += torch.dot(w_t, w_t).item()
        a['nd'] += torch.dot(w_t - w_i, w_t - w_i).item()
        a['n'] += w_i.numel()
    out = {}
    for g, a in acc.items():
        ni, nt = a['ni'] ** .5, a['nt'] ** .5
        out[g] = dict(params=a['n'],
                      cosine=a['dot'] / (ni * nt + 1e-12),
                      norm_ratio=nt / (ni + 1e-12),
                      rel_drift=a['nd'] ** .5 / (ni + 1e-12))
    return out


def measure_grad(model, loader, device, steps, lr):
    criterion, suploss = ASDLoss().to(device), SupConLoss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    no_grad_groups = {}
    gsum = {n: torch.zeros_like(p) for n, p in named}
    step_norms, step_updates, wnorms = {}, {}, {}
    done = 0
    while done < steps:
        for x_wavs, x_mels, labels in loader:
            if done >= steps:
                break
            model.train()
            x_wavs, x_mels, labels = x_wavs.to(device), x_mels.to(device), labels.to(device)
            mx_w, mx_m, y_a, y_b, lam, _ = mixup_data(x_wavs, x_mels, labels, device)
            logits, feature = model(mx_w, mx_m, labels)
            loss = os_scl(criterion, logits, y_a, y_b, lam, suploss, feature)
            opt.zero_grad()
            loss.backward()

            before = {n: p.detach().clone() for n, p in named}
            per_group = {}
            for n, p in named:
                if p.grad is None:
                    no_grad_groups.setdefault(group_of(n), set()).add(n)
                    continue
                gsum[n] += p.grad.detach()
                g = group_of(n)
                d = per_group.setdefault(g, [0., 0.])
                d[0] += p.grad.detach().float().pow(2).sum().item()
                d[1] += p.detach().float().pow(2).sum().item()
            for g, (gs, ws) in per_group.items():
                step_norms.setdefault(g, []).append(gs ** .5)
                wnorms.setdefault(g, []).append(ws ** .5)

            opt.step()
            upd = {}
            for n, p in named:
                g = group_of(n)
                d = upd.setdefault(g, 0.)
                upd[g] = d + (p.detach() - before[n]).float().pow(2).sum().item()
            for g, v in upd.items():
                step_updates.setdefault(g, []).append(v ** .5)
            done += 1
            if done % 25 == 0:
                print(f'  step {done}/{steps}  loss {loss.item():.4f}', flush=True)

    out = {}
    for g in step_norms:
        mean_norm = float(np.mean(step_norms[g]))
        mean_w = float(np.mean(wnorms[g]))
        sq = sum((gsum[n] / done).float().pow(2).sum().item()
                 for n, _ in named if group_of(n) == g)
        norm_of_mean = sq ** .5
        mean_upd = float(np.mean(step_updates[g]))
        out[g] = dict(grad_norm=mean_norm,
                      rel_grad=mean_norm / (mean_w + 1e-12),
                      snr=norm_of_mean / (mean_norm + 1e-12),
                      rel_step=mean_upd / (mean_w + 1e-12))
    out['_noise_floor'] = 1.0 / done ** .5   # SNR of pure noise over this window
    out['_never_grad'] = {g: len(v) for g, v in no_grad_groups.items()}
    return out


def show(title, rows, cols):
    print(f'\n=== {title} ===')
    w = max(len(r) for r in rows)
    print('  ' + 'group'.ljust(w) + ''.join(c.rjust(13) for c, _ in cols))
    for r in sorted(rows, key=lambda x: (x != 'TFgram_live', x)):
        print('  ' + r.ljust(w) + ''.join(f(rows[r]).rjust(13) for _, f in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=2)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--ckpt', default='check_points/SCLTFSTgramMFN_sweep_f1_s2024/model.pth')
    ap.add_argument('--out', default='ablation/tfgram-gradient/scores/probe.json')
    a = ap.parse_args()

    cfg = yaml.safe_load(open('config.yaml'))
    cfg.update(fussion=1, m=0.4, ht='basic', num_workers=8, seed=a.seed)
    device = torch.device(f'cuda:{a.gpu}')
    lr = float(cfg.get('lr', 1e-4))

    print(f'building model + dataset (seed {a.seed}) ...', flush=True)
    model, init_state, loader = build(cfg, device, a.seed)
    report = {'seed': a.seed, 'steps': a.steps, 'ckpt': a.ckpt, 'lr': lr}

    # ---- measurement 1: drift of the trained checkpoint from its own init ----
    trained = torch.load(a.ckpt, map_location=device)
    report['drift'] = measure_drift(trained, init_state)
    show('drift of trained checkpoint vs its random init', report['drift'],
         [('params', lambda d: f"{d['params']:,}"),
          ('cosine', lambda d: f"{d['cosine']:.4f}"),
          ('|Wt|/|Wi|', lambda d: f"{d['norm_ratio']:.4f}"),
          ('rel_drift', lambda d: f"{d['rel_drift']:.4f}")])

    # ---- measurement 2a: gradients at random init (epoch 0) ----
    print(f'\ngradient probe @ random init, {a.steps} steps ...', flush=True)
    report['grad_init'] = measure_grad(model, loader, device, a.steps, lr)

    # ---- measurement 2b: gradients at the converged checkpoint ----
    print(f'\ngradient probe @ trained checkpoint, {a.steps} steps ...', flush=True)
    model2 = SCLTFSTgramMFN(num_classes=41, m=cfg['m'], cfg=cfg).to(device)
    model2.load_state_dict(trained)
    report['grad_trained'] = measure_grad(model2, loader, device, a.steps, lr)

    cols = [('|g|', lambda d: f"{d['grad_norm']:.3e}"),
            ('|g|/|W|', lambda d: f"{d['rel_grad']:.3e}"),
            ('SNR', lambda d: f"{d['snr']:.4f}"),
            ('|dW|/|W|', lambda d: f"{d['rel_step']:.3e}")]
    for tag in ('grad_init', 'grad_trained'):
        nf = report[tag]['_noise_floor']
        show(f'{tag}  (pure-noise SNR floor = {nf:.4f})',
             {k: v for k, v in report[tag].items() if not k.startswith('_')}, cols)
        ng = report[tag]['_never_grad']
        if ng:
            print('  tensors that NEVER received a gradient: '
                  + ', '.join(f'{g}={n}' for g, n in sorted(ng.items())))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(report, open(a.out, 'w'), indent=2)
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
