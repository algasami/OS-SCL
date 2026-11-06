"""Full training-step throughput with and without --permix.

    python ablation/permix/bench.py --gpu 2   # writes scores/bench.txt

Unlike ../spectrum-branch/bench.py this times Trainer.train itself, because PerMix
changes the mixing and the loss, not the network: wave_aug noise,shift, mixing,
forward, loss, backward, AdamW, EMA, all at batch 64 under fp16 + cudnn.benchmark,
lean, fussion 1. Inputs are random tensors on the CPU (no decoding), so the H2D copy
is included and identical in both arms. One untimed warm-up epoch, then one timed
epoch of N batches (+1 validation batch). Each config runs in a fresh subprocess.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))


def one(permix, gpu, n):
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from model.net import SCLTFSTgramMFN
    from trainer import Trainer
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = False, True
    dev = torch.device(f'cuda:{gpu}')
    desc = f'bench_tmp_{os.getpid()}'
    cfg = dict(m=0.4, gpu_num=gpu, epoch=1, batch_size=64, fussion=1, ht='basic', desc=desc, lean=1,
               amp='fp16', wave_aug=1, wave_aug_ops='noise,shift', permix=permix)
    b = 64
    w, m = torch.randn((n + 1) * b, 1, 160000) * 0.03, torch.randn((n + 1) * b, 1, 128, 313) * 10
    y = torch.randint(0, 41, ((n + 1) * b,))
    tr = DataLoader(TensorDataset(w[:-b], m[:-b], y[:-b]), batch_size=b)
    va = DataLoader(TensorDataset(w[-b:], m[-b:], y[-b:]), batch_size=b)
    net = SCLTFSTgramMFN(num_classes=41, m=0.4, cfg=cfg).to(dev)
    ema = SCLTFSTgramMFN(num_classes=41, m=0.4, cfg=cfg).to(dev)
    for p in ema.parameters():
        p.detach_()
    tr_ = Trainer(device=dev, net=net, ema_net=ema, epochs=1, cfg=cfg)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        tr_.train(tr, va)                      # warm-up (cudnn autotune, allocator)
        torch.cuda.synchronize()
        t = time.time()
        tr_.train(tr, va)
        torch.cuda.synchronize()
        dt = time.time() - t
    shutil.rmtree(f'check_points/SCLTFSTgramMFN_{desc}', ignore_errors=True)
    print(f'permix {permix} | {n / dt:5.2f} it/s ({n} steps + 1 valid batch) '
          f'| peak {torch.cuda.max_memory_allocated(dev) / 2**30:4.1f} GB')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=2)
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--one', type=int)
    a = ap.parse_args()
    if a.one is not None:
        one(a.one, a.gpu, a.n)
        return
    import torch
    lines = [f'{torch.cuda.get_device_name(a.gpu)} (GPU {a.gpu}), torch {torch.__version__}, batch 64, '
             f'random inputs, full Trainer step (wave_aug noise,shift, fp16, cudnn.benchmark, lean)']
    for rep in range(2):
        for permix in (0, 1):
            r = subprocess.run([sys.executable, __file__, '--gpu', str(a.gpu), '--n', str(a.n), '--one', str(permix)],
                               capture_output=True, text=True, cwd=ROOT)
            lines.append(f'rep {rep} ' + (r.stdout.strip() or f'permix {permix}: FAILED {r.stderr.strip().splitlines()[-1:]}'))
            print(lines[-1], flush=True)
    with open(os.path.join(HERE, 'scores', 'bench.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
