"""Training-step throughput per recipe (README, "Why this exists").

    python ablation/fast-recipe/bench.py --gpu 1      # writes scores/bench.txt

Random inputs at batch 64; forward + ArcFace CE + backward + AdamW. No data
loading, mixup, SupCon or EMA, so it isolates the network's compute: the real
loop is ~15-30% slower. Each config runs in a fresh subprocess, so cudnn flags
and the caching allocator don't leak between configs.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
CONFIGS = [  # (fussion, lean, cudnn, amp)
    (1, 0, 'det', 'none'),      # today's recipe
    (1, 0, 'bench', 'none'),
    (1, 0, 'bench', 'fp16'),
    (1, 1, 'bench', 'fp16'),    # fr_f1
    (4, 1, 'bench', 'fp16'),    # fr_f4
    (2, 1, 'bench', 'fp16'),
    (2, 0, 'det', 'none'),
]


def one(fussion, lean, cudnn, amp, gpu):
    sys.path.insert(0, ROOT)
    import torch
    from losses import ASDLoss
    from model.net import SCLTFSTgramMFN
    torch.backends.cudnn.deterministic = cudnn == 'det'
    torch.backends.cudnn.benchmark = cudnn == 'bench'
    dev = torch.device(f'cuda:{gpu}')
    cfg = {'fussion': fussion, 'ht': 'basic', 'm': 0.4, 'lean': bool(lean)}
    net = SCLTFSTgramMFN(num_classes=41, m=0.4, cfg=cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    crit = ASDLoss().to(dev)
    w = torch.randn(64, 1, 160000, device=dev) * 0.1
    m = torch.randn(64, 1, 128, 313, device=dev) * 10
    y = torch.randint(0, 41, (64,), device=dev)
    scaler = torch.amp.GradScaler('cuda', enabled=amp == 'fp16')

    def step():
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp == 'fp16'):
            logits, _ = net(w, m, y)
        opt.zero_grad()
        scaler.scale(crit(logits, y)).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    t, n = time.time(), 30
    for _ in range(n):
        step()
    torch.cuda.synchronize()
    params = sum(p.numel() for p in net.parameters())
    print(f'fussion {fussion}  lean {lean}  cudnn {cudnn:5s}  amp {amp:4s} | {n / (time.time() - t):5.2f} it/s'
          f' | {params / 1e6:6.1f} M params | peak {torch.cuda.max_memory_allocated(dev) / 2**30:4.1f} GB')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--one', nargs=4)
    a = ap.parse_args()
    if a.one:
        f, lean, cudnn, amp = a.one
        one(int(f), int(lean), cudnn, amp, a.gpu)
        return
    import torch
    lines = [f'{torch.cuda.get_device_name(a.gpu)}, torch {torch.__version__}, batch 64, random inputs']
    for c in CONFIGS:
        r = subprocess.run([sys.executable, __file__, '--gpu', str(a.gpu), '--one', *map(str, c)],
                           capture_output=True, text=True, cwd=ROOT)
        lines.append(r.stdout.strip() or f'{c}: FAILED {r.stderr.strip().splitlines()[-1:]}')
        print(lines[-1], flush=True)
    with open(os.path.join(HERE, 'scores', 'bench.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
