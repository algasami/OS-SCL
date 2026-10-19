"""Compute cost per input configuration (README, "Cost").

    python ablation/no-logmel/bench.py --gpu 1     # writes scores/bench.csv and scores/bench.txt

Three parts, each config in a fresh subprocess so cudnn flags, thread counts and
the caching allocator don't leak between them:

  train   training step, batch 64, random inputs, forward + ArcFace CE + backward
          + AdamW under the fast recipe (lean, cudnn.benchmark, fp16 autocast).
          The same harness as ../fast-recipe/bench.py, so rows are comparable.
  feat    per-clip cost of what every config pays before the network: decoding a
          real wav, and Wave2Mel (dataloader.py's exact transform) on the CPU at 1
          and 8 threads and on the GPU at batch 64.
  infer   network forward in eval mode, no grad, ArcFace logits included: GPU
          batch 64 (fp32 and fp16), GPU batch 1 (eval.py's path, train=False),
          CPU batch 1 at 1 and 8 threads. Log-Mel is an input here, not computed:
          add a `feat` row to get a config's end-to-end cost.

Times are medians over repeats after warm-up, in ms per clip.
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
MODES = [1, 4, 2, 5, 6, 7]   # with Log-Mel first, then the mel-free arms
FIELDS = ['part', 'what', 'fussion', 'device', 'threads', 'batch', 'precision', 'ms_per_clip', 'it_s', 'params_M']


def med_time(fn, reps, sync):
    for _ in range(3):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2]


def emit(**row):
    print('ROW ' + ','.join(str(row.get(k, '')) for k in FIELDS), flush=True)


def net_for(fussion, dev):
    from model.net import SCLTFSTgramMFN
    cfg = {'fussion': fussion, 'ht': 'basic', 'm': 0.4, 'lean': True}
    net = SCLTFSTgramMFN(num_classes=41, m=0.4, cfg=cfg).to(dev)
    return net, sum(p.numel() for p in net.parameters()) / 1e6


def part_train(fussion, gpu):
    import torch
    from losses import ASDLoss
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    dev = torch.device(f'cuda:{gpu}')
    net, params = net_for(fussion, dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    crit = ASDLoss().to(dev)
    w = torch.randn(64, 1, 160000, device=dev) * 0.1
    m = torch.randn(64, 1, 128, 313, device=dev) * 10
    y = torch.randint(0, 41, (64,), device=dev)
    scaler = torch.amp.GradScaler('cuda')

    def step():
        with torch.autocast('cuda', dtype=torch.float16):
            logits, _ = net(w, m, y)
        opt.zero_grad()
        scaler.scale(crit(logits, y)).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    t, n = time.time(), 40
    for _ in range(n):
        step()
    torch.cuda.synchronize()
    it_s = n / (time.time() - t)
    emit(part='train', what='step', fussion=fussion, device='gpu', batch=64, precision='fp16',
         ms_per_clip=f'{1000 / it_s / 64:.4f}', it_s=f'{it_s:.2f}', params_M=f'{params:.4f}')


def part_feat(gpu):
    import torch
    from dataloader import Wave2Mel, load_wav
    files = sorted(glob.glob(os.path.join(ROOT, 'data/dataset/fan/train/*.wav')))[:32]
    i = [0]

    def decode():
        load_wav(files[i[0] % len(files)])
        i[0] += 1
    torch.set_num_threads(1)
    emit(part='feat', what='decode', device='cpu', threads=1, batch=1, precision='fp32',
         ms_per_clip=f'{1000 * med_time(decode, 64, lambda: None):.4f}')

    w2m = Wave2Mel()
    x = load_wav(files[0])[0]                                      # (1, 160000)
    for th in (1, 8):
        torch.set_num_threads(th)
        emit(part='feat', what='logmel', device='cpu', threads=th, batch=1, precision='fp32',
             ms_per_clip=f'{1000 * med_time(lambda: w2m(x), 64, lambda: None):.4f}')
    torch.set_num_threads(8)
    xb = torch.stack([load_wav(f)[0] for f in files] * 2)          # (64, 1, 160000)
    emit(part='feat', what='logmel', device='cpu', threads=8, batch=64, precision='fp32',
         ms_per_clip=f'{1000 * med_time(lambda: w2m(xb), 10, lambda: None) / 64:.4f}')

    dev = torch.device(f'cuda:{gpu}')
    w2m.mel_transform.to(dev)
    w2m.amplitude_to_db.to(dev)
    xg = xb.to(dev)
    sync = torch.cuda.synchronize
    emit(part='feat', what='logmel', device='gpu', batch=64, precision='fp32',
         ms_per_clip=f'{1000 * med_time(lambda: w2m(xg), 30, sync) / 64:.4f}')
    # sanity: the GPU mel is the CPU mel (README of ../lit-arms: ~1e-5 dB)
    diff = (w2m(xg[:4]).cpu() - Wave2Mel()(xb[:4])).abs().max().item()
    print(f'gpu-vs-cpu logmel max |diff| = {diff:.2e} dB', flush=True)


def part_infer(fussion, gpu, device, threads):
    import torch
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(threads)
    if device == 'gpu':
        dev = torch.device(f'cuda:{gpu}')
        sync = torch.cuda.synchronize
    else:
        dev = torch.device('cpu')
        sync = lambda: None   # noqa: E731
    net, params = net_for(fussion, dev)
    net.eval()

    def run(b, amp, reps):
        w = torch.randn(b, 1, 160000, device=dev) * 0.1
        m = torch.randn(b, 1, 128, 313, device=dev) * 10
        y = torch.randint(0, 41, (b,), device=dev)

        @torch.no_grad()
        def fwd():
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                net(w, m, y, train=b > 1)
        return 1000 * med_time(fwd, reps, sync) / b

    if device == 'gpu':
        for b, amp, reps in ((64, False, 30), (64, True, 30), (1, False, 100)):
            emit(part='infer', what='net', fussion=fussion, device='gpu', batch=b,
                 precision='fp16' if amp else 'fp32', ms_per_clip=f'{run(b, amp, reps):.4f}',
                 params_M=f'{params:.4f}')
    else:
        emit(part='infer', what='net', fussion=fussion, device='cpu', threads=threads, batch=1,
             precision='fp32', ms_per_clip=f'{run(1, False, 20):.4f}', params_M=f'{params:.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--one', nargs='+')
    a = ap.parse_args()
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    if a.one:
        kind = a.one[0]
        if kind == 'train':
            part_train(int(a.one[1]), a.gpu)
        elif kind == 'feat':
            part_feat(a.gpu)
        elif kind == 'infer':
            part_infer(int(a.one[1]), a.gpu, a.one[2], int(a.one[3]))
        return

    import torch
    jobs = [['feat']] + [['train', str(f)] for f in MODES] + [['infer', str(f), 'gpu', '1'] for f in MODES] \
        + [['infer', str(f), 'cpu', str(t)] for t in (1, 8) for f in MODES]
    rows, notes = [], []
    for j in jobs:
        r = subprocess.run([sys.executable, __file__, '--gpu', str(a.gpu), '--one', *j],
                           capture_output=True, text=True, cwd=ROOT,
                           env={**os.environ, 'OMP_NUM_THREADS': j[3] if j[0] == 'infer' and j[2] == 'cpu' else '8'})
        out = [ln[4:] for ln in r.stdout.splitlines() if ln.startswith('ROW ')]
        notes += [ln for ln in r.stdout.splitlines() if not ln.startswith('ROW ') and ln.strip()]
        if r.returncode or not out:
            notes.append(f'{j}: FAILED {r.stderr.strip().splitlines()[-1:]}')
        rows += out
        for ln in out:
            print(ln, flush=True)
    os.makedirs(os.path.join(HERE, 'scores'), exist_ok=True)
    with open(os.path.join(HERE, 'scores', 'bench.csv'), 'w') as f:
        f.write(','.join(FIELDS) + '\n' + '\n'.join(rows) + '\n')
    cpu = subprocess.run("lscpu | grep -E '^Model name|^CPU\\(s\\)'", shell=True, capture_output=True, text=True)
    head = [f'{torch.cuda.get_device_name(a.gpu)}, torch {torch.__version__}, ' +
            ' / '.join(x.split(':', 1)[1].strip() for x in cpu.stdout.splitlines())]
    with open(os.path.join(HERE, 'scores', 'bench.txt'), 'w') as f:
        f.write('\n'.join(head + notes) + '\n')
        w = csv.DictReader(open(os.path.join(HERE, 'scores', 'bench.csv')))
        for r in w:
            f.write(f"{r['part']:5s} {r['what']:6s} f{r['fussion'] or '-':2s} {r['device']:3s} "
                    f"thr {r['threads'] or '-':2s} b {r['batch']:>2s} {r['precision']:4s} | "
                    f"{float(r['ms_per_clip']):9.3f} ms/clip" + (f" | {r['it_s']} it/s" if r['it_s'] else '')
                    + (f" | {r['params_M']} M" if r['params_M'] else '') + '\n')


if __name__ == '__main__':
    main()
