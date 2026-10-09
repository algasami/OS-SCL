"""Frame-level redundancy of the TFgram branch, across training.

For each run: the exact init train.py started from (replayed), every student
snapshot (`student_e<N>.pth`, saved by --student_snapshots) and the EMA
checkpoint (`model.pth`, the deliverable). Each state is probed on the same
656 training clips (16 per machine ID) as ../tfgram-redesign.

Frame level only (N*313 rows x 128 dims). Clip-level CKA is biased toward 1
at N << D (../tfgram-redesign, "A correction to the metric").

  cka/<A>~TFgram   linear CKA, A in Log-Mel, Tgram, Tgram-front
  r2/<A>->TFgram   share of TFgram's variance linearly predictable from A
  r2/TFgram->Tgram-front

Tgram-front is Tgram's `conv_extrctor` alone (a linear filterbank, like
linear_avg's stack). Reference: two independently seeded Tgram front ends
predict each other at R^2 0.86 (../tfgram-redesign/scores/tgram_probe.json).

    python ablation/tfgram-linear/branch_probe.py <gpu> <out.json> <model_name>...

The init is replayed in train.py's RNG order: random_seed -> train_dataset ->
dataset_split (random_split consumes the torch RNG) -> model -> model_ema.
WeightEMA.__init__ copies ema_net into the student, so model_ema's draw is the
init the run actually starts from.
"""
import glob, json, os, random, re, sys
import numpy as np
import torch

sys.path.insert(0, '.')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from model.net import SCLTFSTgramMFN
from dataloader import train_dataset
from utils import dataset_split
from scores_path import scores_dir

DEV = torch.device(f'cuda:{int(sys.argv[1])}')
OUT = sys.argv[2]
NAMES = sys.argv[3:]
NAME_LIST = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']


def random_seed(seed):  # same as train.py
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cka(X, Y):
    """linear CKA, X/Y are N x D (rows = examples)"""
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    return ((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm())).item()


def lin_r2(X, Y):
    """share of Y's frame-wise variance that is a linear function of X"""
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    X1 = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype)], 1)
    W = torch.linalg.lstsq(X1, Y).solution
    return (1 - ((Y - X1 @ W) ** 2).sum() / (Y ** 2).sum()).item()


def frames(net, wavs, mel):
    net.eval()
    TG, FR, TF = [], [], []
    with torch.no_grad():
        for s in range(0, len(wavs), 32):
            w = wavs[s:s + 32].to(DEV)
            TG.append(net.tgramnet(w).cpu())
            FR.append(net.tgramnet.conv_extrctor(w).cpu())
            TF.append(net.TFgramNet(w, True).cpu())
    reps = {'Log-Mel': mel, 'Tgram': torch.cat(TG), 'Tgram-front': torch.cat(FR),
            'TFgram': torch.cat(TF)}
    return {k: v.double().permute(0, 2, 1).reshape(-1, 128) for k, v in reps.items()}


def measure(f):
    out = {f'cka/{a}~TFgram': cka(f[a], f['TFgram']) for a in ('Log-Mel', 'Tgram', 'Tgram-front')}
    for a in ('Log-Mel', 'Tgram', 'Tgram-front'):
        out[f'r2/{a}->TFgram'] = lin_r2(f[a], f['TFgram'])
    out['r2/TFgram->Tgram-front'] = lin_r2(f['TFgram'], f['Tgram-front'])
    out['tfgram_std'] = f['TFgram'].std().item()
    out['tfgram_frac_neg'] = (f['TFgram'] < 0).double().mean().item()
    return out


results = {}
if os.path.exists(OUT):
    with open(OUT) as fh:
        results = json.load(fh)

cache = None
for name in NAMES:
    with open(os.path.join(scores_dir(name), 'config.json')) as fh:
        cfg = json.load(fh)
    seed = cfg.get('seed', 2024)

    random_seed(seed)
    ds = train_dataset('data/dataset', NAME_LIST, cfg)
    dataset_split(ds)
    SCLTFSTgramMFN(num_classes=41, m=cfg['m'], cfg=cfg)
    net = SCLTFSTgramMFN(num_classes=41, m=cfg['m'], cfg=cfg).to(DEV)

    if cache is None:  # the clips are the same for every run
        labels = ds.labels.numpy()
        rng = np.random.default_rng(0)  # same subsample as ../tfgram-redesign
        idx = np.concatenate([rng.choice(np.where(labels == c)[0], 16, replace=False)
                              for c in np.unique(labels)])
        cache = (torch.stack([ds[i][0] for i in idx]),
                 torch.stack([ds[i][1] for i in idx]).squeeze(1))
    wavs, mel = cache

    ckpt_dir = f'check_points/{name}'
    snaps = sorted(glob.glob(f'{ckpt_dir}/student_e*.pth'),
                   key=lambda p: int(re.search(r'student_e(\d+)', p).group(1)))
    states = [('init', None)] + [(os.path.basename(p)[:-4], p) for p in snaps] + \
             [('ema_best', f'{ckpt_dir}/model.pth')]

    r = {'tf_act': cfg.get('tf_act', 'relu'), 'tf_pool': cfg.get('tf_pool', 'max'),
         'seed': seed, 'epoch': cfg.get('epoch'), 'states': {}}
    for label, path in states:
        if path is not None:
            net.load_state_dict(torch.load(path, map_location=DEV), strict=True)
        m = r['states'][label] = measure(frames(net, wavs, mel))
        print(f"{name:36s} {label:14s} cka mel {m['cka/Log-Mel~TFgram']:.3f}  tgram {m['cka/Tgram~TFgram']:.3f}"
              f"  front {m['cka/Tgram-front~TFgram']:.3f} | r2 front->TF {m['r2/Tgram-front->TFgram']:.3f}"
              f"  TF->front {m['r2/TFgram->Tgram-front']:.3f}  mel->TF {m['r2/Log-Mel->TFgram']:.3f}", flush=True)
    results[name] = r

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        json.dump(results, fh, indent=2)
