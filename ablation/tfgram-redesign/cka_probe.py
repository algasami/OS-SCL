"""CKA of the TFgram branch against Log-Mel and Tgram, at init and trained.

Generalises ../gram-branch-redundancy/cka_probe.py to any run: each run is
rebuilt from its own config.json (so tf_act/tf_pool are honoured), probed once
at the exact init train.py started it from and once at its checkpoint.

    python ablation/tfgram-redesign/cka_probe.py <gpu> <out.json> <model_name>...

The init is replayed in train.py's RNG order: random_seed -> train_dataset ->
dataset_split (random_split consumes the torch RNG) -> model -> model_ema.
WeightEMA.__init__ copies ema_net into the student, so model_ema's draw is the
init the run actually starts from.
"""
import json, os, random, sys
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
PAIRS = (('Log-Mel', 'TFgram'), ('Tgram', 'TFgram'), ('Log-Mel', 'Tgram'))


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


def branches(net, ds, idx):
    net.eval()
    MEL, TG, TF = [], [], []
    with torch.no_grad():
        for s in range(0, len(idx), 32):
            b = idx[s:s + 32]
            wav = torch.stack([ds[i][0] for i in b]).to(DEV)
            MEL.append(torch.stack([ds[i][1] for i in b]).squeeze(1))
            TG.append(net.tgramnet(wav).cpu())
            TF.append(net.TFgramNet(wav, True).cpu())
    return {'Log-Mel': torch.cat(MEL).double(), 'Tgram': torch.cat(TG).double(),
            'TFgram': torch.cat(TF).double()}


def measure(reps):
    clip = {k: v.reshape(v.shape[0], -1) for k, v in reps.items()}
    frame = {k: v.permute(0, 2, 1).reshape(-1, 128) for k, v in reps.items()}
    out = {}
    for a, b in PAIRS:
        out[f'cka_clip/{a}~{b}'] = cka(clip[a], clip[b])
        out[f'cka_frame/{a}~{b}'] = cka(frame[a], frame[b])
    out['r2/Log-Mel->TFgram'] = lin_r2(frame['Log-Mel'], frame['TFgram'])
    out['r2/Tgram->TFgram'] = lin_r2(frame['Tgram'], frame['TFgram'])
    out['tfgram_std'] = reps['TFgram'].std().item()
    out['tfgram_frac_neg'] = (reps['TFgram'] < 0).double().mean().item()
    return out


results = {}
if os.path.exists(OUT):
    with open(OUT) as f:
        results = json.load(f)

ds = None
for name in NAMES:
    with open(os.path.join(scores_dir(name), 'config.json')) as f:
        cfg = json.load(f)
    seed = cfg.get('seed', 2024)

    random_seed(seed)
    ds = train_dataset('data/dataset', NAME_LIST, cfg)
    dataset_split(ds)
    SCLTFSTgramMFN(num_classes=41, m=cfg['m'], cfg=cfg)
    init = SCLTFSTgramMFN(num_classes=41, m=cfg['m'], cfg=cfg).to(DEV)

    trained = SCLTFSTgramMFN(num_classes=41, m=cfg['m'], cfg=cfg).to(DEV)
    trained.load_state_dict(torch.load(f'check_points/{name}/model.pth', map_location=DEV), strict=True)

    # stratified subsample, ~16 clips per machine ID; fixed across runs
    labels = ds.labels.numpy()
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(labels == c)[0], 16, replace=False)
                          for c in np.unique(labels)])

    results[name] = {'tf_act': cfg.get('tf_act', 'relu'), 'tf_pool': cfg.get('tf_pool', 'max'),
                     'seed': seed, 'epoch': cfg.get('epoch'),
                     'init': measure(branches(init, ds, idx)),
                     'trained': measure(branches(trained, ds, idx))}
    r = results[name]
    print(f"{name:40s} {r['tf_act']:5s} {r['tf_pool']:3s} | clip CKA(mel,TF) init {r['init']['cka_clip/Log-Mel~TFgram']:.3f}"
          f" -> {r['trained']['cka_clip/Log-Mel~TFgram']:.3f} | CKA(Tg,TF) {r['init']['cka_clip/Tgram~TFgram']:.3f}"
          f" -> {r['trained']['cka_clip/Tgram~TFgram']:.3f}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
