"""Which frequencies does a fully linear, avg-pooled TFgram respond to? (init only)

A linear stack followed by ~511x average pooling is a linear filter with a
low-pass of roughly 16000/511 ~ 31 Hz on its output. If its decorrelation from
Log-Mel and Tgram comes from discarding the audio band, removing the low band
from the input should remove most of its output variance.

Measures the share of output variance (eval mode, random init, 3 seeds) that
survives high-passing the input waveform at several cutoffs, for the linear
avg-pooled stack and, as a reference, the paper's ReLU/max stack.

    python ablation/tfgram-redesign/band_probe.py <gpu> <out.json>
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
import model.waveq as waveq
from dataloader import train_dataset

DEV = torch.device(f'cuda:{int(sys.argv[1])}')
OUT = sys.argv[2]
CUTOFFS = [0, 31, 100, 300, 1000]
SEEDS = [2024, 2025, 2026]
NAME_LIST = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
SR = 16000

ds = train_dataset('data/dataset', NAME_LIST, {})
labels = ds.labels.numpy()
rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(np.where(labels == c)[0], 4, replace=False) for c in np.unique(labels)])
wavs = torch.stack([ds[i][0] for i in idx])          # (N, 1, 160000)


def highpass(w, fc):
    if fc == 0:
        return w
    spec = torch.fft.rfft(w, dim=-1)
    freqs = torch.fft.rfftfreq(w.shape[-1], 1 / SR)
    spec[..., freqs < fc] = 0
    return torch.fft.irfft(spec, n=w.shape[-1], dim=-1)


results = []
for label, slope, pool in (('relu_max', 0.0, 'max'), ('linear_avg', 1.0, 'avg')):
    waveq.tf_act = lambda x, act, a=slope: F.leaky_relu(x, a)
    for seed in SEEDS:
        torch.manual_seed(seed)
        net = waveq.TFgram(41, act='leaky', pool=pool).to(DEV).eval()
        out = {}
        with torch.no_grad():
            for fc in CUTOFFS:
                y = torch.cat([net(highpass(wavs[s:s + 16], fc).to(DEV), True).cpu()
                               for s in range(0, len(idx), 16)]).double()
                out[fc] = (y - y.mean((0, 2), keepdim=True)).pow(2).mean().item()
        row = {'branch': label, 'seed': seed,
               'var_share_surviving': {str(fc): out[fc] / out[0] for fc in CUTOFFS}}
        results.append(row)
        print(label, seed, ' '.join(f'hp{fc}Hz={v:.4f}' for fc, v in row['var_share_surviving'].items()), flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump({'note': 'Share of TFgram output variance (per-channel-centred) that survives high-passing '
                       'the input at each cutoff (Hz). Random init, eval mode, 164 clips (4 per ID).',
               'results': results}, f, indent=2)
