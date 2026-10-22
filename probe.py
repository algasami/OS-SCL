"""Dev-test probe: eval.py's metrics, batched, from the cache_data.py arrays.

Used by trainer.py's --probe_every (a diagnostic during training, never read by
checkpoint selection) and by ablation gates. The score is eval.py's R0: the
per-clip cross-entropy of the margin ArcFace logits against the clip's own ID.
"""
import csv
import os

import numpy as np
import torch
from sklearn import metrics

from dataloader import Wave2Mel, cached_item_ok, load_wav
from losses import ASDLoss


def load_devtest(cache_dir):
    """(wav memmap, mel memmap, meta rows) for the dev test set."""
    with open(os.path.join(cache_dir, 'devtest_meta.csv')) as f:
        meta = list(csv.DictReader(f))
    return (np.load(os.path.join(cache_dir, 'devtest_wav.npy'), mmap_mode='r'),
            np.load(os.path.join(cache_dir, 'devtest_mel.npy'), mmap_mode='r'), meta)


@torch.no_grad()
def devtest_metrics(net, data, device, amp_dtype=None):
    """Per machine AUC / pAUC (max FPR 0.1) / mAUC (min per-ID AUC), and their means.

    Batched through the train=True path, which differs from eval.py's batch-1
    train=False path only in TFgram's reshape (G2: per-machine AUC within 1.3e-5).
    Draws no random numbers, so a probe does not perturb training.
    """
    wav, mel, meta = data
    net = net.eval()
    criterion = ASDLoss(reduction=False)
    labels = torch.tensor([int(r['label']) for r in meta])
    scores = []
    # chunk sizes differ by at most 1 and are all >= 2: TFgram's squeeze() breaks on batch 1
    for idx in np.array_split(np.arange(len(meta)), -(-len(meta) // 64)):
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        w, m = np.array(wav[lo:hi]), np.array(mel[lo:hi])
        for k in range(len(w)):
            if not cached_item_ok(w[k], m[k]):   # host-RAM bit flip: recompute from the file
                y, _ = load_wav(meta[lo + k]['file'])
                w[k], m[k] = y.numpy(), Wave2Mel()(y).numpy()
        y = labels[lo:hi].to(device)
        with torch.autocast('cuda', dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            logits, _ = net(torch.from_numpy(w).to(device), torch.from_numpy(m).to(device), y, train=True)
        scores.append(criterion(logits.float(), y).cpu())
    scores = torch.cat(scores).numpy()

    out, aucs, paucs, maucs = {}, [], [], []
    for machine in dict.fromkeys(r['machine'] for r in meta):
        sel = np.array([i for i, r in enumerate(meta) if r['machine'] == machine])
        yt = np.array([int(meta[i]['y_true']) for i in sel])
        ys = scores[sel]
        ids = np.array([meta[i]['id'] for i in sel])
        per_id = [metrics.roc_auc_score(yt[ids == i], ys[ids == i]) for i in sorted(set(ids))]
        out[f'{machine}_AUC'] = metrics.roc_auc_score(yt, ys)
        out[f'{machine}_pAUC'] = metrics.roc_auc_score(yt, ys, max_fpr=0.1)
        out[f'{machine}_mAUC'] = min(per_id)
        aucs.append(out[f'{machine}_AUC']), paucs.append(out[f'{machine}_pAUC']), maucs.append(min(per_id))
    out.update(AUC=float(np.mean(aucs)), pAUC=float(np.mean(paucs)), mAUC=float(np.mean(maucs)))
    return out


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
