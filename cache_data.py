"""Precompute waveforms and Log-Mels into .npy memmaps (ablation/fast-recipe).

    python cache_data.py [--out data/cache] [--workers 16]

Writes, for the training set (train_dataset's file order) and the dev test set
(test_dataset's order, all six machines concatenated):

    train_wav.npy  (N, 1, 160000) float32      devtest_wav.npy
    train_mel.npy  (N, 1, 128, 313) float32    devtest_mel.npy
    train_files.txt                            devtest_meta.csv  file,machine,id,label,y_true

The values are what dataloader.py computes per item in a DataLoader worker
(same load_wav, same single-threaded Wave2Mel on CPU). Each array is written to
a .tmp file and renamed only when complete; an existing array is kept.

On smartlab1, run `--verify` after a build: the host's RAM flips bits, and a
verify pass recomputes every item and rewrites any that differ. dataloader.py
also range-checks each cached item as it reads it.
"""
import argparse
import csv
import os
from multiprocessing import Pool

import numpy as np
import torch

from dataloader import Wave2Mel, load_wav, test_index, train_dataset

NAME_LIST = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
WAV_SHAPE, MEL_SHAPE = (1, 160000), (1, 128, 313)

_mel = None


def _load(path):
    global _mel
    if _mel is None:
        torch.set_num_threads(1)
        _mel = Wave2Mel()
    y, sr = load_wav(path)
    m = _mel(y)
    assert tuple(y.shape) == WAV_SHAPE and tuple(m.shape) == MEL_SHAPE, (path, y.shape, m.shape)
    return y.numpy(), m.numpy()


def write_arrays(files, out, prefix, workers):
    wav_p, mel_p = os.path.join(out, f'{prefix}_wav.npy'), os.path.join(out, f'{prefix}_mel.npy')
    if os.path.exists(wav_p) and os.path.exists(mel_p):
        print(f'{prefix}: exists, kept')
        return
    n = len(files)
    wav = np.lib.format.open_memmap(wav_p + '.tmp', 'w+', np.float32, (n,) + WAV_SHAPE)
    mel = np.lib.format.open_memmap(mel_p + '.tmp', 'w+', np.float32, (n,) + MEL_SHAPE)
    with Pool(workers) as pool:
        for i, (y, m) in enumerate(pool.imap(_load, files, chunksize=32)):
            wav[i], mel[i] = y, m
            if i % 2000 == 0:
                print(f'{prefix}: {i}/{n}', flush=True)
    wav.flush(), mel.flush()
    del wav, mel
    os.replace(wav_p + '.tmp', wav_p)
    os.replace(mel_p + '.tmp', mel_p)
    print(f'{prefix}: wrote {n}')


def verify(files, out, prefix, workers):
    """Recompute every item and compare bit for bit; rewrite any that differ.

    The first build left 2 of 36,283 training mels corrupt (|x| up to 1.3e11, wav
    intact, a recompute is normal), so a cache is not trusted until a full pass
    reproduces it exactly. Returns the indices that were repaired.
    """
    wav = np.load(os.path.join(out, f'{prefix}_wav.npy'), mmap_mode='r+')
    mel = np.load(os.path.join(out, f'{prefix}_mel.npy'), mmap_mode='r+')
    fixed = []
    with Pool(workers) as pool:
        for i, (y, m) in enumerate(pool.imap(_load, files, chunksize=32)):
            if not (np.array_equal(wav[i], y) and np.array_equal(mel[i], m)):
                print(f'{prefix}[{i}] {files[i]}: cached |mel| max {np.abs(mel[i]).max():.4g}, '
                      f'recomputed {np.abs(m).max():.4g}, wav equal {np.array_equal(wav[i], y)}', flush=True)
                fixed.append(i)
                wav[i], mel[i] = y, m
    wav.flush(), mel.flush()
    print(f'{prefix}: verified {len(files)}, repaired {len(fixed)}')
    return fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/cache')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--verify', action='store_true', help='recompute all items and repair mismatches')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ds = train_dataset('data/dataset', NAME_LIST, {})
    with open(os.path.join(a.out, 'train_files.txt'), 'w') as f:
        f.write('\n'.join(ds.unrolled_files_list) + '\n')
    write_arrays(ds.unrolled_files_list, a.out, 'train', a.workers)
    if a.verify:
        verify(ds.unrolled_files_list, a.out, 'train', a.workers)

    rows = test_index('data/dataset', NAME_LIST)
    with open(os.path.join(a.out, 'devtest_meta.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'machine', 'id', 'label', 'y_true'])
        w.writerows(rows)
    write_arrays([r[0] for r in rows], a.out, 'devtest', a.workers)
    if a.verify:
        verify([r[0] for r in rows], a.out, 'devtest', a.workers)


if __name__ == '__main__':
    main()
