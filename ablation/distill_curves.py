"""Distil training curves out of the raw logs into the committed scores record.

A 300-epoch training log is ~10 MB, of which ~300 lines matter, and `.gitignore`
excludes `*.log` -- so the curves vanish the moment the logs are cleaned. This
writes `scores/<model_name>/curve.csv` (epoch, train_loss, train_acc,
valid_loss, valid_acc), a few KB per run, so training-curve figures stay
regenerable from a clone.

Usage (from the repo root):
    python ablation/distill_curves.py <ablation-name> [<ablation-name> ...]
    python ablation/distill_curves.py --all
"""
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAT = re.compile(r'EPOCH:\s*(\d+)\s*\|\s*Train_loss:\s*([\d.]+)\s*\|\s*'
                 r'Train_accuracy:\s*([\d.]+)\s*\|\s*Valid_loss:\s*([\d.]+)\s*\|\s*'
                 r'Valid_accuracy:\s*([\d.]+)')
HEADER = ['epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc']


def distil(log_path):
    txt = open(log_path, errors='ignore').read().replace('\r', '\n')
    rows = PAT.findall(txt)
    # a log may repeat an epoch if a run was resumed; last write wins
    return [list(r) for r in {int(r[0]): r for r in rows}.values()]


def run(ablation):
    root = os.path.join(HERE, ablation)
    results, scores = os.path.join(root, 'results'), os.path.join(root, 'scores')
    if not os.path.isdir(scores):
        print(f'{ablation}: no scores/ -- skipped')
        return
    for model in sorted(os.listdir(scores)):
        # the log name is not derivable from the model name, so match on the
        # descriptor: model_name is 'SCLTFSTgramMFN_<desc>'
        desc = model.replace('SCLTFSTgramMFN', '').lstrip('_')
        cands = [p for p in glob.glob(os.path.join(results, 'train_*.log'))
                 if desc and desc in os.path.basename(p)]
        if not cands:
            print(f'  {model}: no training log found -- skipped')
            continue
        rows = sorted(distil(cands[0]), key=lambda r: int(r[0]))
        if not rows:
            print(f'  {model}: log has no EPOCH lines -- skipped')
            continue
        out = os.path.join(scores, model, 'curve.csv')
        with open(out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerows(rows)
        print(f'  {model}: {len(rows)} epochs -> {os.path.relpath(out, os.path.dirname(HERE))}')


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if args == ['--all']:
        args = [d for d in sorted(os.listdir(HERE))
                if os.path.isdir(os.path.join(HERE, d, 'scores'))]
    for a in args:
        print(a)
        run(a)
