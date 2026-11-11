"""R0 two ways: score.py's R0 (extract.py's batched pass) against eval.py's own CSVs (batch size 1).

Reads scores/ only. Writes <src>/r0_crosscheck.csv and prints the largest gap. A run without eval.py CSVs is
skipped. This is a check on the extraction, not a decision: F1-F6 use score.py's numbers.

    python crosscheck.py                     fresh runs (after `run.sh score` and `run.sh evalpy`)
    python crosscheck.py scores/dryrun_p1    the dry run's phase-1 seeds against their archived eval.py CSVs
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'ablation'))
from scores_path import scores_csv  # noqa: E402

SRC = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else 'scores')


def main():
    with open(os.path.join(SRC, 'single.csv'), encoding='UTF-8') as f:
        rows = [r for r in csv.DictReader(f) if r['rule'] == 'R0']
    out, worst = [], 0.0
    for r in rows:
        try:
            with open(scores_csv(r['run'], r['dataset']), encoding='UTF-8') as f:
                o = [x for x in csv.DictReader(f) if x['scope'] == 'overall']
        except FileNotFoundError:
            o = []
        if not o:
            print('skip (no eval.py CSV):', r['run'], r['dataset'])
            continue
        line = [r['run'], r['dataset']]
        for k in ('AUC', 'pAUC', 'mAUC'):
            a, b = float(r[k]), 100 * float(o[0][k])
            worst = max(worst, abs(a - b))
            line += [f'{a:.4f}', f'{b:.3f}', f'{a - b:+.4f}']
        out.append(line)
    if not out:
        return print('nothing to compare')
    with open(os.path.join(SRC, 'r0_crosscheck.csv'), 'w', newline='', encoding='UTF-8') as f:
        w = csv.writer(f)
        w.writerow(['run', 'dataset'] + [f'{k}_{s}' for k in ('AUC', 'pAUC', 'mAUC') for s in ('score', 'evalpy', 'diff')])
        w.writerows(out)
    print(f'{len(out)} run x set pairs, max |score.py - eval.py| = {worst:.4f} points -> r0_crosscheck.csv')


if __name__ == '__main__':
    main()
