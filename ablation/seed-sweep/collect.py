"""Collect the sweep's eval CSVs into per-arm mean/std tables.

Reads ablation/seed-sweep/scores/SCLTFSTgramMFN_sweep_<arm>_s<seed>/eval_{dev,eval}.csv
and writes scores/summary.csv (one row per arm x dataset x metric) plus a
per-machine table. Nothing is hardcoded: arms and seeds come from jobs.txt.
"""
import csv, os, sys, statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SWEEP = os.path.join(ROOT, 'ablation', 'seed-sweep')
sys.path.insert(0, os.path.join(ROOT, 'ablation'))
from scores_path import scores_csv  # noqa: E402
jobs = [l.split() for l in open(os.path.join(SWEEP, 'jobs.txt')) if l.strip()]

# overall[(arm, dataset)][metric] -> list over seeds
overall = defaultdict(lambda: defaultdict(list))
machine = defaultdict(lambda: defaultdict(list))   # [(arm, dataset, machine)][metric]
missing = []

for arm, seed, fussion, freeze in jobs:
    for ds in ('dev', 'eval'):
        path = scores_csv(f'SCLTFSTgramMFN_sweep_{arm}_s{seed}', ds)
        if not os.path.exists(path):
            missing.append(f'{arm} s{seed} {ds}')
            continue
        for row in csv.DictReader(open(path)):
            vals = {}
            for m in ('AUC', 'pAUC', 'mAUC'):
                if row[m] not in ('', None):
                    v = float(row[m])
                    vals[m] = v * 100 if v <= 1.0 else v   # CSVs store fractions
            if row['scope'] == 'overall':
                for m, v in vals.items():
                    overall[(arm, ds)][m].append(v)
            elif row['scope'] == 'machine':
                for m, v in vals.items():
                    machine[(arm, ds, row['machine'])][m].append(v)

def agg(vs):
    if not vs:
        return '', '', 0
    return (round(st.mean(vs), 3),
            round(st.stdev(vs), 3) if len(vs) > 1 else 0.0,
            len(vs))

os.makedirs(os.path.join(SWEEP, 'scores'), exist_ok=True)
with open(os.path.join(SWEEP, 'scores', 'summary.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['arm', 'dataset', 'metric', 'mean', 'std', 'n'])
    for (arm, ds), d in sorted(overall.items()):
        for m in ('AUC', 'pAUC', 'mAUC'):
            mean, sd, n = agg(d[m])
            w.writerow([arm, ds, m, mean, sd, n])

with open(os.path.join(SWEEP, 'scores', 'per_machine.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['arm', 'dataset', 'machine', 'metric', 'mean', 'std', 'n'])
    for (arm, ds, mc), d in sorted(machine.items()):
        for m in ('AUC', 'pAUC', 'mAUC'):
            mean, sd, n = agg(d[m])
            w.writerow([arm, ds, mc, m, mean, sd, n])

print(f'{"arm":<12}{"set":<6}{"AUC":>18}{"pAUC":>18}{"mAUC":>18}')
for (arm, ds), d in sorted(overall.items()):
    cells = []
    for m in ('AUC', 'pAUC', 'mAUC'):
        mean, sd, n = agg(d[m])
        cells.append(f'{mean:.2f} +/- {sd:.2f}' if n else '--')
    n = len(d['AUC'])
    print(f'{arm:<12}{ds:<6}' + ''.join(f'{c:>18}' for c in cells) + f'   n={n}')
if missing:
    print(f'\nmissing ({len(missing)}): ' + ', '.join(missing))
