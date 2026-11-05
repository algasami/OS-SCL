"""spectrum-branch summary: sb_spec vs ao_nogain, seed-paired, R0 (eval.py CSVs).

    /home/f74134118/anaconda3/bin/python ablation/spectrum-branch/score.py

Reads only scores/ (sb_spec here, ao_nogain from ../aug-ops/scores via
scores_path). Prints 4-seed means, paired deltas with their se, the per-seed
pump deltas and the D1 checklist from the README. Seeds without a scored
sb_spec run are skipped, so it runs on a partial set; D1 is only evaluated
when all four stage-1 seeds are present.
"""
import csv
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from scores_path import scores_csv  # noqa: E402

MACHINES = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
STAGE1 = [2040, 2041, 2042, 2043]
PAPER = {'AUC': 95.71, 'pAUC': 90.23, 'mAUC': 91.23, 'pump': 96.25}


def load(name, ds):
    """{(machine, id, scope): (AUC, pAUC, mAUC)} in points, or None if not scored."""
    try:
        path = scores_csv(name, ds)
    except FileNotFoundError:
        return None
    if not os.path.exists(path):
        return None
    out = {}
    for r in csv.DictReader(open(path)):
        out[(r['machine'], r['id'], r['scope'])] = tuple(100 * float(r[k]) if r[k] else None
                                                         for k in ('AUC', 'pAUC', 'mAUC'))
    return out


def metrics(d):
    """The numbers the README's predictions and rules are stated in."""
    o = d[('ALL', 'ALL', 'overall')]
    m = {'AUC': o[0], 'pAUC': o[1], 'mAUC': o[2], 'mean': (o[0] + o[1]) / 2}
    for mac in MACHINES:
        m[mac] = d[(mac, 'ALL', 'machine')][0]
    for (mac, i, scope), v in d.items():
        if scope == 'id' and mac in ('pump', 'ToyConveyor', 'ToyCar'):
            m[f'{mac} {i}'] = v[0]
    return m


def mean_se(xs):
    return st.mean(xs), (st.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else float('nan'))


def main():
    seeds = sorted(int(s) for s in os.environ.get('SB_SEEDS', ' '.join(map(str, STAGE1))).split())
    res = {}
    for ds in ('dev', 'eval'):
        pairs = []
        for s in seeds:
            a, b = load(f'SCLTFSTgramMFN_sb_spec_s{s}', ds), load(f'SCLTFSTgramMFN_ao_nogain_s{s}', ds)
            if a is None:
                continue
            if b is None:
                print(f'!! no ao_nogain partner for seed {s} ({ds}); skipped')
                continue
            pairs.append((s, metrics(a), metrics(b)))
        res[ds] = pairs
        if not pairs:
            print(f'== {ds}: no scored sb_spec runs yet')
            continue
        print(f'== {ds}, R0, seeds {[p[0] for p in pairs]}: sb_spec | ao_nogain | paired delta (se)')
        keys = [k for k in pairs[0][1] if all(k in p[1] and k in p[2] for p in pairs)]
        for k in keys:
            a = [p[1][k] for p in pairs]
            b = [p[2][k] for p in pairs]
            d, se = mean_se([x - y for x, y in zip(a, b)])
            print(f'  {k:18s} {st.mean(a):7.2f} | {st.mean(b):7.2f} | {d:+6.2f} ({se:.2f})')
        if ds == 'dev':
            per = ' '.join(f'{p[1]["pump"] - p[2]["pump"]:+.2f}' for p in pairs)
            print(f'  per-seed dev pump delta: {per}')

    dev, ev = res['dev'], res['eval']
    if [p[0] for p in dev] != STAGE1 or [p[0] for p in ev] != STAGE1:
        print('D1: not evaluated (needs all four stage-1 seeds on dev and eval)')
        return

    def delta(pairs, k):
        return st.mean(p[1][k] - p[2][k] for p in pairs)

    pump_d = delta(dev, 'pump')
    pump_pos = sum(p[1]['pump'] > p[2]['pump'] for p in dev)
    checks = [
        ('(a) dev pump delta >= +0.40', pump_d >= 0.40, f'{pump_d:+.2f}'),
        ('(a) dev pump delta > 0 on >= 3/4 seeds', pump_pos >= 3, f'{pump_pos}/4'),
        ('(b) dev AUC delta >= -0.10', delta(dev, 'AUC') >= -0.10, f'{delta(dev, "AUC"):+.2f}'),
        ('(b) dev mean(AUC,pAUC) delta >= -0.10', delta(dev, 'mean') >= -0.10, f'{delta(dev, "mean"):+.2f}'),
        ('(c) eval mean(AUC,pAUC) delta >= -0.20', delta(ev, 'mean') >= -0.20, f'{delta(ev, "mean"):+.2f}'),
        ('(d) dev ToyConveyor delta >= -0.80', delta(dev, 'ToyConveyor') >= -0.80,
         f'{delta(dev, "ToyConveyor"):+.2f}'),
    ]
    for name, ok, val in checks:
        print(f'  {name:42s} {val:>7s}  {"ok" if ok else "FAIL"}')
    print('D1:', 'PASS -> stage 2 (fresh seeds 2044-2047)' if all(c[1] for c in checks) else 'FAIL -> D0')
    sb = [p[1] for p in dev]
    print('vs paper (dev, sb_spec 4-seed mean): ' + ', '.join(
        f'{k} {st.mean(m[k] for m in sb):.2f} vs {v}' for k, v in PAPER.items()))


if __name__ == '__main__':
    main()
