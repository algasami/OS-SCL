"""permix summary: pm vs ao_nogain, seed-paired, under S1′ (primary), R0 (guard) and K.

    /home/f74134118/anaconda3/bin/python ablation/permix/score.py

Reads only scores/: R0 from eval.py CSVs (pm here, ao_nogain from ../aug-ops/scores
via scores_path), S1′ and K from scores/s1p.csv (score_s1p.py). Prints 4-seed means,
paired deltas with their se, per-seed deltas and the D1 checklist from the README.
Seeds without a scored pm run are skipped; D1 is evaluated only when all four
stage-1 seeds are scored under both rules. PM_SEEDS="2044 2045 2046 2047" prints
stage 2 (D2 is then read off the README's rules by hand).
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
PAPER_OSSCL = {'AUC': 95.71, 'pAUC': 90.23, 'eval mean': 93.22}
PAPER_PERMIX = {'dev AUC': 95.89, 'dev pAUC': 90.98, 'eval AUC': 96.45, 'eval pAUC': 91.49}
S1P = os.path.join(HERE, 'scores', 's1p.csv')


def load_r0(name, ds):
    try:
        path = scores_csv(name, ds)
    except FileNotFoundError:
        return None
    if not os.path.exists(path):
        return None
    d = {}
    for r in csv.DictReader(open(path)):
        v = (100 * float(r['AUC']), 100 * float(r['pAUC']) if r['pAUC'] else None)
        if r['scope'] == 'overall':
            d['ALL'] = v
        elif r['scope'] == 'machine':
            d[r['machine']] = v
        else:
            d[f"{r['machine']} {r['id']}"] = v
    return d


def load_s1p():
    """{(run, rule, ds): {machine | 'machine id' | 'ALL': (AUC, pAUC)}}"""
    out = {}
    if not os.path.exists(S1P):
        return out
    for r in csv.DictReader(open(S1P)):
        k = r['machine'] if r['id'] == 'ALL' else f"{r['machine']} {r['id']}"
        out.setdefault((r['run'], r['rule'], r['dataset']), {})[k] = (
            float(r['AUC']), float(r['pAUC']) if r['pAUC'] else None)
    return out


def metrics(d):
    m = {'AUC': d['ALL'][0], 'pAUC': d['ALL'][1], 'mean': sum(d['ALL']) / 2}
    for mac in MACHINES:
        m[mac] = d[mac][0]
        m[f'{mac} mean'] = sum(d[mac]) / 2
    for k, v in d.items():
        if ' ' in k and k.split()[0] in ('ToyCar', 'ToyConveyor', 'pump'):
            m[k] = v[0]
    return m


def mean_se(xs):
    return st.mean(xs), (st.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else float('nan'))


def pairs_for(rule, ds, seeds, s1p):
    out = []
    for s in seeds:
        a_n, b_n = f'SCLTFSTgramMFN_pm_s{s}', f'SCLTFSTgramMFN_ao_nogain_s{s}'
        if rule == 'R0':
            a, b = load_r0(a_n, ds), load_r0(b_n, ds)
        else:
            a, b = s1p.get((a_n, rule, ds)), s1p.get((b_n, rule, ds))
        if a is not None and b is not None:
            out.append((s, metrics(a), metrics(b)))
    return out


def main():
    seeds = sorted(int(s) for s in os.environ.get('PM_SEEDS', ' '.join(map(str, STAGE1))).split())
    s1p = load_s1p()
    res = {}
    for rule in ('S1p', 'R0', 'K'):
        for ds in ('dev', 'eval'):
            pairs = pairs_for(rule, ds, seeds, s1p)
            res[(rule, ds)] = pairs
            if not pairs:
                print(f'== {rule} {ds}: no scored pm/ao_nogain pair yet')
                continue
            print(f'== {rule} {ds}, seeds {[p[0] for p in pairs]}: pm | ao_nogain | paired delta (se) | per seed')
            keys = [k for k in pairs[0][1] if all(k in p[1] and k in p[2] for p in pairs)]
            for k in keys:
                a = [p[1][k] for p in pairs]
                b = [p[2][k] for p in pairs]
                dd = [x - y for x, y in zip(a, b)]
                d, se = mean_se(dd)
                print(f'  {k:18s} {st.mean(a):7.2f} | {st.mean(b):7.2f} | {d:+6.2f} ({se:.2f}) | '
                      + ' '.join(f'{x:+.2f}' for x in dd))

    def dlt(rule, ds, k):
        return st.mean(p[1][k] - p[2][k] for p in res[(rule, ds)])

    if any([p[0] for p in res[(r, ds)]] != STAGE1 for r in ('S1p', 'R0') for ds in ('dev', 'eval')) \
            or seeds != STAGE1:
        print('D1: not evaluated (needs all four stage-1 seeds on dev and eval, under S1′ and R0)')
        return
    pos = sum(p[1]['AUC'] > p[2]['AUC'] for p in res[('S1p', 'dev')])
    checks = [
        ("(a) S1′ dev AUC delta >= +0.20", dlt('S1p', 'dev', 'AUC') >= 0.20, dlt('S1p', 'dev', 'AUC')),
        ("(a) S1′ dev AUC delta > 0 on >= 3/4 seeds", pos >= 3, pos),
        ("(a) S1′ dev pAUC delta >= 0", dlt('S1p', 'dev', 'pAUC') >= 0, dlt('S1p', 'dev', 'pAUC')),
        ("(b) S1′ eval mean(AUC,pAUC) delta >= -0.20", dlt('S1p', 'eval', 'mean') >= -0.20,
         dlt('S1p', 'eval', 'mean')),
        ("(c) R0 dev AUC delta >= -0.20", dlt('R0', 'dev', 'AUC') >= -0.20, dlt('R0', 'dev', 'AUC')),
        ("(c) R0 eval mean(AUC,pAUC) delta >= -0.30", dlt('R0', 'eval', 'mean') >= -0.30,
         dlt('R0', 'eval', 'mean')),
        ("(d) S1′ dev ToyConveyor delta >= -0.80", dlt('S1p', 'dev', 'ToyConveyor') >= -0.80,
         dlt('S1p', 'dev', 'ToyConveyor')),
    ]
    print('\nD1 checklist (README):')
    for name, ok, val in checks:
        v = f'{val}/4' if isinstance(val, int) else f'{val:+.2f}'
        print(f'  {name:44s} {v:>7s}  {"ok" if ok else "FAIL"}')
    print('D1:', 'PASS -> stage 2 (fresh seeds 2044-2047)' if all(c[1] for c in checks) else 'FAIL -> D0')
    for rule in ('R0', 'S1p', 'K'):
        dv = [p[1] for p in res[(rule, 'dev')]]
        ev = [p[1] for p in res[(rule, 'eval')]]
        if dv and ev:
            print(f'pm 4-seed {rule}: dev AUC {st.mean(m["AUC"] for m in dv):.2f} pAUC {st.mean(m["pAUC"] for m in dv):.2f}'
                  f' | eval AUC {st.mean(m["AUC"] for m in ev):.2f} pAUC {st.mean(m["pAUC"] for m in ev):.2f}'
                  f' mean {st.mean(m["mean"] for m in ev):.2f}')
    print(f'reference: PerMix paper (TFSTgram-MFN, kNN, 3 seeds) {PAPER_PERMIX}; OS-SCL paper {PAPER_OSSCL}')


if __name__ == '__main__':
    main()
