"""Collect the final systems into one table, from committed scores/ only.

    python synthesis/final/collect.py

Writes synthesis/final/scores/summary.csv (overall) and machines.csv (per machine).
Every number is read from a committed record of the study that produced it; nothing is
recomputed here except the 8-seed mean of eval.py's R0 rows. Fresh-confirm rows are
skipped until synthesis/fresh-confirm/scores/ has them.
"""
import csv
import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
SYN = os.path.join(REPO, 'synthesis')
OUT = os.path.join(HERE, 'scores')
MACHINES = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
METRICS = ['AUC', 'pAUC', 'mAUC', 'Amean']


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def pick(path, **kw):
    """All rows of a CSV whose columns equal kw."""
    return [r for r in rows(path) if all(r.get(k) == v for k, v in kw.items())]


def num(r, keys=METRICS):
    out = {}
    for k in keys:
        v = r.get(k, r.get('mean_AUC_pAUC') if k == 'Amean' else None)
        out[k] = float(v) if v not in (None, '') else None
    return out


summary, machines = [], []


def add(track, system, members, status, source, dev=None, ev=None):
    rec = dict(track=track, system=system, members=members, status=status, source=source)
    for ds, m in (('dev', dev), ('eval', ev)):
        for k in METRICS:
            rec[f'{ds}_{k}'] = '' if not m or m.get(k) is None else f'{m[k]:.2f}'
    summary.append(rec)


def add_machines(system, ds, per):
    for mach in MACHINES:
        if mach in per:
            machines.append(dict(system=system, dataset=ds, machine=mach,
                                 **{k: f'{per[mach][k]:.2f}' for k in METRICS if per[mach].get(k) is not None}))


# ---- paper (arXiv:2509.13853): dev AUC/pAUC/mAUC, eval only as mean(AUC, pAUC)
add('paper', 'OS-SCL (paper)', 1, 'reported', 'arXiv:2509.13853',
    dev=dict(AUC=95.71, pAUC=90.23, mAUC=91.23, Amean=(95.71 + 90.23) / 2), ev=dict(Amean=93.22))

# ---- single network, R0 = eval.py's per-sample ArcFace CE, 8 ao_nogain seeds (ablation/aug-ops)
r0 = {}
for ds in ('dev', 'eval'):
    per_seed, per_mach = [], {m: [] for m in MACHINES}
    for s in range(2040, 2048):
        f = os.path.join(REPO, 'ablation', 'aug-ops', 'scores', f'SCLTFSTgramMFN_ao_nogain_s{s}', f'eval_{ds}.csv')
        for r in rows(f):
            if r['scope'] not in ('overall', 'machine'):
                continue
            a, p, mn = (100 * float(r[k]) for k in ('AUC', 'pAUC', 'mAUC'))
            v = dict(AUC=a, pAUC=p, mAUC=mn, Amean=(a + p) / 2)
            if r['scope'] == 'overall':
                per_seed.append(v)
            elif r['scope'] == 'machine':
                per_mach[r['machine']].append(v)
    assert len(per_seed) == 8, (ds, len(per_seed))
    r0[ds] = {k: statistics.mean(v[k] for v in per_seed) for k in METRICS}
    add_machines('R0 (single, mean of 8)', ds,
                 {m: {k: statistics.mean(v[k] for v in vs) for k in METRICS} for m, vs in per_mach.items() if vs})
add('single network', 'R0 = eval.py score, ao_nogain (mean of 8 seeds)', 1, 'phase-1 baseline rule',
    'ablation/aug-ops', dev=r0['dev'], ev=r0['eval'])

# ---- single network, C1 and C1ᵀ (synthesis/single-final, 8-seed means)
sf = os.path.join(SYN, 'single-final', 'scores', 'summary.csv')
for rule, name, status in (('C1', 'C1 = zR0′ + zK′ + zGmean', 'phase-1 single-model rule'),
                           ('C1T', 'C1ᵀ = C1 + small TTA', 'FINAL single-model rule')):
    got = {}
    for ds in ('dev', 'eval'):
        per = {r['machine']: {k: float(r[f'{k}_mean8']) for k in METRICS} for r in pick(sf, dataset=ds, rule=rule)}
        got[ds] = per.pop('overall')
        add_machines(f'{rule} (single, mean of 8)', ds, per)
    add('single network', f'{name}, ao_nogain (mean of 8 seeds)', 1, status, 'synthesis/single-final',
        dev=got['dev'], ev=got['eval'])

# ---- from-scratch ensembles (synthesis/grand-ensemble, E2′ = Σ per-member S1′)
ge = os.path.join(SYN, 'grand-ensemble', 'scores')
ens = [('A8', 'E2′[A8] (8 la_aug)', 'reported best (from scratch)'),
       ('G8', 'E2′[G8] (8 ao_nogain)', 'phase-1 dev pick'),
       ('GS12', 'E2′[GS12] (8 ao_nogain + 4 sb_spec, equal weights)', 'phase-2 dev pick; ties A8 on eval'),
       ('NG18', 'E2′[NG18]', 'secondary (not dev pick)'),
       ('FAST42', 'E2′[FAST42]', 'secondary (dev rank 8)'),
       ('ALL50', 'E2′[ALL50]', 'secondary (dev rank 10)')]
for sysname, label, status in ens:
    got = {}
    for ds in ('dev', 'eval'):
        rs = pick(os.path.join(ge, f'systems_{ds}.csv'), system=sysname)
        got[ds] = num([r for r in rs if r['scope'] == 'overall'][0])
        add_machines(sysname, ds, {r['scope']: num(r) for r in rs if r['scope'] in MACHINES and not r['id']})
    n = pick(os.path.join(ge, 'systems_dev.csv'), system=sysname)[0]['n']
    add('ensemble (from scratch)', label, n, status, 'synthesis/grand-ensemble', dev=got['dev'], ev=got['eval'])

# ---- external pretraining (BEATs; never mixed into a from-scratch table)
for sysname, label, status in (('F1k', 'F1k = E2′[A8] + ΣzK′[BEATs ×3]', 'phase-1 external best'),
                               ('X1[r=0.375]', 'X1 = E2′[GS12] + 1.5·ΣzK′[BEATs ×3]', 'FINAL external system')):
    got = {}
    for ds in ('dev', 'eval'):
        rs = pick(os.path.join(ge, f'x1_{ds}.csv'), system=sysname)
        got[ds] = num([r for r in rs if r['scope'] == 'overall'][0])
        add_machines(sysname, ds, {r['scope']: num(r) for r in rs if r['scope'] in MACHINES and not r['id']})
    n = pick(os.path.join(ge, 'x1_dev.csv'), system=sysname)[0]['n']
    add('WITH EXTERNAL PRETRAINING', label, n, status, 'synthesis/grand-ensemble', dev=got['dev'], ev=got['eval'])

# ---- fresh seeds 2060+ (synthesis/fresh-confirm), only once scored
fc = os.path.join(SYN, 'fresh-confirm', 'scores')
if os.path.exists(os.path.join(fc, 'single.csv')):
    single = rows(os.path.join(fc, 'single.csv'))
    g_runs = sorted({r['run'] for r in single if '_fc_nogain_' in r['run']})
    for rule in ('R0', 'C1', 'C1T'):
        got = {}
        for ds in ('dev', 'eval'):
            rs = [r for r in single if r['run'] in g_runs and r['rule'] == rule and r['dataset'] == ds]
            got[ds] = {k: statistics.mean(float(r[k]) for r in rs) for k in METRICS} if rs else None
        add('fresh seeds (2060+)', f'{rule}, fresh ao_nogain (mean of {len(g_runs)})', 1, 'confirmation',
            'synthesis/fresh-confirm', dev=got['dev'], ev=got['eval'])
if os.path.exists(os.path.join(fc, 'ensembles.csv')):
    E = rows(os.path.join(fc, 'ensembles.csv'))
    n_fresh = {}
    if os.path.exists(os.path.join(fc, 'decisions.json')):
        with open(os.path.join(fc, 'decisions.json')) as f:
            sd = json.load(f)['seeds']
        n_fresh = {'G4': len(sd['G']), 'GS8': len(sd['G']) + len(sd['S']), 'X1': len(sd['G']) + len(sd['S']) + 3}
    for sysname in sorted({r['system'] for r in E}, key=lambda s: ('X1' in s, s)):
        got = {}
        for ds in ('dev', 'eval'):
            rs = [r for r in E if r['system'] == sysname and r['dataset'] == ds]
            o = [r for r in rs if r['machine'] in ('all', 'overall', 'ALL', '')]
            got[ds] = num(o[0]) if o else None
            add_machines(f'fresh {sysname}', ds, {r['machine']: num(r) for r in rs if r['machine'] in MACHINES})
        track = 'WITH EXTERNAL PRETRAINING' if 'X1' in sysname else 'fresh seeds (2060+)'
        add(track, f'fresh {sysname}', n_fresh.get(sysname, ''), 'confirmation', 'synthesis/fresh-confirm',
            dev=got['dev'], ev=got['eval'])

os.makedirs(OUT, exist_ok=True)
cols = ['track', 'system', 'members', 'status'] + [f'{d}_{k}' for d in ('dev', 'eval') for k in METRICS] + ['source']
with open(os.path.join(OUT, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, cols)
    w.writeheader()
    w.writerows(summary)
with open(os.path.join(OUT, 'machines.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, ['system', 'dataset', 'machine'] + METRICS)
    w.writeheader()
    w.writerows(machines)
for r in summary:
    print(f"{r['track'][:26]:26s} {r['system'][:52]:52s} dev {r['dev_Amean']:>6s}  eval {r['eval_Amean']:>6s}")
