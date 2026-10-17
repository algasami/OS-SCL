#!/usr/bin/env python
"""Check the pre-registered predictions P1-P6 and decisions D1-D3; write scores/.

    python ablation/distill/score.py [ablation/distill/results/emb]

Primary (R0, eval.py's rule) reads only the committed eval CSVs, so it reruns from a
clone. Delta = student kd_relu_max_s<seed> - archived tfl_relu_max_s<seed>, overall AUC.
The noise floor is kdnull_s2024 (the baseline recipe rerun at seed 2024) - archived.

Secondary, reported only (README, Scoring): with the embedding dump from
../ensemble-backend/extract.py, S1' per student and the 4-model ensembles E1 = sum zR0
and E2' = sum S1' over students vs over baselines, with ../ensemble-backend's code.
Paired stratified bootstrap (2000 replicates, one resample shared by all vectors) of the
mean student - baseline delta; descriptive, not a decision input.

Writes: summary.csv, scorecard.txt, and with the dump also secondary.csv, bootstrap.json,
reproduction.json.
"""
import csv
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from scores_path import scores_csv    # noqa: E402

SCORES = os.path.join(HERE, 'scores')
SEEDS = (2024, 2025, 2026, 2027)
STU = {s: f'SCLTFSTgramMFN_kd_relu_max_s{s}' for s in SEEDS}
BASE = {s: f'SCLTFSTgramMFN_tfl_relu_max_s{s}' for s in SEEDS}
NULL = 'SCLTFSTgramMFN_kdnull_s2024'
MACH = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
PAPER = {'AUC': 95.71, 'pAUC': 90.23, 'mAUC': 91.23, 'eval_mean': 93.22}
N_BOOT = 2000


def read(run, ds):
    """Overall and per-machine metrics (x100) from a committed eval CSV."""
    out = {}
    with open(scores_csv(run, ds), encoding='UTF-8') as f:
        for row in csv.DictReader(f):
            if row['scope'] == 'overall':
                out.update({k: 100 * float(row[k]) for k in ('AUC', 'pAUC', 'mAUC')})
            elif row['scope'] == 'machine':
                out[row['machine']] = 100 * float(row['AUC'])
    out['mean_AUC_pAUC'] = (out['AUC'] + out['pAUC']) / 2
    return out


def e1_relu4_dev_auc():
    """E1[relu4] dev AUC as committed by ../ensemble-backend (P3's denominator)."""
    with open(os.path.join(HERE, '..', 'ensemble-backend', 'scores', 'summary.csv'), encoding='UTF-8') as f:
        for row in csv.DictReader(f):
            if row['system'] == 'relu4' and row['dataset'] == 'dev' and row['rule'] == 'E1':
                return float(row['AUC'])


def primary(lines):
    R = {(r, ds): read(r, ds) for r in list(STU.values()) + list(BASE.values()) + [NULL]
         for ds in ('dev', 'eval')}
    rows = []
    for r in list(STU.values()) + list(BASE.values()) + [NULL]:
        for ds in ('dev', 'eval'):
            rows.append(dict(run=r, dataset=ds, rule='R0', **{k: round(v, 2) for k, v in R[(r, ds)].items()}))
    with open(os.path.join(SCORES, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    d = {ds: np.array([R[(STU[s], ds)]['AUC'] - R[(BASE[s], ds)]['AUC'] for s in SEEDS]) for ds in ('dev', 'eval')}
    mean = lambda runs, ds, k: np.mean([R[(r, ds)][k] for r in runs])    # noqa: E731
    say = lines.append
    say('R0 overall AUC, student - same-seed baseline')
    say(f'{"seed":>6} {"dev":>8} {"eval":>8}')
    for i, s in enumerate(SEEDS):
        say(f'{s:>6} {d["dev"][i]:+8.2f} {d["eval"][i]:+8.2f}')
    say(f'{"mean":>6} {d["dev"].mean():+8.2f} {d["eval"].mean():+8.2f}')
    nf = {ds: R[(NULL, ds)]['AUC'] - R[(BASE[2024], ds)]['AUC'] for ds in ('dev', 'eval')}
    say(f'{"floor":>6} {nf["dev"]:+8.2f} {nf["eval"]:+8.2f}   kdnull_s2024 - archived s2024 (pure rerun noise)')
    say('')

    out = {}
    npos = int((d['dev'] > 0).sum())
    out['P1'] = npos >= 3 and d['dev'].mean() >= 0.3
    say(f'P1 dev: delta > 0 on {npos}/4 (need >= 3), mean {d["dev"].mean():+.2f} (need >= +0.30) -> '
        f'{"HOLDS" if out["P1"] else "FAILS"}')
    npos_e = int((d['eval'] > 0).sum())
    out['P2'] = npos_e >= 3 and d['eval'].mean() > 0
    say(f'P2 eval: delta > 0 on {npos_e}/4 (need >= 3), mean {d["eval"].mean():+.2f} (need > 0) -> '
        f'{"HOLDS" if out["P2"] else "FAILS"}')
    b_dev = mean(BASE.values(), 'dev', 'AUC')
    gain = e1_relu4_dev_auc() - b_dev
    frac = (mean(STU.values(), 'dev', 'AUC') - b_dev) / gain
    out['P3'] = 0.1 <= frac <= 0.5
    say(f'P3 dev fraction of E1[relu4] gain (+{gain:.2f}) recovered: {frac:+.2f} (need [0.10, 0.50]) -> '
        f'{"HOLDS" if out["P3"] else "FAILS"}')
    pm = {m: np.mean([R[(STU[s], 'dev')][m] - R[(BASE[s], 'dev')][m] for s in SEEDS]) for m in MACH}
    top = max(pm, key=pm.get)
    say('P4 dev per-machine mean delta: ' + ', '.join(f'{m} {pm[m]:+.2f}' for m in MACH))
    say(f'   largest: {top} (predicted ToyConveyor); pump {pm["pump"]:+.2f} (predicted < +0.5). '
        f'Conditional on P1, which {"holds" if out["P1"] else "fails"}: '
        f'{"read" if out["P1"] else "reported, not read"}')
    sm = {k: mean(STU.values(), 'dev', k) for k in ('AUC', 'pAUC', 'mAUC')}
    sm['eval_mean'] = mean(STU.values(), 'eval', 'mean_AUC_pAUC')
    bm = {k: mean(BASE.values(), 'dev', k) for k in ('AUC', 'pAUC', 'mAUC')}
    bm['eval_mean'] = mean(BASE.values(), 'eval', 'mean_AUC_pAUC')
    worst = {k: min(R[(r, 'dev')][k] for r in STU.values()) for k in ('AUC', 'pAUC', 'mAUC')}
    worst['eval_mean'] = min(R[(r, 'eval')]['mean_AUC_pAUC'] for r in STU.values())
    out['P5_dev'] = sm['AUC'] >= PAPER['AUC']
    out['P5_eval'] = sm['eval_mean'] >= PAPER['eval_mean']
    say(f'P5 mean of 4 students: dev AUC {sm["AUC"]:.2f} vs paper 95.71 -> {"above" if out["P5_dev"] else "below"}; '
        f'eval mean {sm["eval_mean"]:.2f} vs 93.22 -> {"above" if out["P5_eval"] else "below"}')
    sd_s = np.std([R[(r, 'dev')]['ToyConveyor'] for r in STU.values()], ddof=1)
    sd_b = np.std([R[(r, 'dev')]['ToyConveyor'] for r in BASE.values()], ddof=1)
    say(f'P6 ToyConveyor dev AUC sd: students {sd_s:.2f}, baselines {sd_b:.2f} (reported, not read)')
    say('')
    out['D1'] = out['P1'] and out['P2']
    say(f'D1 adopt KD recipe (P1 and P2): {"ADOPT" if out["D1"] else "NOT ADOPTED"}')
    claims = [k for k in sm if sm[k] > PAPER[k]]
    say('D2 single-model claims vs paper (4-student mean > paper): ' + (', '.join(
        f'{k} {sm[k]:.2f} > {PAPER[k]} ({"clearly" if worst[k] > PAPER[k] else "worst student below"})'
        for k in claims) or 'none'))
    say('   means  students / baselines / paper: ' + ', '.join(
        f'{k} {sm[k]:.2f} / {bm[k]:.2f} / {PAPER[k]}' for k in sm))
    say(f'Noise floor: |rerun gap| dev {abs(nf["dev"]):.2f}, eval {abs(nf["eval"]):.2f} vs |mean delta| '
        f'dev {abs(d["dev"].mean()):.2f}, eval {abs(d["eval"].mean()):.2f}')
    return out


def secondary(emb, lines):
    spec = importlib.util.spec_from_file_location('eb_score', os.path.join(HERE, '..', 'ensemble-backend', 'score.py'))
    eb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eb)
    runs = list(STU.values()) + list(BASE.values()) + [NULL]
    meta = dict(np.load(os.path.join(emb, 'meta.npz')))
    S = {r: eb.per_run_scores(meta, dict(np.load(os.path.join(emb, f'{r}.npz')))) for r in runs}
    ev = {ds: eb.Evaluator(meta, ds) for ds in ('dev', 'eval')}

    # P0-style check: the dump's R0 reproduces the committed per-ID AUCs
    repro = {}
    for r in runs:
        for ds in ev:
            res = ev[ds](S[r]['S0'])
            com = {}
            with open(scores_csv(r, ds), encoding='UTF-8') as f:
                for row in csv.DictReader(f):
                    if row['scope'] == 'id':
                        com[(row['machine'], row['id'])] = float(row['AUC'])
            repro[f'{r}/{ds}'] = max(abs(res[m]['ids'][i] - com[(m, i)]) for m in MACH for i in res[m]['ids'])
    json.dump(repro, open(os.path.join(SCORES, 'reproduction.json'), 'w'), indent=2)

    vec = {}
    for r in runs:
        vec[(r, 'R0')], vec[(r, 'S1p')] = S[r]['S0'], S[r]['S1p']
    for name, grp in (('stu4', STU), ('base4', BASE)):
        vec[(name, 'E1')] = sum(S[r]['zR0'] for r in grp.values())
        vec[(name, 'E2p')] = sum(S[r]['S1p'] for r in grp.values())
    rows, M = [], {}
    for (sysname, rule), v in vec.items():
        for ds in ev:
            o = ev[ds](v)['overall']
            M[(sysname, rule, ds)] = o
            rows.append(dict(system=sysname, rule=rule, dataset=ds, **{k: round(x, 2) for k, x in o.items()}))
    with open(os.path.join(SCORES, 'secondary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    say = lines.append
    say('')
    say(f'Secondary (reported only). Dump R0 vs committed per-ID AUC: max |diff| {max(repro.values()):.1e}')
    say(f'{"":28} {"dev AUC/pAUC/mAUC":>22} {"eval mean":>10}')
    fmt = lambda o, e: f'{o["AUC"]:6.2f}/{o["pAUC"]:5.2f}/{o["mAUC"]:5.2f}  {e["mean_AUC_pAUC"]:9.2f}'  # noqa: E731
    for rule in ('R0', 'S1p'):
        for name, grp in (('students', STU), ('baselines', BASE)):
            o = {k: np.mean([M[(r, rule, 'dev')][k] for r in grp.values()]) for k in ('AUC', 'pAUC', 'mAUC')}
            e = {'mean_AUC_pAUC': np.mean([M[(r, rule, 'eval')]['mean_AUC_pAUC'] for r in grp.values()])}
            say(f'{rule + " mean of 4 " + name:28} {fmt(o, e)}')
    for rule in ('E1', 'E2p'):
        for name in ('stu4', 'base4'):
            say(f'{rule + " over " + name:28} {fmt(M[(name, rule, "dev")], M[(name, rule, "eval")])}')
    for rule in ('R0', 'S1p'):
        dd = [M[(STU[s], rule, 'dev')]['AUC'] - M[(BASE[s], rule, 'dev')]['AUC'] for s in SEEDS]
        de = [M[(STU[s], rule, 'eval')]['AUC'] - M[(BASE[s], rule, 'eval')]['AUC'] for s in SEEDS]
        say(f'{rule} AUC delta per seed: dev ' + ' '.join(f'{x:+.2f}' for x in dd)
            + '  eval ' + ' '.join(f'{x:+.2f}' for x in de))

    # paired stratified bootstrap of the mean student - baseline delta
    rng = np.random.default_rng(0)
    keys = {'R0': 'R0', 'S1p': 'S1p'}
    bs = {f'{k}_{ds}_{m}': [] for k in keys for ds in ev for m in ('AUC', 'mean_AUC_pAUC')}
    bs.update({f'E2p_stu4_minus_base4_{ds}_{m}': [] for ds in ev for m in ('AUC', 'mean_AUC_pAUC')})
    for _ in range(N_BOOT):
        for ds in ev:
            idx = ev[ds].resample(rng)
            for k in keys:
                o = {s: (ev[ds](vec[(STU[s], k)], idx)['overall'], ev[ds](vec[(BASE[s], k)], idx)['overall'])
                     for s in SEEDS}
                for m in ('AUC', 'mean_AUC_pAUC'):
                    bs[f'{k}_{ds}_{m}'].append(np.mean([o[s][0][m] - o[s][1][m] for s in SEEDS]))
            a, b = ev[ds](vec[('stu4', 'E2p')], idx)['overall'], ev[ds](vec[('base4', 'E2p')], idx)['overall']
            for m in ('AUC', 'mean_AUC_pAUC'):
                bs[f'E2p_stu4_minus_base4_{ds}_{m}'].append(a[m] - b[m])
    ci = {k: dict(lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)),
                  frac_pos=float(np.mean(np.array(v) > 0))) for k, v in bs.items()}
    json.dump(ci, open(os.path.join(SCORES, 'bootstrap.json'), 'w'), indent=2)
    say('Paired bootstrap, mean student - baseline (95% CI over test clips; seeds fixed):')
    for k, v in ci.items():
        say(f'  {k:36} [{v["lo"]:+.2f}, {v["hi"]:+.2f}]  P(>0) {v["frac_pos"]:.2f}')


def main():
    os.makedirs(SCORES, exist_ok=True)
    lines = []
    primary(lines)
    emb = sys.argv[1] if len(sys.argv) > 1 else None
    if emb and os.path.exists(os.path.join(emb, 'meta.npz')):
        secondary(emb, lines)
    txt = '\n'.join(lines) + '\n'
    open(os.path.join(SCORES, 'scorecard.txt'), 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
