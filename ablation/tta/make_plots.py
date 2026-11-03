"""Figures for ablation/tta. Reads scores/ only; skips a figure whose inputs are missing.

Run with /home/f74134118/anaconda3/bin/python (matplotlib lives in base).
Colour = member type, as in ../aug-ops and ../nogain-ensemble: nogain green, aug blue; fr_f1 (plain, the
contrast) achromatic. Green is below 3:1 on the surface, so green marks carry direct labels.
In stage figures colour = TTA scheme: match (primary) blue, small (secondary) orange.
"""
import csv
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(HERE, 'scores')
PLOTS = os.path.join(HERE, 'plots')
os.makedirs(PLOTS, exist_ok=True)

SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8983', '#e4e3df'
BLUE, ORANGE, GREEN = '#2a78d6', '#eb6834', '#1baf7a'
MACH = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
TYPE = {'ao_nogain': (GREEN, 'nogain'), 'la_aug': (BLUE, 'aug'), 'fr_f1': (INK_2, 'plain (no shift aug)')}
SCHEME = {'Tm': (BLUE, 'match  0, ±3, ±5.5, ±8%'), 'Ts': (ORANGE, 'small  0, ±1%')}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'text.color': INK, 'axes.labelcolor': INK_2, 'axes.edgecolor': GRID,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False,
})


def read(name):
    p = os.path.join(SCORES, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding='UTF-8') as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(os.path.join(PLOTS, name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def fig1():
    rows = read('pilot_shift.csv')
    if not rows:
        return
    circ = lambda s: s - 1 if s >= 0.5 + 1e-9 else s   # noqa: E731  (0.875 -> -0.125; 0.5 stays +0.5)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True)
    for ax, m in zip(axes.flat, MACH):
        ax.axvspan(0.03, 0.08, color=GRID, alpha=0.6, lw=0)
        ax.axvspan(-0.08, -0.03, color=GRID, alpha=0.6, lw=0)
        for t, (c, lab) in TYPE.items():
            runs = sorted({r['run'] for r in rows if r['run'].startswith(t)})
            if not runs:
                continue
            sh = sorted({float(r['shift']) for r in rows}, key=circ)
            y = [np.mean([float(r['penalty']) for r in rows if r['run'] in runs and r['machine'] == m
                          and abs(float(r['shift']) - s) < 1e-6]) for s in sh]
            ax.plot([circ(s) for s in sh], y, color=c, lw=1.6, marker='o', ms=3, label=f'{lab} (n={len(runs)})')
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axhline(0.25, color=MUTED, lw=0.8, ls=':')
        ax.set_title(m, loc='left', fontsize=10, color=INK)
        ax.grid(axis='y')
        ax.set_xticks([-0.375, -0.25, -0.125, 0, 0.125, 0.25, 0.375, 0.5])
        ax.set_xticklabels(['−⅜', '−¼', '−⅛', '0', '⅛', '¼', '⅜', '½'])
    for ax in axes[1]:
        ax.set_xlabel('circular shift (fraction of the 10 s clip)')
    for ax in axes[:, 0]:
        ax.set_ylabel('R0 rise on normal held-out clips\n(sd of unshifted R0)')
    axes[1, 0].legend(loc='upper center', fontsize=8)
    fig.suptitle('Rolling a clip past the training shift range (grey) makes normal ToyCar look anomalous (~60 sd at half a clip);\n'
                 'fan, pump, slider do not move. Dotted: the 0.25 sd penalty bar of the pilot rule',
                 x=0.01, ha='left', fontsize=10.5, color=INK)
    fig.tight_layout()
    save(fig, 'fig1_pilot_shift.png')


def runs_table(ds):
    rows = read(f'{ds}_runs.csv')
    if not rows:
        return None
    out = {}
    for x in rows:
        k = (x['run'], x['version'], x['rule'])
        d = out.setdefault(k, {})
        if x['scope'] == 'overall':
            d['mean'] = float(x['mean_AUC_pAUC'])
        else:
            d[x['scope']] = float(x['AUC'])
    return out


def fig_stage(ds, fname):
    R = runs_table(ds)
    bpath = os.path.join(SCORES, f'{ds}_bootstrap.json')
    if not R or not os.path.exists(bpath):
        return
    boot = json.load(open(bpath))
    sysr = read(f'{ds}_systems.csv') or []
    E = {x['system']: float(x['mean_AUC_pAUC']) for x in sysr if x['scope'] == 'overall'}
    runs = sorted({k[0] for k in R if '_ao_nogain_' in k[0]})
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    cols = [('S0', 'Tm'), ('S0', 'Ts'), ('S1p', 'Tm'), ('S1p', 'Ts')]
    labels = []
    for i, (rule, v) in enumerate(cols):
        c = SCHEME[v][0]
        d = [R[(r, v, rule)]['mean'] - R[(r, 'T0', rule)]['mean'] for r in runs]
        ax.scatter(np.full(len(d), i) + np.linspace(-0.12, 0.12, len(d)), d, s=14, color=c, alpha=0.55, lw=0)
        b = boot.get(f'mean single nogain {rule}: {v}−T0', {}).get('mean_AUC_pAUC')
        if b:
            ax.errorbar(i + 0.28, b[0], yerr=[[b[0] - b[1]], [b[2] - b[0]]], fmt='o', color=c, ms=5, capsize=3)
        labels.append(f'single, {"R0" if rule == "S0" else "S1′"}\n{v}')
    for j, e in enumerate(('G8', 'A8')):
        for v in ('Tm', 'Ts'):
            b = boot.get(f'{e} E2′: {v}−T0', {}).get('mean_AUC_pAUC')
            if not b:
                continue
            i = len(labels)
            ax.errorbar(i, b[0], yerr=[[b[0] - b[1]], [b[2] - b[0]]], fmt='s', color=SCHEME[v][0], ms=6, capsize=3)
            labels.append(f'E2′[{e}]\n{v}')
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f'Δ {ds} mean(AUC, pAUC) vs no TTA (points)')
    ax.grid(axis='y')
    for v, (c, lab) in SCHEME.items():
        ax.plot([], [], 'o', color=c, label=lab)
    ax.legend(loc='lower left', fontsize=8)
    tm = boot.get('mean single nogain S1p: Tm−T0', {}).get('mean_AUC_pAUC', [0])[0]
    ax.set_title(f'{ds}: shift TTA vs none, ao_nogain ×8 (dots = seeds; bars = paired 95% bootstrap). '
                 f'match on S1′: {tm:+.2f}', loc='left', fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, fname)


def fig_machines(ds, fname):
    R = runs_table(ds)
    if not R:
        return
    runs = sorted({k[0] for k in R if '_ao_nogain_' in k[0]})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    x = np.arange(len(MACH))
    for ax, rule in zip(axes, ('S0', 'S1p')):
        for j, v in enumerate(('Tm', 'Ts')):
            d = [[100 * 0 + R[(r, v, rule)][m] - R[(r, 'T0', rule)][m] for r in runs] for m in MACH]
            mu = [np.mean(z) for z in d]
            se = [np.std(z, ddof=1) / np.sqrt(len(z)) for z in d]
            ax.bar(x + (j - 0.5) * 0.36, mu, 0.34, yerr=se, color=SCHEME[v][0], label=SCHEME[v][1],
                   error_kw=dict(ecolor=INK_2, lw=0.8, capsize=2))
        ax.axhline(0, color=MUTED, lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(MACH, fontsize=8)
        ax.set_title(f'single model, {"R0" if rule == "S0" else "S1′"}', loc='left', fontsize=10, color=INK)
        ax.grid(axis='y')
    axes[0].set_ylabel(f'Δ {ds} AUC vs no TTA (points, 8-seed mean ± se)')
    axes[0].legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    save(fig, fname)


if __name__ == '__main__':
    fig1()
    for ds, n in (('dev', 2), ('eval', 4)):
        fig_stage(ds, f'fig{n}_{ds}_delta.png')
        fig_machines(ds, f'fig{n + 1}_{ds}_machines.png')
