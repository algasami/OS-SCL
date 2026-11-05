"""Figures for ablation/permix. Reads scores/ only (via scores_path); skips what is missing.

    /home/f74134118/anaconda3/bin/python ablation/permix/make_plots.py

Run with base's python (matplotlib lives there). `ao_nogain` keeps its colour from
../aug-ops (#1baf7a, aqua: direct labels); `pm` takes the first categorical slot.
Titles are computed from the data, so they state only what the scored runs show.
PM_ARM=<desc prefix> and PM_S1P=<s1p.csv> substitute another arm (rendering checks only).
"""
import csv
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from scores_path import scores_csv, scores_dir  # noqa: E402

PLOTS = os.path.join(HERE, 'plots')
os.makedirs(PLOTS, exist_ok=True)
SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8983', '#e4e3df'
SEEDS = [int(s) for s in os.environ.get('PM_SEEDS', '2040 2041 2042 2043').split()]
PM = os.environ.get('PM_ARM', 'pm')
S1P = os.environ.get('PM_S1P', os.path.join(HERE, 'scores', 's1p.csv'))
ARMS = [('ao_nogain', 'ao_nogain', '#1baf7a'), (PM, 'PerMix' if PM == 'pm' else PM, '#2a78d6')]
MACH = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
# PerMix paper, TFSTgram-MFN, kNN scoring, 3 seeds (Table 3); OS-SCL paper (one network, CE score)
PAPER_PM = {('dev', 'AUC'): 95.89, ('dev', 'pAUC'): 90.98, ('eval', 'AUC'): 96.45, ('eval', 'pAUC'): 91.49}
RULES = [('S1p', 'S1′ (primary)'), ('R0', 'R0 (guard)')]

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'text.color': INK, 'axes.labelcolor': INK_2, 'axes.edgecolor': GRID,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'legend.frameon': False,
})


def name(arm, s):
    return f'SCLTFSTgramMFN_{arm}_s{s}'


def load_r0(arm, s, ds):
    """{machine | 'ALL': (AUC, pAUC)} in points; None if missing."""
    try:
        p = scores_csv(name(arm, s), ds)
    except FileNotFoundError:
        return None
    if not os.path.exists(p):
        return None
    out = {}
    for r in csv.DictReader(open(p)):
        if r['scope'] in ('machine', 'overall'):
            out[r['machine'] if r['scope'] == 'machine' else 'ALL'] = (100 * float(r['AUC']), 100 * float(r['pAUC']))
    return out


_S1P = None


def load_s1p(arm, s, ds, rule='S1p'):
    global _S1P
    if _S1P is None:
        _S1P = {}
        if os.path.exists(S1P):
            for r in csv.DictReader(open(S1P)):
                if r.get('id', 'ALL') == 'ALL':
                    _S1P.setdefault((r['run'], r['rule'], r['dataset']), {})[r['machine']] = (
                        float(r['AUC']), float(r['pAUC']))
    return _S1P.get((name(arm, s), rule, ds))


def load(arm, s, ds, rule):
    return load_r0(arm, s, ds) if rule == 'R0' else load_s1p(arm, s, ds, rule)


def paired(ds, rule):
    out = []
    for s in SEEDS:
        a, b = load(PM, s, ds, rule), load('ao_nogain', s, ds, rule)
        if a and b:
            out.append((s, a, b))
    return out


def save(fig, fn):
    fig.savefig(os.path.join(PLOTS, fn), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('wrote plots/' + fn)


def fig1_machines():
    """Seed-paired delta AUC per machine, dev and eval, under S1′ and R0."""
    data = {(rule, ds): paired(ds, rule) for rule, _ in RULES for ds in ('dev', 'eval')}
    if not any(data.values()):
        print('skip fig1: no paired runs')
        return
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2), sharey=True)
    for row, (rule, rlabel) in enumerate(RULES):
        for col, ds in enumerate(('dev', 'eval')):
            ax, pairs = axes[row, col], data[(rule, ds)]
            ax.axhline(0, color=INK_2, lw=0.8)
            ax.grid(axis='y')
            if not pairs:
                ax.set_title(f'{rlabel}, {ds}: not scored')
                continue
            for i, mac in enumerate(MACH + ['ALL']):
                d = np.array([a[mac][0] - b[mac][0] for _, a, b in pairs])
                ax.scatter(np.full(len(d), i) + np.linspace(-0.1, 0.1, len(d)), d, s=16, color=ARMS[1][2],
                           alpha=0.5, edgecolor='none', zorder=2)
                ax.plot([i - 0.25, i + 0.25], [d.mean()] * 2, color=ARMS[1][2], lw=2.4, zorder=3)
                ax.annotate(f'{d.mean():+.2f}', (i + 0.28, d.mean()), va='center', fontsize=7.5, color=INK)
            ax.axvline(len(MACH) - 0.5, color=GRID, lw=0.8)
            ax.set_xticks(range(len(MACH) + 1), MACH + ['all'], rotation=20)
            ax.set_title(f'{rlabel}, {ds} (n={len(pairs)} seed pairs)')
        axes[row, 0].set_ylabel(f'AUC, {ARMS[1][1]} − ao_nogain (points)')
    tc = data[('S1p', 'dev')] or data[('R0', 'dev')]
    rule = 'S1′' if data[('S1p', 'dev')] else 'R0'
    if tc:
        d_all = np.mean([a['ALL'][0] - b['ALL'][0] for _, a, b in tc])
        d_tc = np.mean([a['ToyConveyor'][0] - b['ToyConveyor'][0] for _, a, b in tc])
        title = (f'{ARMS[1][1]} − ao_nogain under {rule}: dev AUC {d_all:+.2f}, dev ToyConveyor {d_tc:+.2f} '
                 f'(dots: seeds, bar: mean)')
        fig.suptitle(title, x=0.01, ha='left', fontsize=10.5, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, 'fig1_machines.png')


def fig2_overall():
    """Overall AUC and pAUC, both arms, per rule, with the PerMix paper's reported numbers."""
    cells = [(ds, k) for ds in ('dev', 'eval') for k in ('AUC', 'pAUC')]
    rules = [('S1p', 'S1′'), ('R0', 'R0'), ('K', 'K (paper rule)')]
    have = {r: paired('dev', r) for r, _ in rules}
    if not any(have.values()):
        print('skip fig2: no paired runs')
        return
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.5))
    for ax, (ds, k) in zip(axes, cells):
        j = 0 if k == 'AUC' else 1
        ax.grid(axis='y')
        for ri, (rule, rlabel) in enumerate(rules):
            pairs = paired(ds, rule)
            if not pairs:
                continue
            for ai, (arm, label, c) in enumerate(ARMS):
                v = np.array([p[2 if arm == 'ao_nogain' else 1]['ALL'][j] for p in pairs])
                x = ri + (ai - 0.5) * 0.34
                ax.scatter(np.full(len(v), x) + np.linspace(-0.05, 0.05, len(v)), v, s=12, color=c, alpha=0.5,
                           edgecolor='none', zorder=2)
                ax.plot([x - 0.13, x + 0.13], [v.mean()] * 2, color=c, lw=2.4, zorder=3,
                        label=label if ri == 0 else None)
                ax.annotate(f'{v.mean():.2f}', (x, v.mean()), xytext=(0, 5), textcoords='offset points',
                            ha='center', fontsize=6.5, color=INK)
        ax.axhline(PAPER_PM[(ds, k)], color=MUTED, lw=0.9, ls=(0, (4, 3)))
        ax.annotate(f'PerMix paper {PAPER_PM[(ds, k)]}', (-0.55, PAPER_PM[(ds, k)]), ha='left', va='bottom',
                    fontsize=7, color=INK_2)
        ax.set_xticks(range(len(rules)), [r[1] for r in rules], fontsize=7.5)
        ax.set_xlim(-0.6, 2.6)
        ax.set_title(f'{ds} {k}')
    axes[0].set_ylabel('points (seeds and 4-seed mean)')
    axes[0].legend(loc='lower left', fontsize=8)
    s1 = have['S1p'] or have['R0']
    gap = np.mean([p[1]['ALL'][0] for p in s1]) - PAPER_PM[('dev', 'AUC')]
    fig.suptitle(f'Overall scores by rule; {ARMS[1][1]} dev AUC under {"S1′" if have["S1p"] else "R0"} is {gap:+.2f} '
                 f'vs the PerMix paper (its baseline was weaker)', x=0.01, ha='left', fontsize=10.5,
                 fontweight='bold', y=1.04)
    fig.tight_layout()
    save(fig, 'fig2_overall.png')


def fig3_probe():
    """Dev-test probe (R0) over training, mean over seeds: overall AUC and ToyConveyor AUC."""
    curves = {}
    for arm, _, _ in ARMS:
        rows = []
        for s in SEEDS:
            try:
                p = os.path.join(scores_dir(name(arm, s)), 'probe.csv')
            except FileNotFoundError:
                continue
            if os.path.exists(p):
                rows.append({int(r['epoch']): r for r in csv.DictReader(open(p))})
        if rows:
            curves[arm] = rows
    if len(curves) < 2:
        print('skip fig3: probe logs missing for an arm')
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.3), sharex=True,
                             gridspec_kw={'wspace': 0.5})
    finals = {}
    for ax, key, title in zip(axes, ('AUC', 'ToyCar_AUC', 'ToyConveyor_AUC'),
                              ('dev AUC, all machines', 'dev ToyCar AUC', 'dev ToyConveyor AUC')):
        ax.grid(axis='y')
        ends = []
        for arm, label, c in ARMS:
            rows = curves[arm]
            ep = sorted(set.intersection(*(set(r) for r in rows)))
            y = np.array([[100 * float(r[e][key]) for e in ep] for r in rows]).mean(0)
            ax.plot(np.array(ep) + 1, y, color=c, lw=1.6)
            ends.append((y[-1], ep[-1] + 1, f'{label} ({len(rows)}) {y[-1]:.2f}', c))
            finals[(arm, key)] = y[-1]
        ends.sort()
        for i, (yv, x, text, c) in enumerate(ends):
            ax.annotate(text, (x, yv), xytext=(4, (i - (len(ends) - 1) / 2) * 11), textcoords='offset points',
                        va='center', fontsize=7.5, color=c if c != '#1baf7a' else INK_2)
        ax.set_ylim({'AUC': 85, 'ToyCar_AUC': 85, 'ToyConveyor_AUC': 70}[key], 100)
        ax.set_xlim(0, 300)
        ax.set_title(title)
        ax.set_xlabel('epoch')
    axes[0].set_ylabel('EMA-net probe AUC, R0 (points), mean over seeds')
    d = finals[(PM, 'AUC')] - finals[('ao_nogain', 'AUC')]
    fig.suptitle(f'Dev-test probe over training (R0, diagnostic only): {ARMS[1][1]} ends {d:+.2f} on all machines',
                 x=0.01, ha='left', fontsize=10.5, fontweight='bold', y=1.03)
    save(fig, 'fig3_probe.png')


if __name__ == '__main__':
    fig1_machines()
    fig2_overall()
    fig3_probe()
