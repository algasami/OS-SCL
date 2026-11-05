"""Figures for ablation/spectrum-branch. Reads scores/ only (via scores_path); skips what is missing.

    /home/f74134118/anaconda3/bin/python ablation/spectrum-branch/make_plots.py

Run with base's python (matplotlib lives there). `ao_nogain` keeps its colour from
../aug-ops (#1baf7a, aqua: direct labels); `sb_spec` takes the first categorical slot.
SB_ARM=<desc prefix> substitutes another arm for sb_spec (rendering checks only).
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
SEEDS = [int(s) for s in os.environ.get('SB_SEEDS', '2040 2041 2042 2043').split()]
SB = os.environ.get('SB_ARM', 'sb_spec')
ARMS = [('ao_nogain', 'ao_nogain', '#1baf7a'), (SB, 'spectrum branch' if SB == 'sb_spec' else SB, '#2a78d6')]
MACH = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
PAPER_PUMP = 96.25
D1_PUMP = 0.40   # README D1(a), fixed before any run

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


def load(arm, s, ds):
    """{(machine, id): AUC} in points, machine rows under id 'ALL'; None if missing."""
    try:
        p = scores_csv(name(arm, s), ds)
    except FileNotFoundError:
        return None
    if not os.path.exists(p):
        return None
    return {(r['machine'], r['id']): 100 * float(r['AUC']) for r in csv.DictReader(open(p))}


def paired(ds):
    """[(seed, sb, nogain)] for seeds where both are scored."""
    out = []
    for s in SEEDS:
        a, b = load(SB, s, ds), load('ao_nogain', s, ds)
        if a and b:
            out.append((s, a, b))
    return out


def save(fig, fn):
    fig.savefig(os.path.join(PLOTS, fn), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('wrote plots/' + fn)


def fig1_machines():
    """Seed-paired delta AUC per machine, dev and eval."""
    data = {ds: paired(ds) for ds in ('dev', 'eval')}
    if not any(data.values()):
        print('skip fig1: no paired runs')
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    for ax, ds in zip(axes, ('dev', 'eval')):
        pairs = data[ds]
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.grid(axis='y')
        if not pairs:
            ax.set_title(f'{ds}: not scored')
            continue
        for i, mac in enumerate(MACH):
            d = np.array([a[(mac, 'ALL')] - b[(mac, 'ALL')] for _, a, b in pairs])
            ax.scatter(np.full(len(d), i) + np.linspace(-0.1, 0.1, len(d)), d, s=16, color=ARMS[1][2],
                       alpha=0.5, edgecolor='none', zorder=2)
            ax.plot([i - 0.25, i + 0.25], [d.mean()] * 2, color=ARMS[1][2], lw=2.4, zorder=3)
            ax.annotate(f'{d.mean():+.2f}', (i + 0.28, d.mean()), va='center', fontsize=7.5, color=INK)
        ax.set_xticks(range(len(MACH)), MACH, rotation=20)
        ax.set_title(f'{ds} (n={len(pairs)} seed pairs)')
    axes[0].set_ylabel(f'AUC, {ARMS[1][1]} − ao_nogain (points)')
    fig.suptitle('The spectrum branch lowers AUC on most machines; eval ToyCar −1.7 on 4/4 seeds (dots: seeds, bar: mean)',
                 x=0.01, ha='left', fontsize=10.5, fontweight='bold', y=1.03)
    save(fig, 'fig1_machines.png')


def fig2_pump():
    """Pump per ID, both arms, seeds and means; the paper's machine-level dev pump AUC."""
    ids = {'dev': ['id_00', 'id_02', 'id_04', 'id_06'], 'eval': ['id_01', 'id_03', 'id_05']}
    data = {ds: paired(ds) for ds in ('dev', 'eval')}
    if not any(data.values()):
        print('skip fig2: no paired runs')
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={'width_ratios': [5, 4]})
    for ax, ds in zip(axes, ('dev', 'eval')):
        pairs = data[ds]
        ax.grid(axis='y')
        if not pairs:
            ax.set_title(f'{ds}: not scored')
            continue
        cols = ['ALL'] + ids[ds]
        for j, (arm, label, c) in enumerate(ARMS):
            k = 2 if arm == 'ao_nogain' else 1   # index into (seed, sb, nogain)
            for i, col in enumerate(cols):
                v = np.array([p[k][('pump', col)] for p in pairs])
                x = i + (j - 0.5) * 0.36
                ax.scatter(np.full(len(v), x) + np.linspace(-0.06, 0.06, len(v)), v, s=14, color=c,
                           alpha=0.5, edgecolor='none', zorder=2)
                ax.plot([x - 0.14, x + 0.14], [v.mean()] * 2, color=c, lw=2.4, zorder=3,
                        label=label if i == 0 else None)
                ax.annotate(f'{v.mean():.1f}', (x, v.mean()), xytext=(0, 5), textcoords='offset points',
                            ha='center', fontsize=7, color=INK)
        if ds == 'dev':
            ax.axhline(PAPER_PUMP, color=MUTED, lw=0.9, ls=(0, (4, 3)))
            ax.annotate(f'paper, pump overall {PAPER_PUMP}', (len(cols) - 0.5, PAPER_PUMP), ha='right',
                        va='bottom', fontsize=7.5, color=INK_2)
        ax.set_xticks(range(len(cols)), ['pump overall' if c == 'ALL' else c for c in cols])
        ax.set_title(f'{ds} pump AUC by ID')
    axes[0].set_ylabel('AUC (points)')
    axes[0].legend(loc='lower left', fontsize=8)
    fig.suptitle('Pump: the branch lifts dev id_02 by +2.9 (4/4 seeds) but costs id_00 and id_06, so pump overall drops', x=0.01, ha='left', fontsize=10.5,
                 fontweight='bold', y=1.03)
    save(fig, 'fig2_pump.png')


def fig3_probe():
    """Dev-test probe over training, mean over seeds: pump AUC and overall AUC."""
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
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.3), sharex=True)
    for ax, key, title in zip(axes, ('pump_AUC', 'AUC'), ('dev pump AUC', 'dev AUC, all machines')):
        ax.grid(axis='y')
        ends = []
        for arm, label, c in ARMS:
            rows = curves[arm]
            ep = sorted(set.intersection(*(set(r) for r in rows)))
            y = np.array([[100 * float(r[e][key]) for e in ep] for r in rows]).mean(0)
            ax.plot(np.array(ep) + 1, y, color=c, lw=1.6)
            ends.append((y[-1], ep[-1] + 1, f'{label} ({len(rows)}) {y[-1]:.2f}', c))
        # end labels, pushed apart vertically so close curves stay readable
        ends.sort()
        for i, (yv, x, text, c) in enumerate(ends):
            ax.annotate(text, (x, yv), xytext=(4, (i - (len(ends) - 1) / 2) * 11), textcoords='offset points',
                        va='center', fontsize=7.5, color=c if c != '#1baf7a' else INK_2)
        ax.set_ylim(80 if key == 'pump_AUC' else 85, 100)
        ax.set_xlim(100, 375)
        ax.set_title(title)
        ax.set_xlabel('epoch')
    axes[0].set_ylabel('EMA-net probe AUC (points), mean over seeds')
    fig.suptitle('The branch learns faster early but plateaus lower (dev-test probe; diagnostic, not used for selection)', x=0.01, ha='left',
                 fontsize=10.5, fontweight='bold', y=1.03)
    save(fig, 'fig3_probe.png')


if __name__ == '__main__':
    fig1_machines()
    fig2_pump()
    fig3_probe()
