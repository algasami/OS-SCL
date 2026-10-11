#!/usr/bin/env python
"""Figures for the scoring probe.

Run from the repo root:
    /home/f74134118/anaconda3/bin/python ablation/scoring-probe/make_plots.py

Reads scores/ only (rule_metrics.csv, cross_type.csv). Picks up every run
present, so stage-2 seeds land in the same figures. A figure whose inputs are
missing is skipped, not fatal.

Arm colours follow ../tfgram-linear: relu_max (the paper's TFgram) achromatic,
linear_avg the first categorical slot. Every figure plots a rule *minus R0 on
the same checkpoint*, so each dot is a paired difference.
"""
import csv
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(HERE, 'scores')
PLOTS = os.path.join(HERE, 'plots')
os.makedirs(PLOTS, exist_ok=True)

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8983'
GRID = '#e4e3df'
BAND = '#efeeea'
ARMS = [('relu_max', 'ReLU + max  (paper)', '#52514e'),
        ('linear_avg', 'linear + avg', '#2a78d6')]
COLOR = {a: c for a, _, c in ARMS}
LABEL = {a: l for a, l, _ in ARMS}
ANOMALY = '#eb6834'
MACHINES = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
RULES = ['R1', 'R2', 'R3', 'R4']
RULE_NAME = {'R1': 'R1  41 cls, no margin', 'R2': 'R2  own type, margin',
             'R3': 'R3  own type, no margin', 'R4': 'R4  1 − cos(own)'}
RULE_TICK = {'R1': 'R1\n41 classes\nno margin', 'R2': 'R2\nown type\nmargin',
             'R3': 'R3\nown type\nno margin', 'R4': 'R4\n1 − cos(own)\n'}
DECISION = 0.3   # README decision rule: |mean delta| >= 0.3 AUC points

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'text.color': INK, 'axes.labelcolor': INK_2, 'axes.edgecolor': GRID,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False,
})


def save(fig, name):
    out = os.path.join(PLOTS, name)
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote', os.path.relpath(out))


def arm_seed(name):
    m = re.search(r'tfl_(\w+?)_(\w+?)_s(\d+)$', name)
    return (f'{m.group(1)}_{m.group(2)}', int(m.group(3))) if m else (None, None)


def read(name):
    path = os.path.join(SCORES, name)
    if not os.path.exists(path):
        print(f'skip: {name} missing')
        return None
    with open(path, encoding='UTF-8') as f:
        return list(csv.DictReader(f))


def deltas(rows, scope, metric):
    """{(model, dataset, machine, rule): 100 * (rule - R0)} for one scope."""
    v = {(r['model'], r['dataset'], r['machine'], r['rule']): float(r[metric])
         for r in rows if r['scope'] == scope and r[metric] != ''}
    return {(mo, ds, ma, ru): 100 * (x - v[(mo, ds, ma, 'R0')])
            for (mo, ds, ma, ru), x in v.items() if ru != 'R0'}


def jitter(i, n, width=0.22):
    return 0.0 if n == 1 else (i / (n - 1) - 0.5) * width


def by_arm(models):
    order = [a for a, _, _ in ARMS]
    return sorted(models, key=lambda m: (order.index(arm_seed(m)[0]) if arm_seed(m)[0] in order else 9,
                                         arm_seed(m)[1] or 0))


def fig1_overall(rows):
    models = by_arm({r['model'] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), sharey='row')
    for row, (metric, mlabel) in enumerate([('AUC', 'pooled AUC (eval.py)'), ('idAUC', 'mean per-ID AUC')]):
        d = deltas(rows, 'overall', metric)
        for col, ds in enumerate(['dev', 'eval']):
            ax = axes[row, col]
            ax.axhspan(-DECISION, DECISION, color=BAND, lw=0, zorder=0)
            ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
            for i, mo in enumerate(models):
                arm, seed = arm_seed(mo)
                xs = np.arange(len(RULES)) + jitter(i, len(models))
                ys = [d[(mo, ds, 'ALL', ru)] for ru in RULES]
                ax.scatter(xs, ys, s=46, color=COLOR.get(arm, MUTED), edgecolor=SURFACE, lw=1.5, zorder=3,
                           label=f'{LABEL.get(arm, arm)}, s{seed}' if (row, col) == (0, 0) else None)
            ax.set_xticks(range(len(RULES)), [RULE_TICK[r] for r in RULES], fontsize=8)
            ax.grid(axis='y')
            ax.set_title(f'{ds}: {mlabel}', loc='left', fontsize=9.5, color=INK)
            if col == 0:
                ax.set_ylabel('rule − R0  (AUC points)')
    axes[0, 0].legend(loc='upper left', bbox_to_anchor=(0, 1.24), ncol=4, fontsize=8)
    fig.suptitle(overall_title(rows), x=0.07, y=1.06, ha='left', fontsize=11, color=INK)
    fig.text(0.07, 1.005, f'Each dot is one checkpoint scored two ways; shaded band = ±{DECISION} '
             'decision threshold (README).', fontsize=8.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig1_overall_delta.png')


def overall_title(rows):
    """State only what the data supports: which rules clear the README decision rule."""
    d = deltas(rows, 'overall', 'AUC')
    cells = sorted({(mo, ds) for mo, ds, _, _ in d})
    movers = []
    for ru in RULES:
        v = [d[(mo, ds, 'ALL', ru)] for mo, ds in cells]
        if (all(x > 0 for x in v) or all(x < 0 for x in v)) and abs(np.mean(v)) >= DECISION:
            movers.append(f'{ru} {np.mean(v):+.2f}')
    if not movers:
        return 'No scoring rule moves overall AUC consistently past ±0.3 vs. eval.py\'s R0'
    worse = all(m.split()[1].startswith('-') for m in movers)
    verb = 'loses to' if worse else 'moves vs.'
    return (f'{" and ".join(movers)}: {"each" if len(movers) > 1 else ""} {verb} R0 in every cell; '
            'no rule beats R0 consistently').replace('  ', ' ')


def fig2_machines(rows):
    models = by_arm({r['model'] for r in rows})
    d = deltas(rows, 'machine', 'AUC')
    fig, axes = plt.subplots(2, len(RULES), figsize=(11, 5.6), sharex=True, sharey=True)
    lim = max(1.0, max(abs(v) for v in d.values()) * 1.1)
    for row, ds in enumerate(['dev', 'eval']):
        for col, ru in enumerate(RULES):
            ax = axes[row, col]
            ax.axvspan(-DECISION, DECISION, color=BAND, lw=0, zorder=0)
            ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
            for i, mo in enumerate(models):
                arm, _ = arm_seed(mo)
                ys = np.arange(len(MACHINES)) + jitter(i, len(models), 0.3)
                xs = [d[(mo, ds, ma, ru)] for ma in MACHINES]
                ax.scatter(xs, ys, s=34, color=COLOR.get(arm, MUTED), edgecolor=SURFACE, lw=1.2, zorder=3)
            ax.set_yticks(range(len(MACHINES)), MACHINES)
            ax.invert_yaxis()
            ax.set_xlim(-lim, lim)
            ax.grid(axis='x')
            if row == 0:
                ax.set_title(RULE_NAME[ru], loc='left', fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel(ds, fontsize=10, color=INK)
            if row == 1:
                ax.set_xlabel('rule − R0  (machine AUC points)')
    handles = [plt.Line2D([], [], ls='', marker='o', ms=6, color=c, label=l) for _, l, c in ARMS]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.06, 1.02), ncol=2, fontsize=8.5)
    fig.suptitle('Per-machine change in pooled AUC under each rule, paired against R0',
                 x=0.06, y=1.08, ha='left', fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, 'fig2_machine_delta.png')


def fig3_cross_type(rows):
    agg = defaultdict(list)
    for r in rows:
        agg[(r['dataset'], r['machine'], r['condition'])].append(r)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.8), sharex=True)
    panels = [('p_cross_median', 'median R0 softmax mass on other machine types', False),
              ('frac_nearest_rival_foreign', 'share of clips whose nearest rival centre is another type', False)]
    for row, (key, title, logy) in enumerate(panels):
        for col, ds in enumerate(['dev', 'eval']):
            ax = axes[row, col]
            for k, (cond, color) in enumerate([('normal', INK_2), ('anomaly', ANOMALY)]):
                for x, ma in enumerate(MACHINES):
                    vals = [float(r[key]) for r in agg[(ds, ma, cond)]]
                    if logy:
                        vals = [max(v, 1e-12) for v in vals]
                    xs = x + (k - 0.5) * 0.28 + np.linspace(-0.05, 0.05, len(vals))
                    ax.scatter(xs, vals, s=30, color=color, edgecolor=SURFACE, lw=1.2, zorder=3,
                               label=cond if (x, row, col) == (0, 0, 0) else None)
            if logy:
                ax.set_yscale('log')
            else:
                ax.set_ylim(-0.03, 1.03)
            ax.set_xticks(range(len(MACHINES)), MACHINES, fontsize=8)
            ax.grid(axis='y')
            ax.set_title(f'{ds}: {title}', loc='left', fontsize=9, color=INK)
    axes[0, 0].legend(loc='upper left', bbox_to_anchor=(0, 1.3), ncol=2, fontsize=8.5)
    fig.suptitle('Other machine types are far from negligible in eval.py\'s score',
                 x=0.07, y=1.06, ha='left', fontsize=11, color=INK)
    fig.text(0.07, 1.005, '~15% of softmax mass on normal clips; 50–80% on anomalous fan/pump/slider/valve. '
             'One dot per checkpoint.', fontsize=8.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig3_cross_type.png')


def main():
    rows = read('rule_metrics.csv')
    if rows:
        fig1_overall(rows)
        fig2_machines(rows)
    rows = read('cross_type.csv')
    if rows:
        fig3_cross_type(rows)


if __name__ == '__main__':
    main()
