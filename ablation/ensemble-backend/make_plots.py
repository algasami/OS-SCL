#!/usr/bin/env python
"""Figures for ensemble-backend.

Run from the repo root:
    /home/f74134118/anaconda3/bin/python ablation/ensemble-backend/make_plots.py

Reads scores/ only (summary.csv, ensemble_curve.csv, bootstrap.json). A figure
whose inputs are missing is skipped, not fatal. Palette follows the repo's
make_plots.py files: the headline ensemble E1[relu4] takes the first
categorical slot (blue), the training-statistics variant the second (orange);
single models and the paper are references and stay achromatic.
"""
import csv
import json
import os

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
BLUE = '#2a78d6'
ORANGE = '#eb6834'
MACH = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
PAPER_DEV = {'AUC': 95.71, 'pAUC': 90.23, 'mAUC': 91.23}
PAPER_EVAL_MEAN = 93.22
PAPER_MACH = {'fan': 98.41, 'pump': 96.25, 'slider': 99.63, 'ToyCar': 96.55,
              'ToyConveyor': 83.56, 'valve': 99.85}
RELU = [f'SCLTFSTgramMFN_tfl_relu_max_s{s}' for s in (2024, 2025, 2026, 2027)]

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


def read(name):
    path = os.path.join(SCORES, name)
    if not os.path.exists(path):
        print(f'skip: {name} missing')
        return None
    if name.endswith('.json'):
        return json.load(open(path))
    with open(path, encoding='UTF-8') as f:
        return list(csv.DictReader(f))


def summary_lookup(rows):
    return {(r['system'], r['dataset'], r['rule']): r for r in rows}


def paper_line(ax, y, text):
    ax.axhline(y, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.annotate(text, (1.0, y), xycoords=('axes fraction', 'data'), xytext=(-4, 4),
                textcoords='offset points', ha='right', fontsize=8, color=INK)


def fig1_size(curve, summ):
    rows = [r for r in curve if r['rule'] == 'E1']
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    panels = [('dev', 'AUC', 'dev: overall AUC', PAPER_DEV['AUC'], 'paper 95.71'),
              ('eval', 'mean_AUC_pAUC', 'eval: mean of AUC and pAUC (the paper\'s eval number)',
               PAPER_EVAL_MEAN, 'paper 93.22')]
    for ax, (ds, key, title, pap, plab) in zip(axes, panels):
        rr = [r for r in rows if r['dataset'] == ds]
        rng = np.random.default_rng(0)
        for r in rr:
            ax.scatter(int(r['n']) + rng.uniform(-0.18, 0.18), float(r[key]), s=9, color=MUTED,
                       alpha=0.45, lw=0, zorder=2)
        ns = range(1, 9)
        means = [np.mean([float(r[key]) for r in rr if int(r['n']) == n]) for n in ns]
        ax.plot(ns, means, color=INK_2, lw=2, zorder=3, marker='o', ms=4)
        ax.annotate('mean over all subsets', xy=(6, means[5]), xytext=(0.97, 0.12),
                    textcoords='axes fraction', ha='right', fontsize=8, color=INK_2,
                    arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
        head = float(summ[('relu4', ds, 'E1')][key])
        ax.scatter([4], [head], s=70, color=BLUE, edgecolor=SURFACE, lw=1.8, zorder=5)
        ax.annotate(f'E1[relu4] {head:.2f}', (4, head), xytext=(-10, 0), textcoords='offset points',
                    ha='right', va='center', fontsize=8.5, color=INK,
                    bbox=dict(boxstyle='round,pad=0.15', fc=SURFACE, ec='none', alpha=0.85))
        paper_line(ax, pap, plab)
        ax.set_xticks(list(ns))
        ax.set_xlabel('models in the ensemble (n)')
        ax.grid(axis='y')
        ax.set_title(title, loc='left', fontsize=9.5, color=INK)
    axes[0].set_ylabel('score (%)')
    fig.suptitle('Every added model raises the mean score; four seeds of the paper\'s model '
                 'clear its dev AUC and eval number',
                 x=0.06, y=1.09, ha='left', fontsize=11, color=INK)
    fig.text(0.06, 1.01, 'Grey dots: every subset of the 8 checkpoints (E1, per-ID z-scored R0). '
             'Blue: the pre-registered headline, the 4 relu_max seeds.', fontsize=8.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig1_ensemble_size.png')


def fig2_machines(summ):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharey=True)
    for ax, ds in zip(axes, ['dev', 'eval']):
        y = np.arange(len(MACH))
        single = [np.mean([float(summ[(r, ds, 'S0')][m]) for r in RELU]) for m in MACH]
        ens = [float(summ[('relu4', ds, 'E1')][m]) for m in MACH]
        for i in range(len(MACH)):
            ax.plot([single[i], ens[i]], [y[i], y[i]], color=GRID, lw=2.5, zorder=1)
        ax.scatter(single, y, s=40, color=INK_2, edgecolor=SURFACE, lw=1.2, zorder=3,
                   label='single model (mean of 4 relu_max)')
        ax.scatter(ens, y, s=55, color=BLUE, edgecolor=SURFACE, lw=1.5, zorder=4, label='E1[relu4]')
        if ds == 'dev':
            ax.scatter([PAPER_MACH[m] for m in MACH], y, s=70, marker='|', color=INK, lw=1.8, zorder=5,
                       label='paper')
        ax.set_yticks(y, MACH)
        ax.invert_yaxis()
        ax.set_xscale('function', functions=(lambda v: -np.log(100.5 - v), lambda v: 100.5 - np.exp(-v)))
        ax.set_xticks([80, 85, 90, 95, 98, 99, 100])
        ax.set_xlim(78, 100.2)
        ax.grid(axis='x')
        ax.set_xlabel('machine AUC (%)  —  axis stretched near 100')
        ax.set_title(f'{ds}' + ('' if ds == 'dev' else '  (the paper gives no per-machine eval AUC)'),
                     loc='left', fontsize=9.5, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper left', bbox_to_anchor=(0.06, 1.0), ncol=3, fontsize=8.5)
    d = summ[('relu4', 'dev', 'E1')]
    fig.suptitle(f'On dev the ensemble gains most on ToyConveyor; pump stays '
                 f'{PAPER_MACH["pump"] - float(d["pump"]):.1f} below the paper',
                 x=0.06, y=1.16, ha='left', fontsize=11, color=INK)
    fig.text(0.06, 1.075, 'On eval, pooled valve AUC drops although every valve ID improves: held-out '
             'z-statistics (~44 clips per ID) mis-scale the IDs against each other.',
             fontsize=8.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig2_per_machine.png')


def fig3_fusion(summ):
    runs = RELU + [r.replace('relu_max', 'linear_avg') for r in RELU]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharey=True)
    for ax, ds in zip(axes, ['dev', 'eval']):
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        ax.axhspan(0, 0.2, color=BAND, lw=0, zorder=0)
        x = np.arange(len(runs))
        s1 = [float(summ[(r, ds, 'S1')]['AUC']) - float(summ[(r, ds, 'S0')]['AUC']) for r in runs]
        s1p = [float(summ[(r, ds, 'S1p')]['AUC']) - float(summ[(r, ds, 'S0')]['AUC']) for r in runs]
        ax.scatter(x - 0.12, s1, s=46, color=BLUE, edgecolor=SURFACE, lw=1.4, zorder=3,
                   label='S1: statistics from held-out clips (pre-registered)')
        ax.scatter(x + 0.12, s1p, s=46, color=ORANGE, edgecolor=SURFACE, lw=1.4, zorder=3,
                   label='S1′: statistics from training clips (sensitivity)')
        ax.set_xticks(x, [r.split('tfl_')[1].replace('_s', '\ns') for r in runs], fontsize=7.5)
        ax.grid(axis='y')
        pos = sum(v > 0 for v in s1)
        posp = sum(v > 0 for v in s1p)
        ax.set_title(f'{ds}: S1 positive {pos}/8, S1′ positive {posp}/8', loc='left', fontsize=9.5, color=INK)
    axes[0].set_ylabel('fusion − R0, overall AUC (points)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper left', bbox_to_anchor=(0.06, 1.0), ncol=2, fontsize=8.5)
    fig.suptitle('Fusing R0 with kNN: inconsistent with held-out statistics, '
                 'positive in every cell with training-clip statistics',
                 x=0.06, y=1.16, ha='left', fontsize=11, color=INK)
    fig.text(0.06, 1.075, 'Shaded band: 0 to +0.2, the D3 adoption threshold for the mean. '
             'Each pair of dots is one checkpoint scored three ways.', fontsize=8.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig3_fusion_delta.png')


def fig4_vs_paper(boot):
    items = [('dev', 'E1_AUC', 'dev AUC', PAPER_DEV['AUC']),
             ('dev', 'E1_pAUC', 'dev pAUC', PAPER_DEV['pAUC']),
             ('dev', 'E1_mAUC', 'dev mAUC', PAPER_DEV['mAUC']),
             ('eval', 'E1_mean_AUC_pAUC', 'eval mean(AUC, pAUC)', PAPER_EVAL_MEAN)]
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 3.4))
    n_above = 0
    for ax, (ds, key, lab, pap) in zip(axes, items):
        pt, lo, hi = boot[ds][key]
        n_above += pt > pap
        ax.plot([0, 0], [lo, hi], color=BLUE, lw=2.2, solid_capstyle='round', zorder=3)
        ax.scatter([0], [pt], s=60, color=BLUE, edgecolor=SURFACE, lw=1.5, zorder=4)
        ax.annotate(f'{pt:.2f}\n[{lo:.2f}, {hi:.2f}]', (0, pt), xytext=(-10, 0), textcoords='offset points',
                    ha='right', va='center', fontsize=8, color=INK)
        ax.axhline(pap, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.annotate(f'paper {pap:.2f}', (0.98, pap), xycoords=('axes fraction', 'data'), xytext=(0, 4),
                    textcoords='offset points', ha='right', fontsize=8, color=INK)
        pad = 0.25 * (hi - lo)
        ax.set_ylim(min(lo, pap) - pad, max(hi, pap) + pad)
        ax.set_xlim(-1.3, 0.9)
        ax.set_xticks([])
        ax.grid(axis='y')
        ax.set_title(lab, loc='left', fontsize=9.5, color=INK)
    fig.suptitle(f'E1[relu4] exceeds the paper on {n_above} of 4 numbers by point estimate; '
                 'every 95% interval still includes the paper\'s value',
                 x=0.06, y=1.13, ha='left', fontsize=11, color=INK)
    fig.text(0.06, 1.03, 'Stratified bootstrap over test clips (2000 draws): test-set sampling only, '
             'not seed variance. Each panel has its own scale.', fontsize=8.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig4_vs_paper.png')


def main():
    summ_rows = read('summary.csv')
    curve = read('ensemble_curve.csv')
    boot = read('bootstrap.json')
    summ = summary_lookup(summ_rows) if summ_rows else None
    if curve and summ:
        fig1_size(curve, summ)
    if summ:
        fig2_machines(summ)
        fig3_fusion(summ)
    if boot:
        fig4_vs_paper(boot)


if __name__ == '__main__':
    main()
