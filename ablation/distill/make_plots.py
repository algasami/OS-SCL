#!/usr/bin/env python
"""Figures for distill.

Run from the repo root:
    /home/f74134118/anaconda3/bin/python ablation/distill/make_plots.py

Reads scores/ only: the per-run eval CSVs and curve.csv (via scores_path), and
summary.csv / secondary.csv from score.py. A figure whose inputs are missing is
skipped, not fatal. The KD students take the first categorical slot (blue); the
tfl_relu_max baselines, the kdnull rerun and the paper are achromatic.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from scores_path import scores_csv, scores_dir    # noqa: E402

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
MACH = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
PAPER = {'AUC': 95.71, 'pAUC': 90.23, 'mAUC': 91.23, 'eval_mean': 93.22}
SEEDS = (2024, 2025, 2026, 2027)
STU = {s: f'SCLTFSTgramMFN_kd_relu_max_s{s}' for s in SEEDS}
BASE = {s: f'SCLTFSTgramMFN_tfl_relu_max_s{s}' for s in SEEDS}
NULL = 'SCLTFSTgramMFN_kdnull_s2024'

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


def metrics(run, ds):
    out = {}
    with open(scores_csv(run, ds), encoding='UTF-8') as f:
        for row in csv.DictReader(f):
            if row['scope'] == 'overall':
                out.update({k: 100 * float(row[k]) for k in ('AUC', 'pAUC', 'mAUC')})
            elif row['scope'] == 'machine':
                out[row['machine']] = 100 * float(row['AUC'])
    out['mean_AUC_pAUC'] = (out['AUC'] + out['pAUC']) / 2
    return out


def curve(run):
    with open(os.path.join(scores_dir(run), 'curve.csv'), encoding='UTF-8') as f:
        rows = list(csv.DictReader(f))
    return np.array([int(r['epoch']) for r in rows]), np.array([float(r['valid_loss']) for r in rows])


def fig1_paired_delta():
    """Per-seed student - baseline R0 AUC, with P1's threshold and the rerun noise floor."""
    try:
        d = {ds: [metrics(STU[s], ds)['AUC'] - metrics(BASE[s], ds)['AUC'] for s in SEEDS] for ds in ('dev', 'eval')}
        nf = {ds: metrics(NULL, ds)['AUC'] - metrics(BASE[2024], ds)['AUC'] for ds in ('dev', 'eval')}
    except FileNotFoundError as e:
        print('skip fig1:', e)
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    for ax, ds in zip(axes, ('dev', 'eval')):
        x = np.arange(4)
        ax.axhline(0, color=MUTED, lw=0.8)
        if ds == 'dev':
            ax.axhline(0.3, color=INK_2, lw=1, ls='--')
            ax.text(3.45, 0.3, 'P1 needs\nmean ≥ +0.30', color=INK_2, va='center', fontsize=7.5)
        ax.axhspan(-abs(nf[ds]), abs(nf[ds]), color=BAND, zorder=0)
        ax.text(4.25, abs(nf[ds]) + 0.01, f'one rerun gap ±{abs(nf[ds]):.2f}', color=INK_2, va='bottom',
                ha='right', fontsize=7.5)
        cols = [BLUE if v > 0 else '#a9c9ef' for v in d[ds]]
        ax.bar(x, d[ds], width=0.55, color=cols, zorder=2)
        for xi, v in zip(x, d[ds]):
            ax.text(xi, v + (0.03 if v >= 0 else -0.03), f'{v:+.2f}', ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=8, color=INK)
        m = np.mean(d[ds])
        ax.axhline(m, color=BLUE, lw=1.4)
        ax.text(3.45, m, f'mean {m:+.2f}', color=INK, va='center', fontsize=8)
        ax.set_xticks(x, [f's{s}' for s in SEEDS])
        ax.set_xlim(-0.6, 4.3)
        ax.set_title(f'{ds}: student − baseline, R0 AUC', loc='left', fontsize=9.5)
        ax.grid(axis='y')
    axes[0].set_ylabel('Δ overall AUC (points)')
    fig.suptitle('KD helps on 2/4 seeds on dev (mean −0.05); P1 fails', x=0.02, ha='left', fontsize=11)
    fig.tight_layout()
    save(fig, 'fig1_paired_delta.png')


def fig2_per_machine():
    """Mean per-machine delta over the 4 seeds, dev and eval, with each seed as a dot."""
    try:
        D = {ds: {m: [metrics(STU[s], ds)[m] - metrics(BASE[s], ds)[m] for s in SEEDS] for m in MACH}
             for ds in ('dev', 'eval')}
    except FileNotFoundError as e:
        print('skip fig2:', e)
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    for ax, ds in zip(axes, ('dev', 'eval')):
        x = np.arange(len(MACH))
        means = [np.mean(D[ds][m]) for m in MACH]
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.bar(x, means, width=0.55, color=BLUE, alpha=0.85, zorder=2)
        for xi, m in zip(x, MACH):
            ax.scatter(np.full(4, xi), D[ds][m], s=14, color=INK_2, zorder=3, linewidths=0)
        ax.set_xticks(x, MACH, rotation=30, ha='right')
        ax.set_title(f'{ds}: per-machine Δ AUC (bar = mean, dots = seeds)', loc='left', fontsize=9.5)
        ax.grid(axis='y')
    axes[0].set_ylabel('Δ machine AUC (points)')
    fig.suptitle('No machine moves consistently; ToyConveyor, the ensemble\'s big gain, is flat on dev',
                 x=0.02, ha='left', fontsize=11)
    fig.tight_layout()
    save(fig, 'fig2_per_machine.png')


def fig3_curves():
    """Best-so-far validation loss, student vs baseline, per seed."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), sharey=True)
    ok = False
    for ax, s in zip(axes, SEEDS):
        try:
            eb, vb = curve(BASE[s])
            es, vs = curve(STU[s])
        except FileNotFoundError as e:
            print('skip fig3 panel:', e)
            continue
        ok = True
        ax.step(eb, vb, where='post', color=MUTED, lw=1.4, label='baseline')
        ax.step(es, vs, where='post', color=BLUE, lw=1.4, label='KD student')
        ax.set_yscale('log')
        ax.set_title(f's{s}', loc='left', fontsize=9.5)
        ax.set_xlabel('epoch')
        ax.grid(axis='y', which='both')
    if not ok:
        plt.close(fig)
        return
    axes[0].set_ylabel('best valid loss (hard CE, EMA)')
    axes[0].legend(loc='upper right', fontsize=8)
    fig.suptitle('Student and baseline validation curves nearly coincide on every seed',
                 x=0.02, ha='left', fontsize=11)
    fig.tight_layout()
    save(fig, 'fig3_valid_curves.png')


def fig4_vs_paper():
    """Mean of 4 students vs mean of 4 baselines vs the paper; secondary S1' if scored."""
    try:
        st = {ds: [metrics(r, ds) for r in STU.values()] for ds in ('dev', 'eval')}
        ba = {ds: [metrics(r, ds) for r in BASE.values()] for ds in ('dev', 'eval')}
    except FileNotFoundError as e:
        print('skip fig4:', e)
        return
    keys = [('dev', 'AUC'), ('dev', 'pAUC'), ('dev', 'mAUC'), ('eval', 'mean_AUC_pAUC')]
    labels = ['dev AUC', 'dev pAUC', 'dev mAUC', 'eval mean(AUC, pAUC)']
    pkeys = ['AUC', 'pAUC', 'mAUC', 'eval_mean']
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.2))
    for ax, (ds, k), lab, pk in zip(axes, keys, labels, pkeys):
        b = [m[k] for m in ba[ds]]
        s = [m[k] for m in st[ds]]
        for xi, v, c in ((0, b, MUTED), (1, s, BLUE)):
            ax.scatter(np.full(4, xi), v, s=26, color=c, zorder=3, linewidths=0)
            ax.hlines(np.mean(v), xi - 0.22, xi + 0.22, color=c, lw=2.2, zorder=2)
            ax.text(xi + 0.27, np.mean(v), f'{np.mean(v):.2f}', va='center', fontsize=8)
        ax.axhline(PAPER[pk], color=INK, ls='--', lw=1)
        ax.text(0.5, PAPER[pk] + 0.03, f'paper {PAPER[pk]}', ha='center', va='bottom', fontsize=7.5, color=INK_2)
        lo = min(min(b), min(s), PAPER[pk]) - 0.4
        hi = max(max(b), max(s), PAPER[pk]) + 0.4
        ax.set_ylim(lo, hi)
        ax.set_xlim(-0.5, 1.7)
        ax.set_xticks([0, 1], ['baseline', 'KD'])
        ax.set_title(lab, loc='left', fontsize=9.5)
        ax.grid(axis='y')
    fig.suptitle('Single model, R0 (dots = seeds, line = mean): KD students sit on the baselines; eval mean edges the paper by 0.03',
                 x=0.02, ha='left', fontsize=11)
    fig.tight_layout()
    save(fig, 'fig4_vs_paper.png')


def fig5_secondary():
    """Secondary rules (reported only): R0 / S1' single models and E1 / E2' ensembles."""
    p = os.path.join(SCORES, 'secondary.csv')
    if not os.path.exists(p):
        print('skip fig5: no secondary.csv')
        return
    with open(p, encoding='UTF-8') as f:
        rows = list(csv.DictReader(f))
    M = {(r['system'], r['rule'], r['dataset']): r for r in rows}

    def val(sysnames, rule, ds, k):
        return np.mean([float(M[(n, rule, ds)][k]) for n in sysnames])

    groups = [('R0', 'one model', list(STU.values()), list(BASE.values())),
              ('S1p', "one model S1′", list(STU.values()), list(BASE.values())),
              ('E1', '4-model E1', ['stu4'], ['base4']),
              ('E2p', "4-model E2′", ['stu4'], ['base4'])]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, (ds, k, lab, pk) in zip(axes, (('dev', 'AUC', 'dev AUC', 'AUC'),
                                           ('eval', 'mean_AUC_pAUC', 'eval mean(AUC, pAUC)', 'eval_mean'))):
        x = np.arange(len(groups))
        b = [val(g[3], g[0], ds, k) for g in groups]
        s = [val(g[2], g[0], ds, k) for g in groups]
        ax.bar(x - 0.15, b, width=0.28, color=MUTED, label='baselines', zorder=2)
        ax.bar(x + 0.15, s, width=0.28, color=BLUE, label='KD students', zorder=2)
        for xi, bv, sv in zip(x, b, s):
            ax.text(xi + 0.15, sv + 0.05, f'{sv - bv:+.2f}', ha='center', va='bottom', fontsize=7.5, color=INK)
        ax.axhline(PAPER[pk], color=INK, ls='--', lw=1)
        ax.text(len(groups) - 0.45, PAPER[pk], 'paper', va='bottom', fontsize=7.5, color=INK_2)
        ax.set_ylim(min(b + s + [PAPER[pk]]) - 0.6, max(b + s + [PAPER[pk]]) + 0.5)
        ax.set_xticks(x, [g[1] for g in groups])
        ax.set_title(f'{lab} (label = student − baseline)', loc='left', fontsize=9.5)
        ax.grid(axis='y')
    axes[0].legend(loc='upper left', fontsize=8)
    fig.suptitle('Secondary rules (reported only): flat on dev under every rule; eval up by 0 to 0.28',
                 x=0.02, ha='left', fontsize=11)
    fig.tight_layout()
    save(fig, 'fig5_secondary_rules.png')


if __name__ == '__main__':
    fig1_paired_delta()
    fig2_per_machine()
    fig3_curves()
    fig4_vs_paper()
    fig5_secondary()
