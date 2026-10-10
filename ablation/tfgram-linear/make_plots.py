#!/usr/bin/env python
"""Figures for the TFgram linear ablation.

Run from the repo root:
    /home/f74134118/anaconda3/bin/python ablation/tfgram-linear/make_plots.py

Reads scores/ only (branch_probe_stage*.json, per-run eval CSVs and curve.csv,
and the two-random-Tgram-fronts control in
../tfgram-redesign/scores/tgram_probe.json). Picks up every seed present, so
the same script draws stage 1 (n=1) and stage 1+2 (n=4). A figure whose inputs
are missing is skipped, not fatal.

relu_max is the paper's TFgram, i.e. the baseline, so it is drawn achromatic;
linear_avg takes the first categorical slot.
"""
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(HERE, 'scores')
PLOTS = os.path.join(HERE, 'plots')
os.makedirs(PLOTS, exist_ok=True)

# ---------------------------------------------------------------- palette ---
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8983'
GRID = '#e4e3df'
BASE = '#52514e'     # relu_max, the paper's branch -- achromatic baseline
ARMS = [('relu_max', 'ReLU + max  (paper)', BASE),
        ('linear_avg', 'linear + avg', '#2a78d6')]
COLOR = {a: c for a, _, c in ARMS}
LABEL = {a: l for a, l, _ in ARMS}
MACHINES = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']

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


def probe_runs():
    runs = {}
    for path in sorted(glob.glob(os.path.join(SCORES, 'branch_probe_stage*.json'))):
        runs.update(json.load(open(path)))
    return runs


def tgram_control():
    path = os.path.join(HERE, '..', 'tfgram-redesign', 'scores', 'tgram_probe.json')
    if not os.path.exists(path):
        return None
    rows = json.load(open(path))['results']
    return float(np.mean([r['pairs']['control: Tgram_front ~ Tgram_front_other_seed']['r2_a_to_b']
                          for r in rows]))


def auc_rows(name, split):
    path = os.path.join(SCORES, name, f'eval_{split}.csv')
    if not os.path.exists(path):
        return None
    out = {}
    for r in csv.DictReader(open(path)):
        if r['scope'] in ('machine', 'overall'):
            out[r['machine']] = 100 * float(r['AUC'])
    return out


# ------------------------------------------------------- fig1: trajectory ---
def fig1():
    runs = probe_runs()
    if not runs:
        print('skip fig1 (no branch_probe_stage*.json)'); return
    order = ['init', 'student_e10', 'student_e30', 'student_e90', 'student_e300', 'ema_best']
    ticks = ['init', 'e10', 'e30', 'e90', 'e300', 'EMA']
    ctrl = tgram_control()
    panels = [('cka/Log-Mel~TFgram', 'frame CKA, TFgram ~ Log-Mel',
               'linear_avg never looks like Log-Mel'),
              ('r2/Tgram-front->TFgram', 'frame R², Tgram front → TFgram',
               'nor does it become a Tgram copy')]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for ax, (key, ylabel, title) in zip(axes, panels):
        for name, r in sorted(runs.items()):
            arm, seed = arm_seed(name)
            ys = [r['states'][s][key] if s in r['states'] else np.nan for s in order]
            ax.plot(range(len(order)), ys, '-o', color=COLOR.get(arm, MUTED), lw=1.6, ms=4,
                    alpha=0.9, label=f'{LABEL.get(arm, arm)}' if seed == min(
                        arm_seed(n)[1] for n in runs if arm_seed(n)[0] == arm) else None)
        ax.set_xticks(range(len(order)), ticks)
        ax.set_ylim(-0.03, 1.0)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', fontsize=9.5, color=INK)
        ax.grid(axis='y')
    ax = axes[1]
    if ctrl is not None:
        ax.axhline(ctrl, color=MUTED, lw=1, ls=':')
        ax.text(0.02, ctrl + 0.02, f'two random Tgram fronts: {ctrl:.2f}', color=INK_2, fontsize=8,
                transform=ax.get_yaxis_transform())
    ax.axhline(0.50, color=MUTED, lw=1, ls='--')
    ax.text(0.02, 0.52, 'gate G1: 0.50', color=INK_2, fontsize=8, transform=ax.get_yaxis_transform())
    axes[0].legend(loc='center right', fontsize=8)
    n = len({arm_seed(k)[1] for k in runs})
    fig.suptitle(f'TFgram branch across training: init, student snapshots, EMA (n={n} seed{"s" if n > 1 else ""})',
                 x=0.01, ha='left', fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, 'fig1_branch_trajectory.png')


# ---------------------------------------------------- fig2: paired deltas ---
def fig2():
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(SCORES, 'SCLTFSTgramMFN_tfl_*')))
    seeds = sorted({arm_seed(n)[1] for n in names})
    pairs = [s for s in seeds if all(os.path.isdir(os.path.join(SCORES, f'SCLTFSTgramMFN_tfl_{a}_s{s}'))
                                     for a in ('relu_max', 'linear_avg'))]
    if not pairs:
        print('skip fig2 (no complete seed pair)'); return
    cats = MACHINES + ['ALL']
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, split in zip(axes, ('dev', 'eval')):
        deltas = []
        for s in pairs:
            a = auc_rows(f'SCLTFSTgramMFN_tfl_relu_max_s{s}', split)
            b = auc_rows(f'SCLTFSTgramMFN_tfl_linear_avg_s{s}', split)
            if a is None or b is None:
                continue
            deltas.append([b[c] - a[c] for c in cats])
        if not deltas:
            continue
        d = np.array(deltas)
        ax.axhspan(-1, 1, color=GRID, alpha=0.6, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        for i in range(len(cats)):
            jitter = np.linspace(-0.12, 0.12, len(d)) if len(d) > 1 else [0]
            ax.scatter(i + np.array(jitter), d[:, i], s=22, color=COLOR['linear_avg'], zorder=3)
            if len(d) > 1:
                ax.plot([i - 0.25, i + 0.25], [d[:, i].mean()] * 2, color=INK, lw=1.4, zorder=4)
        ax.set_xticks(range(len(cats)), [c.replace('ToyConveyor', 'ToyConv.') for c in cats], rotation=0, fontsize=8)
        ax.set_title(f'{split}', loc='left', fontsize=9.5, color=INK)
        ax.grid(axis='y')
    axes[0].set_ylabel('AUC, linear_avg − relu_max (points)')
    axes[0].text(0.01, 0.9, 'grey band: ±1 point, run-to-run\nnoise (ToyConveyor: >4 points)', transform=axes[0].transAxes,
                 fontsize=8, color=INK_2, va='top')
    n = len(pairs)
    fig.suptitle(f'Paired AUC difference per machine (n={n} seed{"s" if n > 1 else ""}'
                 f'{", bar = mean" if n > 1 else ""})',
                 x=0.01, ha='left', fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, 'fig2_paired_auc.png')


# ------------------------------------------------------- fig3: curves -------
def fig3():
    curves = {}
    for path in sorted(glob.glob(os.path.join(SCORES, 'SCLTFSTgramMFN_tfl_*', 'curve.csv'))):
        name = os.path.basename(os.path.dirname(path))
        rows = list(csv.DictReader(open(path)))
        if rows:
            curves[name] = rows
    if not curves:
        print('skip fig3 (no curve.csv)'); return
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    for name, rows in curves.items():
        arm, seed = arm_seed(name)
        ep = [int(r['epoch']) for r in rows]
        for ax, key in zip(axes, ('valid_loss', 'valid_acc')):
            ax.step(ep, [float(r[key]) for r in rows], where='post', color=COLOR.get(arm, MUTED),
                    lw=1.4, alpha=0.9, label=LABEL.get(arm) if seed == min(
                        arm_seed(n)[1] for n in curves if arm_seed(n)[0] == arm) else None)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('validation loss (EMA, best so far)')
    axes[1].set_ylabel('validation accuracy (EMA, at best loss)')
    for ax in axes:
        ax.set_xlabel('epoch')
        ax.grid(axis='y')
    axes[1].legend(loc='lower right', fontsize=8)
    fig.suptitle('Both arms train alike: EMA validation curves (rows only at new best loss)',
                 x=0.01, ha='left', fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, 'fig3_valid_curves.png')


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
