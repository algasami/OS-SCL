#!/usr/bin/env python
"""Figures for fast-recipe.

Run from the repo root:
    /home/f74134118/anaconda3/bin/python ablation/fast-recipe/make_plots.py

Reads scores/ only: <run>/{wall,probe,eval_dev,eval_eval}.csv, relu8_wall.csv, and
the relu8 eval CSVs that live in other ablations' scores/ (via scores_path). A
figure whose inputs are missing is skipped, not fatal. Palette as in the repo's
make_plots.py files: relu8 (today's recipe, the reference) achromatic, fr_f1
blue, fr_f4 orange, the paper a dashed ink line.
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
from scores_path import scores_csv  # noqa: E402

SCORES = os.path.join(HERE, 'scores')
PLOTS = os.path.join(HERE, 'plots')
os.makedirs(PLOTS, exist_ok=True)

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8983'
GRID = '#e4e3df'
BLUE = '#2a78d6'
ORANGE = '#eb6834'
MACH = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
PAPER_DEV_AUC = 95.71
SEEDS = (2040, 2041, 2042, 2043)
ARMS = [('fr_f1', 'fr_f1 (TFSTgram, fast)', BLUE), ('fr_f4', 'fr_f4 (Log-Mel + Tgram, fast)', ORANGE)]
RELU8 = [f'SCLTFSTgramMFN_tfl_relu_max_s{s}' for s in (2024, 2025, 2026, 2027)] + \
        [f'SCLTFSTgramMFN_cs_relu_max_s{s}' for s in (2028, 2029, 2030, 2031)]

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'text.color': INK, 'axes.labelcolor': INK_2, 'axes.edgecolor': GRID,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False,
})


def runs(arm):
    return [f'SCLTFSTgramMFN_{arm}_s{s}' for s in SEEDS]


def rows(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding='UTF-8') as f:
        return list(csv.DictReader(f))


def overall(run, ds):
    """Overall AUC (%) from a run's eval.py CSV, wherever its scores live."""
    try:
        path = os.path.join(SCORES, run, f'eval_{ds}.csv')
        r = rows(path) if os.path.exists(path) else rows(scores_csv(run, ds))
    except Exception:
        return None
    if not r:
        return None
    for x in r:
        if x['scope'] == 'overall':
            return 100 * float(x['AUC'])
        # older CSVs use machine=ALL, id=ALL
    return None


def save(fig, name):
    out = os.path.join(PLOTS, name)
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote', os.path.relpath(out))


def paper_line(ax, y, text, xfrac=0.0, ha='left'):
    ax.axhline(y, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.annotate(text, (xfrac, y), xycoords=('axes fraction', 'data'), xytext=(4 if ha == 'left' else -4, 4),
                textcoords='offset points', ha=ha, fontsize=8, color=INK)


def fig1_speed():
    old = rows(os.path.join(SCORES, 'relu8_wall.csv'))
    hours = {'relu8': [int(r['seconds']) / 3600 for r in old] if old else []}
    for arm, _, _ in ARMS:
        hours[arm] = []
        for run in runs(arm):
            w = rows(os.path.join(SCORES, run, 'wall.csv'))
            if w:
                hours[arm].append(int(w[0]['seconds']) / 3600)
    if not hours['relu8'] or not any(hours[a] for a, _, _ in ARMS):
        print('skip fig1: wall-clock inputs missing')
        return
    fig, ax = plt.subplots(figsize=(8.2, 2.9))
    order = [('relu8', 'relu8 (today: fp32, deterministic cudnn)', INK_2)] + ARMS
    base = np.median(hours['relu8'])
    for y, (key, label, col) in enumerate(reversed(order)):
        h = hours[key]
        if not h:
            continue
        med = np.median(h)
        ax.barh(y, med, height=0.5, color=col, alpha=0.18 if key != 'relu8' else 0.25, zorder=1)
        ax.scatter(h, [y] * len(h), s=44, color=col, edgecolor=SURFACE, lw=1.2, zorder=3)
        ax.annotate(f'{med:.1f} h  ({base / med:.1f}×)', (max(h), y), xytext=(8, 0), textcoords='offset points',
                    va='center', fontsize=8.5, color=INK)
    ax.set_yticks(range(len(order)), [o[1] for o in reversed(order)])
    ax.set_xlabel('wall clock per 300-epoch run (hours, one A4000 each)')
    ax.set_xlim(0, base * 1.25)
    ax.grid(axis='x')
    fast = min(np.median(hours[a]) for a, _, _ in ARMS if hours[a])
    ax.set_title(f'The fast recipe cuts a run from {base:.1f} h to {fast:.1f} h', loc='left', fontsize=10.5)
    fig.tight_layout()
    save(fig, 'fig1_speed.png')


def fig2_auc():
    ref = {ds: [v for v in (overall(r, ds) for r in RELU8) if v is not None] for ds in ('dev', 'eval')}
    got = {(a, ds): [v for v in (overall(r, ds) for r in runs(a)) if v is not None]
           for a, _, _ in ARMS for ds in ('dev', 'eval')}
    if not ref['dev'] or not any(got.values()):
        print('skip fig2: eval CSVs missing')
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, ds in zip(axes, ('dev', 'eval')):
        series = [('relu8', ref[ds], INK_2)] + [(a, got[(a, ds)], c) for a, _, c in ARMS]
        for x, (name, vals, col) in enumerate(series):
            if not vals:
                continue
            jit = np.linspace(-0.12, 0.12, len(vals))
            ax.scatter(x + jit, vals, s=40, color=col, edgecolor=SURFACE, lw=1.2, zorder=3)
            m = np.mean(vals)
            ax.plot([x - 0.25, x + 0.25], [m, m], color=col, lw=2, zorder=2)
            ax.annotate(f'{m:.2f}', (x + 0.27, m), va='center', fontsize=8, color=INK)
        if ds == 'dev':
            paper_line(ax, PAPER_DEV_AUC, 'paper 95.71')
        ax.set_xticks(range(3), [f'relu8\n(n={len(ref[ds])})'] +
                      [f'{a}\n(n={len(got[(a, ds)])})' for a, _, _ in ARMS])
        ax.set_xlim(-0.5, 2.7)
        ax.grid(axis='y')
        ax.set_title(f'{ds} set', loc='left', fontsize=9.5)
    axes[0].set_ylabel("overall AUC (%), eval.py's score (R0)")
    d1 = got[('fr_f1', 'dev')]
    title = ('fr_f1 vs today\'s recipe: dev mean ' f'{np.mean(d1) - np.mean(ref["dev"]):+.2f}' if d1
             else 'Single models by recipe')
    fig.suptitle(title + ' (bar = mean, dots = seeds)', x=0.06, ha='left', fontsize=10.5)
    fig.tight_layout()
    save(fig, 'fig2_auc.png')


def probe_rows(run):
    r = rows(os.path.join(SCORES, run, 'probe.csv'))
    return [x for x in r if x.get('AUC')] if r else None


def fig3_probe():
    have = {a: [(run, probe_rows(run)) for run in runs(a)] for a, _, _ in ARMS}
    if not any(p for a in have for _, p in have[a] if p and len(p) > 2):
        print('skip fig3: probe.csv missing')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    best = []
    for ax, (arm, label, col) in zip(axes, ARMS):
        for run, p in have[arm]:
            if not p:
                continue
            ep = [int(x['epoch']) + 1 for x in p]
            auc = [100 * float(x['AUC']) for x in p]
            ax.plot(ep, auc, color=col, lw=1.2, alpha=0.85)
            i = int(np.argmax(auc))
            best.append(auc[i])
            ax.scatter([ep[i]], [auc[i]], s=36, color=col, edgecolor=SURFACE, lw=1.2, zorder=3)
            sel = overall(run, 'dev')
            if sel is not None:
                ax.scatter([ep[-1]], [sel], s=40, marker='s', color=SURFACE, edgecolor=col, lw=1.4, zorder=3)
        paper_line(ax, PAPER_DEV_AUC, 'paper 95.71')
        ax.set_ylim(90, 97)
        ax.set_xlabel('epoch')
        ax.grid(axis='y')
        ax.set_title(label, loc='left', fontsize=9.5)
    axes[0].set_ylabel('dev AUC (%) of the EMA net, probed every 5 epochs')
    h = [plt.Line2D([], [], marker='o', ls='', color=INK_2, ms=6, label='best probed epoch (test-set selection)'),
         plt.Line2D([], [], marker='s', ls='', color=INK_2, mfc=SURFACE, ms=6, label='saved checkpoint (val loss)')]
    fig.legend(handles=h, loc='upper right', bbox_to_anchor=(0.98, 1.0), ncol=2, fontsize=8)
    reach = max(best) if best else float('nan')
    fig.suptitle(f'Picking epochs on the test set reaches at most {reach:.2f} dev AUC '
                 f'({"above" if reach >= PAPER_DEV_AUC else "below"} the paper)', x=0.06, y=1.04, ha='left',
                 fontsize=10.5)
    fig.tight_layout()
    save(fig, 'fig3_probe.png')


def fig4_oracle_by_machine():
    """Oracle gain = metrics at the best probed epoch (by overall AUC) minus the saved checkpoint's."""
    gains = {a: [] for a, _, _ in ARMS}
    for arm, _, _ in ARMS:
        for run in runs(arm):
            p = probe_rows(run)
            ev = rows(os.path.join(SCORES, run, 'eval_dev.csv'))
            if not p or not ev:
                continue
            b = max(p, key=lambda x: float(x['AUC']))
            sel = {x['machine']: 100 * float(x['AUC']) for x in ev if x['scope'] == 'machine'}
            gains[arm].append([100 * float(b[f'{m}_AUC']) - sel[m] for m in MACH])
    if not any(gains.values()):
        print('skip fig4: probe.csv or eval_dev.csv missing')
        return
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    w = 0.36
    for k, (arm, label, col) in enumerate(ARMS):
        g = np.array(gains[arm])
        if not len(g):
            continue
        x = np.arange(len(MACH)) + (k - 0.5) * w
        ax.bar(x, g.mean(0), width=w - 0.04, color=col, label=f'{label}, n={len(g)}', zorder=2)
        for j in range(len(MACH)):
            ax.scatter([x[j]] * len(g), g[:, j], s=10, color=INK_2, zorder=3)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(range(len(MACH)), MACH)
    ax.set_ylabel('dev AUC at best probed epoch\n− saved checkpoint (points)')
    ax.grid(axis='y')
    ax.legend(loc='upper left', fontsize=8)
    allg = np.concatenate([np.array(g) for g in gains.values() if len(g)])
    top = MACH[int(np.argmax(allg.mean(0)))]
    ax.set_title(f'Where test-set selection would gain: most in {top}', loc='left', fontsize=10.5)
    fig.tight_layout()
    save(fig, 'fig4_oracle_by_machine.png')


if __name__ == '__main__':
    fig1_speed()
    fig2_auc()
    fig3_probe()
    fig4_oracle_by_machine()
