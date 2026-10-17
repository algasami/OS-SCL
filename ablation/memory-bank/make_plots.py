#!/usr/bin/env python
"""Figures for memory-bank.

Run from the repo root:
    /home/f74134118/anaconda3/bin/python ablation/memory-bank/make_plots.py

Reads scores/ only (summary.csv, bootstrap.json). A figure whose inputs are
missing is skipped, not fatal. Palette follows the repo's make_plots.py files:
the three coloured slots are the patch arms (MFN blue, TFgram orange, ImageNet
green; the mel-input ImageNet arm shares green and is told apart by its hollow
diamond). The references R0, K' and S1' are achromatic. Row labels carry
identity, so colour is never the only cue.
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
GREEN = '#1baf7a'
MACH = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
CKPTS = ([f'SCLTFSTgramMFN_tfl_relu_max_s{s}' for s in (2024, 2025, 2026, 2027)]
         + [f'SCLTFSTgramMFN_tfl_linear_avg_s{s}' for s in (2024, 2025, 2026, 2027)])

# arm -> (label, colour, marker filled?)
ARMS = {
    'mfn': ('patch: MFN', BLUE, True),
    'tfgram': ('patch: TFgram', ORANGE, True),
    'imagenet': ('patch: ImageNet on input stack', GREEN, True),
    'imagenet_mel': ('patch: ImageNet on mel', GREEN, False),
}
P2_THRESHOLD = 0.2   # pre-registered, dev, mean F_arm - S1'
P3_PUMP = 0.5        # pre-registered, pump

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
    fig.savefig(out, dpi=150, bbox_inches='tight')
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


def header(fig, title, sub):
    fig.text(0.01, 0.985, title, ha='left', va='top', fontsize=11, color=INK)
    fig.text(0.01, 0.925, sub, ha='left', va='top', fontsize=8.5, color=INK_2)


def val(summ, system, ds, rule, col='AUC'):
    return float(summ[(system, ds, rule)][col])


def fig1_standalone(summ):
    # (rule, label, colour, filled, per-checkpoint?)
    rows = [('S0', 'R0 (paper)', INK_2, True),
            ('Kp', 'K′ clip kNN', MUTED, True)]
    rows += [(f'P_{a}', ARMS[a][0], ARMS[a][1], ARMS[a][2]) for a in ARMS]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharey=True)
    gaps = {}
    for ax, ds in zip(axes, ['dev', 'eval']):
        y = np.arange(len(rows))
        means = {}
        for i, (rule, lab, col, filled) in enumerate(rows):
            v = np.array([val(summ, c, ds, rule) for c in CKPTS])
            means[rule] = v.mean()
            seedfree = np.ptp(v) < 1e-9
            if seedfree:
                v = v[:1]
            off = np.linspace(-0.16, 0.16, len(v)) if len(v) > 1 else np.zeros(1)
            ax.scatter(v, i + off, s=30,
                       facecolor=col if filled else SURFACE, edgecolor=col, lw=1.4,
                       marker='o' if filled else 'D', zorder=3, alpha=0.9)
            ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.6, zorder=4)
            ax.annotate(f'mean {v.mean():.1f}' + ('\n(seed-free: one value)' if seedfree else ''),
                        (0.012, i), xycoords=('axes fraction', 'data'), ha='left', va='center',
                        fontsize=8, color=INK_2)
        gaps[ds] = (means['Kp'] - max(means[f'P_{a}'] for a in ARMS),
                    means['S0'] - max(means[f'P_{a}'] for a in ARMS),
                    means['Kp'] - min(means[f'P_{a}'] for a in ARMS),
                    means['S0'] - min(means[f'P_{a}'] for a in ARMS))
        ax.set_yticks(y, [r[1] for r in rows])
        ax.set_ylim(len(rows) - 0.5, -0.6)
        ax.set_xlim(25, 100)
        ax.grid(axis='x')
        ax.set_xlabel('overall AUC (%)')
        ax.set_title(ds, loc='left', fontsize=9.5, color=INK)
    lo = min(min(g) for g in gaps.values())
    hi = max(max(g) for g in gaps.values())
    header(fig, f'Patch scores sit {lo:.0f}\u2013{hi:.0f} AUC points below the clip-level scores as standalone detectors',
           'Dots: one per checkpoint (8); black tick: mean. Range = gap from the R0 / K\u2032 means to the best '
           'and worst patch-arm mean, dev and eval.\nTFgram patches are bimodal: ~50 (chance) on the linear_avg '
           'checkpoints, ~66 on relu_max.')
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save(fig, 'fig1_standalone.png')
    print('fig1 gaps (K\'-best, R0-best, K\'-worst, R0-worst):', gaps)


def fig2_fusion_delta(summ, boot):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True)
    arms = list(ARMS)
    n_pos = 0
    for ax, ds in zip(axes, ['dev', 'eval']):
        ax.axvline(0, color=MUTED, lw=0.9, zorder=1)
        if ds == 'dev':
            ax.axvline(P2_THRESHOLD, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=1)
            ax.annotate(f'P2: +{P2_THRESHOLD}\n(pre-registered)', (P2_THRESHOLD, -0.45), xytext=(4, 0),
                        textcoords='offset points', fontsize=7.5, color=INK, va='top')
        for i, a in enumerate(arms):
            lab, col, filled = ARMS[a]
            d = np.array([val(summ, c, ds, f'F_{a}') - val(summ, c, ds, 'S1p') for c in CKPTS])
            n_pos += int((d > 0).sum())
            ax.scatter(d, i - 0.12 + np.linspace(-0.1, 0.1, len(d)), s=26,
                       facecolor=col if filled else SURFACE, edgecolor=col, lw=1.3,
                       marker='o' if filled else 'D', alpha=0.85, zorder=3)
            m, lo, hi = boot[ds][f'F_{a}_minus_S1p_AUC']
            ax.plot([lo, hi], [i + 0.25] * 2, color=INK, lw=1.8, solid_capstyle='butt', zorder=4)
            ax.scatter([m], [i + 0.25], s=34, color=INK, zorder=5)
            ax.annotate(f'{m:+.2f}  [{lo:+.2f}, {hi:+.2f}]', (lo, i + 0.25), xytext=(-6, 0),
                        textcoords='offset points', ha='right', va='center', fontsize=7.5, color=INK)
        ax.set_yticks(range(len(arms)), [ARMS[a][0] for a in arms])
        ax.set_ylim(len(arms) - 0.4, -0.75)
        ax.set_xlim(-2.3, 0.75)
        ax.grid(axis='x')
        ax.set_xlabel('fusion R0+patch − S1′, overall AUC (points)')
        ax.set_title(ds, loc='left', fontsize=9.5, color=INK)
    header(fig, 'Every fused patch arm is below S1\u2032 on both dev and eval; none comes near the '
           f'+{P2_THRESHOLD} bar',
           'Coloured dots: one per checkpoint (8). Black: bootstrap mean and 95% CI over test clips of the pooled '
           'difference (test-set sampling only).\n'
           f'Checkpoint-level differences above 0: {n_pos} of 64 (4 arms x 8 checkpoints x dev/eval).')
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save(fig, 'fig2_fusion_delta.png')


def fig3_per_machine(summ):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharey=True, sharex=True)
    y = np.arange(len(MACH))
    for ax, ds in zip(axes, ['dev', 'eval']):
        ax.axvline(0, color=MUTED, lw=0.9, zorder=1)
        D = np.array([[val(summ, c, ds, 'F_mfn', m) - val(summ, c, ds, 'S1p', m) for m in MACH]
                      for c in CKPTS])
        mean = D.mean(0)
        ax.barh(y, mean, height=0.5, color=BLUE, alpha=0.9, zorder=2)
        for j in range(len(MACH)):
            ax.scatter(D[:, j], np.full(len(CKPTS), y[j]) + np.linspace(-0.18, 0.18, len(CKPTS)), s=9,
                       color=INK_2, alpha=0.5, lw=0, zorder=3)
            ax.annotate(f'{mean[j]:+.2f}', (0.012, y[j]), xycoords=('axes fraction', 'data'),
                        ha='left', va='center', fontsize=8.5, color=INK)
        pj = MACH.index('pump')
        ax.plot([P3_PUMP] * 2, [pj - 0.4, pj + 0.4], color=INK, lw=1.4, ls=(0, (3, 2)), zorder=4)
        ax.annotate(f'P3: +{P3_PUMP}\n(pump)', (P3_PUMP, pj), xytext=(5, 0), textcoords='offset points',
                    va='center', fontsize=7.5, color=INK)
        ax.set_yticks(y, MACH)
        ax.set_ylim(len(MACH) - 0.5, -0.6)
        ax.set_xlim(-4.2, 1.4)
        ax.grid(axis='x')
        ax.set_xlabel('F_mfn − S1′, machine AUC (points)')
        ax.set_title(ds, loc='left', fontsize=9.5, color=INK)
    d_pump = {ds: np.mean([val(summ, c, ds, 'F_mfn', 'pump') - val(summ, c, ds, 'S1p', 'pump') for c in CKPTS])
              for ds in ('dev', 'eval')}
    tc = np.mean([val(summ, c, 'dev', 'F_mfn', 'ToyConveyor') - val(summ, c, 'dev', 'S1p', 'ToyConveyor')
                  for c in CKPTS])
    header(fig, f'MFN patch fusion never reaches the +{P3_PUMP} pump bar (dev {d_pump["dev"]:+.2f}, '
           f'eval {d_pump["eval"]:+.2f}); on dev ToyConveyor takes the biggest hit ({tc:+.2f})',
           'Bars: mean over the 8 checkpoints (value at left). Grey dots: individual checkpoints. '
           'Dashed tick: the pre-registered P3 pump threshold.')
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, 'fig3_per_machine.png')


def main():
    rows = read('summary.csv')
    boot = read('bootstrap.json')
    summ = {(r['system'], r['dataset'], r['rule']): r for r in rows} if rows else None
    if summ:
        fig1_standalone(summ)
    else:
        print('skip fig1: summary.csv missing')
    if summ and boot:
        fig2_fusion_delta(summ, boot)
    else:
        print('skip fig2: summary.csv or bootstrap.json missing')
    if summ:
        fig3_per_machine(summ)
    else:
        print('skip fig3: summary.csv missing')


if __name__ == '__main__':
    main()
