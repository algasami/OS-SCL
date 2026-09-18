#!/usr/bin/env python
"""Figures for the seed sweep.

Run from the repo root:
    /home/f74134118/anaconda3/bin/python ablation/seed-sweep/make_plots.py

Reads the eval CSVs written by eval.py for all 16 sweep cells, plus the four
pre-existing single-seed checkpoints used by the earlier ablations. Writes PNGs
to ablation/seed-sweep/plots/. Nothing is hardcoded -- the previously reported
n=1 deltas shown for reference in fig2 are recomputed from the old checkpoints,
not copied from the READMEs.

The story is "the effects sit inside the noise", so every figure shows spread,
never a bare mean.
"""
import csv
import os
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from scores_path import scores_csv  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, 'plots')
os.makedirs(PLOTS, exist_ok=True)

# ---------------------------------------------------------------- palette ---
# Categorical slots 1-3 of the reference palette, same assignment as
# ../frozen-tfgram so the figure sets agree. Revalidated for this set
# (scripts/validate_palette.js --mode light --pairs all):
#   CVD dE 9.2 (deutan), normal-vision dE 24.0 -> ALL CHECKS PASS
# Aqua sits at 2.74:1 on the light surface, below 3:1, so the relief rule
# applies: every aqua mark ships a visible direct label.
#
# f1 is the BASELINE, not a fourth category, so it is drawn achromatic. That
# keeps the categorical set at three slots (four would put yellow beside orange
# and fail the all-pairs floors) and is maximally CVD-separable besides.
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8983'
GRID = '#e4e3df'
NEUTRAL = '#c9c8c2'
ARM_COLOR = {'f1': INK_2, 'ft_strict': '#2a78d6', 'ft_bnadapt': '#eb6834', 'f4': '#1baf7a'}
RELIEF = {'f4'}          # below 3:1 -- always direct-labelled

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'text.color': INK, 'axes.labelcolor': INK_2, 'axes.edgecolor': GRID,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False,
})

ARMS = ['f1', 'f4', 'ft_strict', 'ft_bnadapt']
SEEDS = [2024, 2025, 2026, 2027]
DATASETS = ['dev', 'eval']
MACHINES = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
TC = 'ToyConveyor'
LABEL = {'f1': 'f1  trained TFgram\n(baseline)', 'f4': 'f4  TFgram removed',
         'ft_strict': 'ft_strict  frozen, BN pinned',
         'ft_bnadapt': 'ft_bnadapt  frozen, BN adapts'}
SHORT = {'f1': 'f1', 'f4': 'f4', 'ft_strict': 'ft_strict', 'ft_bnadapt': 'ft_bnadapt'}
# the single-seed checkpoints the earlier ablations reported against
OLD = {'f1': 'SCLTFSTgramMFN_test', 'f4': 'SCLTFSTgramMFN_f4_mel_tgram',
       'ft_strict': 'SCLTFSTgramMFN_ft_strict', 'ft_bnadapt': 'SCLTFSTgramMFN_ft_bnadapt'}
T_CRIT_N4 = 3.182        # two-sided 95%, df = 3


def read(path, scope):
    key = (lambda r: r['machine']) if scope == 'machine' else (lambda r: 'ALL')
    return {key(r): {k: (float(r[k]) * 100 if r[k] else float('nan'))
                     for k in ('AUC', 'pAUC', 'mAUC')}
            for r in csv.DictReader(open(path)) if r['scope'] == scope}


def cell(arm, seed, ds, scope='overall'):
    return read(scores_csv(f'SCLTFSTgramMFN_sweep_{arm}_s{seed}', ds), scope)


def old(arm, ds, scope='overall'):
    return read(scores_csv(OLD[arm], ds), scope)


def series(arm, ds, metric='AUC'):
    return [cell(arm, s, ds)['ALL'][metric] for s in SEEDS]


def machine_mean(arm, ds, m, metric='AUC'):
    return st.mean([cell(arm, s, ds, 'machine')[m][metric] for s in SEEDS])


def overall_excl(arm, seed, ds, excl=None):
    d = cell(arm, seed, ds, 'machine')
    ks = [k for k in MACHINES if k != excl]
    return sum(d[k]['AUC'] for k in ks) / len(ks)


def ci(vals):
    """mean and half-width of the 95% t interval."""
    m = st.mean(vals)
    if len(vals) < 2:
        return m, 0.0
    return m, T_CRIT_N4 * st.stdev(vals) / len(vals) ** 0.5


# =============================================================== figure 1 ===
# Per-seed AUC for every arm. The job is identity + spread, and the point is
# that the arms overlap, so: one row per arm, one dot per seed, mean as a rule.
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for ax, ds in zip(axes, DATASETS):
        f1v = series('f1', ds)
        lo, hi = min(f1v), max(f1v)
        ax.axvspan(lo, hi, color=NEUTRAL, alpha=0.38, lw=0, zorder=0)
        ax.text((lo + hi) / 2, 3.62, f"f1's own range\n{hi - lo:.2f} pts",
                ha='center', va='bottom', fontsize=7.5, color=INK_2, linespacing=1.25)
        for i, arm in enumerate(ARMS):
            y = len(ARMS) - 1 - i
            v = series(arm, ds)
            c = ARM_COLOR[arm]
            ax.plot(v, [y] * len(v), 'o', ms=8, color=c, alpha=0.5,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            m = st.mean(v)
            ax.plot([m, m], [y - 0.26, y + 0.26], '-', color=c, lw=2.4, zorder=4,
                    solid_capstyle='round')
            lbl = f'{m:.2f}'
            ax.text(m, y + 0.34, lbl, ha='center', va='bottom', fontsize=8,
                    color=INK if arm in RELIEF else INK_2,
                    fontweight='bold' if arm in RELIEF else 'normal')
        ax.set_yticks(range(len(ARMS)))
        ax.set_yticklabels([SHORT[a] for a in reversed(ARMS)], fontsize=9)
        ax.set_ylim(-0.7, 3.95)
        ax.set_xlabel('overall AUC  (%)')
        ax.set_title(f'{ds} set', loc='left', fontsize=10, color=INK, pad=8)
        ax.grid(axis='x', zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle('Four seeds per arm — the spread of one arm covers the gap between arms',
                 x=0.062, y=0.975, ha='left', fontsize=11.5, color=INK)
    fig.text(0.062, 0.905, 'one dot per seed (2024–2027) · vertical rule = mean · '
             'grey band = the range f1 alone covers',
             ha='left', fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(os.path.join(PLOTS, 'fig1_seed_spread.png'), dpi=200)
    plt.close(fig)


# =============================================================== figure 2 ===
# The claim figure: paired delta vs f1 with a 95% interval, against the noise
# band, with the previously published n=1 deltas as hollow reference marks.
def fig2():
    arms = ['f4', 'ft_strict', 'ft_bnadapt']
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for ax, ds in zip(axes, DATASETS):
        f1v = series('f1', ds)
        half = (max(f1v) - min(f1v)) / 2
        ax.axvspan(-half, half, color=NEUTRAL, alpha=0.38, lw=0, zorder=0)
        ax.axvline(0, color=INK_2, lw=1.1, zorder=2)
        for i, arm in enumerate(arms):
            y = len(arms) - 1 - i
            d = [series(arm, ds)[k] - f1v[k] for k in range(len(SEEDS))]
            m, h = ci(d)
            c = ARM_COLOR[arm]
            ax.plot([m - h, m + h], [y, y], '-', color=c, lw=2.4, zorder=3,
                    solid_capstyle='round')
            ax.plot([m], [y], 'o', ms=9, color=c, markeredgecolor=SURFACE,
                    markeredgewidth=2, zorder=4)
            o = old(arm, ds)['ALL']['AUC'] - old('f1', ds)['ALL']['AUC']
            ax.plot([o], [y], 'o', ms=8, markerfacecolor=SURFACE, markeredgecolor=c,
                    markeredgewidth=1.8, zorder=4)
            ax.text(m, y + 0.3, f'{m:+.2f}', ha='center', va='bottom', fontsize=8,
                    color=INK if arm in RELIEF else INK_2,
                    fontweight='bold' if arm in RELIEF else 'normal')
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([SHORT[a] for a in reversed(arms)], fontsize=9)
        ax.set_ylim(-0.7, len(arms) - 0.15)
        ax.set_xlabel('AUC change vs f1  (percentage points)')
        ax.set_title(f'{ds} set', loc='left', fontsize=10, color=INK, pad=8)
        ax.grid(axis='x', zorder=0)
        ax.set_axisbelow(True)
    axes[1].text(0.97, 0.06, 'hollow = the n=1 value\nthe earlier READMEs reported',
                 transform=axes[1].transAxes, ha='right', va='bottom',
                 fontsize=7.5, color=MUTED, linespacing=1.3)
    fig.suptitle('Five of six intervals cross zero — and none survives correction for six tests',
                 x=0.062, y=0.975, ha='left', fontsize=11.5, color=INK)
    fig.text(0.062, 0.905, 'filled dot = paired mean over 4 seeds, bar = 95% t interval · '
             'grey band = half of f1’s own 4-seed range · Bonferroni threshold p<0.0083',
             ha='left', fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(os.path.join(PLOTS, 'fig2_effect_vs_noise.png'), dpi=200)
    plt.close(fig)


# =============================================================== figure 3 ===
# The decisive result: drop ToyConveyor and every delta collapses onto zero.
def fig3():
    arms = ['f4', 'ft_strict', 'ft_bnadapt']
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for ax, ds in zip(axes, DATASETS):
        ax.axvline(0, color=INK_2, lw=1.1, zorder=2)
        for i, arm in enumerate(arms):
            y = len(arms) - 1 - i
            c = ARM_COLOR[arm]
            full = st.mean([overall_excl(arm, s, ds) - overall_excl('f1', s, ds)
                            for s in SEEDS])
            excl = st.mean([overall_excl(arm, s, ds, excl=TC) -
                            overall_excl('f1', s, ds, excl=TC) for s in SEEDS])
            ax.plot([full, excl], [y, y], '-', color=c, lw=1.6, alpha=0.5, zorder=3)
            ax.plot([full], [y], 'o', ms=9, color=c, markeredgecolor=SURFACE,
                    markeredgewidth=2, zorder=4)
            ax.plot([excl], [y], 'o', ms=9, markerfacecolor=SURFACE,
                    markeredgecolor=c, markeredgewidth=2.2, zorder=5)
            ax.text(full, y + 0.3, f'{full:+.2f}', ha='center', va='bottom',
                    fontsize=8, color=INK if arm in RELIEF else INK_2,
                    fontweight='bold' if arm in RELIEF else 'normal')
            ax.text(excl - 0.035, y, f'{excl:+.2f}', ha='right', va='center',
                    fontsize=8, color=MUTED)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([SHORT[a] for a in reversed(arms)], fontsize=9)
        ax.set_ylim(-0.7, len(arms) - 0.15)
        ax.set_xlim(-0.42, 0.78)
        ax.set_xlabel('mean per-machine AUC change vs f1  (pts)')
        ax.set_title(f'{ds} set', loc='left', fontsize=10, color=INK, pad=8)
        ax.grid(axis='x', zorder=0)
        ax.set_axisbelow(True)
    axes[1].text(0.97, 0.06, 'filled = all 6 machines\nhollow = ToyConveyor dropped',
                 transform=axes[1].transAxes, ha='right', va='bottom',
                 fontsize=7.5, color=MUTED, linespacing=1.3)
    fig.suptitle('Drop ToyConveyor and every arm lands on zero',
                 x=0.062, y=0.975, ha='left', fontsize=11.5, color=INK)
    fig.text(0.062, 0.905, 'mean over 4 seeds of the per-machine AUC change · '
             'every hollow marker is within 0.09 pts of f1',
             ha='left', fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(os.path.join(PLOTS, 'fig3_toyconveyor_collapse.png'), dpi=200)
    plt.close(fig)


# =============================================================== figure 4 ===
# Where the variance actually lives: per-machine seed-to-seed spread.
def fig4():
    """Seed spread per machine. Plotting absolute AUC here does not work: the
    machines sit 18 points apart, so ToyConveyor's level swamps the axis and the
    spread -- the actual subject -- becomes invisible. Plot the spread itself."""
    def spread(arm, ds, m):
        v = [cell(arm, s_, ds, 'machine')[m]['AUC'] for s_ in SEEDS]
        return max(v) - min(v)

    order = sorted(MACHINES,
                   key=lambda m: -st.mean([spread(a, 'eval', m) for a in ARMS]))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for j, m in enumerate(order):
            for i, arm in enumerate(ARMS):
                x = j + (i - 1.5) * 0.19
                v = spread(arm, ds, m)
                c = ARM_COLOR[arm]
                ax.plot([x, x], [0, v], '-', color=c, lw=2.6, alpha=0.9,
                        solid_capstyle='round', zorder=3)
                ax.plot([x], [v], 'o', ms=5.5, color=c, markeredgecolor=SURFACE,
                        markeredgewidth=1.4, zorder=4)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=8.5)
        ax.set_title(f'{ds} set', loc='left', fontsize=10, color=INK, pad=8)
        ax.grid(axis='y', zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.6, len(order) - 0.4)
    axes[0].set_ylabel('AUC spread across the 4 seeds  (pts)')
    handles = [plt.Line2D([], [], color=ARM_COLOR[a], lw=2.6, marker='o', ms=5.5,
                          markeredgecolor=SURFACE, label=SHORT[a]) for a in ARMS]
    axes[1].legend(handles=handles, loc='upper right', fontsize=8, ncol=2,
                   handlelength=1.4, columnspacing=1.2)
    fig.suptitle('ToyConveyor alone swings more than 4 points between seeds',
                 x=0.062, y=0.975, ha='left', fontsize=11.5, color=INK)
    fig.text(0.062, 0.905, 'bar height = max − min AUC over seeds 2024–2027, per arm · '
             'machines ordered by mean spread on eval',
             ha='left', fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(os.path.join(PLOTS, 'fig4_per_machine_variance.png'), dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4()
    for f in sorted(os.listdir(PLOTS)):
        print('wrote', os.path.join('ablation/seed-sweep/plots', f))
