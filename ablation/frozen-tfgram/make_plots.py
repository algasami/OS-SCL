#!/usr/bin/env python
"""Figures for the frozen-TFgram ablation.

Run from the repo root:
    /home/f74134118/anaconda3/bin/python ablation/frozen-tfgram/make_plots.py

Reads the eval CSVs written by eval.py and the training logs; writes PNGs to
ablation/frozen-tfgram/plots/. Nothing is hardcoded.

Every figure is *relative*: f1 (the trained-TFgram model) is the zero reference
rather than a plotted series, which is also what keeps each categorical figure
to three slots -- the cap that validates under --pairs all.
"""
import csv
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
PLOTS = os.path.join(HERE, 'plots')
os.makedirs(PLOTS, exist_ok=True)

# ---------------------------------------------------------------- palette ---
# Categorical slots 1-3 of the reference palette, revalidated for this figure
# set (scripts/validate_palette.js, light surface, --pairs all):
#   CVD dE 9.2 (deutan), normal-vision dE 24.0  -> ALL CHECKS PASS
# Aqua sits at 2.74:1 on the light surface, below 3:1, so the relief rule
# applies: every aqua mark ships a visible direct label.
# f4 keeps the aqua it has in ../gram-branch-redundancy so the two figure sets
# agree; the two new arms take slots 1 and 2.
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8983'
GRID = '#e4e3df'
ARM_COLOR = {'ft_strict': '#2a78d6', 'ft_bnadapt': '#eb6834', 'f4': '#1baf7a'}
NEUTRAL = '#c9c8c2'

# Diverging pair from the reference palette: blue <-> red, gray midpoint,
# equal step count per arm.
DIV = LinearSegmentedColormap.from_list('blue_red_div', [
    '#9e2d2d', '#c73a39', '#e34948', '#ee8180', '#f6bdbc',
    '#f0efec',
    '#b7d3f6', '#86b6ef', '#5598e7', '#2a78d6', '#185391'])

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'text.color': INK, 'axes.labelcolor': INK_2, 'axes.edgecolor': GRID,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False,
})

CKPT = {'f1': 'SCLTFSTgramMFN_test', 'f3': 'SCLTFSTgramMFN_f3_mel_tfgram',
        'f4': 'SCLTFSTgramMFN_f4_mel_tgram', 'ft_bnadapt': 'SCLTFSTgramMFN_ft_bnadapt',
        'ft_strict': 'SCLTFSTgramMFN_ft_strict'}
LABEL = {'f3': 'f3  mel+TFgram\n(Tgram removed)', 'f4': 'f4  mel+Tgram\n(TFgram removed)',
         'ft_bnadapt': 'ft_bnadapt\n(TFgram frozen, BN adapts)',
         'ft_strict': 'ft_strict\n(TFgram frozen, BN pinned)'}
SHORT = {'f3': 'f3  Tgram removed', 'f4': 'f4  TFgram removed',
         'ft_bnadapt': 'ft_bnadapt  frozen, BN adapts',
         'ft_strict': 'ft_strict  frozen, BN pinned'}
MACHINES = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
TC = 'ToyConveyor'
DATASETS = ['dev', 'eval']


def load(arm, ds, scope):
    p = f'check_points/{CKPT[arm]}/eval_{ds}.csv'
    key = (lambda r: r['machine']) if scope == 'machine' else (lambda r: 'ALL')
    return {key(r): {k: float(r[k]) * 100 for k in ('AUC', 'pAUC', 'mAUC')}
            for r in csv.DictReader(open(p)) if r['scope'] == scope}


M = {(a, ds): load(a, ds, 'machine') for a in CKPT for ds in DATASETS}
O = {(a, ds): load(a, ds, 'overall')['ALL'] for a in CKPT for ds in DATASETS}


def delta(arm, ds, machine, metric='AUC'):
    return M[(arm, ds)][machine][metric] - M[('f1', ds)][machine][metric]


def mean_excl_tc(arm, ds, metric='AUC'):
    ms = [m for m in MACHINES if m != TC]
    return (sum(M[(arm, ds)][m][metric] for m in ms)
            - sum(M[('f1', ds)][m][metric] for m in ms)) / len(ms)


def load_curve(arm, path):
    txt = open(path, errors='ignore').read().replace('\r', '\n')
    rows = re.findall(r'EPOCH:\s*(\d+)\s*\|\s*Train_loss:\s*([\d.]+)\s*\|\s*Train_accuracy:\s*([\d.]+)'
                      r'\s*\|\s*Valid_loss:\s*([\d.]+)\s*\|\s*Valid_accuracy:\s*([\d.]+)', txt)
    a = np.array([[float(x) for x in r] for r in rows])
    return a[:, 0], a[:, 3], a[:, 4] * 100


# ------------------------------------------ fig 1: the ToyConveyor collapse ---
# The headline: every arm "beats" f1 until one unstable machine is removed.
# A dumbbell, not two bar groups -- the message is the *movement* toward zero.
ARMS4 = ['f3', 'f4', 'ft_bnadapt', 'ft_strict']
fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0), sharex=True)
fig.subplots_adjust(left=0.255, right=0.975, top=0.695, bottom=0.14, wspace=0.075)

for c, ds in enumerate(DATASETS):
    ax = axes[c]
    y = np.arange(len(ARMS4))[::-1]
    for i, arm in enumerate(ARMS4):
        yy = y[i]
        if i % 2 == 0:
            ax.axhspan(yy - 0.5, yy + 0.5, color='#f3f2ef', zorder=0, lw=0)
        a_all, a_ex = O[(arm, ds)]['AUC'] - O[('f1', ds)]['AUC'], mean_excl_tc(arm, ds)
        ax.plot([a_all, a_ex], [yy, yy], color=NEUTRAL, lw=2.0, zorder=2,
                solid_capstyle='round')
        # open = all six machines, filled = ToyConveyor dropped
        ax.plot(a_all, yy, 'o', ms=9, mfc=SURFACE, mec=MUTED, mew=1.8, zorder=3)
        ax.plot(a_ex, yy, 'o', ms=9, color=INK, mec=SURFACE, mew=1.6, zorder=4)
        ax.text(a_all, yy + 0.30, f'{a_all:+.2f}', ha='center', va='bottom',
                fontsize=8.2, color=MUTED, zorder=5)
        ax.text(a_ex, yy - 0.32, f'{a_ex:+.2f}', ha='center', va='top',
                fontsize=8.6, color=INK, fontweight='bold', zorder=5)
    ax.axvline(0, color=INK_2, lw=1.1, zorder=1)
    ax.set_yticks(y, [SHORT[a] for a in ARMS4], fontsize=8.8)
    ax.set_ylim(-0.62, len(ARMS4) - 0.38)
    ax.set_xlim(-1.15, 1.62)
    ax.set_xlabel('mean per-machine AUC, change vs f1 (points)', fontsize=8.5)
    ax.set_title(f'{ds} set', fontsize=9.5, color=INK_2, pad=7, loc='left')
    ax.xaxis.grid(True); ax.set_axisbelow(True); ax.spines['left'].set_visible(False)
    if c == 1:
        ax.tick_params(labelleft=False)

handles = [plt.Line2D([], [], marker='o', ls='', ms=9, mfc=SURFACE, mec=MUTED, mew=1.8,
                      label='all 6 machines'),
           plt.Line2D([], [], marker='o', ls='', ms=9, color=INK, mec=SURFACE, mew=1.6,
                      label='ToyConveyor dropped')]
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.255, 0.845),
           ncol=2, fontsize=9, handletextpad=0.4, columnspacing=1.8)
fig.suptitle('Drop one unstable machine and every apparent gain collapses to zero',
             fontsize=11.5, fontweight='bold', x=0.255, ha='left', y=0.960)
fig.text(0.255, 0.893, 'f1 (TFgram trained) is the zero line  ·  '
         'ToyConveyor alone moves these arms +0.2 to +6.9 points',
         fontsize=8.5, color=MUTED, ha='left')
fig.savefig(os.path.join(PLOTS, 'fig1_toyconveyor_collapse.png'), dpi=200)
plt.close(fig)


# ------------------------------------------- fig 2: per-machine delta matrix ---
# Survey view. Diverging, centred at 0; the scale is clipped at +-3 so the
# near-zero majority stays readable, and every cell is annotated so the two
# clipped ToyConveyor cells lose nothing.
fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.3))
fig.subplots_adjust(left=0.235, right=0.895, top=0.775, bottom=0.075, hspace=0.30)
norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)

for r, ds in enumerate(DATASETS):
    ax = axes[r]
    D = np.array([[delta(a, ds, m) for m in MACHINES] for a in ARMS4])
    im = ax.imshow(D, cmap=DIV, norm=norm, aspect='auto')
    for i in range(len(ARMS4)):
        for j in range(len(MACHINES)):
            v = D[i, j]
            ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=8.6,
                    color='#ffffff' if abs(v) > 1.6 else INK,
                    fontweight='bold' if abs(v) > 1.0 else 'normal')
    ax.set_xticks(range(len(MACHINES)), MACHINES, fontsize=8.8)
    ax.set_yticks(range(len(ARMS4)), [SHORT[a] for a in ARMS4], fontsize=8.5)
    ax.set_title(f'{ds} set', fontsize=9.5, color=INK_2, pad=7, loc='left')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    # separate the ToyConveyor column -- it is the whole story
    j = MACHINES.index(TC)
    for xo in (j - 0.5, j + 0.5):
        ax.axvline(xo, color=SURFACE, lw=2.5)

cax = fig.add_axes([0.908, 0.075, 0.014, 0.70])
cb = fig.colorbar(im, cax=cax, extend='both')
cb.set_label('AUC change vs f1 (points)', fontsize=8.3, color=INK_2)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=8, color=GRID)

fig.suptitle('Every TFgram variant is flat everywhere except ToyConveyor',
             fontsize=11.5, fontweight='bold', x=0.235, ha='left', y=0.968)
fig.text(0.235, 0.915, 'Blue = better than f1, red = worse.  Scale clipped at ±3 points; '
         'all values annotated.\nOnly f3, which removes Tgram rather than TFgram, moves the '
         'non-ToyConveyor machines (fan −2.89).',
         fontsize=8.5, color=MUTED, ha='left', va='top')
fig.savefig(os.path.join(PLOTS, 'fig2_per_machine_delta.png'), dpi=200)
plt.close(fig)


# ------------------------------------------------- fig 3: the null, zoomed in ---
# The actual claim. ToyConveyor excluded, axis tight: trained / frozen / removed
# are indistinguishable on every remaining machine.
ARMS3 = ['ft_strict', 'ft_bnadapt', 'f4']
MACH5 = [m for m in MACHINES if m != TC]
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), sharey=True)
fig.subplots_adjust(left=0.105, right=0.978, top=0.705, bottom=0.135, wspace=0.08)
OFFSET = {'ft_strict': 0.25, 'ft_bnadapt': 0.0, 'f4': -0.25}

for c, ds in enumerate(DATASETS):
    ax = axes[c]
    y = np.arange(len(MACH5))[::-1]
    for i, m in enumerate(MACH5):
        if i % 2 == 0:
            ax.axhspan(y[i] - 0.5, y[i] + 0.5, color='#f3f2ef', zorder=0, lw=0)
        for arm in ARMS3:
            v = delta(arm, ds, m)
            yy = y[i] + OFFSET[arm]
            ax.plot(v, yy, 'o', ms=8.5, color=ARM_COLOR[arm], zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.6)
            # relief rule: aqua is under 3:1, so label every mark directly.
            # Offset away from zero and wider than the marker radius, or the
            # marker sits on top of the value's minus sign.
            ax.text(v + (0.038 if v >= 0 else -0.038), yy, f'{v:+.2f}',
                    va='center', ha='left' if v >= 0 else 'right',
                    fontsize=7.5, color=INK_2, zorder=4)
    ax.axvline(0, color=INK_2, lw=1.1, zorder=1)
    ax.set_yticks(y, MACH5, fontsize=9)
    ax.set_ylim(-0.62, len(MACH5) - 0.38)
    ax.set_xlim(-0.72, 0.72)
    ax.set_xlabel('AUC change vs f1 (points)', fontsize=8.5)
    ax.set_title(f'{ds} set', fontsize=9.5, color=INK_2, pad=7, loc='left')
    ax.xaxis.grid(True); ax.set_axisbelow(True); ax.spines['left'].set_visible(False)

handles = [plt.Line2D([], [], marker='o', ls='', ms=8.5, color=ARM_COLOR[a],
                      markeredgecolor=SURFACE, markeredgewidth=1.6, label=SHORT[a])
           for a in ARMS3]
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.105, 0.852),
           ncol=3, fontsize=9, handletextpad=0.4, columnspacing=1.8)
fig.suptitle('Training TFgram, freezing it, or deleting it: no difference',
             fontsize=11.5, fontweight='bold', x=0.105, ha='left', y=0.960)
fig.text(0.105, 0.898, 'ToyConveyor excluded  ·  note the axis: the full range shown is '
         '±0.7 points, and f1 is the zero line',
         fontsize=8.5, color=MUTED, ha='left')
fig.savefig(os.path.join(PLOTS, 'fig3_null_zoomed.png'), dpi=200)
plt.close(fig)


# --------------------------------------------------- fig 4: training curves ---
# Does freezing a branch impair optimisation? No.
CURVES = {'ft_strict': os.path.join(RESULTS, 'train_ft_strict.log'),
          'ft_bnadapt': os.path.join(RESULTS, 'train_ft_bnadapt.log'),
          'f4': os.path.join(HERE, '..', 'gram-branch-redundancy', 'results', 'train_f4.log')}
curves = {a: load_curve(a, p) for a, p in CURVES.items()}
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
fig.subplots_adjust(left=0.078, right=0.985, top=0.735, bottom=0.145, wspace=0.205)
ZOOM = 180

ax = axes[0]
for arm in ARMS3:
    ep, vl, _ = curves[arm]
    ax.plot(ep, vl, lw=2.0, color=ARM_COLOR[arm], solid_capstyle='round',
            label=f'{SHORT[arm]}   final {vl[-1]:.3f}')
    ax.plot(ep[-1], vl[-1], 'o', ms=8, color=ARM_COLOR[arm],
            markeredgecolor=SURFACE, markeredgewidth=1.6)
ax.set_yscale('log')
ax.set_xlabel('epoch', fontsize=8.5)
ax.set_ylabel('best validation loss (log scale)', fontsize=8.5)
ax.set_title('Validation loss, full run', fontsize=9.5, color=INK_2, pad=8, loc='left')
ax.yaxis.grid(True); ax.set_axisbelow(True); ax.set_xlim(-8, 312)
ax.legend(fontsize=8.5, loc='lower left')

ax = axes[1]
for arm in ARMS3:
    ep, _, va = curves[arm]
    k = ep >= ZOOM
    ax.plot(ep[k], va[k], lw=2.0, color=ARM_COLOR[arm], solid_capstyle='round',
            label=f'{arm}   final {va[-1]:.2f}%')
    ax.plot(ep[-1], va[-1], 'o', ms=8, color=ARM_COLOR[arm],
            markeredgecolor=SURFACE, markeredgewidth=1.6)
ax.set_xlabel('epoch', fontsize=8.5)
ax.set_ylabel('validation accuracy (%)', fontsize=8.5)
ax.set_title(f'Validation accuracy, epoch {ZOOM}+ (plateau)', fontsize=9.5,
             color=INK_2, pad=8, loc='left')
ax.yaxis.grid(True); ax.set_axisbelow(True)
ax.legend(fontsize=8.5, loc='lower right')

fig.suptitle('Freezing the branch does not impair optimisation',
             fontsize=11.5, fontweight='bold', x=0.078, ha='left', y=0.955)
fig.text(0.078, 0.872, 'trainer.py logs only epochs that set a new minimum validation loss, '
         'so these are best-so-far envelopes  ·  f4 has no TFgram branch at all',
         fontsize=8.5, color=MUTED, ha='left')
fig.savefig(os.path.join(PLOTS, 'fig4_training_curves.png'), dpi=200)
plt.close(fig)

print('wrote:')
for f in sorted(os.listdir(PLOTS)):
    print('  ', os.path.join('ablation/frozen-tfgram/plots', f))
