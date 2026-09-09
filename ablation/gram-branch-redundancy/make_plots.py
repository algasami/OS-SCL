#!/usr/bin/env python
"""Figures for the gram-branch-redundancy ablation.

Run from the repo root:
    /home/f74134118/anaconda3/bin/python ablation/gram-branch-redundancy/make_plots.py

Reads the eval CSVs written by eval.py, the CKA probe output, and the training
logs; writes PNGs to ablation/gram-branch-redundancy/plots/.
"""
import csv
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
PLOTS = os.path.join(HERE, 'plots')
os.makedirs(PLOTS, exist_ok=True)

# ---------------------------------------------------------------- palette ---
# Validated categorical slots 1-3 (scripts/validate_palette.js, light surface):
# CVD dE 9.2, normal-vision dE 24.0. Aqua sits below 3:1 on the light surface,
# so every aqua mark ships a visible direct label (the relief rule).
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8983'
GRID = '#e4e3df'
ARM_COLOR = {'f1': '#2a78d6', 'f3': '#eb6834', 'f4': '#1baf7a'}
POS, NEG = '#2a78d6', '#e34948'          # diverging poles
NEUTRAL = '#c9c8c2'
BLUE_RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
             '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
SEQ = LinearSegmentedColormap.from_list('blue_seq', BLUE_RAMP)

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'text.color': INK, 'axes.labelcolor': INK_2, 'axes.edgecolor': GRID,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False,
})

ARMS = [('f1', 'SCLTFSTgramMFN_test', 'f1  mel+Tgram+TFgram'),
        ('f3', 'SCLTFSTgramMFN_f3_mel_tfgram', 'f3  mel+TFgram'),
        ('f4', 'SCLTFSTgramMFN_f4_mel_tgram', 'f4  mel+Tgram')]
MACHINES = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
BRANCHES = ['Log-Mel', 'Tgram', 'TFgram']


def load_scores():
    out = {}
    for ds in ('dev', 'eval'):
        for arm, d, _ in ARMS:
            p = f'check_points/{d}/eval_{ds}.csv'
            for r in csv.DictReader(open(p)):
                if r['scope'] in ('machine', 'overall'):
                    key = 'ALL' if r['scope'] == 'overall' else r['machine']
                    out[(ds, arm, key)] = {k: float(r[k]) * 100 for k in ('AUC', 'pAUC', 'mAUC')}
    return out


def load_cka():
    txt = open(os.path.join(RESULTS, 'cka.txt')).read()
    cka, r2, section = {'clip': {}, 'frame': {}}, {}, None
    for line in txt.splitlines():
        if 'clip level' in line:
            section = 'clip'
        elif 'frame level' in line:
            section = 'frame'
        elif 'predictability' in line:
            section = 'r2'
        m = re.search(r'CKA\(\s*([\w-]+)\s*,\s*([\w-]+)\s*\)\s*=\s*([\d.]+)', line)
        if m and section in ('clip', 'frame'):
            cka[section][(m.group(1), m.group(2))] = float(m.group(3))
        m = re.search(r'([\w-]+)\s*->\s*([\w-]+)\s*:\s*R\^2\s*=\s*([\d.]+)', line)
        if m:
            r2[(m.group(1), m.group(2))] = float(m.group(3))
    return cka, r2


def cka_matrix(pairs):
    M = np.eye(len(BRANCHES))
    for (a, b), v in pairs.items():
        i, j = BRANCHES.index(a), BRANCHES.index(b)
        M[i, j] = M[j, i] = v
    return M


def load_curve(arm):
    txt = open(os.path.join(RESULTS, f'train_{arm}.log'), errors='ignore').read().replace('\r', '\n')
    rows = re.findall(r'EPOCH:\s*(\d+)\s*\|\s*Train_loss:\s*([\d.]+)\s*\|\s*Train_accuracy:\s*([\d.]+)'
                      r'\s*\|\s*Valid_loss:\s*([\d.]+)\s*\|\s*Valid_accuracy:\s*([\d.]+)', txt)
    a = np.array([[float(x) for x in r] for r in rows])
    return a[:, 0], a[:, 3], a[:, 4] * 100


S = load_scores()
CKA, R2 = load_cka()


# ------------------------------------------------- fig 1: branch redundancy ---
fig = plt.figure(figsize=(11.2, 3.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5], wspace=0.42,
                      left=0.085, right=0.975, top=0.78, bottom=0.16)

for k, (level, title) in enumerate([('clip', 'Clip level\n(N × 128·313)'),
                                    ('frame', 'Frame level\n(N·313 × 128)')]):
    ax = fig.add_subplot(gs[0, k])
    M = cka_matrix(CKA[level])
    ax.imshow(M, cmap=SEQ, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=9.5,
                    color='#ffffff' if M[i, j] > 0.55 else INK,
                    fontweight='bold' if i != j else 'normal')
    ax.set_xticks(range(3), BRANCHES, fontsize=8.5)
    ax.set_yticks(range(3), BRANCHES, fontsize=8.5)
    ax.set_title(title, fontsize=9, color=INK_2, pad=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)

ax = fig.add_subplot(gs[0, 2])
items = sorted(R2.items(), key=lambda kv: kv[1])
lbls = [f'{a} → {b}' for (a, b), _ in items]
vals = [v for _, v in items]
hi = max(range(len(vals)), key=lambda i: vals[i])
cols = [POS if i == hi else NEUTRAL for i in range(len(vals))]
ax.barh(range(len(vals)), vals, color=cols, height=0.6)
for i, v in enumerate(vals):
    ax.text(v + 0.012, i, f'{v:.3f}', va='center', fontsize=9,
            color=INK if i == hi else INK_2, fontweight='bold' if i == hi else 'normal')
ax.set_yticks(range(len(vals)), lbls, fontsize=8.5)
ax.set_xlim(0, 0.78)
ax.set_xlabel('R²  (frame-level linear fit)', fontsize=8.5)
ax.set_title('How much of the target is a linear\nfunction of the source?', fontsize=9,
             color=INK_2, pad=8)
ax.xaxis.grid(True); ax.set_axisbelow(True); ax.spines['left'].set_visible(False)

fig.suptitle('TFgram duplicates the Log-Mel channel; Tgram is orthogonal to both',
             fontsize=11.5, fontweight='bold', x=0.085, ha='left', y=0.955)
fig.text(0.085, 0.885, 'Linear CKA between the three input branches of the trained '
         'fussion=1 model  ·  656 clips, 16 per machine ID',
         fontsize=8.5, color=MUTED, ha='left')
fig.savefig(os.path.join(PLOTS, 'fig1_branch_redundancy.png'), dpi=200)
plt.close(fig)


# ------------------------------------------------ fig 2: per-machine scores ---
# Arms are offset within each machine row: several arms land within 0.1 pp of
# each other (slider, valve), and drawn at one y they occlude each other.
fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.8), sharey=True)
fig.subplots_adjust(left=0.115, right=0.98, top=0.85, bottom=0.07, hspace=0.28, wspace=0.10)
OFFSET = {'f1': 0.26, 'f3': 0.0, 'f4': -0.26}
ypos = {m: y for m, y in zip(MACHINES, np.arange(len(MACHINES))[::-1])}

for r, metric in enumerate(['AUC', 'mAUC']):
    for c, ds in enumerate(['dev', 'eval']):
        ax = axes[r, c]
        lo = min(S[(ds, a, m)][metric] for a, _, _ in ARMS for m in MACHINES)
        for i, m in enumerate(MACHINES):
            y = ypos[m]
            if i % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color='#f3f2ef', zorder=0, lw=0)
            vs = {arm: S[(ds, arm, m)][metric] for arm, _, _ in ARMS}
            spread = max(vs.values()) - min(vs.values())
            for arm, _, _ in ARMS:
                yy = y + OFFSET[arm]
                ax.plot(vs[arm], yy, 'o', ms=8.5, color=ARM_COLOR[arm], zorder=3,
                        markeredgecolor=SURFACE, markeredgewidth=1.6)
                # selective direct labels: only rows where the arms actually differ
                if spread > 1.0:
                    ax.text(vs[arm] + 0.45, yy, f'{vs[arm]:.1f}', va='center',
                            fontsize=7.6, color=INK_2, zorder=4)
        ax.set_yticks(list(ypos.values()), list(ypos.keys()), fontsize=9)
        ax.set_ylim(-0.62, len(MACHINES) - 0.38)
        ax.xaxis.grid(True); ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False)
        ax.set_title(f'{metric}  \u00b7  {ds} set', fontsize=9.5, color=INK_2, pad=8, loc='left')
        ax.set_xlabel(f'{metric} (%)', fontsize=8.5)
        ax.set_xlim(lo - 2.5, 104.0)

handles = [plt.Line2D([], [], marker='o', ls='', ms=8.5, color=ARM_COLOR[a],
                      markeredgecolor=SURFACE, markeredgewidth=1.6, label=lab)
           for a, _, lab in ARMS]
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.113, 0.908),
           ncol=3, fontsize=9, handletextpad=0.4, columnspacing=1.8)
fig.suptitle('Dropping TFgram costs nothing; dropping Tgram costs fan and ToyConveyor',
             fontsize=11.5, fontweight='bold', x=0.115, ha='left', y=0.972)
fig.text(0.115, 0.940, 'Per-machine scores, one seed per arm  \u00b7  higher is better  '
         '\u00b7  values shown where the arms differ by more than 1 point',
         fontsize=8.5, color=MUTED, ha='left')
fig.savefig(os.path.join(PLOTS, 'fig2_per_machine_scores.png'), dpi=200)
plt.close(fig)


# ----------------------------------------------------- fig 3: overall deltas ---
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharex=True)
fig.subplots_adjust(left=0.145, right=0.975, top=0.755, bottom=0.145, wspace=0.30)
metrics = ['AUC', 'pAUC', 'mAUC']
rows = [(m, arm) for m in metrics for arm in ('f3', 'f4')]

for c, ds in enumerate(['dev', 'eval']):
    ax = axes[c]
    y = np.arange(len(rows))[::-1]
    d = [S[(ds, arm, 'ALL')][m] - S[(ds, 'f1', 'ALL')][m] for m, arm in rows]
    ax.barh(y, d, color=[POS if v >= 0 else NEG for v in d], height=0.62)
    for yy, v in zip(y, d):
        off = 0.055 if v >= 0 else -0.055
        ax.text(v + off, yy, f'{v:+.2f}', va='center',
                ha='left' if v >= 0 else 'right', fontsize=8.8, color=INK_2)
    ax.axvline(0, color=MUTED, lw=1.0)
    ax.set_yticks(y, [f'{m}  ·  {arm}' for m, arm in rows], fontsize=8.8)
    ax.set_xlim(-2.3, 2.3)
    ax.set_xlabel('change vs f1 (percentage points)', fontsize=8.5)
    ax.set_title(f'{ds} set', fontsize=9.5, color=INK_2, pad=7, loc='left')
    ax.xaxis.grid(True); ax.set_axisbelow(True); ax.spines['left'].set_visible(False)

fig.suptitle('f4 matches the full model everywhere; f3 loses on dev, gains on eval',
             fontsize=11.5, fontweight='bold', x=0.145, ha='left', y=0.955)
fig.text(0.145, 0.875, 'Blue = better than the 3-channel baseline, red = worse.  '
         'The dev/eval sign flip for f3 is why n=1 cannot settle a ~1-point effect.',
         fontsize=8.5, color=MUTED, ha='left')
fig.savefig(os.path.join(PLOTS, 'fig3_overall_deltas.png'), dpi=200)
plt.close(fig)


# ------------------------------------------------------ fig 4: training curves ---
# Loss spans 18.6 -> 0.34, so the late-training separation the headline is about
# is invisible on a linear axis; panel A is log-scaled and panel B zooms the
# accuracy plateau. Final values ride in the legend, which cannot collide.
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
fig.subplots_adjust(left=0.075, right=0.985, top=0.745, bottom=0.145, wspace=0.20)
curves = {a: load_curve(a) for a in ('f3', 'f4')}
ZOOM = 180

ax = axes[0]
for arm in ('f3', 'f4'):
    ep, vl, _ = curves[arm]
    ax.plot(ep, vl, lw=2.0, color=ARM_COLOR[arm], solid_capstyle='round',
            label=f'{arm}  final {vl[-1]:.3f}')
    ax.plot(ep[-1], vl[-1], 'o', ms=8, color=ARM_COLOR[arm],
            markeredgecolor=SURFACE, markeredgewidth=1.6)
ax.set_yscale('log')
ax.set_xlabel('epoch', fontsize=8.5)
ax.set_ylabel('best validation loss (log scale)', fontsize=8.5)
ax.set_title('Validation loss, full run', fontsize=9.5, color=INK_2, pad=8, loc='left')
ax.yaxis.grid(True); ax.set_axisbelow(True); ax.set_xlim(-8, 312)
ax.legend(fontsize=8.8, loc='lower left')

ax = axes[1]
for arm in ('f3', 'f4'):
    ep, _, va = curves[arm]
    k = ep >= ZOOM
    ax.plot(ep[k], va[k], lw=2.0, color=ARM_COLOR[arm], solid_capstyle='round',
            label=f'{arm}  final {va[-1]:.2f}%')
    ax.plot(ep[-1], va[-1], 'o', ms=8, color=ARM_COLOR[arm],
            markeredgecolor=SURFACE, markeredgewidth=1.6)
ax.set_xlabel('epoch', fontsize=8.5)
ax.set_ylabel('validation accuracy (%)', fontsize=8.5)
ax.set_title(f'Validation accuracy, epoch {ZOOM}+ (plateau)', fontsize=9.5,
             color=INK_2, pad=8, loc='left')
ax.yaxis.grid(True); ax.set_axisbelow(True)
ax.legend(fontsize=8.8, loc='lower right')

fig.suptitle('f4 reaches a lower validation loss than f3, in 59% of the wall-clock time',
             fontsize=11.5, fontweight='bold', x=0.075, ha='left', y=0.955)
fig.text(0.075, 0.872, 'trainer.py logs only epochs that set a new minimum validation '
         'loss, so these are best-so-far envelopes  \u00b7  f3 5h24m, f4 3h10m',
         fontsize=8.5, color=MUTED, ha='left')
fig.savefig(os.path.join(PLOTS, 'fig4_training_curves.png'), dpi=200)
plt.close(fig)


print('wrote:')
for f in sorted(os.listdir(PLOTS)):
    print('  ', os.path.join('ablation/gram-branch-redundancy/plots', f))
