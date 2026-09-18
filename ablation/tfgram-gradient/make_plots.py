#!/usr/bin/env python
"""Figures for the TFgram gradient probe.

Run from the repo root:
    /home/f74134118/anaconda3/bin/python ablation/tfgram-gradient/make_plots.py

Reads scores/probe.json only; nothing is hardcoded. Writes PNGs to plots/.

The argument these figures carry is TFgram vs Tgram, so exactly two categorical
hues are in play and every other branch is context in neutral. The dead-weight
control is drawn as a reference, not a series.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, 'plots')
os.makedirs(PLOTS, exist_ok=True)
R = json.load(open(os.path.join(HERE, 'scores', 'probe.json')))

# ---------------------------------------------------------------- palette ---
# Categorical slots 1-2 of the reference palette, shared with
# ../frozen-tfgram/make_plots.py so the figure sets agree.
# scripts/validate_palette.js "#2a78d6,#eb6834" --mode light --pairs all:
#   CVD dE 24.7 (protan), normal-vision dE 33.6, contrast >= 3:1 -> ALL PASS.
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8983'
GRID = '#e4e3df'
SUBJECT = '#2a78d6'   # TFgram  (slot 1)
BENCH = '#eb6834'     # Tgram   (slot 2)
NEUTRAL = '#c9c8c2'   # context branches
CONTROL = '#8a8983'   # the zero-gradient dead-weight control

# fig3/fig4 vary the *training regime*, which is ordinal (init -> converged),
# not categorical -- so they use one hue in two steps rather than reusing the
# blue/orange identity pair, which stays bound to TFgram/Tgram everywhere.
# validate_palette.js "#86b6ef,#2a78d6" --mode light --ordinal -> ALL PASS
# (monotone L, dL >= 0.06, light end 2.06:1 vs surface).
REGIME = [('grad_init', 'at random init', '#86b6ef'),
          ('grad_trained', 'at convergence', '#2a78d6')]
# Encoder modules only. head/arcface are the classifier and rotate far more;
# including them compresses the comparison that the figure is actually making.
ENCODER = ('TFgram_dead', 'MobileFaceNet', 'TFgram_live', 'Tgram')

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'text.color': INK, 'axes.labelcolor': INK_2, 'axes.edgecolor': GRID,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False,
})

LABEL = {'TFgram_live': 'TFgram', 'TFgram_dead': 'TFgram (dead)', 'Tgram': 'Tgram',
         'MobileFaceNet': 'MobileFaceNet', 'head': 'head', 'arcface': 'arcface'}
COLOR = {'TFgram_live': SUBJECT, 'Tgram': BENCH, 'TFgram_dead': CONTROL}


def color_of(g):
    return COLOR.get(g, NEUTRAL)


def finish(fig, path):
    fig.savefig(os.path.join(PLOTS, path), dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)


# --- fig 1: did TFgram actually rotate away from its init? -------------------
def fig1():
    d = R['drift']
    order = sorted(ENCODER, key=lambda g: 1 - d[g]['cosine'])
    rot = [1 - d[g]['cosine'] for g in order]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.barh(range(len(order)), rot, height=.5, color=[color_of(g) for g in order])
    ax.set_yticks(range(len(order)), [LABEL[g] for g in order])
    ax.set_xlabel('rotation away from random init   (1 - cosine, scale-invariant)')
    ax.grid(axis='x'); ax.set_axisbelow(True)
    span = max(rot)
    for i, (g, v) in enumerate(zip(order, rot)):
        ax.text(v + span * .015, i, f'cos {d[g]["cosine"]:.4f}', va='center',
                fontsize=8, color=INK_2)
    ax.set_xlim(0, span * 1.60)
    ax.axvline(0, color=CONTROL, lw=1.4, zorder=3)
    ax.annotate(f'control: {d["TFgram_dead"]["params"]/1e6:.1f}M params that never\n'
                f'receive a gradient. Pure WeightEMA decay\n'
                f'({d["TFgram_dead"]["norm_ratio"]:.3f}x) rotates nothing.',
                xy=(span * .004, .18), xytext=(span * .72, .58), fontsize=8, color=INK_2,
                va='center', arrowprops=dict(arrowstyle='-', color=CONTROL, lw=.9))
    ax.set_title('TFgram moved: more rotation than the backbone, same order as Tgram',
                 loc='left', fontsize=10.5, color=INK, pad=18)
    ax.text(0, 1.045, 'encoder modules only; the classifier rotates far more '
            f'(head cos {d["head"]["cosine"]:.2f}, arcface cos {d["arcface"]["cosine"]:.2f})',
            transform=ax.transAxes, fontsize=8, color=MUTED)
    finish(fig, 'fig1_rotation_vs_control.png')


# --- fig 2: raw gradient is not the thing that matters; AdamW normalizes it --
def fig2():
    g = R['grad_trained']
    order = sorted([k for k in g if not k.startswith('_')],
                   key=lambda k: -g[k]['grad_norm'])
    cols = [color_of(k) for k in order]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2), sharey=True)
    fig.subplots_adjust(wspace=.08)

    ax = axes[0]
    ax.barh(range(len(order)), [g[k]['grad_norm'] for k in order], height=.58, color=cols)
    ax.set_xscale('log')
    lo = min(g[k]['grad_norm'] for k in order)
    hi = max(g[k]['grad_norm'] for k in order)
    ticks = [t for t in (.2, .5, 1, 2, 5, 10, 20) if lo * .7 <= t <= hi * 1.02]
    ax.set_xticks(ticks, [f'{t:g}' for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(range(len(order)), [LABEL[k] for k in order])
    ax.set_xlabel('raw gradient norm  |g|   (log)')
    ax.set_title('What you would measure naively', loc='left', fontsize=9.5, color=INK_2)
    ax.grid(axis='x'); ax.set_axisbelow(True)
    ax.invert_yaxis()

    ax = axes[1]
    step = [g[k]['rel_step'] for k in order]
    ax.barh(range(len(order)), step, height=.58, color=cols)
    ax.tick_params(labelleft=False)
    ax.set_xlabel('effective AdamW step  |dW|/|W|')
    ax.set_title('What the optimizer actually does', loc='left', fontsize=9.5, color=INK_2)
    ax.grid(axis='x'); ax.set_axisbelow(True)
    ax.invert_yaxis()
    for i, v in enumerate(step):
        ax.text(v + max(step) * .015, i, f'{v:.2e}', va='center', fontsize=7.6, color=INK_2)
    ax.set_xlim(0, max(step) * 1.28)

    r_g = g['MobileFaceNet']['grad_norm'] / g['TFgram_live']['grad_norm']
    r_s = g['TFgram_live']['rel_step'] / g['MobileFaceNet']['rel_step']
    fig.suptitle(f"AdamW divides by sqrt(v): TFgram's gradient is {r_g:.0f}x smaller than the "
                 f"backbone's, its step {r_s:.2f}x larger",
                 x=.012, ha='left', fontsize=10.5, color=INK, y=1.04)
    finish(fig, 'fig2_adam_normalizes.png')


# --- fig 3: parity with Tgram, the branch that demonstrably matters ----------
def fig3():
    metrics = [('rel_grad', 'relative gradient\n|g|/|W|'),
               ('snr', 'gradient SNR'),
               ('rel_step', 'effective step\n|dW|/|W|')]
    regimes = REGIME
    x = np.arange(len(metrics))
    w = .34
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for j, (tag, name, col) in enumerate(regimes):
        vals = [R[tag]['TFgram_live'][m] / R[tag]['Tgram'][m] for m, _ in metrics]
        pos = x + (j - .5) * w
        ax.bar(pos, vals, width=w - .03, color=col, label=name)
        for p, v in zip(pos, vals):
            ax.text(p, v + .03, f'{v:.2f}x', ha='center', fontsize=8, color=INK_2)
    ax.axhline(1.0, color=INK_2, lw=1.2, ls=(0, (4, 3)))
    ax.text(len(metrics) - .42, 1.05, 'parity with Tgram', fontsize=8, color=INK_2, ha='right')
    ax.set_xticks(x, [n for _, n in metrics])
    ax.set_ylabel('TFgram / Tgram')
    ax.set_ylim(0, 1.72)
    ax.grid(axis='y'); ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=8.5)
    ax.set_title('TFgram receives the same order of learning signal as Tgram throughout training',
                 loc='left', fontsize=10.5, color=INK, pad=10)
    finish(fig, 'fig3_parity_with_tgram.png')


# --- fig 4: the honest null -- SNR could not discriminate -------------------
def fig4():
    floor = R['grad_trained']['_noise_floor']
    regimes = REGIME
    order = [k for k in R['grad_trained'] if not k.startswith('_')]
    order.sort(key=lambda k: -R['grad_trained'][k]['snr'])
    x = np.arange(len(order))
    w = .34
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    for j, (tag, name, col) in enumerate(regimes):
        pos = x + (j - .5) * w
        ax.bar(pos, [R[tag][k]['snr'] for k in order], width=w - .03, color=col, label=name)
        for p, k in zip(pos, order):          # selective: only the two branches at issue
            if k in ('TFgram_live', 'Tgram'):
                ax.text(p, R[tag][k]['snr'] + .004, f'{R[tag][k]["snr"]:.3f}',
                        ha='center', fontsize=7.4, color=INK_2)
    ax.axhline(floor, color=INK_2, lw=1.3, ls=(0, (4, 3)))
    ax.text(len(order) - .45, floor + .016,
            f'pure-noise floor over {R["steps"]} steps  = {floor:.3f}',
            fontsize=8, color=INK_2, ha='right')
    ax.set_xticks(x, [LABEL[k] for k in order])
    ax.set_ylabel('|mean g| / mean |g|')
    ax.grid(axis='y'); ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=8.5)
    ax.set_title('Null result: TFgram and Tgram sit at the noise floor in both regimes',
                 loc='left', fontsize=10.5, color=INK, pad=18)
    ax.text(0, 1.045, 'arcface clears the floor at init, so the metric is not broken '
            '-- it simply separates nothing among the feature branches',
            transform=ax.transAxes, fontsize=8, color=MUTED)
    finish(fig, 'fig4_snr_null.png')


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4()
