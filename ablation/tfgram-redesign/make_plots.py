#!/usr/bin/env python
"""Figures for the TFgram redesign ablation.

Run from the repo root:
    /home/f74134118/anaconda3/bin/python ablation/tfgram-redesign/make_plots.py

Reads scores/ only (cka_pilot.json, per-run curve.csv, and the reference CKA in
../gram-branch-redundancy/scores/cka.txt). A figure whose inputs are missing is
skipped, not fatal.

relu_max is the paper's TFgram, i.e. the baseline, so it is drawn achromatic.
The three redesigned cells take the three categorical slots that validate under
--pairs all, in the same order as the other figure sets.
"""
import csv
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
# scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light --pairs all
#   -> passes (CVD dE 9.2, normal-vision dE 24.0); aqua is 2.74:1 on the
#   surface, so every aqua mark is direct-labelled.
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8983'
GRID = '#e4e3df'
BASE = '#52514e'     # relu_max, the paper's branch -- achromatic baseline
CELLS = [('relu_max', 'ReLU + max  (paper)', BASE),
         ('leaky_max', 'Leaky + max', '#2a78d6'),
         ('relu_avg', 'ReLU + avg', '#eb6834'),
         ('leaky_avg', 'Leaky + avg', '#1baf7a')]

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


def pilot_runs():
    path = os.path.join(SCORES, 'cka_pilot.json')
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return {f"{r['tf_act']}_{r['tf_pool']}": (name, r) for name, r in d.items()}


def reference_cka():
    """Trained f1 on the original host (../gram-branch-redundancy), for context."""
    path = os.path.join(HERE, '..', 'gram-branch-redundancy', 'scores', 'cka.txt')
    if not os.path.exists(path):
        return None
    txt = open(path).read()
    out, level = {}, None
    for line in txt.splitlines():
        if 'clip level' in line:
            level = 'clip'
        elif 'frame level' in line:
            level = 'frame'
        m = re.search(r'CKA\((\S+)\s*,\s*(\S+)\s*\)\s*=\s*([\d.]+)', line)
        if m and level:
            out[(level, m.group(1), m.group(2))] = float(m.group(3))
    return out


# ------------------------------------------------- fig1: CKA at random init ---
def fig1():
    runs, ref = pilot_runs(), reference_cka()
    if not runs:
        print('skip fig1: no scores/cka_pilot.json'); return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9), sharey=True)
    for ax, level, title in ((axes[0], 'clip', 'clip level'), (axes[1], 'frame', 'frame level')):
        for i, (key, label, col) in enumerate(CELLS):
            if key not in runs:
                continue
            r = runs[key][1]['init']
            v = r[f'cka_{level}/Log-Mel~TFgram']
            ax.plot([0, v], [i, i], color=GRID, lw=2, solid_capstyle='round', zorder=1)
            ax.scatter([v], [i], s=70, color=col, edgecolor=SURFACE, lw=2, zorder=3)
            t = runs[key][1].get('trained')
            if t:  # EMA ckpt, ~55% init: drawn hollow so it isn't read as a trained result
                tv = t[f'cka_{level}/Log-Mel~TFgram']
                ax.scatter([tv], [i], s=70, facecolor='none', edgecolor=col, lw=1.6, zorder=4)
                ax.text(max(v, tv) + 0.03, i, f'{v:.2f} (ckpt {tv:.2f})', va='center', color=INK_2, fontsize=8)
            else:
                ax.text(v + 0.03, i, f'{v:.2f}', va='center', color=INK_2, fontsize=8.5)
        # context: what a decorrelated branch looks like, and trained f1 elsewhere
        tg = np.mean([runs[k][1]['init'][f'cka_{level}/Log-Mel~Tgram'] for k in runs])
        ax.axvline(tg, color=MUTED, lw=1, ls=(0, (2, 2)), zorder=0)
        ax.text(tg + 0.015, len(CELLS) - 0.35, f'Tgram vs Log-Mel: {tg:.2f}', color=MUTED, fontsize=7.5, va='center')
        if ref and (level, 'Log-Mel', 'TFgram') in ref:
            f1 = ref[(level, 'Log-Mel', 'TFgram')]
            ax.axvline(f1, color=MUTED, lw=1, zorder=0)
            ax.text(f1 - 0.015, -0.75, f'trained f1, PRO 6000 host: {f1:.2f}',
                    color=MUTED, fontsize=7.5, va='center', ha='right')
        ax.set_xlim(0, 1.0)
        ax.set_title(f'CKA(Log-Mel, TFgram), {title}', loc='left', fontsize=9, color=INK_2)
        ax.grid(axis='x'); ax.set_axisbelow(True)
    axes[1].scatter([], [], s=50, facecolor='none', edgecolor=MUTED, lw=1.6,
                    label='hollow: pilot EMA checkpoint, ~55% init weights at save (void gate)')
    fig.legend(*axes[1].get_legend_handles_labels(), loc='lower left', bbox_to_anchor=(0.01, -0.09),
               fontsize=7.5, frameon=False, labelcolor=INK_2)
    axes[0].set_yticks(range(len(CELLS)))
    axes[0].set_yticklabels([c[1] for c in CELLS])
    axes[0].set_ylim(-1.05, len(CELLS) - 0.15)
    axes[0].invert_yaxis()
    fig.suptitle('At random init, all four TFgram variants already look like Log-Mel',
                 x=0.01, ha='left', fontsize=10.5, color=INK, y=1.09)
    fig.text(0.01, 1.01, 'Seed 2024, 656 training clips. The act/pool switches move init CKA by '
             'about 0.01; most of the similarity is there before any training.',
             ha='left', color=INK_2, fontsize=8)
    save(fig, 'fig1_init_cka.png')


# ------------------------------------- fig2: why the 30-epoch gate is void ---
def fig2():
    runs = pilot_runs()
    if not runs:
        print('skip fig2: no scores/cka_pilot.json'); return
    best = {}
    for key, (name, _) in runs.items():
        p = os.path.join(SCORES, name, 'curve.csv')
        if os.path.exists(p):
            rows = list(csv.DictReader(open(p)))
            best[key] = int(rows[-1]['epoch'])
    # WeightEMA: ema = a*ema + (1-a)*student each step, starting from the init,
    # so the init's share of the EMA weights after s steps is a**s.
    alpha, steps_per_epoch = 0.9999, 539   # losses.py; 539 batches/epoch in the train logs
    ep = np.arange(0, 301)
    share = alpha ** (ep * steps_per_epoch)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(ep, share * 100, color='#2a78d6', lw=2)
    for e, lab in ((30, 'pilot ends\n(30 ep)'), (300, 'full run\n(300 ep)')):
        y = alpha ** (e * steps_per_epoch) * 100
        ax.scatter([e], [y], s=60, color='#2a78d6', edgecolor=SURFACE, lw=2, zorder=3)
        ax.annotate(f'{lab}: {y:.1f}%', (e, y), xytext=(8, 10), textcoords='offset points',
                    color=INK_2, fontsize=8, ha='left' if e < 200 else 'right')
    if best:
        e = int(np.median(list(best.values())))
        y = alpha ** (e * steps_per_epoch) * 100
        ax.scatter([e], [y], s=70, color=INK, edgecolor=SURFACE, lw=2, zorder=4)
        ax.annotate(f'saved checkpoints (best valid loss,\nepoch {min(best.values())}-{max(best.values())}): '
                    f'~{y:.0f}% init', (e, y), xytext=(14, 4), textcoords='offset points',
                    color=INK, fontsize=8)
    ax.set_xlim(0, 305); ax.set_ylim(0, 105)
    ax.set_xlabel('epoch'); ax.set_ylabel('share of EMA weights still at init (%)')
    ax.grid(axis='y'); ax.set_axisbelow(True)
    ax.set_title('The pilot checkpoints were mostly random init', loc='left', fontsize=10.5,
                 color=INK, pad=20)
    ax.text(0, 1.03, 'EMA alpha 0.9999 at 539 steps/epoch; the EMA net is what trainer.py validates and saves.',
            transform=ax.transAxes, color=INK_2, fontsize=8)
    save(fig, 'fig2_ema_init_share.png')


# ------------------------- fig3: what it takes to stop computing an envelope ---
def fig3():
    path = os.path.join(SCORES, 'init_probe.json')
    if not os.path.exists(path):
        print('skip fig3: no scores/init_probe.json'); return
    rows = json.load(open(path))['results']
    runs = pilot_runs()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1), sharey=True)
    pools = (('max', 'max pool (paper)', BASE), ('avg', 'avg pool', '#2a78d6'))
    for ax, level in ((axes[0], 'clip'), (axes[1], 'frame')):
        for pool, label, col in pools:
            sl = sorted({r['slope'] for r in rows if r['pool'] == pool})
            v = np.array([[r[f'cka_{level}'] for r in rows if r['pool'] == pool and r['slope'] == x] for x in sl])
            ax.fill_between(sl, v.min(1), v.max(1), color=col, alpha=0.15, lw=0)
            ax.plot(sl, v.mean(1), color=col, lw=2, marker='o', ms=5, mec=SURFACE, mew=1.5)
            ax.text(sl[-1] + 0.03, v.mean(1)[-1], label, color=col, fontsize=8, va='center')
        if runs:
            tg = np.mean([runs[k][1]['init'][f'cka_{level}/Log-Mel~Tgram'] for k in runs])
            ax.axhline(tg, color=MUTED, lw=1, ls=(0, (2, 2)), zorder=0)
            ax.text(0.01, tg + 0.025, f'Tgram vs Log-Mel: {tg:.2f}', color=MUTED, fontsize=7.5)
        for x, lab in ((0.0, 'ReLU'), (0.2, 'Leaky 0.2\n(the 2x2)')):
            ax.axvline(x, color=GRID, lw=1, zorder=0)
            ax.text(x + 0.01, 0.93, lab, color=MUTED, fontsize=7.5, va='top')
        ax.set_xlim(-0.03, 1.32); ax.set_ylim(-0.03, 1.0)
        ax.set_xticks([0, 0.2, 0.5, 0.8, 1.0])
        ax.set_xlabel('activation negative slope  (1 = fully linear)')
        ax.set_title(f'CKA(Log-Mel, TFgram) at init, {level} level', loc='left', fontsize=9, color=INK_2)
        ax.grid(axis='y'); ax.set_axisbelow(True)
    fig.suptitle('Only a fully linear stack with linear pooling stops TFgram looking like Log-Mel',
                 x=0.01, ha='left', fontsize=10.5, color=INK, y=1.09)
    fig.text(0.01, 1.01, 'Random init, no training; mean of 3 seeds, band = min-max. Any rectification, '
             'or max pooling alone, yields an energy envelope.', ha='left', color=INK_2, fontsize=8)
    save(fig, 'fig3_slope_sweep.png')


# ------------- fig4: a decorrelated TFgram is not a second Tgram (at init) ---
def fig4():
    tp, bp = (os.path.join(SCORES, f) for f in ('tgram_probe.json', 'band_probe.json'))
    if not (os.path.exists(tp) and os.path.exists(bp)):
        print('skip fig4: needs scores/tgram_probe.json and scores/band_probe.json'); return
    rows = json.load(open(tp))['results']
    variants = [('relu_max', 'ReLU + max (paper)'), ('leaky_avg', 'Leaky + avg'),
                ('linear_max', 'linear + max'), ('linear_avg', 'linear + avg'),
                ('linear_first', 'linear + avg,\nLeaky after')]
    refs = [('Log-Mel', 'Log-Mel'), ('Tgram', 'Tgram'), ('Tgram_front', 'Tgram front end')]
    M = np.array([[np.mean([r['pairs'][f'{ref} ~ TF:{v}']['r2_a_to_b'] for r in rows]) for ref, _ in refs]
                  for v, _ in variants])
    ctrl = np.mean([r['pairs']['control: Tgram_front ~ Tgram_front_other_seed']['r2_a_to_b'] for r in rows])

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 3.3), gridspec_kw={'width_ratios': [1.05, 1]})
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('seq', ['#f3f2ee', '#2a78d6', '#0d3a73'])
    a.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=8.5,
                   color=SURFACE if M[i, j] > 0.55 else INK)
    a.set_xticks(range(len(refs))); a.set_xticklabels([r[1] for r in refs])
    a.set_yticks(range(len(variants))); a.set_yticklabels([v[1] for v in variants])
    a.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)
    for sp in a.spines.values():
        sp.set_visible(False)
    a.set_title(f'share of TFgram predictable from each branch (frame R²)\n'
                f'control, two random Tgram front ends: {ctrl:.2f}', loc='left', fontsize=8.5,
                color=INK_2, pad=24)

    brows = json.load(open(bp))['results']
    for label, name, col in (('relu_max', 'ReLU + max (paper)', BASE), ('linear_avg', 'linear + avg', '#2a78d6')):
        cut = [int(c) for c in brows[0]['var_share_surviving']]
        v = np.array([[r['var_share_surviving'][str(c)] for c in cut] for r in brows if r['branch'] == label])
        x = np.arange(len(cut))
        b.fill_between(x, v.min(0) * 100, v.max(0) * 100, color=col, alpha=0.15, lw=0)
        b.plot(x, v.mean(0) * 100, color=col, lw=2, marker='o', ms=5, mec=SURFACE, mew=1.5)
        b.text(x[-1] + 0.1, v.mean(0)[-1] * 100 + (6 if label == 'linear_avg' else -6), name,
               color=col, fontsize=8, va='center')
        b.set_xticks(x); b.set_xticklabels(['none' if c == 0 else f'{c} Hz' for c in cut])
    b.set_xlim(-0.2, len(cut) + 0.6); b.set_ylim(0, 108)
    b.set_xlabel('input high-passed at'); b.set_ylabel('output variance surviving (%)')
    b.grid(axis='y'); b.set_axisbelow(True)
    b.set_title('which input band drives the output', loc='left', fontsize=8.5, color=INK_2)

    fig.suptitle('At init, a linear avg-pooled TFgram is neither Log-Mel nor Tgram, yet still responds to 31-300 Hz',
                 x=0.01, ha='left', fontsize=10.5, color=INK, y=1.04)
    fig.text(0.01, 0.95, 'Random init, 3 seeds. Left: frame-level linear R², reference -> TFgram. '
             'Right: it responds to the same 31-300 Hz band as the paper\'s stack.',
             ha='left', color=INK_2, fontsize=8)
    fig.tight_layout()
    save(fig, 'fig4_not_tgram.png')


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    fig4()
