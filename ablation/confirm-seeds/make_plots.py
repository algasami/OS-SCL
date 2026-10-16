#!/usr/bin/env python
"""Figures for confirm-seeds.

Run from the repo root:
    /home/f74134118/anaconda3/bin/python ablation/confirm-seeds/make_plots.py

Reads scores/ only (summary.csv, ensemble_curve.csv, bootstrap.json). A figure
whose inputs are missing is skipped, not fatal. Palette follows the repo's
make_plots.py files: E1 (the old, held-out-statistics ensemble rule) takes the
first categorical slot (blue), the training-statistic rules S1' / E2' (the
adopted ones) the second (orange); R0 / S0 baselines and the paper are
achromatic (paper = dashed line).
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
N4 = [f'SCLTFSTgramMFN_cs_relu_max_s{s}' for s in (2028, 2029, 2030, 2031)]
O4 = [f'SCLTFSTgramMFN_tfl_relu_max_s{s}' for s in (2024, 2025, 2026, 2027)]

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


def paper_line(ax, y, text, xfrac=1.0, ha='right'):
    ax.axhline(y, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.annotate(text, (xfrac, y), xycoords=('axes fraction', 'data'),
                xytext=(-4 if ha == 'right' else 4, 4),
                textcoords='offset points', ha=ha, fontsize=8, color=INK)


def short(r):
    return 's' + r.rsplit('_s', 1)[1]


def fig1_single(summ):
    need = [(r, ds, k) for r in N4 + O4 for ds in ('dev', 'eval') for k in ('S0', 'S1p')]
    if any(k not in summ for k in need):
        print('skip fig1: summary rows missing')
        return
    deltas = {ds: [float(summ[(r, ds, 'S1p')]['AUC']) - float(summ[(r, ds, 'S0')]['AUC'])
                   for r in N4 + O4] for ds in ('dev', 'eval')}
    all_pos = all(d > 0 for ds in deltas for d in deltas[ds])
    n_pos = sum(d > 0 for ds in deltas for d in deltas[ds])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    for ax, ds in zip(axes, ['dev', 'eval']):
        for i, r in enumerate(N4 + O4):
            new = r in N4
            a = float(summ[(r, ds, 'S0')]['AUC'])
            b = float(summ[(r, ds, 'S1p')]['AUC'])
            ax.plot([0, 1], [a, b], color=MUTED, lw=0.9, zorder=1)
            for x, v, col in ((0, a, INK_2), (1, b, ORANGE)):
                ax.scatter([x], [v], s=46, marker='o', zorder=3,
                           facecolor=col if new else SURFACE, edgecolor=col, lw=1.5)
        if ds == 'dev':
            paper_line(ax, PAPER_DEV['AUC'], 'paper 95.71', xfrac=0.0, ha='left')
        ax.set_xticks([0, 1], ['S0 (R0 baseline)', 'S1′ (train-clip statistics)'])
        ax.set_xlim(-0.35, 1.35)
        ax.grid(axis='y')
        d = deltas[ds]
        ax.set_title(f'{ds}: S1′ − S0 = {min(d):+.2f} to {max(d):+.2f} points (8 checkpoints)', loc='left',
                     fontsize=9.5, color=INK)
    axes[0].set_ylabel('overall AUC (%)')
    h = [plt.Line2D([], [], marker='o', ls='', color=INK_2, mfc=INK_2, ms=7, label='N4 (fresh seeds 2028–2031)'),
         plt.Line2D([], [], marker='o', ls='', color=INK_2, mfc=SURFACE, ms=7, mew=1.5,
                    label='O4 (seeds 2024–2027, already seen)')]
    fig.legend(handles=h, loc='upper left', bbox_to_anchor=(0.06, 1.0), ncol=2, fontsize=8.5)
    fig.suptitle(('S1′ beats R0 on all 8 checkpoints on both sets' if all_pos else
                  f'S1′ beats R0 on {n_pos} of 16 checkpoint-set pairs'),
                 x=0.06, y=1.12, ha='left', fontsize=11, color=INK)
    fig.text(0.06, 1.045, "Each line is one checkpoint scored two ways: grey S0 = R0, eval.py's raw score, "
             'orange S1′ = R0 fused with kNN using training-clip statistics.', fontsize=8.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig1_single_model.png')


def fig2_eval_mean(summ, boot):
    if boot is None or any((r, 'eval', k) not in summ for r in N4 for k in ('S0', 'S1p')):
        print('skip fig2: inputs missing')
        return
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    s0 = [float(summ[(r, 'eval', 'S0')]['mean_AUC_pAUC']) for r in N4]
    s1 = [float(summ[(r, 'eval', 'S1p')]['mean_AUC_pAUC']) for r in N4]
    x = np.arange(4)
    for i in range(4):
        ax.plot([i, i], [s0[i], s1[i]], color=GRID, lw=2.5, zorder=1)
    ax.scatter(x, s0, s=50, color=INK_2, edgecolor=SURFACE, lw=1.2, zorder=3, label='S0 (R0 baseline)')
    ax.scatter(x, s1, s=60, color=ORANGE, edgecolor=SURFACE, lw=1.5, zorder=4, label='S1′')
    for i in range(4):
        ax.annotate(f'{s1[i]:.2f}', (i, s1[i]), xytext=(9, 0), textcoords='offset points',
                    va='center', fontsize=8, color=INK)
        ax.annotate(f'{s0[i]:.2f}', (i, s0[i]), xytext=(9, 0), textcoords='offset points',
                    va='center', fontsize=8, color=INK_2)
    ev = boot['eval']
    items = [(4.8, ev['mean_S1p_mean_AUC_pAUC'], ORANGE, 'D', 'mean of the\n4 S1′ models'),
             (5.8, ev['E1_mean_AUC_pAUC'], BLUE, 'o', 'E1[N4]'),
             (6.8, ev['E2p_mean_AUC_pAUC'], ORANGE, 'o', 'E2′[N4]')]
    for xx, (pt, lo, hi), col, mk, _ in items:
        ax.plot([xx, xx], [lo, hi], color=col, lw=2.2, solid_capstyle='round', zorder=3)
        ax.scatter([xx], [pt], s=62, marker=mk, color=col, edgecolor=SURFACE, lw=1.5, zorder=4)
        ax.annotate(f'{pt:.2f}', (xx, hi), xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=8, color=INK)
    ax.axvline(3.9, color=GRID, lw=0.8)
    ax.set_xticks(list(x) + [i[0] for i in items],
                  [short(r) for r in N4] + [i[4] for i in items], fontsize=8)
    ax.set_xlim(-0.6, 7.5)
    paper_line(ax, PAPER_EVAL_MEAN, 'paper 93.22')
    ax.set_ylim(92.0, 95.2)
    ax.grid(axis='y')
    ax.set_ylabel('eval mean(AUC, pAUC) (%)')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.93), fontsize=8.5)
    n_above = sum(v > PAPER_EVAL_MEAN for v in s1)
    n_s0 = sum(v > PAPER_EVAL_MEAN for v in s0)
    fig.suptitle(f'On fresh seeds S1′ lifts {n_above} of 4 single models past the paper\'s eval number '
                 f'(S0: {n_s0} of 4)', x=0.06, y=1.07, ha='left', fontsize=11, color=INK)
    fig.text(0.06, 1.0, 'Left: the 4 N4 checkpoints, no interval. Right: bootstrap 95% CIs over test clips '
             '(test-set sampling only, not seed variance).', fontsize=8.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig2_eval_mean_vs_paper.png')


def fig3_size(curve):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    panels = [('dev', 'AUC', 'dev: overall AUC', PAPER_DEV['AUC'], 'paper 95.71'),
              ('eval', 'mean_AUC_pAUC', 'eval: mean of AUC and pAUC', PAPER_EVAL_MEAN, 'paper 93.22')]
    rules = [('E1', BLUE, -0.17, 'E1 (held-out statistics)'), ('E2p', ORANGE, 0.17, 'E2′ (training statistics)')]
    ns = list(range(1, 9))
    for ax, (ds, key, title, pap, plab) in zip(axes, panels):
        for rule, col, off, lab in rules:
            rr = [r for r in curve if r['rule'] == rule and r['dataset'] == ds]
            if not rr:
                continue
            rng = np.random.default_rng(0)
            xs = [int(r['n']) + off + rng.uniform(-0.1, 0.1) for r in rr]
            ax.scatter(xs, [float(r[key]) for r in rr], s=7, color=col, alpha=0.2, lw=0, zorder=2)
            means = [np.mean([float(r[key]) for r in rr if int(r['n']) == n]) for n in ns]
            ax.plot(ns, means, color=col, lw=2, marker='o', ms=4, zorder=4, label=lab)
        paper_line(ax, pap, plab, xfrac=0.0, ha='left')
        ax.set_xticks(ns)
        ax.set_xlabel('models in the ensemble (n)')
        ax.grid(axis='y')
        ax.set_title(title, loc='left', fontsize=9.5, color=INK)
    axes[0].set_ylabel('score (%)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper left', bbox_to_anchor=(0.06, 1.0), ncol=2, fontsize=8.5)
    fig.suptitle('E2′ sits above E1 at every ensemble size; both rise with n',
                 x=0.06, y=1.12, ha='left', fontsize=11, color=INK)
    fig.text(0.06, 1.045, 'Lines: mean over all subsets of the 8 relu checkpoints (N4 + O4) of size n. '
             'Dots: the individual subsets.', fontsize=8.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig3_ensemble_size.png')


def fig4_vs_paper(boot):
    items = [('dev', 'AUC', 'dev AUC', PAPER_DEV['AUC']),
             ('dev', 'pAUC', 'dev pAUC', PAPER_DEV['pAUC']),
             ('dev', 'mAUC', 'dev mAUC', PAPER_DEV['mAUC']),
             ('eval', 'mean_AUC_pAUC', 'eval mean(AUC, pAUC)', PAPER_EVAL_MEAN)]
    systems = [('E2p', 'E2′[N4]', ORANGE, 'o'), ('E1', 'E1[N4]', BLUE, 'o'),
               ('mean_S1p', 'mean S1′', ORANGE, 'D')]
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.8))
    for ax, (ds, m, lab, pap) in zip(axes, items):
        lo_all, hi_all = [pap], [pap]
        for i, (pre, name, col, mk) in enumerate(systems):
            pt, lo, hi = boot[ds][f'{pre}_{m}']
            lo_all.append(lo)
            hi_all.append(hi)
            ax.plot([i, i], [lo, hi], color=col, lw=2.2, solid_capstyle='round', zorder=3)
            ax.scatter([i], [pt], s=52, marker=mk, color=col, edgecolor=SURFACE, lw=1.4, zorder=4)
            ax.annotate(f'{pt:.2f}', (i, pt), xytext=(8, 0), textcoords='offset points',
                        ha='left', va='center', fontsize=7.5, color=INK,
                        bbox=dict(boxstyle='round,pad=0.1', fc=SURFACE, ec='none', alpha=0.85))
        ax.axhline(pap, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.annotate(f'paper\n{pap:.2f}', (0.0, pap), xycoords=('axes fraction', 'data'),
                    xytext=(2, 3), textcoords='offset points', ha='left', va='bottom', fontsize=8, color=INK)
        pad = 0.25 * (max(hi_all) - min(lo_all))
        ax.set_ylim(min(lo_all) - pad, max(hi_all) + pad * 1.4)
        ax.set_xlim(-1.0, 2.9)
        ax.set_xticks(range(3), [s[1].replace(' ', '\n') if ' ' in s[1] else s[1] for s in systems],
                      fontsize=7.5)
        ax.grid(axis='y')
        ax.set_title(lab, loc='left', fontsize=9.5, color=INK)
    e2 = [boot[ds][f'E2p_{m}'][0] > pap for ds, m, _, pap in items]
    e1 = [boot[ds][f'E1_{m}'][0] > pap for ds, m, _, pap in items]
    fig.suptitle(f'E2′[N4] exceeds the paper on {sum(e2)} of 4 numbers by point estimate, E1[N4] on {sum(e1)}', x=0.04, y=1.12, ha='left', fontsize=11, color=INK)
    fig.text(0.04, 1.03, 'Stratified bootstrap 95% CIs over test clips: test-set sampling only, not seed variance. '
             'Each panel has its own scale. E2′ dev AUC lower bound = the paper value; dev mAUC interval spans it.',
             fontsize=8.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig4_vs_paper.png')


def fig5_machines(summ):
    if any((r, ds, k) not in summ for r in N4 for ds in ('dev', 'eval') for k in ('S0', 'S1p')):
        print('skip fig5: summary rows missing')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharey=True)
    x = np.arange(len(MACH))
    markers = ['o', 's', '^', 'v']
    neg_valve = None
    for ax, ds in zip(axes, ['dev', 'eval']):
        ax.axhline(0, color=INK_2, lw=0.9, zorder=1)
        D = np.array([[float(summ[(r, ds, 'S1p')][m]) - float(summ[(r, ds, 'S0')][m]) for m in MACH]
                      for r in N4])
        for j, r in enumerate(N4):
            ax.scatter(x + (j - 1.5) * 0.09, D[j], s=34, marker=markers[j], color=ORANGE,
                       edgecolor=SURFACE, lw=1.0, zorder=3, alpha=0.9, label=short(r))
        ax.hlines(D.mean(0), x - 0.3, x + 0.3, color=INK, lw=2.2, zorder=4, label='mean of 4')
        ax.set_xticks(x, MACH)
        ax.grid(axis='y')
        if ds == 'eval':
            neg_valve = int((D[:, MACH.index('valve')] < 0).sum())
        ax.set_title(ds, loc='left', fontsize=9.5, color=INK)
    axes[0].set_ylabel('S1′ − S0, machine AUC (points)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper left', bbox_to_anchor=(0.06, 1.0), ncol=5, fontsize=8.5)
    ev = np.array([[float(summ[(r, 'eval', 'S1p')][m]) - float(summ[(r, 'eval', 'S0')][m]) for m in MACH]
                   for r in N4]).mean(0)
    top = [MACH[i] for i in np.argsort(-ev)[:2]]
    fig.suptitle(f'The S1′ gain comes from {top[0]} and {top[1]}; eval valve falls on {neg_valve} of 4 checkpoints',
                 x=0.06, y=1.12, ha='left', fontsize=11, color=INK)
    fig.text(0.06, 1.045, 'Dots: the four N4 checkpoints (shape = seed). Black bar: their mean.',
             fontsize=8.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig5_per_machine.png')


def main():
    summ_rows = read('summary.csv')
    curve = read('ensemble_curve.csv')
    boot = read('bootstrap.json')
    summ = summary_lookup(summ_rows) if summ_rows else None
    if summ:
        fig1_single(summ)
        fig5_machines(summ)
    if summ and boot:
        fig2_eval_mean(summ, boot)
    if curve:
        fig3_size(curve)
    if boot:
        fig4_vs_paper(boot)


if __name__ == '__main__':
    main()
