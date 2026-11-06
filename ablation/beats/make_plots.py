"""Figures for ablation/beats (WITH EXTERNAL PRETRAINING). Reads scores/ (and pooled-knn's and
nogain-ensemble's committed scores/) only; skips a figure whose inputs are missing.

Run with /home/f74134118/anaconda3/bin/python (matplotlib lives in base).
Colour = system: our from-scratch MFN (C1 single / E2′[A8]) blue; BEATs orange; the fusion
green (direct labels: green is below 3:1 on the surface); literature achromatic.
"""
import csv
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.environ.get('BT_SCORES', os.path.join(HERE, 'scores'))      # the smoke test points these at results/
PLOTS = os.environ.get('BT_PLOTS', os.path.join(HERE, 'plots'))
PKNN = os.path.join(HERE, '..', 'pooled-knn', 'scores', 'confirm_metrics.csv')
os.makedirs(PLOTS, exist_ok=True)

SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8983', '#e4e3df'
BLUE, ORANGE, GREEN = '#2a78d6', '#eb6834', '#1baf7a'
MACH = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
RULES = ['S0', 'K', 'zKp', 'S1p', 'C1b']
TICKS = ['fan', 'pump', 'slider', 'ToyCar', 'Toy-\nConveyor', 'valve', 'mean']
TAG = 'with external pretraining (BEATs_iter3)'
# AnoPatch (arXiv 2406.11364) Table 1, mean of AUC and pAUC per machine
ANOPATCH = {'dev': dict(fan=86.46, pump=93.10, slider=99.20, ToyCar=96.10, ToyConveyor=73.20, valve=97.53),
            'eval': dict(fan=95.56, pump=94.34, slider=99.77, ToyCar=96.00, ToyConveyor=83.74, valve=96.26)}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'text.color': INK, 'axes.labelcolor': INK_2, 'axes.edgecolor': GRID,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False,
})


def read(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='UTF-8') as f:
        return list(csv.DictReader(f)) if path.endswith('.csv') else json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(PLOTS, name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def c1_machine(ds):
    rows = read(PKNN)
    if not rows:
        return None
    out = {}
    for m in MACH + ['overall']:
        v = [float(r['Amean']) for r in rows if r['rule'] == 'C1' and r['dataset'] == ds and r['machine'] == m
             and r['system'].startswith('SCLTFSTgramMFN_ao_nogain')]
        out[m] = np.mean(v)
    return out


def fig1():
    """Per-machine Amean: BEATs single model (selected rule) vs our C1 single model vs AnoPatch."""
    sel = read(os.path.join(SCORES, 'selection.json'))
    if not sel:
        return
    rule = sel['rule']
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for ax, ds in zip(axes, ('dev', 'eval')):
        rows, c1 = read(os.path.join(SCORES, f'single_{ds}.csv')), c1_machine(ds)
        if not rows or not c1:
            ax.set_visible(False)
            continue
        runs = [r for r in rows if r['rule'] == rule and r['run'] != 'mean']
        x = np.arange(len(MACH) + 1)
        bt = [[float(r[m]) for r in runs] for m in MACH] + [[float(r['Amean']) for r in runs]]
        ours = [c1[m] for m in MACH] + [c1['overall']]
        ano = [ANOPATCH[ds][m] for m in MACH] + [np.mean(list(ANOPATCH[ds].values()))]
        ax.bar(x - 0.2, ours, 0.38, color=BLUE, label='ours, C1 single (8 seeds)')
        ax.bar(x + 0.2, [np.mean(b) for b in bt], 0.38, color=ORANGE, label=f'BEATs, {rule} (n={len(runs)})')
        for j, b in enumerate(bt):
            ax.scatter([x[j] + 0.2] * len(b), b, s=9, color=INK, zorder=3, lw=0)
        ax.scatter(x, ano, marker='_', s=160, color=INK_2, zorder=4, label='AnoPatch (published)')
        ax.set_xticks(x, TICKS, fontsize=8)
        lo = min(min(min(b) for b in bt), min(ours), min(ano))
        ax.set_ylim(min(70, 5 * np.floor((lo - 2) / 5)), 100.5)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(f'{ds}: mean of AUC and pAUC per machine', fontsize=9, loc='left')
        ax.set_ylabel('Amean (%)')
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f'Single models — BEATs {TAG}', fontsize=10, x=0.01, ha='left')
    save(fig, 'fig1_single_per_machine.png')


def fig2():
    """Every single rule, per seed, dev vs eval."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    drawn = False
    for ax, ds in zip(axes, ('dev', 'eval')):
        rows = read(os.path.join(SCORES, f'single_{ds}.csv'))
        if not rows:
            ax.set_visible(False)
            continue
        drawn = True
        for j, rule in enumerate(RULES):
            v = [float(r['Amean']) for r in rows if r['rule'] == rule and r['run'] != 'mean']
            ax.scatter([j] * len(v), v, s=16, color=ORANGE, alpha=0.6, lw=0)
            ax.plot([j - 0.25, j + 0.25], [np.mean(v)] * 2, color=ORANGE, lw=2.2)
            ax.text(j, max(v) + 0.15, f'{np.mean(v):.2f}', ha='center', va='bottom', fontsize=8, color=INK_2)
        c1 = c1_machine(ds)
        if c1:
            ax.axhline(c1['overall'], color=BLUE, lw=1.2, ls='--')
            ax.text(len(RULES) - 0.5, c1['overall'], ' ours C1', color=BLUE, va='center', fontsize=8)
        ax.set_xticks(range(len(RULES)), ['R0', 'kNN\n(AnoPatch)', 'z kNN', 'S1′', 'C1b'])
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(f'{ds}: overall Amean by scoring rule', fontsize=9, loc='left')
    if drawn:
        fig.suptitle(f'BEATs scoring rules — {TAG}', fontsize=10, x=0.01, ha='left')
        save(fig, 'fig2_rules.png')
    else:
        plt.close(fig)


def fig3():
    """Fusion with E2′[A8]: per-machine Δ Amean of F1 / F2 vs A8, dev and eval."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    drawn = False
    for ax, ds in zip(axes, ('dev', 'eval')):
        rows = read(os.path.join(SCORES, f'systems_{ds}.csv'))
        if not rows:
            ax.set_visible(False)
            continue
        drawn = True
        S = {r['system']: r for r in rows}
        x = np.arange(len(MACH) + 1)
        for k, (f, col) in enumerate((('F1', GREEN), ('F2', INK_2))):
            d = [float(S[f][m]) - float(S['A8'][m]) for m in MACH] + [float(S[f]['Amean']) - float(S['A8']['Amean'])]
            ax.bar(x + (k - 0.5) * 0.38, d, 0.38, color=col, label={'F1': 'F1 = E2′[A8 + B3]',
                                                                     'F2': 'F2 = E2′[A8] + 8/3 ΣS1′[B3]'}[f])
            ax.text(x[-1] + (k - 0.5) * 0.38, d[-1], f'{d[-1]:+.2f}', ha='center',
                    va='bottom' if d[-1] >= 0 else 'top', fontsize=7, color=INK_2)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x, TICKS, fontsize=8)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(f'{ds}: Δ Amean vs E2′[A8]', fontsize=9, loc='left')
    if drawn:
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc='lower center', ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.06))
        fig.suptitle(f'Fusing BEATs into our best ensemble — {TAG}', fontsize=10, x=0.01, ha='left')
        save(fig, 'fig3_fusion.png')
    else:
        plt.close(fig)


def fig4():
    """Training curves: validation loss and accuracy per optimizer step."""
    runs = sorted(d for d in os.listdir(SCORES) if os.path.isdir(os.path.join(SCORES, d))) if os.path.isdir(SCORES) else []
    curves = {r: read(os.path.join(SCORES, r, 'curve.csv')) for r in runs}
    curves = {r: c for r, c in curves.items() if c}
    if not curves:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    shades = [ORANGE, '#b24a1f', '#f3a27c']
    for (r, c), col in zip(curves.items(), shades):
        st = [int(x['epoch']) for x in c]
        axes[0].plot(st, [float(x['valid_loss']) for x in c], color=col, lw=1.4, label=r.split('_')[-1])
        axes[0].plot(st, [float(x['train_loss']) for x in c], color=col, lw=0.8, ls=':')
        axes[1].plot(st, [float(x['valid_acc']) for x in c], color=col, lw=1.4)
    axes[0].set_yscale('log')
    axes[0].set_title('loss (solid valid, dotted train; ArcFace CE)', fontsize=9, loc='left')
    axes[1].set_title('validation accuracy (41 IDs)', fontsize=9, loc='left')
    for ax in axes:
        ax.set_xlabel('optimizer step')
        ax.yaxis.grid(True)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'BEATs fine-tuning curves — {TAG}', fontsize=10, x=0.01, ha='left')
    save(fig, 'fig4_curves.png')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4):
        f()
