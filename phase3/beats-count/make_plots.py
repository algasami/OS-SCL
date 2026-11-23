"""Figures for phase3/beats-count. Reads scores/ (and ../beats-router/scores/cost.json) only; a figure whose
inputs are missing is skipped.

Run with /home/f74134118/anaconda3/bin/python (matplotlib lives in base):
    make_plots.py

Colour: primary weighting ("block", BEATs block 4.5) = blue, secondary ("member", 1.5 per net) = orange, the
two categorical slots. In fig2, dev = blue and eval = orange, as in ../beats-router. GS12f and X1f are ink with a
direct label.
"""
import csv
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'scores')
COST = os.path.join(HERE, '..', 'beats-router', 'scores', 'cost.json')
PLOTS = os.path.join(HERE, 'plots')
os.makedirs(PLOTS, exist_ok=True)

SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8983', '#e4e3df'
BLUE, ORANGE = '#2a78d6', '#eb6834'
WCOL = {'block': BLUE, 'member': ORANGE}
WLAB = {'block': 'block weight 4.5 (primary)', 'member': '1.5 per net (secondary)'}
DCOL = {'dev': BLUE, 'eval': ORANGE}
MACH = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']

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
        return list(csv.DictReader(f))


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='UTF-8') as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(PLOTS, name), dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote', os.path.join('plots', name))


def sg(x):
    return f'{x:+.2f}'.replace('-', '−')


def legend_dots(ax, items, x0, y):
    """Dot + label pairs above an axes, in axes coordinates (labels ~0.0095 axes-width per character)."""
    x = x0
    for lab, c in items:
        ax.scatter([x + 0.006], [y], s=34, color=c, transform=ax.transAxes, clip_on=False)
        ax.text(x + 0.02, y, lab, color=INK_2, fontsize=8.5, va='center', transform=ax.transAxes)
        x += 0.02 + 0.0095 * len(lab) + 0.035


def flops_fractions():
    """{k: forward FLOPs of GS12f + k BEATs nets, fraction of X1f} from beats-router's cost.json, or None."""
    c = read_json(COST)
    try:
        mb = c['members']
        cG = 8 * mb['fc_nogain']['flops'] + 4 * mb['fc_spec']['flops']
        cB = mb['beats']['flops']
    except (TypeError, KeyError):
        return None
    return {k: (cG + k * cB) / (cG + 3 * cB) for k in range(4)}


# ------------------------------------------------------------------ fig 1: Amean against the number of BEATs nets
def fig1():
    rows = read(os.path.join(SRC, 'systems.csv')) or read(os.path.join(SRC, 'dev_systems.csv'))
    if not rows:
        return print('skip fig1 (no systems.csv / dev_systems.csv)')
    sel = read_json(os.path.join(SRC, 'selection.json'))
    picks = {v['system']: k for k, v in sel['picks'].items()} if sel else {}
    R = [r for r in rows if r['machine'] == 'all']
    sets = [ds for ds in ('dev', 'eval') if any(r['dataset'] == ds for r in R)]
    A = {(r['system'], r['dataset']): float(r['Amean']) for r in R}
    fr = flops_fractions()
    fig, axes = plt.subplots(1, len(sets), figsize=(5.4 * len(sets), 4.2), squeeze=False,
                             gridspec_kw=dict(wspace=0.26))
    means = {}
    for ax, ds in zip(axes[0], sets):
        g, x = A[('GS12f', ds)], A[('X1f', ds)]
        labels = []
        for wt, off in (('block', -0.07), ('member', 0.07)):
            mk = [g]
            for k in (1, 2):
                ys = [A[(r['system'], ds)] for r in R if r['dataset'] == ds and r['weighting'] == wt and r['k'] == str(k)]
                names = [r['system'] for r in R if r['dataset'] == ds and r['weighting'] == wt and r['k'] == str(k)]
                ax.scatter([k + off] * len(ys), ys, s=26, color=WCOL[wt], alpha=0.75, linewidths=0, zorder=3)
                if wt == 'block':
                    labels.append((k + off, names, ys))
                mk.append(float(np.mean(ys)))
            mk.append(x)
            means[(wt, ds)] = mk
            ax.plot([0, 1 + off, 2 + off, 3], mk, color=WCOL[wt], lw=1.6, zorder=2)
        for k, y, lab in ((0, g, 'GS12f'), (3, x, 'X1f')):
            ax.scatter([k], [y], s=48, color=INK, zorder=4, edgecolors=SURFACE, linewidths=1.2)
            ax.annotate(f'{lab}  {y:.2f}', (k, y), xytext=(8 if k == 0 else -8, -10 if k == 0 else 10),
                        textcoords='offset points', ha='left' if k == 0 else 'right', va='center', fontsize=8,
                        fontweight='bold')
        lo_, hi_ = ax.get_ylim()
        gap = 0.045 * (hi_ - lo_)
        for xk, names, ys in labels:      # subset labels left of the dots, spread so they never collide
            order = sorted(range(len(ys)), key=lambda i: ys[i])
            pos = []
            for i in order:
                pos.append(max(ys[i], pos[-1] + gap) if pos else ys[i])
            for i, yl in zip(order, pos):
                n = names[i]
                lab = n.split('_')[0] + (f'  ({picks[n]})' if n in picks else '')
                ax.annotate(lab, (xk, ys[i]), xytext=(xk - 0.09, yl), textcoords='data', ha='right', va='center',
                            fontsize=7, color=INK if n in picks else MUTED,
                            fontweight='bold' if n in picks else 'normal',
                            arrowprops=dict(arrowstyle='-', color=GRID, lw=0.6, shrinkA=0, shrinkB=3))
        ax.set_xticks(range(4))
        if fr:
            ax.set_xticklabels([f'{k}\n{fr[k]:.3f}' for k in range(4)])
            ax.set_xlabel('BEATs networks k  /  forward FLOPs, fraction of X1f')
        else:
            ax.set_xticklabels([str(k) for k in range(4)])
            ax.set_xlabel('BEATs networks k')
        ax.set_xlim(-0.35, 3.35)
        ax.set_ylabel(f'{ds} Amean (%)')
        ax.grid(axis='y')
        ax.set_title(ds, loc='left', fontsize=9, color=INK_2, fontweight='bold')
    from matplotlib.lines import Line2D
    axes[0][0].legend([Line2D([], [], color=WCOL[w], lw=1.6, marker='o', ms=5) for w in ('block', 'member')],
                      [WLAB[w] for w in ('block', 'member')], loc='lower right', fontsize=8)
    fig.text(0.125, 0.955, 'dots = BEATs subsets, line = mean over the subsets of size k; (X1f-k) = the dev pick '
             'under the primary weighting', fontsize=8, color=MUTED)
    if 'eval' in sets:
        mk = means[('block', 'eval')]
        g12, g23 = mk[2] - mk[1], mk[3] - mk[2]
        shape = ('the second net adds more than the third' if g12 > g23 else 'the third net adds as much or more '
                 'than the second')
        title = (f'Eval Amean rises with every BEATs net under the primary weighting: mean '
                 f'{mk[1]:.2f} (k=1), {mk[2]:.2f} (k=2), {mk[3]:.2f} (X1f)' if mk[0] < mk[1] < mk[2] < mk[3] else
                 f'Eval Amean, primary weighting: GS12f {mk[0]:.2f}, mean {mk[1]:.2f} (k=1), {mk[2]:.2f} (k=2), '
                 f'X1f {mk[3]:.2f}')
        title += (f'\nGain {sg(g12)} from k=1 to 2 and {sg(g23)} from 2 to 3: {shape}, '
                  f'{"both" if max(abs(g12), abs(g23)) < 0.15 else "not both"} under the 0.15 ensemble noise floor')
    else:
        title = 'Dev Amean against the number of BEATs nets (eval not yet scored)'
    fig.suptitle(title, x=0.125, y=1.06 if '\n' in title else 1.02, ha='left', fontsize=10.5, fontweight='bold')
    save(fig, 'fig1_count.png')


# ------------------------------------------------------------------ fig 2: per-machine delta vs X1f of the picks
def fig2():
    rows, D = read(os.path.join(SRC, 'systems.csv')), read_json(os.path.join(SRC, 'decisions.json'))
    if not rows or not D:
        return print('skip fig2 (no systems.csv / decisions.json)')
    A = {(r['system'], r['dataset'], r['machine']): float(r['Amean']) for r in rows}
    P = D['selection']['picks']
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), sharey=True, gridspec_kw=dict(wspace=0.08))
    allv = []
    for ax, key in zip(axes, ('X1f-1', 'X1f-2')):
        n = P[key]
        for i, mm in enumerate(MACH + ['all']):
            for ds, off in (('eval', -0.13), ('dev', 0.13)):
                d = A[(n, ds, mm)] - A[('X1f', ds, mm)]
                allv.append(d)
                ax.bar(i + off, d, width=0.24, color=DCOL[ds], zorder=2,
                       alpha=1.0 if ds == 'eval' else 0.55)
                if ds == 'eval':
                    ax.annotate(sg(d), (i + off, d), xytext=(0, 3 if d >= 0 else -3), textcoords='offset points',
                                ha='center', va='bottom' if d >= 0 else 'top', fontsize=7, color=ORANGE)
        ax.axhline(0, color=INK, lw=0.9)
        ax.axvline(len(MACH) - 0.5, color=GRID, lw=0.9)
        ax.set_xticks(range(len(MACH) + 1))
        ax.set_xticklabels(MACH + ['all'], rotation=0, fontsize=8)
        ax.grid(axis='y')
        d1 = D[f'D1_{key[-1]}']
        lo, hi = d1['a']['eval_ci_minus_X1f'][1:]
        ax.set_title(f'{key} = {n.split("_")[0]}, {4.5 / int(key[-1]):g} per BEATs net\n'
                     f'eval 95 % CI of Δ all [{sg(lo)}, {sg(hi)}];  D1_{key[-1]} {"passes" if d1["passed"] else "fails"}',
                     loc='left', fontsize=8.5, color=INK_2, fontweight='bold')
    axes[0].set_ylabel('Amean(X1f-k) − Amean(X1f), points')
    lim = max(abs(v) for v in allv) * 1.25
    axes[0].set_ylim(-lim, lim)
    from matplotlib.patches import Patch
    axes[1].legend([Patch(color=ORANGE), Patch(color=BLUE, alpha=0.55)], ['eval', 'dev'], loc='lower right',
                   fontsize=8)
    e = {k: A[(P[k], 'eval', 'all')] - A[('X1f', 'eval', 'all')] for k in ('X1f-1', 'X1f-2')}
    neg = {k: [mm for mm in MACH if A[(P[k], 'eval', mm)] < A[('X1f', 'eval', mm)]] for k in e}
    title = (f'Against X1f on eval, X1f-1 is {sg(e["X1f-1"])} (lower on {len(neg["X1f-1"])} of 6 types) and '
             f'X1f-2 is {sg(e["X1f-2"])} (lower on {len(neg["X1f-2"])} of 6)')
    fig.suptitle(title, x=0.125, y=1.06, ha='left', fontsize=10.5, fontweight='bold')
    save(fig, 'fig2_machine_delta.png')


if __name__ == '__main__':
    for f in (fig1, fig2):
        f()
