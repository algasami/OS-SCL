"""Figures for phase3/beats-router. Reads scores/ only; a figure whose inputs are missing is skipped.

Run with /home/f74134118/anaconda3/bin/python (matplotlib lives in base):
    make_plots.py

Colour: dev = blue, eval = orange, as in synthesis/ensemble-g8. Named systems (X1f, GS12f, R*, R_ci) are
ink with a direct label; everything else takes the dataset colour.
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
PLOTS = os.path.join(HERE, 'plots')
os.makedirs(PLOTS, exist_ok=True)

SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8983', '#e4e3df'
BLUE, ORANGE = '#2a78d6', '#eb6834'
COL = {'dev': BLUE, 'eval': ORANGE}
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


def setname(types):
    return '{' + ', '.join(types) + '}' if types else '∅'


def legend_dots(ax, items, x0, y):
    """Dot + label pairs above an axes, in axes coordinates (labels ~0.0095 axes-width per character)."""
    x = x0
    for lab, c in items:
        ax.scatter([x + 0.006], [y], s=34, color=c, transform=ax.transAxes, clip_on=False)
        ax.text(x + 0.02, y, lab, color=INK_2, fontsize=8.5, va='center', transform=ax.transAxes)
        x += 0.02 + 0.0095 * len(lab) + 0.035


# ------------------------------------------------------------------ fig 1: per-machine BEATs effect
def fig1():
    md, B = read(os.path.join(SRC, 'machine_delta.csv')), read_json(os.path.join(SRC, 'bootstrap.json'))
    if not md or not B:
        return print('skip fig1 (no machine_delta.csv / bootstrap.json)')
    R = {(r['dataset'], r['machine']): r for r in md}
    rs, rci = B['Rstar'], B['Rci']
    fig, ax = plt.subplots(figsize=(9.4, 3.9))
    for i, mm in enumerate(MACH):
        if mm in rs:
            ax.axvspan(i - 0.42, i + 0.42, color=GRID, alpha=0.55, lw=0, zorder=0)
        d, e = R[('dev', mm)], R[('eval', mm)]
        x = i - 0.1
        lo, hi, pt = float(d['ci_lo']), float(d['ci_hi']), float(d['delta_X1f_minus_GS12f'])
        ax.plot([x, x], [lo, hi], color=BLUE, lw=2.2, solid_capstyle='round', zorder=2)
        ax.scatter([x], [pt], s=48, color=BLUE, zorder=3, edgecolors=SURFACE, linewidths=1.2)
        ax.annotate(sg(pt), (x, hi), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=7.5, color=BLUE)
        ev = float(e['delta_X1f_minus_GS12f'])
        ax.scatter([i + 0.14], [ev], s=48, color=ORANGE, zorder=3, edgecolors=SURFACE, linewidths=1.2)
        ax.annotate(sg(ev), (i + 0.14, ev), xytext=(7, 0), textcoords='offset points', va='center', fontsize=7.5,
                    color=ORANGE)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(range(len(MACH)))
    ax.set_xticklabels([f'{m}\n(R* on)' if m in rs else m for m in MACH])
    for t, m in zip(ax.get_xticklabels(), MACH):
        if m in rs:
            t.set_color(INK)
            t.set_fontweight('bold')
    ax.set_xlim(-0.55, len(MACH) - 0.3)
    ax.grid(axis='y')
    ax.set_ylabel('Amean(X1f) − Amean(GS12f), points')
    legend_dots(ax, [('dev, 95 % paired clip-bootstrap CI', BLUE), ('eval point estimate', ORANGE)], 0.0, 1.03)
    ax.text(1.0, 1.03, 'shaded = R*, chosen on dev point estimates', transform=ax.transAxes, ha='right', va='center',
            fontsize=8, color=MUTED)
    flips = [m for m in MACH if (float(R[('dev', m)]['delta_X1f_minus_GS12f']) > 0) != (float(R[('eval', m)]['delta_X1f_minus_GS12f']) > 0)]
    if rci == rs:
        title = f'BEATs raises dev Amean on {", ".join(rs)}; each of the three has a dev CI above zero (R_ci = R*)'
    elif rci:
        title = f'BEATs raises dev Amean on {", ".join(rs)}; only {", ".join(rci)} has a dev CI above zero'
    else:
        title = f'BEATs raises dev Amean on {", ".join(rs)}; no type has a dev CI above zero'
    title += f'\nOn eval the sign differs from dev on {len(flips)} of 6 types ({", ".join(flips)})'
    ax.set_title(title + '\n', loc='left', fontsize=10.5, fontweight='bold')
    save(fig, 'fig1_machine_delta.png')


# ------------------------------------------------------------------ fig 2: cost-accuracy, all 64 routers
def fig2():
    rows, B = read(os.path.join(SRC, 'routers.csv')), read_json(os.path.join(SRC, 'bootstrap.json'))
    cost = read_json(os.path.join(SRC, 'cost.json'))
    if not rows or not B:
        return print('skip fig2 (no routers.csv / bootstrap.json)')
    A = {(r['router'], r['dataset']): float(r['Amean']) for r in rows if r['machine'] == 'all'}
    masks = sorted({r['router'] for r in rows})
    if cost:
        mb = cost['members']
        cG = 8 * mb['fc_nogain']['flops'] + 4 * mb['fc_spec']['flops']
        cB = 3 * mb['beats']['flops']
        X = {k: (cG + cB * k.count('1') / 6) / (cG + cB) for k in masks}
        xlabel = 'type-weighted forward FLOPs, fraction of X1f'
    else:
        X = {k: k.count('1') / 6 for k in masks}
        xlabel = 'machine types routed to BEATs / 6  (cost.json missing: not a FLOPs fraction)'
    rs, rci = B['Rstar_mask'], B['Rci_mask']
    named = {'111111': 'X1f', '000000': 'GS12f', rs: 'R*'}
    if rci not in named:
        named[rci] = 'R_ci'
    elif rci == rs:
        named[rs] = 'R* = R_ci'
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), sharey=False, gridspec_kw=dict(wspace=0.22))
    rng = np.random.default_rng(0)
    for ax, ds in zip(axes, ('dev', 'eval')):
        xs = np.array([X[k] for k in masks])
        ys = np.array([A[(k, ds)] for k in masks])
        jit = rng.uniform(-0.006, 0.006, len(masks))
        other = np.array([k not in named for k in masks])
        ax.scatter(xs[other] + jit[other], ys[other], s=16, color=COL[ds], alpha=0.55, linewidths=0, zorder=2)
        # upper envelope over cost levels (descriptive; picking from it on eval is test-set selection)
        lv = sorted(set(xs))
        env = [max(ys[xs == v]) for v in lv]
        ax.plot(lv, np.maximum.accumulate(env), color=COL[ds], lw=1, alpha=0.6, zorder=1, ls=(0, (3, 2)))
        for k, lab in named.items():
            ax.scatter([X[k]], [A[(k, ds)]], s=46, color=INK, zorder=4, edgecolors=SURFACE, linewidths=1.2)
            left = X[k] > 0.9
            ax.annotate(f'{lab}  {A[(k, ds)]:.2f}', (X[k], A[(k, ds)]), xytext=(-8 if left else 8, -8 if k == rs else 0),
                        textcoords='offset points', ha='right' if left else 'left', va='center', fontsize=8,
                        color=INK, fontweight='bold')
        ax.set_title(f'{ds}', loc='left', fontsize=9, color=COL[ds], fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f'{ds} Amean (%)')
        ax.grid(True)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo - 0.02, hi + 0.08)
    d = {ds: A[(rs, ds)] - A[('111111', ds)] for ds in ('dev', 'eval')}
    frac = X[rs]
    unit = 'of X1f\'s FLOPs' if cost else 'of the BEATs-routed types'
    fig.suptitle(f'R* runs BEATs on half the types for {frac:.0%} {unit}; against X1f it is {sg(d["dev"])} on dev '
                 f'and {sg(d["eval"])} on eval', x=0.125, y=1.02, ha='left', fontsize=10.5, fontweight='bold')
    _, lo, hi = B['pairs']['eval']['Rstar-X1f']['mean_AUC_pAUC']
    fig.text(0.125, 0.955, f'all 64 machine-type routers; dashed = best router at or below each cost (descriptive only)'
             f';  eval 95 % CI of R* − X1f [{sg(lo)}, {sg(hi)}]', fontsize=8, color=MUTED)
    save(fig, 'fig2_frontier.png')


# ------------------------------------------------------------------ fig 3: D3, is the dev routing stable?
def fig3():
    md, B = read(os.path.join(SRC, 'machine_delta.csv')), read_json(os.path.join(SRC, 'bootstrap.json'))
    if not md or not B:
        return print('skip fig3 (no machine_delta.csv / bootstrap.json)')
    R = {(r['dataset'], r['machine']): float(r['frac_draws_pos']) for r in md}
    rs = B['Rstar']
    fig, ax = plt.subplots(figsize=(9.4, 3.6))
    x = np.arange(len(MACH))
    dv = np.array([R[('dev', m)] for m in MACH])
    ax.bar(x, dv, width=0.56, color=BLUE, zorder=2)
    for i, v in enumerate(dv):
        ax.text(i, v + 0.015, f'{v:.0%}', ha='center', va='bottom', fontsize=8, color=BLUE)
    ev = np.array([R[('eval', m)] for m in MACH])
    ax.scatter(x + 0.36, ev, s=34, color=ORANGE, zorder=3, edgecolors=SURFACE, linewidths=1.0)
    for y, lab in ((0.9, '90 %: decided "on"'), (0.1, '10 %: decided "off"')):
        ax.axhline(y, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.text(1.01, y, lab, ha='left', va='center', fontsize=7.5, color=INK, transform=ax.get_yaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels([f'{m}\n(R* on)' if m in rs else m for m in MACH])
    ax.set_ylim(0, 1.12)
    ax.set_xlim(-0.5, len(MACH) - 0.4)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0', '25 %', '50 %', '75 %', '100 %'])
    ax.grid(axis='y')
    ax.set_ylabel('share of 2000 bootstrap draws\nwith Δ_m = X1f − GS12f > 0')
    legend_dots(ax, [('dev (D3)', BLUE), ('eval (descriptive)', ORANGE)], 0.0, 1.06)
    und = [m for m, v in zip(MACH, dv) if 0.1 < v < 0.9]
    same = B['d3']['dev']['frac_draws_set_equals_dev_Rstar']
    if und:
        title = (f'D3 fails: {len(und)} of 6 types are undecided on dev ({", ".join(und)}); '
                 f'a resampled dev picks R* itself in {same:.0%} of draws')
    else:
        title = f'D3 passes: every type is decided in ≥ 90 % of dev draws; a resampled dev picks R* in {same:.0%}'
    ax.set_title(title + '\n', loc='left', fontsize=10.5, fontweight='bold')
    save(fig, 'fig3_stability.png')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3):
        f()
