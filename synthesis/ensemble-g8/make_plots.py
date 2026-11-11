"""Figures for synthesis/ensemble-g8. Reads scores/ only; a figure whose inputs are missing is skipped.

Run with /home/f74134118/anaconda3/bin/python (matplotlib lives in base):
    make_plots.py                      the fresh scores (scores/*.csv, written by `run.sh score`)
    make_plots.py --src scores/dryrun_p1 --out <dir>
                                       layout check on the dry run (seen seeds); never commit these

Colour: dev = blue, eval = orange, as in ../final and ../fresh-confirm. Systems scored before this study
are drawn hollow. The paper's numbers are dotted ink lines. Systems with BEATs say so on the axis.
"""
import csv
import json
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = dict(zip(sys.argv[1::2], sys.argv[2::2]))
SRC = os.path.join(HERE, ARGS.get('--src', 'scores'))
DRY = 'dryrun' in SRC
OLD = os.path.join(HERE, '..', 'fresh-confirm', 'scores', 'dryrun_p1' if DRY else '')
PLOTS = ARGS.get('--out', os.path.join(HERE, 'plots'))
os.makedirs(PLOTS, exist_ok=True)

SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8983', '#e4e3df'
BLUE, ORANGE = '#2a78d6', '#eb6834'
COL = {'dev': BLUE, 'eval': ORANGE}
PAPER = {'dev': 92.97, 'eval': 93.22}                     # Amean; ../final/PUBLISHED.md
PAPER3 = {'AUC': 95.71, 'pAUC': 90.23, 'mAUC': 91.23}

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
    print('wrote', os.path.join(os.path.relpath(PLOTS, HERE), name))


def sg(x):
    return f'{x:+.2f}'.replace('-', '−')


# ------------------------------------------------------------------ fig 1: every system, dev and eval
ROWS = [('G4', '4 ao_nogain  (first four)', False), ('G4b', '4 ao_nogain  (new four)', True),
        ('G8f', '8 ao_nogain', True), ('GS8', '4 ao_nogain ×2 + 4 sb_spec', False),
        ('GS12f', '8 ao_nogain + 4 sb_spec', True), None,
        ('XG8', '8 ao_nogain + BEATs', True), ('X1', '4 ao_nogain ×2 + 4 sb_spec + BEATs', False),
        ('X1f', '8 ao_nogain + 4 sb_spec + BEATs', True)]


def fig1():
    ens, D = read(os.path.join(SRC, 'ensembles.csv')), read_json(os.path.join(SRC, 'decisions.json'))
    if not ens or not D:
        return print('skip fig1 (no ensembles.csv / decisions.json)')
    A = {(r['system'], r['dataset']): float(r['Amean']) for r in ens if r['machine'] == 'all'}
    pick = D['E2']['pick']
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    ys, labels = [], []
    y = 0
    for row in ROWS:
        if row is None:
            y -= 0.9
            ax.axhline(y + 0.05, color=GRID, lw=0.8)
            ax.text(0.0, y - 0.02, 'with external pretraining (three BEATs networks)', transform=ax.get_yaxis_transform(),
                    ha='left', va='top', fontsize=7.5, color=MUTED)
            y -= 1.1
            continue
        name, lab, new = row
        d, e = A[(name, 'dev')], A[(name, 'eval')]
        ax.plot([d, e], [y, y], color=GRID, lw=2, zorder=1, solid_capstyle='round')
        for ds, v in (('dev', d), ('eval', e)):
            ax.scatter([v], [y], s=64, zorder=3, linewidths=1.6, edgecolors=COL[ds],
                       facecolors=COL[ds] if new else SURFACE)
            right = v >= (d if ds == 'eval' else e)
            ax.annotate(f'{v:.2f}', (v, y), xytext=(8 if right else -8, 0), textcoords='offset points',
                        ha='left' if right else 'right', va='center', fontsize=8, color=INK_2)
        ys.append(y)
        labels.append(f'{name}   {lab}' + ('   ← dev pick' if name == pick else ''))
        y -= 1
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    for t, row in zip(ax.get_yticklabels(), [r for r in ROWS if r]):
        if row[0] == pick:
            t.set_fontweight('bold')
            t.set_color(INK)
    vals = list(A.values())
    ax.set_xlim(min(vals) - 0.22, max(vals) + 0.22)
    ax.set_ylim(y + 0.3, 0.8)
    ax.grid(axis='x')
    ax.set_xlabel('Amean (%)  ·  filled = scored in this study, hollow = scored in fresh-confirm and recomputed')
    for ds, x in (('dev', 0.0), ('eval', 0.07)):
        ax.scatter([x + 0.012], [1.045], s=40, color=COL[ds], transform=ax.transAxes, clip_on=False)
        ax.text(x + 0.027, 1.045, ds, transform=ax.transAxes, va='center', fontsize=8.5, color=INK_2)
    c = {k: A[(k, 'eval')] for k in ('G8f', 'GS8', 'GS12f')}
    spread = max(c.values()) - min(c.values())
    seeds = 'seen seeds 2040–47 (dry run)' if DRY else 'fresh seeds'
    ax.set_title(f'{pick} has the highest dev Amean; on eval the three from-scratch candidates span {spread:.2f} points\n',
                 loc='left', fontsize=10.5, fontweight='bold', x=-0.62 if False else 0)
    ax.text(1.0, 1.045, seeds, transform=ax.transAxes, ha='right', va='center', fontsize=8, color=MUTED)
    save(fig, 'fig1_ensembles.png')


# ------------------------------------------------------------------ fig 2: does sb_spec add anything?
def fig2():
    D, B, sub = (read_json(os.path.join(SRC, 'decisions.json')), read_json(os.path.join(SRC, 'bootstrap.json')),
                 read(os.path.join(SRC, 'subsets.csv')))
    if not D or not B or not sub:
        return print('skip fig2 (no decisions.json / bootstrap.json / subsets.csv)')
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 3.9), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.42))
    # left: paired differences against G8f with clip-bootstrap CIs
    pairs = [('GS12f-G8f', '+ 4 sb_spec  (12 nets − 8)', 'GS12f_minus_G8f'), ('GS8-G8f', 'swap 4 ao_nogain for 4 sb_spec\n(GS8 − G8f)', 'GS8_minus_G8f')]
    for i, (key, lab, dk) in enumerate(pairs):
        for j, ds in enumerate(('dev', 'eval')):
            pt = D['E3'][dk][ds]
            _, lo, hi = B[ds][key]['mean_AUC_pAUC']
            yy = -i + (0.16 if ds == 'dev' else -0.16)
            a1.plot([lo, hi], [yy, yy], color=COL[ds], lw=2, solid_capstyle='round')
            a1.scatter([pt], [yy], s=52, color=COL[ds], zorder=3, edgecolors=SURFACE, linewidths=1.2)
            a1.annotate(sg(pt), (hi, yy), xytext=(6, 0), textcoords='offset points', va='center', fontsize=8, color=INK_2)
    a1.axvline(0, color=INK, lw=0.9)
    a1.set_yticks([0, -1])
    a1.set_yticklabels([p[1] for p in pairs])
    a1.set_ylim(-1.6, 0.6)
    a1.grid(axis='x')
    lo, hi = a1.get_xlim()
    a1.set_xlim(lo - 0.02, hi + 0.09)
    a1.set_xlabel('Δ Amean against G8f  (95 % clip-bootstrap CI)')
    a1.set_title('Whole systems', loc='left', fontsize=9, color=INK_2)
    # right: count-matched subsets, 8 members with j sb_spec, every subset
    rng = np.random.default_rng(0)
    pos, labs = [], []
    for i, j in enumerate((2, 4)):
        for ds in ('dev', 'eval'):
            v = np.array([float(r['delta_vs_G8f']) for r in sub if int(r['n_S']) == j and r['dataset'] == ds])
            yy = -i + (0.2 if ds == 'dev' else -0.2)
            a2.scatter(v, yy + rng.uniform(-0.13, 0.13, len(v)), s=9, color=COL[ds], alpha=0.45, linewidths=0)
            a2.plot([v.mean()] * 2, [yy - 0.19, yy + 0.19], color=INK, lw=1.6)
            a2.annotate(f'{sg(v.mean())}  ({int((v > 0).sum())}/{len(v)} above)', (v.max(), yy), xytext=(7, 0),
                        textcoords='offset points', va='center', fontsize=8, color=INK_2)
        pos.append(-i)
        labs.append(f'{8 - j} ao_nogain\n+ {j} sb_spec')
    a2.axvline(0, color=INK, lw=0.9)
    a2.set_yticks(pos)
    a2.set_yticklabels(labs)
    a2.set_ylim(-1.6, 0.6)
    a2.grid(axis='x')
    lo, hi = a2.get_xlim()
    a2.set_xlim(lo, hi + 0.36 * (hi - lo))
    a2.set_xlabel('Δ Amean against G8f, every 8-network subset (bar = mean)')
    a2.set_title('Same size, same weights: 8 networks', loc='left', fontsize=9, color=INK_2)
    for ds, x in (('dev', 0.0), ('eval', 0.05)):
        fig.text(0.125 + x, 0.93, '●', color=COL[ds], fontsize=10, va='center')
        fig.text(0.139 + x, 0.93, ds, color=INK_2, fontsize=8.5, va='center')
    e3 = D['E3']
    te = e3['type_effect']
    dev_yes = all(te[k]['dev'] for k in te)
    eval_yes = any(te[k]['eval'] for k in te)
    if e3['confirmed_on_eval']:
        title = 'Adding sb_spec networks raises eval as well as dev: the eval CI excludes zero'
    elif dev_yes and not eval_yes:
        title = 'sb_spec networks raise the dev score at fixed size; on eval the same swap does not'
    else:
        title = 'sb_spec against ao_nogain members: paired differences and count-matched subsets'
    fig.suptitle(title, x=0.125, y=1.02, ha='left', fontsize=10.5, fontweight='bold')
    save(fig, 'fig2_sb_spec.png')


# ------------------------------------------------------------------ fig 3: the eight fresh single networks
def fig3():
    new, old = read(os.path.join(SRC, 'single.csv')), read(os.path.join(OLD, 'single.csv'))
    D = read_json(os.path.join(SRC, 'decisions.json'))
    if not new or not old or not D:
        return print('skip fig3 (no single.csv)')
    kind = 'ao_nogain' if DRY else 'fc_nogain'
    rows = [dict(r, new=False) for r in old if f'_{kind}_' in r['run']] + [dict(r, new=True) for r in new]
    runs = sorted({r['run'] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), sharey=False, gridspec_kw=dict(wspace=0.28))
    for ax, ds in zip(axes, ('dev', 'eval')):
        for rule, mk, off in (('R0', 'o', -0.14), ('C1T', 'D', 0.14)):
            vals = []
            for i, run in enumerate(runs):
                r = next(x for x in rows if x['run'] == run and x['dataset'] == ds and x['rule'] == rule)
                v = float(r['Amean'])
                vals.append(v)
                ax.scatter([i + off], [v], s=40, marker=mk, zorder=3, linewidths=1.5,
                           edgecolors=COL[ds] if rule == 'C1T' else MUTED,
                           facecolors=(COL[ds] if rule == 'C1T' else MUTED) if r['new'] else SURFACE)
            m = float(np.mean(vals))
            ax.axhline(m, color=COL[ds] if rule == 'C1T' else MUTED, lw=1.1, alpha=0.8)
            ax.text(len(runs) - 0.45, m, f' {"C1ᵀ" if rule == "C1T" else "R0"} mean {m:.2f}', va='center', fontsize=8,
                    color=INK_2)
        ax.axhline(PAPER[ds], color=INK, lw=1, ls=(0, (1, 2)))
        ax.text(-0.45, PAPER[ds], f'paper {PAPER[ds]:.2f} ', va='bottom', ha='left', fontsize=8, color=INK)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([r[-4:] for r in runs])
        ax.set_xlim(-0.5, len(runs) + 1.4)
        ax.set_xlabel('training seed')
        ax.set_ylabel(f'{ds} Amean (%)')
        ax.grid(axis='y')
    fig.text(0.125, 0.925, '◆ C1ᵀ (final rule)    ● R0 (eval.py)    filled = the four networks of this study',
             fontsize=8.5, color=INK_2, va='center')
    c = D['E6']['above_paper_C1T_fresh8']
    n = c['n']
    if D['E6']['passed'] and c['all4'] == n:
        title = f'All {n} fresh networks beat every paper number under C1ᵀ; the new four repeat the first four'
    elif D['E6']['passed']:
        title = f'The new four clear the replication bars; {c["all4"]} of {n} fresh networks beat all four paper numbers under C1ᵀ'
    else:
        title = f'The new four miss a replication bar; {c["all4"]} of {n} fresh networks beat all four paper numbers under C1ᵀ'
    fig.suptitle(title, x=0.125, y=1.02, ha='left', fontsize=10.5, fontweight='bold')
    save(fig, 'fig3_singles.png')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3):
        f()
