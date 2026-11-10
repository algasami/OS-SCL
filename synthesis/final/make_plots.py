"""Figures for synthesis/final. Reads scores/ (and fresh-confirm's scores/) only; skips missing inputs.

Run with /home/f74134118/anaconda3/bin/python (matplotlib lives in base). Run collect.py first.
Dev = blue, eval = orange throughout fig1/fig3. In fig2 the R0 baseline is the zero line (achromatic),
C1ᵀ blue, GS12 orange, X1 green (external pretraining, directly labelled).
"""
import csv
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(HERE, 'scores')
FC = os.path.join(HERE, '..', 'fresh-confirm', 'scores')
PLOTS = os.path.join(HERE, 'plots')
os.makedirs(PLOTS, exist_ok=True)

SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8983', '#e4e3df'
BLUE, ORANGE, GREEN = '#2a78d6', '#eb6834', '#1baf7a'
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


def fl(x):
    return float(x) if x not in (None, '') else None


def save(fig, name):
    fig.savefig(os.path.join(PLOTS, name), dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote plots/' + name)


# Short labels for the adopted systems, in progression order (secondary pools left out of fig1).
SHORT = [('OS-SCL (paper)', 'paper (reported)'),
         ('R0 =', 'single · R0 (eval.py)'),
         ('C1 =', 'single · C1'),
         ('C1ᵀ', 'single · C1ᵀ  [final]'),
         ('E2′[A8]', 'ensemble · A8 (8 nets)'),
         ('E2′[G8]', 'ensemble · G8 (8 nets)'),
         ('E2′[GS12]', 'ensemble · GS12 (12 nets)'),
         ('F1k', 'BEATs* · F1k'),
         ('X1 =', 'BEATs* · X1  [final]')]


def fig1_progression():
    S = read(os.path.join(SCORES, 'summary.csv'))
    if not S:
        return print('skip fig1 (no summary.csv)')
    rows = []
    for key, short in SHORT:
        r = next((x for x in S if x['system'].startswith(key) and x['track'] != 'fresh seeds (2060+)'), None)
        if r:
            rows.append((short, fl(r['dev_Amean']), fl(r['eval_Amean']), r['track']))
    fresh = [x for x in S if x['track'] == 'fresh seeds (2060+)' or x['system'].startswith('fresh')]
    FRESH = {'R0': 'single · R0 (eval.py)', 'C1': 'single · C1', 'C1T': 'single · C1ᵀ  [final]',
             'G4': 'ensemble · G4 (4 nets)', 'GS8': 'ensemble · GS8 (8 nets)', 'X1': 'BEATs* · X1 (GS8 + 3)'}
    n_seen = len(rows)
    for x in fresh:
        key = x['system'].replace('fresh ', '').split(',')[0]
        rows.append((FRESH.get(key, key), fl(x['dev_Amean']), fl(x['eval_Amean']), x['track']))
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * (len(rows) + 1) + 1.1))
    y = np.arange(len(rows))[::-1].astype(float)
    y[:n_seen] += 1.0 if len(rows) > n_seen else 0.0     # one empty row between seen and fresh
    for yi, (lab, d, e, tr) in zip(y, rows):
        if d is not None and e is not None:
            ax.plot([d, e], [yi, yi], color=GRID, lw=2, zorder=1)
        if d is not None:
            ax.scatter(d, yi, s=34, color=BLUE, zorder=3)
        if e is not None:
            ax.scatter(e, yi, s=34, color=ORANGE, zorder=3)
            ax.text(e + 0.06 if (d is None or e >= d) else e - 0.06, yi, f'{e:.2f}', va='center',
                    ha='left' if (d is None or e >= d) else 'right', fontsize=7.5, color=INK_2)
    paper_eval = rows[0][2]
    ax.axvline(paper_eval, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=0)
    ax.text(paper_eval + 0.03, y[0] + 0.62, f'paper eval {paper_eval:.2f}', fontsize=7.5, color=MUTED, ha='left')
    if len(rows) > n_seen:
        yd = y[n_seen - 1] - 0.62
        ax.axhline(yd, color=MUTED, lw=0.8)
        ax.text(0.012, yd - 0.16, 'below: fresh seeds 2060–2063, rules frozen first, scored once'
                '      above: seeds seen during development', transform=ax.get_yaxis_transform(),
                fontsize=7.5, color=INK_2, ha='left', va='top')
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    for t, r in zip(ax.get_yticklabels(), rows):
        if '[final]' in r[0]:
            t.set_fontweight('bold')
    ax.set_xlabel('Amean = mean over machines of (AUC + pAUC) / 2   (%)')
    ax.grid(axis='x')
    ax.set_ylim(y[-1] - 0.7, y[0] + 1.0)
    ax.scatter([], [], color=BLUE, s=34, label='dev')
    ax.scatter([], [], color=ORANGE, s=34, label='eval (labelled)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.085), ncol=2, fontsize=8)
    ev = {r[0]: r[2] for r in rows[:n_seen]}
    fe = {r[0]: r[2] for r in rows[n_seen:]}
    if fe:
        ttl = (f"On eval, fresh seeds land where the seen ones did: one network {fe['single · C1ᵀ  [final]']:.1f} (C1ᵀ), "
               f"from-scratch ensemble {fe['ensemble · GS8 (8 nets)']:.1f},\nwith BEATs* {fe['BEATs* · X1 (GS8 + 3)']:.1f}; "
               f"the paper reports {paper_eval:.2f}   (* external pretraining, separate track)")
    else:
        ttl = (f"Over the paper's eval mean: one network {ev['single · C1ᵀ  [final]'] - paper_eval:+.1f} (C1ᵀ), "
               f"with BEATs* {ev['BEATs* · X1  [final]'] - paper_eval:+.1f}   (* external pretraining, separate track)")
    ax.set_title(ttl, fontsize=9.5, loc='left')
    save(fig, 'fig1_progression.png')


def fig2_machines():
    M = read(os.path.join(SCORES, 'machines.csv'))
    if not M:
        return print('skip fig2 (no machines.csv)')
    get = {(r['system'], r['dataset'], r['machine']): fl(r['Amean']) for r in M}
    base = 'R0 (single, mean of 8)'
    series = [('C1T (single, mean of 8)', 'single · C1ᵀ', BLUE),
              ('GS12', 'ensemble · GS12', ORANGE),
              ('X1[r=0.375]', 'BEATs* · X1', GREEN)]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    x = np.arange(len(MACH))
    w = 0.26
    for i, (key, lab, col) in enumerate(series):
        d = [get.get((key, 'eval', m), np.nan) - get.get((base, 'eval', m), np.nan) for m in MACH]
        ax.bar(x + (i - 1) * w, d, w * 0.92, color=col, label=lab, zorder=2)
        if key.startswith('X1'):
            for xi, v in zip(x, d):
                ax.text(xi + w, v + 0.08, f'{v:+.1f}', ha='center', fontsize=7, color=INK_2)
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(MACH)
    ax.set_ylabel('eval Amean − R0 single net (points)')
    ax.grid(axis='y', zorder=0)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title('Over eval.py\'s score the gain is mostly ToyConveyor and ToyCar (BEATs* adds most on ToyCar);\n'
                 'the single-net kNN rule gives up 1.9 points on valve, which the ensembles keep', fontsize=9.5, loc='left')
    save(fig, 'fig2_machines_eval.png')


def fig3_fresh():
    single = read(os.path.join(FC, 'single.csv'))
    dec = json.load(open(os.path.join(FC, 'decisions.json'))) if os.path.exists(os.path.join(FC, 'decisions.json')) else None
    if not single or not dec:
        return print('skip fig3 (fresh-confirm not scored yet)')
    S = read(os.path.join(SCORES, 'summary.csv'))
    p1 = next(x for x in S if x['system'].startswith('C1ᵀ'))
    paper = next(x for x in S if x['track'] == 'paper')
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.3), sharey=True)

    def margin(ax, yv, text, colour):
        ax.annotate(text, (1.0, yv), xycoords=('axes fraction', 'data'), xytext=(4, 0), textcoords='offset points',
                    va='center', fontsize=7.5, color=colour, annotation_clip=False)

    for ax, ds, col in zip(axs, ('dev', 'eval'), (BLUE, ORANGE)):
        rs = sorted((r for r in single if r['rule'] == 'C1T' and r['dataset'] == ds and '_fc_nogain_' in r['run']),
                    key=lambda r: r['run'])
        v = [float(r['Amean']) for r in rs]
        ax.scatter(range(len(v)), v, color=col, s=40, zorder=3, edgecolor=SURFACE, linewidth=1)
        ax.axhline(float(np.mean(v)), color=col, lw=1.6, zorder=2)
        ref = float(p1[f'{ds}_Amean'])
        ax.axhline(ref, color=MUTED, lw=0.9, ls=(0, (3, 3)))
        margin(ax, ref, f'seeds 2040–47 mean {ref:.2f}', INK_2)
        bar = dec.get('F1', {}).get(f'bar_{ds}')
        if bar is not None:
            ax.axhline(bar, color=INK, lw=1.0, ls=(0, (5, 2)))
            margin(ax, bar, f'F1 bar {bar:.2f}', INK_2)
        pv = float(paper[f'{ds}_Amean'])
        ax.axhline(pv, color=INK, lw=1.0, ls=(0, (1, 2)))
        margin(ax, pv, f'paper, one run {pv:.2f}', INK_2)
        ax.set_xticks(range(len(v)))
        ax.set_xticklabels([r['run'].rsplit('_s', 1)[-1] for r in rs])
        ax.set_xlabel('training seed')
        ax.set_xlim(-0.5, len(v) - 0.5)
        ax.set_title(f'{ds} test set: fresh mean {np.mean(v):.2f} (solid line)', fontsize=9, loc='left', color=INK_2)
        ax.grid(axis='y')
        ax.set_axisbelow(True)
    axs[0].set_ylabel('C1ᵀ Amean, one network (%)')
    ok = dec.get('F1', {}).get('passed')
    fig.suptitle(('Every fresh network is above the paper under the frozen rule C1ᵀ, and the mean clears the F1 bar'
                  if ok else 'Fresh networks under the frozen rule C1ᵀ: the mean does not clear the F1 bar'),
                 fontsize=10.5, x=0.02, ha='left', fontweight='bold')
    fig.tight_layout(w_pad=7.5)
    save(fig, 'fig3_fresh_single.png')


if __name__ == '__main__':
    fig1_progression()
    fig2_machines()
    fig3_fresh()
