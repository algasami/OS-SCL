"""Figures for synthesis/single-final. Reads scores/ only; skips a figure whose inputs are missing.

Run with /home/f74134118/anaconda3/bin/python (matplotlib lives in base).
Colour = rule: C1 (the incumbent baseline) achromatic; S1′ᵀ orange (TTA without the pooled term,
tta's arm colour slot 2); C1ᵀ blue (the candidate). S1′ (phase-1 reference) light achromatic.
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
PLOTS = os.path.join(HERE, 'plots')
os.makedirs(PLOTS, exist_ok=True)

SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8983', '#e4e3df'
BLUE, ORANGE, GREEN = '#2a78d6', '#eb6834', '#1baf7a'
MACH = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
SEL, CONF = (2040, 2041, 2042, 2043), (2044, 2045, 2046, 2047)
ALL8 = SEL + CONF
RUN = 'SCLTFSTgramMFN_ao_nogain_s{}'.format
COL = {'S1p': '#b9b8b3', 'C1': INK_2, 'S1pT': ORANGE, 'C1T': BLUE}
NAME = {'S1p': 'S1′', 'C1': 'C1', 'S1pT': 'S1′ᵀ', 'C1T': 'C1ᵀ'}
PAPER = {'dev': (95.71 + 90.23) / 2, 'eval': 93.22}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'text.color': INK, 'axes.labelcolor': INK_2, 'axes.edgecolor': GRID,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False,
})


def read(name):
    p = os.path.join(SCORES, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding='UTF-8') as f:
        return list(csv.DictReader(f)) if name.endswith('.csv') else json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(PLOTS, name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def table(*names):
    """{(seed, ds, rule, machine): row} over the given metrics files (later files win)."""
    out = {}
    for n in names:
        for r in read(n) or []:
            s = int(r['system'].rsplit('_s', 1)[1])
            out[(s, r['dataset'], r['rule'], r['machine'])] = {k: float(r[k]) for k in ('AUC', 'pAUC', 'mAUC', 'Amean')}
    return out


def fig1():
    """Per-seed and bootstrap Δ Amean, C1ᵀ − C1 (and S1′ᵀ − C1), by stage."""
    A, B, E = read('stageA.json'), read('stageB.json'), read('eval.json')
    T = table('metrics_stageA.csv', 'metrics_stageB.csv', 'metrics_eval.csv')
    cells = []
    if A:
        cells.append(('stage A dev\n2040–43', A['sel_dev'], 'dev', SEL))
    if B:
        cells.append(('stage B dev\n2044–47', B['conf_dev'], 'dev', CONF))
        cells.append(('dev\n8 seeds', B['dev8'], 'dev', ALL8))
    if E:
        cells.append(('eval\n8 seeds', E['eval8'], 'eval', ALL8))
    if not cells:
        return
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    for j, (name, d, ds, seeds) in enumerate(cells):
        lo, hi = d['ci95']['Amean']
        ax.plot([j, j], [lo, hi], color=BLUE, lw=2.2, solid_capstyle='butt')
        ax.scatter([j], [d['d_Amean']], color=BLUE, s=34, zorder=3)
        ax.text(j + 0.1, d['d_Amean'], f'{d["d_Amean"]:+.3f}\n{d["n_pos_Amean"]}/{d["n"]} > 0', va='center', fontsize=8)
        pts = [T[(s, ds, 'C1T', 'overall')]['Amean'] - T[(s, ds, 'C1', 'overall')]['Amean'] for s in seeds]
        ax.scatter([j - 0.16] * len(pts), pts, color=BLUE, s=11, alpha=0.5, lw=0)
        sp = [T[(s, ds, 'S1pT', 'overall')]['Amean'] - T[(s, ds, 'C1', 'overall')]['Amean'] for s in seeds]
        ax.scatter([j - 0.3], [np.mean(sp)], marker='_', s=120, color=ORANGE, lw=2)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=BLUE, marker='o', lw=2.2, label='C1ᵀ − C1: mean, 95% CI, seeds'),
                       Line2D([], [], color=ORANGE, marker='_', ms=12, mew=2, lw=0, label='S1′ᵀ − C1: seed mean')],
              loc='lower right', fontsize=7.5)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([c[0] for c in cells], fontsize=8)
    ax.set_xlim(-0.6, len(cells) - 0.1)
    ax.set_ylabel('C1ᵀ − C1, Amean (points)\nbar = paired 95% clip-bootstrap; dots = seeds')
    ax.grid(axis='y')
    if E:
        v = 'adopted' if E['D1']['adopt_C1T'] else 'not adopted, C1 stays'
        title = f'C1ᵀ vs C1: dev {cells[2][1]["d_Amean"]:+.2f}, eval {E["eval8"]["d_Amean"]:+.2f} Amean; C1ᵀ {v}'
    else:
        title = 'C1ᵀ − C1 by stage (eval not yet scored)'
    ax.set_title(title, loc='left', fontsize=10)
    fig.tight_layout()
    save(fig, 'fig1_delta.png')


def fig2():
    """Per-machine Δ AUC, C1ᵀ − C1, dev and eval 8-seed."""
    B, E = read('stageB.json'), read('eval.json')
    cells = [(n, d) for n, d in (('dev', B and B['dev8']), ('eval', E and E['eval8'])) if d]
    if not cells:
        return
    T = table('metrics_stageB.csv', 'metrics_eval.csv')
    fig, axes = plt.subplots(1, len(cells), figsize=(5.0 * len(cells), 3.6), sharey=True, squeeze=False)
    for ax, (ds, d) in zip(axes[0], cells):
        x = np.arange(len(MACH))
        v = [d['per_machine_dAUC'][m] for m in MACH]
        ax.bar(x, v, color=BLUE, width=0.55)
        for m_i, m in enumerate(MACH):
            pts = [T[(s, ds, 'C1T', m)]['AUC'] - T[(s, ds, 'C1', m)]['AUC'] for s in ALL8]
            ax.scatter([m_i + 0.36] * 8, pts, color=INK_2, s=7, alpha=0.6, lw=0)
            ax.text(m_i, v[m_i] + (0.02 if v[m_i] >= 0 else -0.02), f'{v[m_i]:+.2f}', ha='center',
                    va='bottom' if v[m_i] >= 0 else 'top', fontsize=8)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(MACH, fontsize=8)
        ax.set_title(f'{ds}, 8 seeds (bar = mean, dots = seeds)', loc='left', fontsize=9.5)
        ax.grid(axis='y')
    axes[0][0].set_ylabel('C1ᵀ − C1, AUC (points)')
    fig.suptitle(f'Where TTA moves C1: ToyConveyor {cells[0][1]["per_machine_dAUC"]["ToyConveyor"]:+.2f} dev'
                 + (f', {cells[1][1]["per_machine_dAUC"]["ToyConveyor"]:+.2f} eval' if len(cells) > 1 else ''),
                 x=0.01, ha='left', fontsize=10)
    fig.tight_layout()
    save(fig, 'fig2_per_machine.png')


def fig3():
    """Absolute 8-seed Amean of S1′, S1′ᵀ, C1, C1ᵀ on dev and eval, against the paper."""
    chk, B, E = read('repro_check.json'), read('stageB.json'), read('eval.json')
    if not chk:
        return
    T = table('metrics_check.csv', 'metrics_stageB.csv', 'metrics_eval.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, ds in zip(axes, ('dev', 'eval')):
        rules = ['S1p', 'S1pT', 'C1'] + (['C1T'] if (B if ds == 'dev' else E) else [])
        for j, r in enumerate(rules):
            pts = [T[(s, ds, r, 'overall')]['Amean'] for s in ALL8 if (s, ds, r, 'overall') in T]
            ax.scatter([j] * len(pts), pts, color=COL[r], s=12, alpha=0.55, lw=0)
            mu = np.mean(pts)
            ax.plot([j - 0.25, j + 0.25], [mu, mu], color=COL[r], lw=2.4)
            ax.text(j + 0.28, mu, f'{mu:.2f}', va='center', fontsize=8)
        ax.axhline(PAPER[ds], color=MUTED, lw=0.9, ls='--')
        ax.text(len(rules) - 0.5, PAPER[ds], f'paper {PAPER[ds]:.2f}', fontsize=7.5, color=INK_2, va='bottom', ha='right')
        ax.set_xticks(range(len(rules)))
        ax.set_xticklabels([NAME[r] for r in rules])
        ax.set_xlim(-0.5, len(rules) - 0.2)
        ax.set_title(f'{ds}: Amean, 8 seeds (bar = mean)', loc='left', fontsize=9.5)
        ax.grid(axis='y')
    axes[0].set_ylabel('Amean = mean(AUC, pAUC) (points)')
    fig.suptitle('One ao_nogain network under each single-model rule: every rule\'s 8-seed mean is above the paper on both sets',
                 x=0.01, ha='left', fontsize=10)
    fig.tight_layout()
    save(fig, 'fig3_absolute.png')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3):
        try:
            f()
        except (KeyError, TypeError, ValueError) as e:     # degrade gracefully on partial scores/
            print(f'skipped {f.__name__}: {e!r}')
