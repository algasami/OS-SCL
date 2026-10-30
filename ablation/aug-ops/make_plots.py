"""Figures and summary for ablation/aug-ops. Reads scores/ only (via scores_path); skips what is missing.

Run with /home/f74134118/anaconda3/bin/python (matplotlib lives in base).
Colours: the three leave-one-out arms take the three categorical slots; `la_aug` (all ops)
is achromatic dark grey and `fr_f1` (no ops) light grey.
"""
import csv
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from scores_path import scores_csv  # noqa: E402

PLOTS = os.path.join(HERE, 'plots')
os.makedirs(PLOTS, exist_ok=True)
SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8983', '#e4e3df'
LIGHT = '#bdbcb6'
SEEDS = [2040, 2041, 2042, 2043]
ARMS = [('fr_f1', 'plain (no ops)', LIGHT), ('la_aug', 'aug (all ops)', INK_2),
        ('ao_nonoise', 'no noise', '#2a78d6'), ('ao_noshift', 'no shift', '#eb6834'),
        ('ao_nogain', 'no gain', '#1baf7a')]
NEW = [a for a, _, _ in ARMS[2:]]
MACH = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
# D1 thresholds, fixed in the README before any run
D1_ID05, D1_EVAL, D1_DEV = 86.0, 95.45, 95.30

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'text.color': INK, 'axes.labelcolor': INK_2, 'axes.edgecolor': GRID,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False,
})


def load(arm, seed, ds):
    """{(machine, id, metric): value x100} for one run and dataset, or None."""
    try:
        p = scores_csv(f'SCLTFSTgramMFN_{arm}_s{seed}', ds)
    except Exception:
        return None
    if not p or not os.path.exists(p):
        return None
    out = {}
    with open(p, encoding='UTF-8') as f:
        for r in csv.DictReader(f):
            for k in ('AUC', 'pAUC', 'mAUC'):
                if r[k]:
                    out[(r['machine'], r['id'], k)] = 100 * float(r[k])
    return out


def table():
    """T[arm][ds] = array (seeds,) per key; only arms with all 4 seeds on both sets."""
    T = {}
    for arm, _, _ in ARMS:
        runs = {ds: [load(arm, s, ds) for s in SEEDS] for ds in ('dev', 'eval')}
        if all(r is not None for v in runs.values() for r in v):
            T[arm] = {ds: {k: np.array([r[k] for r in v]) for k in v[0]} for ds, v in runs.items()}
    return T


def g(T, arm, ds, mach='ALL', id_='ALL', k='AUC'):
    return T[arm][ds][(mach, id_, k)]


def mean_ap(T, arm, ds):
    return (g(T, arm, ds) + g(T, arm, ds, k='pAUC')) / 2


def paired(a, b):
    d = a - b
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def summary(T):
    if not all(a in T for a in ('fr_f1', 'la_aug')) or not any(a in T for a in NEW):
        print('skip summary: needs fr_f1, la_aug and at least one aug-ops arm with 4 seeds')
        return
    arms = [a for a, _, _ in ARMS if a in T]
    print('R0, 4-seed means (seeds 2040-2043). Δ = seed-paired vs la_aug, (se).\n')
    rows = [('dev', 'ALL', 'ALL', 'AUC'), ('dev', 'ALL', 'ALL', 'pAUC'), ('eval', 'ALL', 'ALL', 'AUC'),
            ('eval', 'ALL', 'ALL', 'pAUC'),
            ('dev', 'ToyCar', 'id_01', 'AUC'), ('dev', 'ToyCar', 'ALL', 'AUC'),
            ('eval', 'ToyCar', 'id_05', 'AUC'), ('eval', 'ToyCar', 'id_06', 'AUC'), ('eval', 'ToyCar', 'id_07', 'AUC'),
            ('eval', 'ToyCar', 'ALL', 'AUC'), ('dev', 'ToyConveyor', 'ALL', 'AUC'),
            ('eval', 'ToyConveyor', 'ALL', 'AUC'), ('dev', 'pump', 'ALL', 'AUC'), ('eval', 'pump', 'ALL', 'AUC')]
    print(f'{"":28s}' + ''.join(f'{a:>20s}' for a in arms))
    for ds, m, i, k in rows:
        lab = f'{ds} {m}{"" if i == "ALL" else " " + i} {k}'
        cells = []
        for a in arms:
            v = g(T, a, ds, m, i, k)
            if a == 'la_aug':
                cells.append(f'{v.mean():8.2f}            ')
            else:
                d, se = paired(v, g(T, 'la_aug', ds, m, i, k))
                cells.append(f'{v.mean():8.2f} {d:+6.2f}({se:.2f})')
        print(f'{lab:28s}' + ''.join(f'{c:>20s}' for c in cells))
    print('\nPer-seed eval ToyCar id_05:')
    for a in arms:
        print(f'  {a:11s} ' + ' '.join(f'{x:6.2f}' for x in g(T, a, 'eval', 'ToyCar', 'id_05')))
    print('\nPer machine, mean Δ AUC vs fr_f1 (points):')
    print(f'{"":20s}' + ''.join(f'{m:>12s}' for m in MACH))
    for a in arms[1:]:
        for ds in ('dev', 'eval'):
            print(f'{a + " " + ds:20s}' + ''.join(
                f'{(g(T, a, ds, m) - g(T, "fr_f1", ds, m)).mean():+12.2f}' for m in MACH))
    print(f'\nD1 (README): eval id_05 ≥ {D1_ID05}, eval AUC ≥ {D1_EVAL}, dev AUC ≥ {D1_DEV}')
    cands = []
    for a in NEW:
        if a not in T:
            continue
        i5, ev, dv = (g(T, a, 'eval', 'ToyCar', 'id_05').mean(), g(T, a, 'eval').mean(), g(T, a, 'dev').mean())
        ok = [i5 >= D1_ID05, ev >= D1_EVAL, dv >= D1_DEV]
        print(f'  {a:11s} id_05 {i5:6.2f} {"ok" if ok[0] else "no"} | eval {ev:6.2f} {"ok" if ok[1] else "no"}'
              f' | dev {dv:6.2f} {"ok" if ok[2] else "no"} -> {"CANDIDATE" if all(ok) else "out"}')
        if all(ok):
            cands.append((mean_ap(T, a, 'dev').mean(), a))
    print('  D1 pick:', max(cands)[1] if cands else 'none (D0)')


def dots(ax, x, v, c, label=None):
    ax.scatter(np.full(len(v), x) + np.linspace(-0.12, 0.12, len(v)), v, s=16, color=c, alpha=0.55,
               edgecolor='none', zorder=2)
    ax.plot([x - 0.25, x + 0.25], [v.mean()] * 2, color=c, lw=2.4, zorder=3, label=label)
    ax.annotate(f'{v.mean():.1f}', (x + 0.27, v.mean()), va='center', fontsize=7.5, color=INK)


# ------------------------------------------------- fig1: the two hard ToyCar IDs
def fig1_toycar(T):
    arms = [(a, l, c) for a, l, c in ARMS if a in T]
    if len(arms) < 3:
        print('skip fig1_toycar')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharey=False)
    for ax, (ds, i, title) in zip(axes, [('eval', 'id_05', 'eval ToyCar id_05 (aug\'s loss)'),
                                         ('dev', 'id_01', 'dev ToyCar id_01 (aug\'s gain)')]):
        for x, (a, l, c) in enumerate(arms):
            dots(ax, x, g(T, a, ds, 'ToyCar', i), c)
        if ds == 'eval':
            ax.axhline(D1_ID05, color=MUTED, lw=0.9, ls=(0, (4, 3)))
            ax.annotate(f'D1 bar {D1_ID05:.0f}', (len(arms) - 0.5, D1_ID05), va='bottom', ha='right',
                        fontsize=7.5, color=INK_2)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([l for _, l, _ in arms], rotation=20, ha='right')
        ax.set_xlim(-0.5, len(arms) - 0.2)
        ax.grid(axis='y')
        ax.set_title(title, loc='left', fontsize=9.5)
    axes[0].set_ylabel('AUC (%), R0')
    i5 = {a: g(T, a, 'eval', 'ToyCar', 'id_05').mean() for a, _, _ in arms}
    best = max((a for a in NEW if a in T), key=lambda a: i5[a])
    lab = dict((a, l) for a, l, _ in ARMS)
    fig.suptitle(f'Eval ToyCar id_05: best repair is "{lab[best]}" ({i5[best]:.1f} vs aug {i5["la_aug"]:.1f}, '
                 f'plain {i5["fr_f1"]:.1f})', x=0.02, ha='left', fontsize=10.5)
    fig.text(0.02, -0.04, 'Dots = seeds 2040–2043; bar = mean. Every arm shares seeds with plain and aug.',
             fontsize=7.5, color=INK_2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'fig1_toycar.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote plots/fig1_toycar.png')


# ------------------------------------------------- fig2: per machine, Δ vs plain
def fig2_machines(T):
    arms = [(a, l, c) for a, l, c in ARMS[1:] if a in T]
    if 'fr_f1' not in T or len(arms) < 2:
        print('skip fig2_machines')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    w = 0.8 / len(arms)
    for ax, ds in zip(axes, ('dev', 'eval')):
        for j, (a, l, c) in enumerate(arms):
            d = [(g(T, a, ds, m) - g(T, 'fr_f1', ds, m)).mean() for m in MACH]
            ax.bar(np.arange(len(MACH)) + (j - (len(arms) - 1) / 2) * w, d, width=w - 0.02, color=c,
                   label=l, zorder=2)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_xticks(range(len(MACH)))
        ax.set_xticklabels([m.replace('ToyConveyor', 'ToyConv.') for m in MACH])
        ax.grid(axis='y')
        ax.set_title(f'{ds} set', loc='left', fontsize=9.5)
    axes[0].set_ylabel('Δ machine AUC vs plain (points)')
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8)
    tc = {a: (g(T, a, 'eval', 'ToyConveyor') - g(T, 'fr_f1', 'eval', 'ToyConveyor')).mean() for a, _, _ in arms}
    fig.suptitle(f'Noise carries the ToyConveyor gain: eval ToyConveyor vs plain is aug {tc["la_aug"]:+.1f}, '
                 f'no noise {tc["ao_nonoise"]:+.1f}; every arm still loses eval ToyCar', x=0.02, ha='left',
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'fig2_machines.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote plots/fig2_machines.png')


# ------------------------------------------------- fig3: D1 on one page
def fig3_d1(T):
    arms = [(a, l, c) for a, l, c in ARMS if a in T]
    if len(arms) < 3:
        print('skip fig3_d1')
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for a, l, c in arms:
        dv, ev = g(T, a, 'dev'), g(T, a, 'eval')
        ax.scatter(dv, ev, s=14, color=c, alpha=0.4, edgecolor='none')
        ax.scatter([dv.mean()], [ev.mean()], s=70, color=c, edgecolor=SURFACE, lw=1.2, zorder=3)
        ax.annotate(l, (dv.mean(), ev.mean()), xytext=(7, 4), textcoords='offset points', fontsize=8, color=INK)
    ax.axvline(D1_DEV, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.axhline(D1_EVAL, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.fill_between([D1_DEV, xl[1]], D1_EVAL, yl[1], color=GRID, alpha=0.45, zorder=0, lw=0)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.text(xl[1], yl[1], 'D1 (b)+(c) region ', ha='right', va='top', fontsize=7.5, color=INK_2)
    ax.set_xlabel('dev AUC (%), R0')
    ax.set_ylabel('eval AUC (%), R0')
    ax.grid()
    both = [a for a in NEW if a in T and g(T, a, 'dev').mean() >= D1_DEV and g(T, a, 'eval').mean() >= D1_EVAL]
    lab = dict((a, l) for a, l, _ in ARMS)
    i5 = ', '.join(f'{lab[a]} id_05 {g(T, a, "eval", "ToyCar", "id_05").mean():.1f}' for a in both)
    ax.set_title(f'No arm passes D1: {len(both)} of 3 clear the dev and eval bars\nbut miss id_05 ≥ 86 ({i5})',
                 loc='left', fontsize=10)
    fig.text(0.02, -0.03, 'Small dots = seeds; large = 4-seed mean. D1 also needs eval ToyCar id_05 ≥ 86 (fig1).',
             fontsize=7.5, color=INK_2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'fig3_d1.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote plots/fig3_d1.png')


if __name__ == '__main__':
    T = table()
    summary(T)
    fig1_toycar(T)
    fig2_machines(T)
    fig3_d1(T)
