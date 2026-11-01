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


# ------------------------------------------------- replication of ao_nogain (seeds 2044-2047)
PAPER_DEV = {'AUC': 95.71, 'pAUC': 90.23, 'mAUC': 91.23}
PAPER_EVAL_MEAN = 93.22
REP = [2044, 2045, 2046, 2047]


def overall(arm, seeds, ds):
    """{metric: array over seeds} of overall R0 numbers, or None if a run is missing."""
    runs = [load(arm, s, ds) for s in seeds]
    if any(r is None for r in runs):
        return None
    return {k: np.array([r[('ALL', 'ALL', k)] for r in runs]) for k in ('AUC', 'pAUC', 'mAUC')}


def s1p():
    """{(run, rule, ds): {metric: %}} from scores/s1p.csv (already in %), or None."""
    p = os.path.join(HERE, 'scores', 's1p.csv')
    if not os.path.exists(p):
        return None
    with open(p, encoding='UTF-8') as f:
        return {(r['run'], r['rule'], r['dataset']): {k: float(r[k]) for k in ('AUC', 'pAUC', 'mAUC',
                                                                                      'mean_AUC_pAUC')}
                for r in csv.DictReader(f)}


def summary_rep():
    ng = {ds: overall('ao_nogain', REP, ds) for ds in ('dev', 'eval')}
    ag = {ds: overall('la_aug', REP, ds) for ds in ('dev', 'eval')}
    ng1 = overall('ao_nogain', SEEDS, 'dev')
    if not all(ng.values()) or not all(ag.values()) or ng1 is None:
        print('skip summary_rep: needs ao_nogain and la_aug on seeds 2044-2047')
        return
    print('\n== Replication: ao_nogain on fresh seeds 2044-2047, R0 (paired la_aug on the same seeds)')
    for k in ('AUC', 'pAUC', 'mAUC'):
        print(f'  dev {k:5s} nogain ' + ' '.join(f'{x:6.2f}' for x in ng['dev'][k]) +
              f'  mean {ng["dev"][k].mean():6.2f} | aug {ag["dev"][k].mean():6.2f} | paper {PAPER_DEV[k]:.2f}'
              f' -> {"above" if ng["dev"][k].mean() > PAPER_DEV[k] else "below"}')
    d = ng['dev']['AUC'] - ag['dev']['AUC']
    print('  R2 paired dev AUC Δ vs la_aug: ' + ' '.join(f'{x:+.2f}' for x in d) +
          f'  mean {d.mean():+.2f} (se {d.std(ddof=1) / 2:.2f}) -> {"holds" if d.mean() >= 0.20 else "fails"}')
    em = (ng['eval']['AUC'] + ng['eval']['pAUC']) / 2
    ea = (ag['eval']['AUC'] + ag['eval']['pAUC']) / 2
    de = ng['eval']['AUC'] - ag['eval']['AUC']
    print('  eval mean(AUC,pAUC) nogain ' + ' '.join(f'{x:6.2f}' for x in em) +
          f'  mean {em.mean():6.2f} | aug {ea.mean():6.2f} | paper {PAPER_EVAL_MEAN}')
    print('  paired eval AUC Δ vs la_aug: ' + ' '.join(f'{x:+.2f}' for x in de) + f'  mean {de.mean():+.2f}')
    p8 = np.concatenate([ng1['AUC'], ng['dev']['AUC']])
    lo = p8.mean() - 1.96 * p8.std(ddof=1) / np.sqrt(len(p8))
    print(f'  C1 pooled 8 seeds dev AUC {p8.mean():.2f}, mean − 1.96·se = {lo:.2f} vs 95.71 -> '
          f'{"holds" if lo > 95.71 else "fails"}')
    i5 = load('ao_nogain', REP[0], 'eval') and np.array(
        [load('ao_nogain', s, 'eval')[('ToyCar', 'id_05', 'AUC')] for s in REP])
    print('  R5 eval ToyCar id_05: ' + ' '.join(f'{x:6.2f}' for x in i5) + f'  mean {i5.mean():.2f}')


def fig4_replication():
    seeds8 = SEEDS + REP
    ng = {ds: overall('ao_nogain', seeds8, ds) for ds in ('dev', 'eval')}
    ag = {ds: overall('la_aug', seeds8, ds) for ds in ('dev', 'eval')}
    if not all(ng.values()) or not all(ag.values()):
        print('skip fig4_replication')
        return
    S = s1p()
    panels = [('dev', 'AUC'), ('dev', 'pAUC'), ('dev', 'mAUC'), ('eval', 'mean')]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.6))
    green, grey = ARMS[4][2], ARMS[1][2]
    for ax, (ds, k) in zip(axes, panels):
        def vals(d):
            return (d[ds]['AUC'] + d[ds]['pAUC']) / 2 if k == 'mean' else d[ds][k]
        for x, (d, c, lab) in enumerate([(ag, grey, 'aug'), (ng, green, 'no gain')]):
            v = vals(d)
            ax.scatter(np.full(4, x) - 0.1, v[:4], s=18, color=c, alpha=0.45, edgecolor='none', zorder=2)
            ax.scatter(np.full(4, x) + 0.1, v[4:], s=18, color=c, alpha=0.95, edgecolor='none', zorder=2,
                       marker='D')
            ax.plot([x - 0.25, x + 0.25], [v.mean()] * 2, color=c, lw=2.2, zorder=3)
            ax.annotate(f'{v.mean():.2f}', (x + 0.27, v.mean()), va='center', fontsize=7.5, color=INK)
        if S is not None:
            for x, arm in ((0, 'la_aug'), (1, 'ao_nogain')):
                kk = 'mean_AUC_pAUC' if k == 'mean' else k
                v = np.array([S[(f'SCLTFSTgramMFN_{arm}_s{s}', 'S1p', ds)][kk] for s in seeds8])
                ax.scatter([x + 0.38], [v.mean()], s=34, facecolor=SURFACE, edgecolor=green if x else grey,
                           lw=1.3, zorder=3, marker='o')
        ref = PAPER_EVAL_MEAN if k == 'mean' else PAPER_DEV[k]
        ax.axhline(ref, color=MUTED, lw=0.9, ls=(0, (4, 3)))
        lo, hi = ax.get_ylim()
        lo, hi = min(lo, ref), max(hi, ref)
        ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.04 * (hi - lo))
        ax.annotate(f'paper {ref:.2f}', (1.72, ref), va='bottom', ha='right', fontsize=7.5, color=INK_2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['aug', 'no gain'])
        ax.set_xlim(-0.5, 1.75)
        ax.grid(axis='y')
        ax.set_title(f'{ds} {"mean(AUC, pAUC)" if k == "mean" else k}', loc='left', fontsize=9.5)
    axes[0].set_ylabel('% (R0 unless marked)')
    n = sum(int((ng['dev'][k][4:] > PAPER_DEV[k]).all()) for k in PAPER_DEV)
    head = ('Replicated: "no gain" clears the paper\'s dev AUC, pAUC and mAUC on all 4 fresh seeds (R0)'
            if n == 3 else f'"no gain" clears the paper on all 4 fresh seeds for only {n} of 3 dev metrics (R0)')
    em = {a: ((d['eval']['AUC'] + d['eval']['pAUC']) / 2).mean() for a, d in (('aug', ag), ('ng', ng))}
    dd = em['ng'] - em['aug']
    verb = 'ties' if abs(dd) < 0.1 else ('beats' if dd > 0 else 'trails')
    fig.suptitle(f'{head}; on eval it {verb} aug ({em["ng"]:.2f} vs {em["aug"]:.2f}, 8 seeds)', x=0.02,
                 ha='left', fontsize=10.5)
    fig.text(0.02, -0.05, 'Circles (faded) = seeds 2040–2043 (stage 1); diamonds = fresh seeds 2044–2047; '
             'bar = 8-seed mean (R0). Hollow circle = 8-seed mean under S1′. Dashed = paper.',
             fontsize=7.5, color=INK_2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'fig4_replication.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote plots/fig4_replication.png')


if __name__ == '__main__':
    T = table()
    summary(T)
    summary_rep()
    fig1_toycar(T)
    fig2_machines(T)
    fig3_d1(T)
    fig4_replication()
