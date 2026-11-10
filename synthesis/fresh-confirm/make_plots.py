"""Figures for synthesis/fresh-confirm. Reads scores/ only; a figure whose inputs are missing is skipped.

Run with /home/f74134118/anaconda3/bin/python (matplotlib lives in base):
    make_plots.py                      the fresh scores (scores/*.csv, written by `run.sh score`)
    make_plots.py --src scores/dryrun_p1 --out <dir>
                                       layout check on the dry run (phase-1 seeds, all seen); never commit these

Colour: dev = blue, eval = orange, as in ../final. Everything seen in phase 1 is grey.
The paper's numbers are dotted ink lines. X1 uses external pretraining and says so on the axis.
"""
import csv
import json
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'ablation'))
from scores_path import scores_csv  # noqa: E402

ARGS = dict(zip(sys.argv[1::2], sys.argv[2::2]))
SRC = os.path.join(HERE, ARGS.get('--src', 'scores'))
REF = os.path.join(HERE, 'scores', 'dryrun_p1')           # this scorer on phase-1 seeds 2040-2043
P1 = os.path.join(HERE, '..', 'single-final', 'scores')   # phase-1 single networks, 8 seeds
PLOTS = ARGS.get('--out', os.path.join(HERE, 'plots'))
os.makedirs(PLOTS, exist_ok=True)

SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8983', '#e4e3df'
BLUE, ORANGE = '#2a78d6', '#eb6834'
COL = {'dev': BLUE, 'eval': ORANGE}
MACH = ['fan', 'pump', 'slider', 'valve', 'ToyCar', 'ToyConveyor']
PAPER = {'dev': 92.97, 'eval': 93.22}                     # Amean; ../final/PUBLISHED.md
P1_SEEDS = [f'SCLTFSTgramMFN_ao_nogain_s{s}' for s in range(2040, 2048)]
RULE_LABEL = {'R0': 'R0\n(eval.py)', 'C1': 'C1', 'C1T': 'C1ᵀ\n(final rule)'}

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


def paper_line(ax, ds, what):
    """The paper's Amean as a dotted line, labelled in the right margin so it never sits on a dot."""
    ax.axhline(PAPER[ds], color=INK, lw=1, ls=':', zorder=1)
    ax.annotate(f'paper,\n{what}\n{PAPER[ds]:.2f}', (1.0, PAPER[ds]), xycoords=('axes fraction', 'data'),
                xytext=(4, -1), textcoords='offset points', va='top', fontsize=7.6, color=INK_2,
                annotation_clip=False)


def spread(n, width):
    """Deterministic horizontal offsets for n dots that share a category."""
    return np.zeros(1) if n == 1 else np.linspace(-width / 2, width / 2, n)


def dots(ax, x, ys, colour, width=0.16, size=34, label=None, z=3):
    ys = np.sort(np.asarray(ys, float))
    ax.scatter(x + spread(len(ys), width), ys, s=size, color=colour, edgecolor=SURFACE, linewidth=1.0,
               zorder=z, label=label)
    ax.plot([x - width * 0.8, x + width * 0.8], [ys.mean()] * 2, color=colour, lw=2, solid_capstyle='round',
            zorder=z - 1)
    return ys.mean()


def title(fig, head, sub, y=1.0):
    fig.text(0.01, y + 0.075, head, fontsize=11.5, fontweight='bold', ha='left', va='bottom')
    fig.text(0.01, y + 0.005, sub, fontsize=8.8, color=INK_2, ha='left', va='bottom')


def p1_single(ds, rule):
    """Phase-1 per-seed Amean of the 8 ao_nogain networks under one rule, or None."""
    if rule == 'R0':
        out = []
        for r in P1_SEEDS:
            try:
                rows = read(scores_csv(r, ds))
            except FileNotFoundError:
                return None
            o = [x for x in rows or [] if x['scope'] == 'overall']
            if not o:
                return None
            out.append(50 * (float(o[0]['AUC']) + float(o[0]['pAUC'])))
        return out
    rows = read(os.path.join(P1, 'metrics_stageB.csv' if ds == 'dev' else 'metrics_eval.csv'))
    if not rows:
        return None
    out = [float(x['Amean']) for x in rows
           if x['dataset'] == ds and x['rule'] == rule and x['machine'] == 'overall' and x['system'] in P1_SEEDS]
    return out if len(out) == len(P1_SEEDS) else None


def is_g(run):
    return '_nogain_' in run


# ------------------------------------------------------------------ fig 1: single networks, seed by seed
def fig1_single():
    S, D = read(os.path.join(SRC, 'single.csv')), read_json(os.path.join(SRC, 'decisions.json'))
    if not S or not D:
        return print('skip fig1 (no single.csv / decisions.json)')
    rules = ['R0', 'C1', 'C1T']
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), sharey=True)
    means = {}
    for ax, ds in zip(axes, ('dev', 'eval')):
        for i, rule in enumerate(rules):
            old = p1_single(ds, rule)
            new = [float(x['Amean']) for x in S if x['dataset'] == ds and x['rule'] == rule and is_g(x['run'])]
            if old:
                dots(ax, i - 0.2, old, MUTED, size=26, label='phase 1, 8 seeds (seen)' if i == 0 else None)
            means[(ds, rule)] = dots(ax, i + 0.2, new, COL[ds], label=f'fresh, {len(new)} seeds' if i == 0 else None)
            ax.annotate(f'{means[(ds, rule)]:.2f}', (i + 0.2 + 0.15, means[(ds, rule)]), xytext=(4, 0),
                        textcoords='offset points', va='center', fontsize=8.5, color=INK)
        bar = D['F1'][f'bar_{ds}']
        ax.plot([1.55, 2.62], [bar] * 2, color=INK, lw=1.2, ls=(0, (4, 2)))
        ax.annotate(f'F1 bar\n{bar:.2f}', (1.0, bar), xycoords=('axes fraction', 'data'), xytext=(4, 1),
                    textcoords='offset points', va='bottom', fontsize=7.6, color=INK_2, annotation_clip=False)
        paper_line(ax, ds, 'one run')
        ax.set_xticks(range(len(rules)))
        ax.set_xticklabels([RULE_LABEL[r] for r in rules])
        ax.set_xlim(-0.5, 2.62)
        ax.grid(axis='y')
        ax.set_axisbelow(True)
        ax.set_title('dev test set' if ds == 'dev' else 'eval test set', loc='left', fontsize=9.5, color=INK_2)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, handletextpad=0.2)
    axes[0].set_ylabel('Amean = (AUC + pAUC) / 2, %')
    f1 = D['F1']
    verdict = 'holds' if f1['passed'] else 'does not hold'
    title(fig, f"F1 {verdict}: fresh single networks under C1ᵀ average {f1['dev']:.2f} on dev and "
               f"{f1['eval']:.2f} on eval",
          'One dot per network, the short line is the mean. The dashed line is the frozen non-inferiority bar '
          'for the C1ᵀ mean.')
    fig.tight_layout(w_pad=4.5)
    save(fig, 'fig1_single_seeds.png')


# ------------------------------------------------------------------ fig 2: the three ensembles
def fig2_ensembles():
    E, D = read(os.path.join(SRC, 'ensembles.csv')), read_json(os.path.join(SRC, 'decisions.json'))
    if not E or not D:
        return print('skip fig2 (no ensembles.csv / decisions.json)')
    ref = read(os.path.join(REF, 'ensembles.csv')) if os.path.abspath(SRC) != os.path.abspath(REF) else None
    names = ['G4', 'GS8', 'X1']
    n_s = D['F4'].get('n_S', 4)
    lab = ['G4\n4 nogain', f'GS8\n4 nogain + {n_s} spec', 'X1\nGS8 + 3 BEATs*']

    def val(rows, sysname, ds):
        return next(float(x['Amean']) for x in rows if x['system'] == sysname and x['dataset'] == ds
                    and x['machine'] == 'all')

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    for ax, ds in zip(axes, ('dev', 'eval')):
        ys = [val(E, n, ds) for n in names]
        if ref:
            ax.scatter(np.arange(3) - 0.14, [val(ref, n, ds) for n in names], s=46, marker='D', facecolor=SURFACE,
                       edgecolor=MUTED, linewidth=1.5, zorder=3, label='same build, phase-1 seeds 2040–43 (seen)')
        ax.scatter(np.arange(3) + (0.14 if ref else 0), ys, s=64, color=COL[ds], edgecolor=SURFACE, linewidth=1.0,
                   zorder=4, label='fresh seeds 2060–63')
        for i, y in enumerate(ys):
            ax.annotate(f'{y:.2f}', (i + (0.14 if ref else 0), y), xytext=(8, 0), textcoords='offset points',
                        va='center', fontsize=8.5)
        paper_line(ax, ds, 'one net')
        ax.set_xticks(range(3))
        ax.set_xticklabels(lab)
        ax.set_xlim(-0.5, 2.6)
        ax.grid(axis='y')
        ax.set_axisbelow(True)
        ax.set_title('dev test set' if ds == 'dev' else 'eval test set', loc='left', fontsize=9.5, color=INK_2)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=1, fontsize=8, handletextpad=0.2)
    axes[0].set_ylabel('Amean, %')
    f4, f6 = D['F4'], D['F6']
    title(fig, f"Fresh ensembles on eval: GS8 − G4 = {sg(f4['delta']['eval'])} "
               f"[{sg(f4['eval_ci'][0])}, {sg(f4['eval_ci'][1])}],  X1 − GS8 = {sg(f6['delta']['eval'])} "
               f"[{sg(f6['eval_ci'][0])}, {sg(f6['eval_ci'][1])}]",
          'Brackets: 95 % bootstrap over test clips, these networks held fixed. * BEATs = external pretraining; '
          'the BEATs networks are the phase-1 ones.')
    fig.tight_layout(w_pad=4.5)
    save(fig, 'fig2_ensembles.png')


# ------------------------------------------------------------------ fig 3: F4b, member type at equal size
def fig3_f4b():
    F, D = read(os.path.join(SRC, 'f4b_subsets.csv')), read_json(os.path.join(SRC, 'decisions.json'))
    if not F or not D or 'F4b' not in D:
        return print('skip fig3 (no f4b_subsets.csv)')
    same = os.path.abspath(SRC) == os.path.abspath(REF)
    Fr, Dr = (None, None) if same else (read(os.path.join(REF, 'f4b_subsets.csv')),
                                        read_json(os.path.join(REF, 'decisions.json')))
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    rng = np.random.RandomState(0)
    for i, ds in enumerate(('dev', 'eval')):
        groups = [(i + 0.2 if Fr else i, F, D, COL[ds], 'fresh seeds')]
        if Fr:
            groups.insert(0, (i - 0.2, Fr, Dr, MUTED, 'phase-1 seeds (seen)'))
        for x, rows, dec, colour, name in groups:
            d = np.array([float(r['delta_vs_G4']) for r in rows if r['dataset'] == ds])
            ax.scatter(x + rng.uniform(-0.11, 0.11, len(d)), d, s=16, color=colour, alpha=0.75, edgecolor='none',
                       zorder=3)
            ax.plot([x - 0.15, x + 0.15], [d.mean()] * 2, color=colour, lw=2.2, solid_capstyle='round', zorder=4)
            ax.scatter([x], [dec['F4']['delta'][ds]], s=70, marker='D', facecolor=SURFACE, edgecolor=INK,
                       linewidth=1.4, zorder=5)
            ax.annotate(f"{name}\n{int((d > 0).sum())} of {len(d)} above", (x, ax.get_ylim()[0]), xytext=(0, 4),
                        textcoords='offset points', ha='center', va='bottom', fontsize=7.8, color=INK_2)
    ax.axhline(0, color=INK, lw=1)
    ax.annotate('G4\n(4 nogain)', (1.0, 0), xycoords=('axes fraction', 'data'), xytext=(4, 0),
                textcoords='offset points', va='center', fontsize=7.8, color=INK_2, annotation_clip=False)
    ax.scatter([], [], s=16, color=INK_2, label='one 2 nogain + 2 spec subset (4 networks)')
    ax.plot([], [], color=INK_2, lw=2.2, label='mean of the 36 subsets')
    ax.scatter([], [], s=70, marker='D', facecolor=SURFACE, edgecolor=INK, linewidth=1.4,
               label='GS8 (8 networks)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8, handletextpad=0.3,
              columnspacing=1.2)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.16 * (hi - lo), hi)
    for t in ax.texts[:-1]:
        t.xy = (t.xy[0], ax.get_ylim()[0])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.1f}'.replace('-', '−')))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['dev test set', 'eval test set'])
    ax.set_xlim(-0.52, 1.52)
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.set_ylabel('Amean minus G4, points')
    b = D['F4b']
    title(fig, f"At equal size, mixing in spec networks moves Amean by {sg(b['dev']['mean_delta'])} on dev and "
               f"{sg(b['eval']['mean_delta'])} on eval",
          'Each dot swaps two of the four nogain networks for two spec networks. The 36 subsets share networks, '
          'so they are not independent.', y=1.0)
    fig.tight_layout()
    save(fig, 'fig3_f4b_type_vs_count.png')


# ------------------------------------------------------------------ fig 4: where fresh differs, by machine
def fig4_machines():
    M, ref = read(os.path.join(SRC, 'single_machines.csv')), read(os.path.join(P1, 'summary.csv'))
    if not M or not ref:
        return print('skip fig4 (no single_machines.csv or phase-1 summary.csv)')
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    worst = (0.0, '', '')
    for k, ds in enumerate(('dev', 'eval')):
        for j, mm in enumerate(MACH):
            base = [float(x['Amean_mean8']) for x in ref if x['dataset'] == ds and x['rule'] == 'C1T'
                    and x['machine'] == mm]
            new = [float(x['Amean']) for x in M if x['dataset'] == ds and x['rule'] == 'C1T' and x['machine'] == mm
                   and is_g(x['run'])]
            if not base or not new:
                continue
            d = np.array(new) - base[0]
            y = j + (-0.17 if k == 0 else 0.17)
            ax.scatter(d, [y] * len(d), s=18, color=COL[ds], alpha=0.45, edgecolor='none', zorder=3)
            ax.scatter([d.mean()], [y], s=62, color=COL[ds], edgecolor=SURFACE, linewidth=1.0, zorder=4,
                       label=('dev' if k == 0 else 'eval') + ': mean of the fresh seeds' if j == 0 else None)
            if abs(d.mean()) > abs(worst[0]):
                worst = (d.mean(), mm, ds)
    ax.scatter([], [], s=18, color=INK_2, alpha=0.6, label='one fresh network')
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(MACH)))
    ax.set_yticklabels(MACH)
    ax.invert_yaxis()
    ax.grid(axis='x')
    ax.set_axisbelow(True)
    ax.set_xlabel('C1ᵀ Amean minus the phase-1 mean of 8 seeds, points')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8, handletextpad=0.3)
    title(fig, f'By machine type, the largest shift of the fresh mean is {sg(worst[0])} ({worst[1]}, {worst[2]})',
          'Single networks under C1ᵀ. Zero is the phase-1 mean for that machine type and test set.')
    fig.tight_layout()
    save(fig, 'fig4_machines.png')


if __name__ == '__main__':
    for f in (fig1_single, fig2_ensembles, fig3_f4b, fig4_machines):
        f()
