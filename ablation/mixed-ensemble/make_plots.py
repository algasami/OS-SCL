"""Figures for ablation/mixed-ensemble. Reads scores/ only; skips a figure whose inputs are missing.

Run with /home/f74134118/anaconda3/bin/python (matplotlib lives in base).
Colour = what kind of ensemble: `aug` only (blue, aug's colour in ../lit-arms), mixed types
(orange), one plain type (achromatic: fast-recipe plain dark grey, old recipe light grey).
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
BLUE, ORANGE = '#2a78d6', '#eb6834'
PAPER = {'dev': 95.71, 'eval_mean': 93.22}
KIND = {'P4': INK_2, 'A8': BLUE, 'PA12': ORANGE, 'F20': ORANGE, 'ALL32': ORANGE, 'N4': MUTED, 'relu8': MUTED,
        'hom4_plain': INK_2, 'hom4_aug': BLUE, 'hom4_old': MUTED, 'mix4_PAGE': ORANGE, 'mix4_PGE+P': ORANGE,
        'hom8_aug': BLUE, 'hom8_old': MUTED, 'mix8_PAGE': ORANGE}
LABEL = {'P4': 'P4  plain ×4', 'A8': 'A8  aug ×8', 'PA12': 'PA12  plain+aug', 'F20': 'F20  all fast-recipe',
         'ALL32': 'ALL32  everything', 'N4': 'N4  old ×4 (adopted)', 'relu8': 'relu8  old ×8 (bar)'}
SYSTEMS = ['P4', 'A8', 'PA12', 'F20', 'ALL32', 'N4', 'relu8']
FAMILIES = [('hom4_plain', 'plain'), ('hom4_old', 'old'), ('hom4_aug', 'aug'), ('mix4_PGE+P', 'mix, no aug'),
            ('mix4_PAGE', 'mix P+A+G+E'), ('hom8_old', 'old'), ('hom8_aug', 'aug'), ('mix8_PAGE', 'mix P+A+G+E')]
WEAK = ['ToyCar', 'ToyConveyor', 'pump']

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
        return list(csv.DictReader(f))


def systems(ds):
    r = read(f'systems_{ds}.csv')
    if not r:
        return None
    out = {}
    for x in r:
        d = out.setdefault(x['system'], {})
        if x['scope'] == 'overall':
            d.update({k: float(x[k]) for k in ('AUC', 'pAUC', 'mAUC', 'mean_AUC_pAUC')})
        else:
            d[x['scope']] = float(x['AUC'])
    return out


def families(ds):
    r = read(f'combos_{ds}.csv')
    if not r:
        return None
    out = {}
    for x in r:
        out.setdefault(x['family'], {})[(x['scope'], x['metric'])] = (float(x['mean']), float(x['p05']),
                                                                      float(x['p95']))
    return out


def save(fig, name):
    fig.savefig(os.path.join(PLOTS, name), dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote plots/' + name)


# ------------------------------------------------- fig1: the dev ranking against the eval ranking
def fig1_ladder():
    s = {ds: systems(ds) for ds in ('dev', 'eval')}
    if not all(s.values()):
        print('skip fig1_ladder: needs systems_dev.csv and systems_eval.csv')
        return
    boot = json.load(open(os.path.join(SCORES, 'bootstrap_eval.json')))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    y = np.arange(len(SYSTEMS))[::-1]
    for ax, ds in zip(axes, ('dev', 'eval')):
        bar = s[ds]['relu8']['mean_AUC_pAUC']
        ax.axvline(bar, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        for yi, n in zip(y, SYSTEMS):
            v = s[ds][n]['mean_AUC_pAUC']
            ax.plot([bar, v], [yi, yi], color=KIND[n], lw=1.2, alpha=0.5, zorder=2)
            ax.scatter([v], [yi], s=50, color=KIND[n], edgecolor=SURFACE, lw=1.2, zorder=3)
            txt = f'{v:.2f}' + (f'  ({v - bar:+.2f})' if n != 'relu8' else '')
            if ds == 'eval' and n in boot:
                lo, hi = boot[n]['mean_AUC_pAUC'][1:]
                txt += f' [{lo:+.2f}, {hi:+.2f}]'
            ax.annotate(txt, (max(v, bar), yi), xytext=(8, 0), textcoords='offset points', va='center',
                        fontsize=7.5, color=INK)
        ax.set_title(f'{ds} set: mean(AUC, pAUC), E2′', loc='left', fontsize=9.5)
        ax.grid(axis='x')
        lo = min(s[ds][n]['mean_AUC_pAUC'] for n in SYSTEMS)
        hi = max(s[ds][n]['mean_AUC_pAUC'] for n in SYSTEMS)
        ax.set_xlim(lo - 0.15, hi + 0.75)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABEL[n] for n in SYSTEMS])
    a8 = s['eval']['A8']['mean_AUC_pAUC'] - s['eval']['relu8']['mean_AUC_pAUC']
    best = max(SYSTEMS, key=lambda n: s['eval'][n]['mean_AUC_pAUC'])
    fig.suptitle(f'The dev pick A8 leads relu8 by {s["dev"]["A8"]["mean_AUC_pAUC"] - s["dev"]["relu8"]["mean_AUC_pAUC"]:+.2f} '
                 f'on dev and {a8:+.2f} on eval; best on eval: {best}', x=0.02, ha='left', fontsize=10.5)
    fig.text(0.02, -0.03, 'Dashed = E2′[relu8], the bar. Eval brackets: paired clip-bootstrap 95% vs relu8. '
             'Blue = aug only, orange = mixed types, grey = one plain type (dark: fast recipe, light: old).',
             fontsize=7.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig1_ladder.png')


# ------------------------------------------------- fig2: size-matched families
def fig2_families():
    f = {ds: families(ds) for ds in ('dev', 'eval')}
    if not all(f.values()):
        print('skip fig2_families: needs combos_dev.csv and combos_eval.csv')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    x = np.array([0, 1, 2, 3, 4, 5.6, 6.6, 7.6])
    for ax, ds in zip(axes, ('dev', 'eval')):
        for xi, (fam, lab) in zip(x, FAMILIES):
            m, lo, hi = f[ds][fam][('overall', 'mean_AUC_pAUC')]
            ax.plot([xi, xi], [lo, hi], color=KIND[fam], lw=1.4, zorder=2)
            ax.scatter([xi], [m], s=46, color=KIND[fam], edgecolor=SURFACE, lw=1.2, zorder=3)
            ax.annotate(f'{m:.2f}', (xi, hi), xytext=(0, 4), textcoords='offset points', ha='center',
                        fontsize=7.5, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in FAMILIES], rotation=35, ha='right')
        ax.axvline(4.8, color=GRID, lw=0.8)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.12 * (ymax - ymin))
        top = ax.get_ylim()[1]
        ax.text(2, top, '4 members', ha='center', va='top', fontsize=8, color=INK_2)
        ax.text(6.6, top, '8 members', ha='center', va='top', fontsize=8, color=INK_2)
        ax.grid(axis='y')
        ax.set_title(f'{ds} set', loc='left', fontsize=9.5)
    axes[0].set_ylabel('mean(AUC, pAUC), E2′ (%)')
    d4 = f['eval']['mix4_PAGE'][('overall', 'mean_AUC_pAUC')][0] - max(
        f['eval']['hom4_aug'][('overall', 'mean_AUC_pAUC')][0], f['eval']['hom4_plain'][('overall', 'mean_AUC_pAUC')][0])
    d8 = f['eval']['mix8_PAGE'][('overall', 'mean_AUC_pAUC')][0] - f['eval']['hom8_aug'][('overall', 'mean_AUC_pAUC')][0]
    fig.suptitle(f'Same size, mixed vs one type: on eval, mixing is {d4:+.2f} at 4 members and {d8:+.2f} at 8 '
                 f'(D1 needs +0.20)', x=0.02, ha='left', fontsize=10.5)
    fig.text(0.02, -0.08, 'Dot = mean over member combinations; line = 5–95% across combinations (which models '
             'were picked), not test-set noise.', fontsize=7.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig2_families.png')


# ------------------------------------------------- fig3: the weak categories
def fig3_weak():
    s = {ds: systems(ds) for ds in ('dev', 'eval')}
    if not all(s.values()):
        print('skip fig3_weak: needs systems_dev.csv and systems_eval.csv')
        return
    show = ['P4', 'A8', 'F20', 'ALL32', 'relu8']
    fig, axes = plt.subplots(1, len(WEAK), figsize=(10.5, 3.6))
    w = 0.16
    for ax, m in zip(axes, WEAK):
        for i, n in enumerate(show):
            vals = [s['dev'][n][m], s['eval'][n][m]]
            xs = np.arange(2) + (i - (len(show) - 1) / 2) * w
            # F20 and ALL32 are both mixed (orange): ALL32 is drawn hollow-hatched to tell them apart
            hollow = n == 'ALL32'
            ax.bar(xs, vals, width=w - 0.02, color=SURFACE if hollow else KIND[n], edgecolor=KIND[n],
                   hatch='////' if hollow else None, lw=1.0, alpha=0.55 if n == 'relu8' else 1.0,
                   label=LABEL[n].split('  ')[0], zorder=2)
        allv = [s[ds][n][m] for ds in ('dev', 'eval') for n in show]
        ax.set_ylim(min(allv) - 1.0, max(allv) + 0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['dev', 'eval'])
        ax.grid(axis='y')
        ax.set_title(m, loc='left', fontsize=9.5)
    axes[0].set_ylabel('machine AUC, E2′ (%)')
    axes[-1].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.suptitle('aug-only A8 is best on ToyConveyor but worst on eval ToyCar; mixing repairs ToyCar; '
                 'eval pump barely moves', x=0.02, ha='left', fontsize=10.5)
    fig.tight_layout()
    save(fig, 'fig3_weak.png')


if __name__ == '__main__':
    fig1_ladder()
    fig2_families()
    fig3_weak()
