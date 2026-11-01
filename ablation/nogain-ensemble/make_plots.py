"""Figures for ablation/nogain-ensemble. Reads scores/ only; skips a panel or figure whose inputs are missing.

Run with /home/f74134118/anaconda3/bin/python (matplotlib lives in base).
Colour = member type, matching ../aug-ops and ../mixed-ensemble: nogain green, aug blue, mixed types orange,
references achromatic. Green is below 3:1 on the surface, so every green mark carries a direct label.
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
SYSTEMS = ['G8', 'GA16', 'F20g', 'A8', 'F20', 'relu8']
KIND = {'G8': GREEN, 'GA16': ORANGE, 'F20g': ORANGE, 'A8': BLUE, 'F20': INK_2, 'relu8': MUTED,
        'hom4_nogain': GREEN, 'hom4_aug': BLUE}
LABEL = {'G8': 'G8  nogain ×8', 'GA16': 'GA16  nogain + aug', 'F20g': 'F20g  plain/gate/eca + nogain',
         'A8': 'A8  aug ×8 (bar)', 'F20': 'F20  plain/gate/eca + aug', 'relu8': 'relu8  old ×8'}

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
        out.setdefault(x['family'], {})[(x['scope'], x['metric'])] = (
            float(x['mean']), float(x['p05']), float(x['p95']), float(x['frac_pos']) if x['frac_pos'] else None)
    return out


def boot(ds):
    p = os.path.join(SCORES, f'bootstrap_{ds}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(PLOTS, name), dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote plots/' + name)


def present(fn):
    return [ds for ds in ('dev', 'eval') if fn(ds)]


# ------------------------------------------------- fig1: every system against A8, dev and eval
def fig1_ladder():
    dss = present(systems)
    if not dss:
        print('skip fig1_ladder: no systems_*.csv')
        return
    s = {ds: systems(ds) for ds in dss}
    b = {ds: boot(ds) for ds in dss}
    fig, axes = plt.subplots(1, len(dss), figsize=(5.4 * len(dss), 3.7), sharey=True, squeeze=False)
    y = np.arange(len(SYSTEMS))[::-1]
    for ax, ds in zip(axes[0], dss):
        bar = s[ds]['A8']['mean_AUC_pAUC']
        ax.axvline(bar, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        for yi, n in zip(y, SYSTEMS):
            v = s[ds][n]['mean_AUC_pAUC']
            ax.plot([bar, v], [yi, yi], color=KIND[n], lw=1.2, alpha=0.5, zorder=2)
            ax.scatter([v], [yi], s=50, color=KIND[n], edgecolor=SURFACE, lw=1.2, zorder=3)
            txt = f'{v:.2f}' + (f'  ({v - bar:+.2f})' if n != 'A8' else '')
            if b[ds] and n in b[ds]:
                lo, hi = b[ds][n]['mean_AUC_pAUC'][1:]
                txt += f' [{lo:+.2f}, {hi:+.2f}]'
            ax.annotate(txt, (max(v, bar), yi), xytext=(8, 0), textcoords='offset points', va='center',
                        fontsize=7.5, color=INK)
        lo = min(s[ds][n]['mean_AUC_pAUC'] for n in SYSTEMS)
        hi = max(s[ds][n]['mean_AUC_pAUC'] for n in SYSTEMS)
        ax.set_xlim(lo - 0.15, hi + 0.85)
        ax.grid(axis='x')
        ax.set_title(f'{ds} set: mean(AUC, pAUC), E2′', loc='left', fontsize=9.5)
    axes[0][0].set_yticks(y)
    axes[0][0].set_yticklabels([LABEL[n] for n in SYSTEMS])
    parts = [f'{ds} {s[ds]["G8"]["mean_AUC_pAUC"] - s[ds]["A8"]["mean_AUC_pAUC"]:+.2f}' for ds in dss]
    if 'eval' in dss and b['eval']:
        lo, hi = b['eval']['G8']['mean_AUC_pAUC'][1:]
        parts[-1] += f' [{lo:+.2f}, {hi:+.2f}]'
    fig.suptitle('G8 − A8 on mean(AUC, pAUC): ' + ', '.join(parts), x=0.02, ha='left', fontsize=10.5)
    fig.text(0.02, -0.03, 'Dashed = E2′[A8], the bar. Brackets: paired clip-bootstrap 95% vs A8 (2000 resamples). '
             'Green = nogain only, blue = aug only, orange = mixed, grey = references.',
             fontsize=7.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig1_ladder.png')


# ------------------------------------------------- fig2: 4-member families, seed-paired
def fig2_paired():
    dss = present(families)
    if not dss:
        print('skip fig2_paired: no combos_*.csv')
        return
    f = {ds: families(ds) for ds in dss}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    for j, ds in enumerate(dss):
        for k, fam in enumerate(('hom4_aug', 'hom4_nogain')):
            m, lo, hi, _ = f[ds][fam][('overall', 'mean_AUC_pAUC')]
            x = j * 1.2 + k * 0.45
            ax.plot([x, x], [lo, hi], color=KIND[fam], lw=1.4, zorder=2)
            ax.scatter([x], [m], s=46, color=KIND[fam], edgecolor=SURFACE, lw=1.2, zorder=3)
            ax.annotate(f'{fam[5:]} {m:.2f}', (x, hi), xytext=(0, 4), textcoords='offset points', ha='center',
                        fontsize=7.5, color=INK)
    ax.set_xticks([j * 1.2 + 0.225 for j in range(len(dss))])
    ax.set_xticklabels(dss)
    ax.set_xlim(-0.4, len(dss) * 1.2 - 0.3)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.1 * (ymax - ymin))
    ax.set_ylabel('mean(AUC, pAUC), E2′ (%)')
    ax.grid(axis='y')
    ax.set_title('4-member ensembles, all 70 seed subsets', loc='left', fontsize=9.5)

    ax = axes[1]
    keys = [('overall', 'AUC', 'AUC'), ('overall', 'pAUC', 'pAUC'), ('overall', 'mAUC', 'mAUC'),
            ('overall', 'mean_AUC_pAUC', 'mean')]
    w = 0.36
    for j, ds in enumerate(dss):
        for i, (scope, key, lab) in enumerate(keys):
            m, lo, hi, pos = f[ds]['paired4'][(scope, key)]
            x = i + (j - (len(dss) - 1) / 2) * w
            c = INK_2 if ds == 'dev' else INK
            ax.plot([x, x], [lo, hi], color=c, lw=1.4, alpha=0.6 if ds == 'dev' else 1, zorder=2)
            ax.scatter([x], [m], s=40, color=SURFACE if ds == 'dev' else c, edgecolor=c, lw=1.3, zorder=3)
            ax.annotate(f'{m:+.2f}\n{100 * pos:.0f}%>0', (x, hi), xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=6.8, color=INK)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k[2] for k in keys])
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.25 * (ymax - ymin))
    ax.grid(axis='y')
    ax.set_ylabel('nogain − aug, same 4 seeds (points)')
    ax.set_title('paired Δ: hollow = dev, filled = eval', loc='left', fontsize=9.5)
    parts = [f'{ds} {f[ds]["paired4"][("overall", "mean_AUC_pAUC")][0]:+.2f} '
             f'({100 * f[ds]["paired4"][("overall", "mean_AUC_pAUC")][3]:.0f}% of subsets > 0)' for ds in dss]
    fig.suptitle('Seed-paired 4-member ensembles, nogain − aug on mean(AUC, pAUC): ' + ', '.join(parts),
                 x=0.02, ha='left', fontsize=10.5)
    fig.text(0.02, -0.04, 'Dot = mean over the 70 four-seed subsets of 2040–2047; line = 5–95% across subsets '
             '(model-choice noise, not test-set noise).', fontsize=7.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig2_paired.png')


# ------------------------------------------------- fig3: per machine, against A8
def fig3_machines():
    dss = present(systems)
    if not dss:
        print('skip fig3_machines: no systems_*.csv')
        return
    s = {ds: systems(ds) for ds in dss}
    show = ['G8', 'GA16', 'F20g', 'F20']
    fig, axes = plt.subplots(1, len(dss), figsize=(5.4 * len(dss), 3.6), sharey=True, squeeze=False)
    w = 0.19
    for ax, ds in zip(axes[0], dss):
        for i, n in enumerate(show):
            d = [s[ds][n][m] - s[ds]['A8'][m] for m in MACH]
            xs = np.arange(len(MACH)) + (i - (len(show) - 1) / 2) * w
            hollow = n == 'GA16'   # GA16 and F20g are both mixed (orange): GA16 drawn hatched
            ax.bar(xs, d, width=w - 0.02, color=SURFACE if hollow else KIND[n], edgecolor=KIND[n],
                   hatch='////' if hollow else None, lw=1.0, label=n, zorder=2)
            if n == 'G8':
                for x, v in zip(xs, d):
                    ax.annotate(f'{v:+.1f}', (x, v), xytext=(0, 2 if v >= 0 else -9), textcoords='offset points',
                                ha='center', fontsize=6.5, color=INK)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_xticks(range(len(MACH)))
        ax.set_xticklabels(MACH, rotation=20)
        ax.grid(axis='y')
        ax.set_title(f'{ds} set: machine AUC − A8', loc='left', fontsize=9.5)
    axes[0][0].set_ylabel('Δ AUC vs E2′[A8] (points)')
    axes[0][-1].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8)
    ds = dss[-1]
    tc = s[ds]['G8']['ToyCar'] - s[ds]['A8']['ToyCar']
    tv = s[ds]['G8']['ToyConveyor'] - s[ds]['A8']['ToyConveyor']
    fig.suptitle(f'Per machine vs A8 ({ds}): G8 ToyCar {tc:+.2f}, ToyConveyor {tv:+.2f}; '
                 f'F20g ToyCar {s[ds]["F20g"]["ToyCar"] - s[ds]["A8"]["ToyCar"]:+.2f}',
                 x=0.02, ha='left', fontsize=10.5)
    fig.tight_layout()
    save(fig, 'fig3_machines.png')


if __name__ == '__main__':
    fig1_ladder()
    fig2_paired()
    fig3_machines()
