"""Figures for synthesis/grand-ensemble. Reads scores/ only; skips a figure whose inputs are missing.

Run with /home/f74134118/anaconda3/bin/python (matplotlib lives in base).
Colour = whether a candidate contains `sb_spec` (orange) or not (blue); the pick is labelled in bold;
references (A8, F1k) are achromatic. Dev vs eval series: blue = dev, orange = eval.
"""
import csv
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(HERE, 'scores')
PLOTS = os.path.join(HERE, 'plots')
os.makedirs(PLOTS, exist_ok=True)

SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8983', '#e4e3df'
BLUE, ORANGE, GREEN = '#2a78d6', '#eb6834', '#1baf7a'
MACH = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
ORDER = ['G8', 'GA16', 'GS12', 'GT12', 'GST16', 'GAST24', 'NG18', 'FAST42', 'ALL50', 'GREEDY']
HAS_S = {'GS12', 'GST16', 'GAST24', 'NG18', 'FAST42', 'ALL50', 'GREEDY'}
TYPE_NAME = {'G': 'ao_nogain', 'A': 'la_aug', 'P': 'fr_f1', 'Ga': 'la_gate', 'E': 'la_eca', 'S': 'sb_spec',
             'T': 'tp', 'PM': 'pm', 'TS': 'soup', 'O': 'old relu', 'B': 'BEATs*'}

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


def jload(name):
    p = os.path.join(SCORES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def systems(name):
    """{system: {'Amean':..., 'AUC':..., machine: AUC, machine+'_Amean': ..., (machine, id): AUC}}"""
    r = read(name)
    if not r:
        return None
    out = {}
    for x in r:
        d = out.setdefault(x['system'], {})
        if x['scope'] == 'overall':
            d.update({k: float(x[k]) for k in ('AUC', 'pAUC', 'mAUC', 'Amean')})
        elif x['id']:
            d[(x['scope'], x['id'])] = float(x['AUC'])
        else:
            d[x['scope']] = float(x['AUC'])
            d[x['scope'] + '_Amean'] = float(x['Amean'])
    return out


def save(fig, name):
    fig.savefig(os.path.join(PLOTS, name), dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote plots/' + name)


# ------------------------------------------------- fig1: every candidate, dev and eval
def fig1_candidates():
    sd, se = systems('systems_dev.csv'), systems('systems_eval.csv')
    sel = jload('selection_dev.json')
    if not sd or not sel:
        print('skip fig1_candidates: no systems_dev.csv / selection_dev.json')
        return
    pick = sel['x0_pick']
    be = jload('bootstrap_eval.json')
    panels = [('dev', sd)] + ([('eval', se)] if se else [])
    rows = ORDER + ['A8']
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.2), sharey=True, squeeze=False)
    for ax, (ds, s) in zip(axes[0], panels):
        g8, a8 = s['G8']['Amean'], s['A8']['Amean']
        ax.axvline(a8, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.axvline(g8, color=INK_2, lw=0.8, ls=':', zorder=1)
        for yi, n in zip(y, rows):
            v = s[n]['Amean']
            c = MUTED if n == 'A8' else (ORANGE if n in HAS_S else BLUE)
            ax.plot([a8, v], [yi, yi], color=c, lw=1.2, alpha=0.45, zorder=2)
            ax.scatter([v], [yi], s=52, color=c, edgecolor=SURFACE, lw=1.4, zorder=3)
            txt = f'{v:.2f}'
            if n != 'A8':
                txt += f'  ({v - a8:+.2f})'
            if ds == 'eval' and be and f'{n}-A8' in be:
                lo, hi = be[f'{n}-A8']['mean_AUC_pAUC'][1:]
                txt += f' [{lo:+.2f}, {hi:+.2f}]'
            ax.annotate(txt, (max(v, a8), yi), xytext=(8, 0), textcoords='offset points', va='center',
                        fontsize=7.5, color=INK, fontweight='bold' if n == pick else 'normal')
        vals = [s[n]['Amean'] for n in rows]
        ax.set_xlim(min(vals) - 0.1, max(vals) + (0.95 if ds == 'eval' else 0.45))
        ax.grid(axis='x')
        ax.set_title(f'{ds} Amean, Σ S1′ (Δ vs A8' + (', paired 95%)' if ds == 'eval' else ')'),
                     loc='left', fontsize=9.5)
    axes[0][0].set_yticks(y)
    axes[0][0].set_yticklabels([f'{n}  ← pick' if n == pick else ('A8 (bar)' if n == 'A8' else n) for n in rows])
    for t in axes[0][0].get_yticklabels():
        if t.get_text().startswith(pick + ' '):
            t.set_fontweight('bold')
    dec = jload('decisions.json')
    title = f'Dev pick {pick}: +{sd[pick]["Amean"] - sd["G8"]["Amean"]:.2f} over G8 on dev'
    if se and dec:
        d1 = dec['D1']
        title += (f'; on eval {d1["eval_delta_vs_A8"]:+.2f} vs A8 [{d1["ci95_vs_A8"][0]:+.2f}, '
                  f'{d1["ci95_vs_A8"][1]:+.2f}] — ' + ('replaces A8' if d1['replaces_A8'] else 'tie, A8 stays'))
    fig.suptitle(title, x=0.02, ha='left', fontsize=10.5)
    fig.text(0.02, -0.03, 'Orange = contains sb_spec, blue = does not, grey = E2′[A8] (dashed line). Dotted line = G8. '
             'All candidates were fixed before dev was scored; the pick was made on dev only.', fontsize=7.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig1_candidates.png')


# ------------------------------------------------- fig2: between-type diversity
def fig2_diversity():
    sets = [(ds, read(f'type_corr_{ds}.csv')) for ds in ('dev', 'eval')]
    sets = [(ds, r) for ds, r in sets if r]
    if not sets:
        print('skip fig2_diversity: no type_corr_*.csv')
        return
    cmap = LinearSegmentedColormap.from_list('blue1', ['#f2f6fc', '#9cc1ec', BLUE, '#123c70'])
    fig, axes = plt.subplots(1, len(sets), figsize=(5.0 * len(sets), 4.4), squeeze=False)
    for ax, (ds, r) in zip(axes[0], sets):
        types = [x['type'] for x in r]
        C = np.array([[float(x[t]) for t in types] for x in r])
        lo = np.floor(C[~np.eye(len(types), dtype=bool)].min() * 20) / 20
        im = ax.imshow(C, cmap=cmap, vmin=lo, vmax=1.0)
        for i in range(len(types)):
            for j in range(len(types)):
                if i != j:
                    v = C[i, j]
                    ax.text(j, i, f'{v:.2f}'[1:] if v < 1 else '1', ha='center', va='center', fontsize=6.3,
                            color=SURFACE if (v - lo) / (1 - lo) > 0.55 else INK)
        lab = [TYPE_NAME.get(t, t) for t in types]
        ax.set_xticks(range(len(types)))
        ax.set_xticklabels(lab, rotation=55, ha='right', fontsize=7.5)
        ax.set_yticks(range(len(types)))
        ax.set_yticklabels(lab, fontsize=7.5)
        ax.spines[:].set_visible(False)
        ax.set_title(f'{ds}: score correlation between type means', loc='left', fontsize=9.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7)
        if 'S' in types and 'G' in types:
            ix = types.index('G')
            near = sorted(((C[ix, j], types[j]) for j in range(len(types)) if j != ix and types[j] not in ('B',)),
                          reverse=True)
            sub = ', '.join(f'{TYPE_NAME[t]} {v:.2f}' for v, t in near[:4])
    fig.suptitle('sb_spec is the least correlated strong type (≈0.90 with ao_nogain); tp, soup and la_aug are near-copies',
                 x=0.02, ha='left', fontsize=10.5)
    fig.text(0.02, -0.04, 'Pearson correlation of type-mean S1′ (BEATs*: zK′, external pretraining) within each machine ID, '
             f'averaged over IDs. Closest to ao_nogain on {sets[0][0]}: {sub}.', fontsize=7.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig2_diversity.png')


# ------------------------------------------------- fig3: greedy trace and selection optimism
def fig3_selection():
    sel = jload('selection_dev.json')
    if not sel:
        print('skip fig3_selection: no selection_dev.json')
        return
    gt = read('greedy_trace.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    ax = axes[0]
    tr = [t for t in sel['greedy_trace']]
    xs = [t['step'] for t in tr]
    ax.plot([t['step'] for t in tr if t['taken']], [t['amean'] for t in tr if t['taken']], color=BLUE, lw=2,
            marker='o', ms=7, mec=SURFACE, mew=1.4, label='dev (selection)')
    nt = [t for t in tr if not t['taken']]
    if nt:
        last = [t for t in tr if t['taken']][-1]
        ax.plot([last['step'], nt[0]['step']], [last['amean'], nt[0]['amean']], color=BLUE, lw=1.2, ls=':')
        ax.scatter([nt[0]['step']], [nt[0]['amean']], s=50, facecolor=SURFACE, edgecolor=BLUE, lw=1.4, zorder=3)
    cum = []
    for t in tr:
        ax.annotate(('+' if t['taken'] else 'stop: +') + TYPE_NAME[t['type']], (t['step'], t['amean']),
                    xytext=(0, 8) if t['step'] > 1 else (6, 10), textcoords='offset points',
                    ha='center' if t['step'] > 1 else 'left', fontsize=7.5, color=INK)
    if gt:
        ax.plot([int(r['step']) for r in gt], [float(r['eval_Amean']) for r in gt], color=ORANGE, lw=2, marker='o',
                ms=7, mec=SURFACE, mew=1.4, label='eval (descriptive)')
    ax.set_xticks(xs)
    ax.set_xlabel('greedy step (one type-mean block added)')
    ax.set_ylabel('Amean (%)')
    ax.grid(axis='y')
    ax.legend(loc='lower right', fontsize=7.5)
    ax.set_title('GREEDY type-block selection', loc='left', fontsize=9.5)

    ax = axes[1]
    o1 = sel['O1']['splits']
    for name, c in (('GS12', ORANGE), ('GREEDY', BLUE)):
        pts = [(r['delta_A'], r['delta_B']) for r in o1 if r['pick'] == name]
        if pts:
            a, b = np.array(pts).T
            ax.scatter(a, b, s=40, color=c, edgecolor=SURFACE, lw=1.2, zorder=3,
                       label=f'half-A pick {name} ({len(pts)}): {a.mean():+.2f} → {b.mean():+.2f}')
    other = [(r['delta_A'], r['delta_B']) for r in o1 if r['pick'] not in ('GS12', 'GREEDY')]
    if other:
        a, b = np.array(other).T
        ax.scatter(a, b, s=40, color=MUTED, edgecolor=SURFACE, lw=1.2, label=f'other picks ({len(other)})')
    lim = [0, max(max(r['delta_A'], r['delta_B']) for r in o1) + 0.05]
    ax.plot(lim, lim, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('Δ Amean pick − G8 on half A (selection)')
    ax.set_ylabel('Δ on half B (held out)')
    ax.grid()
    ax.legend(loc='upper left', fontsize=7.2)
    ax.set_title(f'O1: 20 clip split-halves, optimism {sel["O1"]["optimism"]:+.2f}', loc='left', fontsize=9.5)
    fig.suptitle(f'On held-out dev halves GREEDY\'s lead over G8 shrinks, GS12\'s holds '
                 f'(pick\'s honest margin over G8 {sel["O1"]["honest_margin_vs_G8"]:+.2f})', x=0.02, ha='left', fontsize=10.5)
    fig.text(0.02, -0.03, 'Dashed diagonal = no shrinkage. Points below it lost part of their selection-half lead.',
             fontsize=7.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig3_selection.png')


# ------------------------------------------------- fig4: per machine, dev vs eval
def fig4_machines():
    sd, se = systems('systems_dev.csv'), systems('systems_eval.csv')
    sel = jload('selection_dev.json')
    if not sd or not sel:
        print('skip fig4_machines: inputs missing')
        return
    pick = sel['x0_pick']
    xd, xe = systems('x1_dev.csv'), systems('x1_eval.csv')
    x1 = sel['x1_pick']
    n_pan = 2 if (xd and xe) else 1
    fig, axes = plt.subplots(1, n_pan, figsize=(5.4 * n_pan, 3.7), squeeze=False)
    x = np.arange(len(MACH))
    w = 0.36
    series = [('dev', sd, BLUE)] + ([('eval', se, ORANGE)] if se else [])
    ax = axes[0][0]
    for k, (ds, s, c) in enumerate(series):
        d = [s[pick][m + '_Amean'] - s['G8'][m + '_Amean'] for m in MACH]
        xx = x + (k - (len(series) - 1) / 2) * w
        ax.bar(xx, d, width=w - 0.04, color=c, label=ds, zorder=2)
        for xi, v in zip(xx, d):
            ax.annotate(f'{v:+.2f}', (xi, v), xytext=(0, 3 if v >= 0 else -9), textcoords='offset points',
                        ha='center', fontsize=6.8, color=INK)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(MACH, fontsize=8)
    ax.set_ylabel(f'Δ Amean per machine, {pick} − G8')
    ax.grid(axis='y')
    ax.legend(fontsize=7.5, loc='upper right')
    ax.set_title(f'from scratch: {pick} − G8', loc='left', fontsize=9.5)
    if n_pan == 2:
        ax = axes[0][1]
        for k, (ds, s, c) in enumerate((('dev', xd, BLUE), ('eval', xe, ORANGE))):
            d = [s[x1][m + '_Amean'] - s['F1k'][m + '_Amean'] for m in MACH]
            xx = x + (k - 0.5) * w
            ax.bar(xx, d, width=w - 0.04, color=c, label=ds, zorder=2)
            for xi, v in zip(xx, d):
                ax.annotate(f'{v:+.2f}', (xi, v), xytext=(0, 3 if v >= 0 else -9), textcoords='offset points',
                            ha='center', fontsize=6.8, color=INK)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(MACH, fontsize=8)
        ax.set_ylabel(f'Δ Amean per machine, X1 − F1k')
        ax.grid(axis='y')
        ax.legend(fontsize=7.5, loc='upper right')
        ax.set_title('WITH EXTERNAL PRETRAINING: X1 − F1k\n(D2 passes, thinly; mostly the GS12-vs-A8 swap; pending fresh seeds)',
                     loc='left', fontsize=9)
    sub = ''
    if se:
        dp = se[pick][('ToyCar', 'id_05')] - se['G8'][('ToyCar', 'id_05')]
        sub = f' Eval ToyCar id_05 AUC {se[pick][("ToyCar", "id_05")]:.2f} ({dp:+.2f} vs G8).'
    ttl = f'Where {pick} gains and loses vs G8, dev vs eval'
    if se:
        ttl = (f'{pick}\'s fan and pump gains hold on eval; its dev ToyConveyor gain reverses. '
               f'X1 − F1k gains fan/pump/ToyConveyor, not ToyCar')
    fig.suptitle(ttl, x=0.02, ha='left', fontsize=10.5)
    fig.text(0.02, -0.03, 'Amean = mean(AUC, pAUC) per machine.' + sub, fontsize=7.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig4_machines.png')


if __name__ == '__main__':
    fig1_candidates()
    fig2_diversity()
    fig3_selection()
    fig4_machines()
