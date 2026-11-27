"""Figures for phase3/id-gate stage 1. Reads scores/ only; a figure whose inputs are missing is skipped.

Run with /home/f74134118/anaconda3/bin/python (matplotlib lives in base):
    make_plots.py

Colour: dev = blue, eval = orange, as in phase3/beats-router. Machine type is never a colour (six types,
three validated slots): it is a marker shape plus a direct text label ("fan-00").
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
MARK = dict(zip(MACH, ['o', 's', '^', 'D', 'v', 'P']))
SYS = [('GS12f_kNN', 'GS12f-kNN\n(= from-scratch block)'), ('X1f_kNN', 'X1f-kNN'), ('beats_block', 'BEATs block')]
SAT = 0.999

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


def sg(x, n=2):
    return f'{x:+.{n}f}'.replace('-', '−')


def tag(r):
    return f'{r["machine"]}-{r["id"][3:]}'


# ------------------------------------------------------------------ fig 1: pi vs Delta on dev IDs
ROUNDOFF = 1e-12      # |pi| below this is float roundoff of an AUC of exactly 1 (stage1.json roundoff_note)


def fig1():
    rows, J = read(os.path.join(SRC, 'proxy_ids.csv')), read_json(os.path.join(SRC, 'stage1.json'))
    if not rows or not J:
        print('skip fig1: scores/proxy_ids.csv or stage1.json missing')
        return
    dev = [r for r in rows if r['set'] == 'dev']
    x = np.array([float(r['pi']) for r in dev])
    y = np.array([float(r['delta']) for r in dev])
    main, d0 = J['rho_pi_delta'], J['D0']
    degenerate = np.abs(x).max() < ROUNDOFF
    if degenerate:
        x = np.zeros_like(x)
    max_pi = J['descriptive'].get('max_abs_pi', {}).get('dev', float(np.abs(x).max()))
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    for mm in MACH:
        k = [i for i, r in enumerate(dev) if r['machine'] == mm]
        ax.scatter(x[k], y[k], marker=MARK[mm], s=46, facecolor=INK, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    # one label per distinct point (IDs on the same point share it), stacked per side so none collide
    groups = {}
    for i, r in enumerate(dev):
        groups.setdefault((x[i], round(y[i], 4)), []).append(tag(r))
    labs = []
    for (gx, gy), names in sorted(groups.items(), key=lambda kv: kv[0][1]):
        if len(names) >= 4:
            txt = '\n'.join([f'{len(names)} IDs at Δ = {gy:g}:'] + [', '.join(names[i:i + 3])
                                                                   for i in range(0, len(names), 3)])
        else:
            txt = ', '.join(names)
        labs.append([gx, gy, txt, txt.count('\n') + 1])
    span = np.ptp(y) or 1.0
    unit = 0.042 * span                                     # one text line in data units
    sides = {1: [], -1: []}
    big = [l for l in labs if l[3] > 1]
    for l in big:
        sides[-1].append(l)
    k = 0
    for l in labs:
        if l in big:
            continue
        sides[1 if k % 2 == 0 else -1].append(l)
        k += 1
    xoff = 0.12 * (ax.get_xlim()[1] - ax.get_xlim()[0]) if not degenerate else 0.12
    for side, ls in sides.items():
        ls.sort(key=lambda l: l[1])
        pos = [l[1] for l in ls]
        for _ in range(200):                                # push apart until every pair clears its half-heights
            moved = False
            for i in range(1, len(ls)):
                need = (ls[i - 1][3] + ls[i][3]) / 2 * unit + 0.35 * unit
                if pos[i] - pos[i - 1] < need:
                    d = (need - (pos[i] - pos[i - 1])) / 2
                    pos[i - 1] -= d
                    pos[i] += d
                    moved = True
            if not moved:
                break
        for l, py in zip(ls, pos):
            ax.annotate(l[2], (l[0], l[1]), xytext=(l[0] + side * xoff, py), textcoords='data',
                        ha='left' if side > 0 else 'right', va='center', fontsize=7.2, color=INK_2, linespacing=1.1,
                        arrowprops=dict(arrowstyle='-', color=GRID, lw=0.7, shrinkA=2, shrinkB=4))
    ax.grid(axis='y', zorder=0)
    ax.set_ylabel('Δ_i  =  dev Amean_i(X1f) − Amean_i(GS12f)   (points)')
    hs = [plt.Line2D([], [], marker=MARK[mm], ls='', color=INK, markersize=6, label=mm) for mm in MACH]
    ax.legend(handles=hs, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8, handletextpad=0.3)
    sat = d0['a']
    if degenerate:
        ax.set_xlim(-1, 1)
        ax.set_xticks([0])
        ax.set_xlabel(f'π_i  =  proxyAUC_i(X1f-kNN) − proxyAUC_i(GS12f-kNN)  '
                      f'(0 on every ID up to roundoff, |π| ≤ {max_pi:.1e})')
        rho = '' if main['rho'] is None else \
            f'; ρ = {sg(main["rho"])} is computed on that roundoff'
        title = (f'π is 0 on all {len(dev)} dev IDs (proxy saturated on {sat["n_saturated"]} of {sat["of"]}), '
                 f'so it cannot track Δ\nΔ spans {sg(y.min())} to {sg(y.max())} points{rho}; '
                 f'D0 {"passes" if d0["passed"] else "fails"}')
    else:
        pad = 0.25 * np.ptp(x)
        ax.set_xlim(x.min() - pad, x.max() + pad)
        ax.set_xlabel('π_i  =  proxyAUC_i(X1f-kNN) − proxyAUC_i(GS12f-kNN)')
        rho = 'ρ undefined' if main['rho'] is None else \
            f'Spearman ρ = {sg(main["rho"])}, 95 % CI [{sg(main["ci_lo"])}, {sg(main["ci_hi"])}]'
        title = f'π_i against Δ_i on the {len(dev)} dev IDs: {rho}; D0 {"passes" if d0["passed"] else "fails"}'
    ax.set_title(title, loc='left', fontsize=9.5, color=INK)
    save(fig, 'fig1_pi_vs_delta.png')


# ------------------------------------------------------------------ fig 2: proxyAUC per system, saturation
def fig2():
    rows, J = read(os.path.join(SRC, 'proxy_ids.csv')), read_json(os.path.join(SRC, 'stage1.json'))
    if not rows or not J:
        print('skip fig2: scores/proxy_ids.csv or stage1.json missing')
        return
    vals = np.array([float(r[f'proxyAUC_{s}']) for r in rows for s, _ in SYS])
    lo = min(0.995, vals.min() - 0.0005)
    fig, axs = plt.subplots(1, len(SYS), figsize=(9.6, 3.9), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (s, lab) in zip(axs, SYS):
        ax.axhline(SAT, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        for j, mm in enumerate(MACH):
            for ds, dx in (('dev', -0.17), ('eval', 0.17)):
                v = [float(r[f'proxyAUC_{s}']) for r in rows if r['machine'] == mm and r['set'] == ds]
                xs = j + dx + rng.uniform(-0.07, 0.07, len(v))
                ax.scatter(xs, v, s=22, color=COL[ds], edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.set_xticks(range(len(MACH)))
        ax.set_xticklabels(MACH, rotation=35, ha='right', fontsize=8)
        ax.set_title(lab, fontsize=9, color=INK_2)
        ax.grid(axis='y', zorder=0)
        ax.set_ylim(lo, 1.0004)
    axs[0].set_ylabel('proxyAUC_i (other IDs\' training clips vs ID i\'s)')
    axs[-1].text(len(MACH) - 0.5, SAT, '0.999 (D0(a))', color=MUTED, fontsize=7.5, ha='right', va='top')
    nd = sum(r['set'] == 'dev' for r in rows)
    hs = [plt.Line2D([], [], marker='o', ls='', color=COL[ds], markersize=5.5, label=f'{ds} IDs ({n})')
          for ds, n in (('dev', nd), ('eval', len(rows) - nd))]
    axs[0].legend(handles=hs, loc='lower left', fontsize=8, handletextpad=0.2)
    d = J['descriptive']
    n_all = sum(all(float(r[f'proxyAUC_{s}']) >= SAT for s, _ in SYS) for r in rows)
    sep = d.get('n_ids_perfectly_separated')
    line2 = ''
    if sep:
        nsep = min(sep[s]['dev'] + sep[s]['eval'] for s, _ in SYS)
        mz = min(min(d['separation_margin_z'][s]['dev']['min'], d['separation_margin_z'][s]['eval']['min'])
                 for s, _ in SYS)
        line2 = (f'\n{nsep} of {len(rows)} are perfectly separated under every system; the smallest gap between '
                 f'stand-in anomalies and normals is {mz:.0f} (summed-z units)')
    fig.suptitle(f'{n_all} of {len(rows)} machine IDs reach proxyAUC ≥ 0.999 under all three systems{line2}',
                 x=0.06, ha='left', y=1.07, fontsize=9.5, color=INK)
    save(fig, 'fig2_proxyauc_saturation.png')


if __name__ == '__main__':
    fig1()
    fig2()
