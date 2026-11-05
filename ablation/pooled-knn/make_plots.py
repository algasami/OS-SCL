"""Figures for ablation/pooled-knn. Reads scores/ only; skips a figure whose inputs are missing.

Run with /home/f74134118/anaconda3/bin/python (matplotlib lives in base).
Colour = score family: S1′ (the reference) achromatic; time-mean pooled kNN blue; time-max
pooled kNN orange; the chosen rule green (direct labels: green is below 3:1 on the surface).
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
FEAT = {'C1': 'Gmean', 'C2': 'Gmax', 'C3': 'Gmm', 'C4': 'Bmean', 'C5': 'Bmax', 'C6': 'All'}
LABEL = {'Gmean': 'global time-mean', 'Gmax': 'global time-max', 'Gmm': 'global mean+max',
         'Bmean': 'per-band time-mean', 'Bmax': 'per-band time-max', 'All': 'all four'}
# mean-pooled family blue, max-pooled orange, mixtures achromatic
FCOL = {'Gmean': BLUE, 'Bmean': BLUE, 'Gmax': ORANGE, 'Bmax': ORANGE, 'Gmm': INK_2, 'All': INK_2}
SEL, CONF = (2040, 2041, 2042, 2043), (2044, 2045, 2046, 2047)

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


def metric(rows, system, ds, rule, machine='overall', k='Amean'):
    for r in rows:
        if r['system'] == system and r['dataset'] == ds and r['rule'] == rule and r['machine'] == machine:
            return float(r[k])
    return None


def fig1():
    """Selection: each candidate's dev Δ vs S1′ on the 4 selection seeds."""
    rows, sel = read('select_metrics.csv'), read('select.json')
    if not rows or not sel:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axes[0]
    cands = list(FEAT)
    for j, c in enumerate(cands):
        d = [metric(rows, f'SCLTFSTgramMFN_ao_nogain_s{s}', 'dev', c) - metric(rows, f'SCLTFSTgramMFN_ao_nogain_s{s}', 'dev', 'S1p')
             for s in SEL]
        col = FCOL[FEAT[c]]
        ax.scatter([j] * len(d), d, s=16, color=col, alpha=0.55, lw=0)
        ax.plot([j - 0.25, j + 0.25], [np.mean(d)] * 2, color=col, lw=2.2)
        ax.text(j, max(d) + 0.03, f'{np.mean(d):+.2f}', ha='center', va='bottom', fontsize=8,
                color=INK, fontweight='bold' if c == sel['chosen'] else 'normal')
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(cands)))
    ax.set_xticklabels([f'{c}\n{LABEL[FEAT[c]]}' for c in cands], fontsize=7.5)
    ax.set_ylabel('dev Amean − S1′ (points)')
    ax.grid(axis='y')
    ch = sel['chosen']
    ax.set_title(f'Selection (seeds 2040–43, dev): {ch} ({LABEL[FEAT[ch]]}) has the largest mean Δ, '
                 f'{sel["candidates"][ch]["d_Amean"]:+.2f}', loc='left', fontsize=9.5)
    ax = axes[1]
    st = sel['standalone_dev_mean']
    names = ['S0', 'zKp', 'Gmean', 'Gmax', 'Gmm', 'Bmean', 'Bmax', 'All']
    lab = {'S0': 'R0 (eval.py)', 'zKp': "K′ (128-d kNN)"} | {k: LABEL[k] for k in LABEL}
    col = {'S0': INK_2, 'zKp': INK_2} | FCOL
    y = np.arange(len(names))[::-1]
    v = [st[n]['AUC'] for n in names]
    ax.barh(y, v, color=[col[n] for n in names], height=0.6)
    for yy, vv in zip(y, v):
        ax.text(vv + 0.2, yy, f'{vv:.1f}', va='center', fontsize=8)
    ax.axvline(st['S1p']['AUC'], color=INK, lw=0.8, ls=':')
    ax.text(st['S1p']['AUC'], y[0] + 0.55, 'S1′', fontsize=8, ha='center')
    ax.set_yticks(y)
    ax.set_yticklabels([lab[n] for n in names])
    lo = min(v) - 3
    ax.set_xlim(lo, 100)
    ax.set_xlabel('standalone dev AUC (4-seed mean)')
    ax.set_title('Standalone scores', loc='left', fontsize=9.5)
    fig.tight_layout()
    save(fig, 'fig1_selection.png')


def fig2():
    """Confirmation: chosen − S1′ with paired bootstrap CI, single models and G8 ensemble."""
    c, rows = read('confirm.json'), read('confirm_metrics.csv')
    if not c or not rows:
        return
    ch = c['chosen']
    cells = [('confirm dev\n(2044–47)', c['confirm_dev'], 'dev', CONF),
             ('eval\n(8 seeds)', c['eval8'], 'eval', SEL + CONF),
             ('G8 ensemble\ndev', c['ensemble_dev'], None, None),
             ('G8 ensemble\neval', c['ensemble_eval'], None, None)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, k in zip(axes, ('AUC', 'pAUC', 'Amean')):
        for j, (name, d, ds, seeds) in enumerate(cells):
            col = GREEN if j < 2 else INK_2
            lo, hi = d['ci'][k]
            ax.plot([j, j], [lo, hi], color=col, lw=2, solid_capstyle='butt')
            ax.scatter([j], [d[k]], color=col, s=30, zorder=3)
            ax.text(j + 0.12, d[k], f'{d[k]:+.2f}', va='center', fontsize=8)
            if seeds:
                pts = [metric(rows, f'SCLTFSTgramMFN_ao_nogain_s{s}', ds, ch, k=k)
                       - metric(rows, f'SCLTFSTgramMFN_ao_nogain_s{s}', ds, 'S1p', k=k) for s in seeds]
                ax.scatter([j - 0.18] * len(pts), pts, color=col, s=9, alpha=0.5, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([x[0] for x in cells], fontsize=7.5)
        ax.set_title(k, loc='left', fontsize=9.5)
        ax.grid(axis='y')
    axes[0].set_ylabel(f'{ch} − S1′ (points)\nbar = paired 95% clip-bootstrap')
    verdict = 'adopted' if c['D1']['adopt'] else 'not adopted'
    fig.suptitle(f'{ch} = S1′ + z{FEAT[ch]} on the cells it was not selected on: D1 {verdict}',
                 x=0.01, ha='left', fontsize=10)
    fig.tight_layout()
    save(fig, 'fig2_confirm.png')


def fig3():
    """Per machine Δ AUC, chosen − S1′, 8-seed mean, dev and eval."""
    c = read('confirm.json')
    if not c:
        return
    ch = c['chosen']
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    x = np.arange(len(MACH))
    for off, ds, col in ((-0.18, 'dev', GREEN), (0.18, 'eval', INK_2)):
        v = [c['per_machine_dAUC'][ds][m] for m in MACH]
        ax.bar(x + off, v, width=0.34, color=col, label=ds)
        for xx, vv in zip(x + off, v):
            ax.text(xx, vv + (0.03 if vv >= 0 else -0.03), f'{vv:+.2f}', ha='center',
                    va='bottom' if vv >= 0 else 'top', fontsize=7)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(MACH)
    ax.set_ylabel(f'{ch} − S1′ AUC (points, 8 seeds)')
    ax.legend(loc='upper left', ncol=2)
    ax.grid(axis='y')
    pm = c['per_machine_dAUC']
    top = max(MACH, key=lambda m: pm['dev'][m] + pm['eval'][m])
    ax.set_title(f'Per machine: {top} moves most (dev {pm["dev"][top]:+.2f}, eval {pm["eval"][top]:+.2f})',
                 loc='left', fontsize=9.5)
    fig.tight_layout()
    save(fig, 'fig3_per_machine.png')


def fig4():
    """ToyCar diagnostic: penalty and AUC vs circular roll."""
    d = read('diag.json')
    if not d:
        return
    ch = read('select.json')['chosen']
    series = [('S1p', 'S1′', INK_2, '-'), ('Gmean', 'zGmean (time-mean)', BLUE, '-'),
              ('Gmax', 'zGmax (time-max)', ORANGE, '-')]
    if ch not in ('Gmean', 'Gmax'):
        series.append((ch, f'{ch} = S1′ + z{FEAT[ch]}', GREEN, '--'))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    k = np.arange(8)
    for j, ds in enumerate(('dev', 'eval')):
        for nm, lab, col, ls in series:
            axes[0, j].plot(k, d[ds][nm]['penalty_by_k'], color=col, ls=ls, marker='o', ms=3, lw=1.6, label=lab)
            axes[1, j].plot(k, d[ds][nm]['AUC_by_k'], color=col, ls=ls, marker='o', ms=3, lw=1.6, label=lab)
        axes[0, j].axhline(0, color=MUTED, lw=0.8)
        axes[0, j].axhline(1, color=MUTED, lw=0.8, ls=':')
        axes[0, j].axvspan(2.5, 5.5, color=GRID, alpha=0.5, lw=0)
        s1, g = d[ds]['S1p']['headline_penalty_sd'], d[ds]['Gmax']['headline_penalty_sd']
        axes[0, j].set_title(f'{ds}: time-max {d[ds]["verdict_Gmax"]} the shift penalty '
                             f'({g:.1f} vs S1′ {s1:.1f} sd at k=3–5)', loc='left', fontsize=9.5)
        axes[1, j].set_xlabel('circular roll (k / 8 of the clip)')
        for a in axes[:, j]:
            a.grid(axis='y')
    axes[0, 0].set_ylabel('rise of normal ToyCar scores\n(sd of the unrolled score)')
    axes[1, 0].set_ylabel('ToyCar AUC, all clips rolled')
    axes[0, 0].legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    save(fig, 'fig4_toycar_roll.png')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4):
        f()
