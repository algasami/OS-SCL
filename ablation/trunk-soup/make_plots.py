"""Figures for ablation/trunk-soup. Reads scores/ only (plus ../aug-ops/scores for the independent
ao_nogain seeds, via scores_path); skips a figure whose inputs are missing.

Run with /home/f74134118/anaconda3/bin/python (matplotlib lives in base).
Colour: forks are ao_nogain models, so green as in ../aug-ops and ../nogain-ensemble; the soup
(the new single network) orange; the fork score-ensemble blue; independent seeds and the trunk's
own second half achromatic. Green is below 3:1 on the surface, so green marks carry a direct label.
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

SCORES = os.path.join(HERE, 'scores')
PLOTS = os.path.join(HERE, 'plots')
os.makedirs(PLOTS, exist_ok=True)

SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8983', '#e4e3df'
BLUE, ORANGE, GREEN = '#2a78d6', '#eb6834', '#1baf7a'
MACH = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
TRUNKS = (2050, 2051)
P = 'SCLTFSTgramMFN_ts_'
INDEP = [f'SCLTFSTgramMFN_ao_nogain_s{s}' for s in range(2040, 2048)]

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


def r0(run, ds, scope='overall', machine='ALL'):
    """eval.py CSV value(s) in points; None if missing."""
    try:
        rows = read(scores_csv(run, ds))
    except Exception:   # noqa: BLE001 -- scores_path raises when the run has no scores dir
        return None
    if not rows:
        return None
    for x in rows:
        if x['scope'] == scope and x['machine'] == machine:
            return {k: 100 * float(x[k]) for k in ('AUC', 'pAUC', 'mAUC')}
    return None


def forks(t):
    return [f'{P}fork_s{t}_f{f}' for f in (1, 2, 3, 4)]


def fig1():
    sysrows = read(os.path.join(SCORES, 'systems.csv'))
    e1 = {}
    if sysrows:
        for x in sysrows:
            if x['rule'] == 'E1p' and x['system'].startswith(f'{P}E1p4_'):
                e1[(int(x['system'][-4:]), x['dataset'])] = float(x['AUC'])
    if r0(f'{P}soup4_s2050', 'dev') is None:
        print('skip fig1 (no soup scores)')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    for ax, ds in zip(axes, ('dev', 'eval')):
        for xi, t in enumerate(TRUNKS):
            fv = [r0(r, ds)['AUC'] for r in forks(t)]
            jit = np.linspace(-0.12, 0.12, len(fv))
            ax.scatter(xi + jit - 0.15, fv, s=22, color=GREEN, zorder=3, label='fork (alone)' if xi == 0 else None)
            ax.hlines(np.mean(fv), xi - 0.32, xi + 0.02, color=GREEN, lw=1.5, zorder=2,
                      label='mean fork' if xi == 0 else None)
            ax.scatter([xi + 0.12], [r0(f'{P}soup4_s{t}', ds)['AUC']], marker='D', s=46, color=ORANGE, zorder=4,
                       label='soup of 4 forks (one network)' if xi == 0 else None)
            ax.scatter([xi + 0.12], [r0(f'{P}soup5_s{t}', ds)['AUC']], marker='D', s=46, facecolor='none',
                       edgecolor=ORANGE, zorder=4, label='soup of 5 (+ trunk)' if xi == 0 else None)
            ax.scatter([xi + 0.26], [r0(f'{P}trunk_s{t}', ds)['AUC']], marker='x', s=36, color=INK_2, zorder=4,
                       label='trunk, own 2nd half' if xi == 0 else None)
            if (t, ds) in e1:
                ax.scatter([xi + 0.38], [e1[(t, ds)]], marker='s', s=36, color=BLUE, zorder=4,
                           label='E1′ over 4 forks (4 networks)' if xi == 0 else None)
            ax.text(xi - 0.15, min(fv) - 0.06, 'forks', color=GREEN, ha='center', va='top', fontsize=8)
        iv = [v['AUC'] for v in (r0(r, ds) for r in INDEP) if v]
        if iv:
            ax.scatter(2 + np.linspace(-0.15, 0.15, len(iv)), iv, s=18, color=MUTED, zorder=3,
                       label='independent ao_nogain seed')
            ax.hlines(np.mean(iv), 1.78, 2.22, color=MUTED, lw=1.5)
        if ds == 'dev':
            ax.axhline(95.71, color=INK_2, lw=0.8, ls=':')
            ax.text(-0.5, 95.71, 'paper 95.71', color=INK_2, fontsize=8, va='bottom', ha='left')
        ax.set_xticks([0, 1, 2], ['trunk s2050', 'trunk s2051', 'independent\n8 seeds'])
        ax.set_xlim(-0.55, 2.5)
        ax.set_ylabel(f'{ds} AUC, R0 (points)')
        ax.grid(axis='y')
        ax.set_title(ds, loc='left', fontsize=10)
    axes[1].legend(loc='lower left', fontsize=7.5, ncol=1)
    fig.suptitle('The soup matches its 4-fork ensemble, but forks are too alike for either to gain much; both trunks sit high on eval',
                 x=0.01, ha='left', fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'fig1_soup_vs_forks.png'), dpi=150)
    plt.close(fig)
    print('wrote fig1')


def fig2():
    if r0(f'{P}soup4_s2050', 'dev', 'machine', 'fan') is None:
        print('skip fig2')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    w = 0.36
    for ax, ds in zip(axes, ('dev', 'eval')):
        for k, t in enumerate(TRUNKS):
            d = [r0(f'{P}soup4_s{t}', ds, 'machine', m)['AUC'] -
                 np.mean([r0(r, ds, 'machine', m)['AUC'] for r in forks(t)]) for m in MACH]
            ax.bar(np.arange(len(MACH)) + (k - 0.5) * w, d, w, color=ORANGE, alpha=1.0 if k == 0 else 0.45,
                   label=f'trunk s{t}')
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_xticks(range(len(MACH)), MACH, fontsize=8)
        ax.set_title(f'{ds}: soup of 4 − mean fork (AUC, R0)', loc='left', fontsize=10)
        ax.grid(axis='y')
    axes[0].set_ylabel('Δ AUC (points)')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'fig2_machines.png'), dpi=150)
    plt.close(fig)
    print('wrote fig2')


def fig3():
    def probe(run):
        return read(os.path.join(SCORES, run, 'probe.csv'))
    if not probe(f'{P}trunk_s2050'):
        print('skip fig3')
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for col, t in enumerate(TRUNKS):
        for row, key in enumerate(('AUC', 'ToyConveyor_AUC')):
            ax = axes[row, col]
            tr = probe(f'{P}trunk_s{t}')
            ep = [int(x['epoch']) for x in tr if int(x['epoch']) >= 100]
            ax.plot(ep, [100 * float(x[key]) for x in tr if int(x['epoch']) >= 100], color=INK_2, lw=1.6,
                    label='trunk (own 2nd half)')
            for i, r in enumerate(forks(t)):
                pr = probe(r) or []
                ax.plot([int(x['epoch']) for x in pr], [100 * float(x[key]) for x in pr], color=GREEN, lw=0.9,
                        alpha=0.85, label='forks 1–4' if i == 0 else None)
            ax.axvline(150, color=MUTED, lw=0.8, ls='--')
            ax.set_title(f'trunk s{t}: dev {key.replace("_", " ")} (probe, EMA net)', loc='left', fontsize=9.5)
            ax.grid()
            if row == 1:
                ax.set_xlabel('epoch')
                lo = min(100 * float(x[key]) for x in tr if int(x['epoch']) >= 200)
                ax.set_ylim(lo - 3, None)
            else:
                ax.set_ylim(90, None)
    axes[0, 0].legend(fontsize=8, loc='lower right')
    fig.suptitle('Forks track their trunk on dev AUC after epoch 150; they spread only on ToyConveyor (~1.5 points)',
                 x=0.01, ha='left', fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'fig3_probe_curves.png'), dpi=150)
    plt.close(fig)
    print('wrote fig3')


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
