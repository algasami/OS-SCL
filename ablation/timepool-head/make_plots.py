"""Figures for ablation/timepool-head. Reads scores/ only; skips a figure whose inputs are missing.

Run with /home/f74134118/anaconda3/bin/python (matplotlib lives in base).
Colour = arm: ao_nogain (the reference) achromatic, tp blue. Rules are told apart by position
and label, never by a further colour. Titles are written from the numbers they plot.
"""
import csv
import glob
import json
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from scores_path import scores_dir  # noqa: E402

SCORES = os.environ.get('TP_OUT', os.path.join(HERE, 'scores'))     # TP_OUT: tests only
PLOTS = os.environ.get('TP_PLOTS', os.path.join(HERE, 'plots'))
os.makedirs(PLOTS, exist_ok=True)

SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8983', '#e4e3df'
BLUE = '#2a78d6'
COL = {'ao': MUTED, 'tp': BLUE}
LAB = {'ao': 'ao_nogain', 'tp': 'tp (time-tied linear7)'}
MACH = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
RULES = ('R0', 'S1p', 'C1')
RLAB = {'R0': 'R0', 'S1p': 'S1′', 'C1': 'C1 (primary)'}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'text.color': INK, 'axes.labelcolor': INK_2, 'axes.edgecolor': GRID,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False,
})


def summary():
    """The summary json with the most seeds (TP_SEEDS of all 8 seeds after stage 2, else stage 1)."""
    files = sorted(glob.glob(os.path.join(SCORES, 'summary_*.json')), key=lambda p: (-p.count('_'), p))
    return json.load(open(files[0])) if files else None


def rows(name):
    p = os.path.join(SCORES, name)
    return list(csv.DictReader(open(p, encoding='UTF-8'))) if os.path.exists(p) else None


def save(fig, fn):
    fig.savefig(os.path.join(PLOTS, fn), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('wrote plots/' + fn)


def dots(ax, x, d, c, w=0.22):
    d = np.asarray(d)
    ax.scatter(np.full(len(d), x) + np.linspace(-w / 3, w / 3, len(d)), d, s=16, color=c, alpha=0.55,
               edgecolor='none', zorder=2)
    ax.plot([x - w, x + w], [d.mean()] * 2, color=c, lw=2.4, zorder=3)
    ax.annotate(f'{d.mean():+.2f}', (x + w + 0.03, d.mean()), va='center', fontsize=7.5, color=INK)


def fig1_rules(S):
    """Paired Δ Amean (tp − ao_nogain) under R0, S1′, C1; dev and eval."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3), sharey=True)
    for ax, ds in zip(axes, ('dev', 'eval')):
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.grid(axis='y')
        for i, r in enumerate(RULES):
            dots(ax, i, S['rules'][r][f'{ds}|overall|Amean']['d_per_seed'], BLUE)
        ax.set_xticks(range(len(RULES)), [RLAB[r] for r in RULES])
        ax.set_xlim(-0.5, len(RULES) - 0.2)
        ax.set_title(f'{ds}, mean(AUC, pAUC)')
    axes[0].set_ylabel('tp − ao_nogain (points)')
    c = S['rules']['C1']
    fig.suptitle(f'Tying linear7 over time: C1 Amean {c["dev|overall|Amean"]["d"]:+.2f} dev, '
                 f'{c["eval|overall|Amean"]["d"]:+.2f} eval, D1 fails; R0 loses most '
                 f'(n={len(S["seeds"])} seed pairs; dots: seeds, bar: mean)',
                 x=0.01, ha='left', fontsize=10.5, fontweight='bold', y=1.03)
    save(fig, 'fig1_rules.png')


def fig2_machines(S):
    """Paired Δ AUC per machine under C1, dev and eval."""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4), sharey=True)
    for ax, ds in zip(axes, ('dev', 'eval')):
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.grid(axis='y')
        for i, m in enumerate(MACH):
            dots(ax, i, S['rules']['C1'][f'{ds}|{m}|AUC']['d_per_seed'], BLUE, w=0.2)
        ax.set_xticks(range(len(MACH)), MACH, rotation=20)
        ax.set_title(f'{ds}, C1')
    axes[0].set_ylabel('AUC, tp − ao_nogain (points)')
    c = S['rules']['C1']
    fig.suptitle(f'Per machine under C1: ToyConveyor {c["dev|ToyConveyor|AUC"]["d"]:+.2f} dev fails D1(d); '
                 f'ToyCar {c["dev|ToyCar|AUC"]["d"]:+.2f} / {c["eval|ToyCar|AUC"]["d"]:+.2f}; non-Toy Amean {c["dev|nontoy|Amean"]["d"]:+.2f} / {c["eval|nontoy|Amean"]["d"]:+.2f}',
                 x=0.01, ha='left', fontsize=10.5, fontweight='bold', y=1.03)
    save(fig, 'fig2_machines.png')


def fig3_toy_ids(S):
    """ToyCar and ToyConveyor per-ID AUC, both arms, under S1′ and C1 (seed means)."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.2), gridspec_kw={'width_ratios': [7, 6]})
    for row, mach in enumerate(('ToyCar', 'ToyConveyor')):
        for ax, ds in zip(axes[row], ('dev', 'eval')):
            ax.grid(axis='y')
            keys = sorted(k for k in S['rules']['C1'] if k.startswith(f'{ds}|{mach} id_'))
            ids = [k.split('|')[1].split()[1] for k in keys]
            for j, (rule, arm) in enumerate([(r, a) for r in ('S1p', 'C1') for a in ('ao', 'tp')]):
                x = np.arange(len(ids)) + (j - 1.5) * 0.19
                v = [S['rules'][rule][k][arm] for k in keys]
                ax.bar(x, v, width=0.17, color=COL[arm], alpha=0.45 if rule == 'S1p' else 1.0,
                       label=f'{LAB[arm]}, {RLAB[rule]}' if row == 0 and ds == 'dev' else None)
            lo = min(min(S['rules'][r][k][a] for k in keys) for r in ('S1p', 'C1') for a in ('ao', 'tp'))
            ax.set_ylim(max(50, np.floor(lo / 5) * 5 - 5), 100.5)
            ax.set_xticks(range(len(ids)), ids)
            ax.set_title(f'{ds} {mach} AUC by ID')
    axes[0][0].set_ylabel('AUC (points), seed mean')
    axes[1][0].set_ylabel('AUC (points), seed mean')
    fig.legend(loc='upper left', bbox_to_anchor=(0.01, 0.965), fontsize=8, ncol=4)
    c = S['rules']['C1']
    fig.suptitle(f'ToyCar under C1: dev id_03 {c["dev|ToyCar id_03|AUC"]["ao"]:.1f} → {c["dev|ToyCar id_03|AUC"]["tp"]:.1f}, '
                 f'eval id_05 {c["eval|ToyCar id_05|AUC"]["ao"]:.1f} → {c["eval|ToyCar id_05|AUC"]["tp"]:.1f} '
                 '(pale: S1′, solid: C1)', x=0.01, ha='left', fontsize=10.5, fontweight='bold', y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, 'fig3_toy_ids.png')


def fig4_roll():
    p = os.path.join(SCORES, 'diag.json')
    if not os.path.exists(p):
        print('skip fig4: no diag.json')
        return
    D = json.load(open(p))
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.0), sharex=True)
    k = np.arange(8) / 8 * 100
    for col, ds in enumerate(('dev', 'eval')):
        o = D[ds]
        for row, (key, ylab) in enumerate((('penalty_by_k', 'normal-clip penalty (sd of unrolled score)'),
                                           ('AUC_by_k', 'ToyCar AUC (points)'))):
            ax = axes[row][col]
            ax.grid(axis='y')
            for arm, nm, ls in (('ao', 'S1p', '-'), ('tp', 'S1p', '-'), ('tp', 'Gmean', ':')):
                y = o[f'{arm}|{nm}'][key]
                ax.plot(k, y, color=COL[arm], lw=1.8, ls=ls, marker='o', ms=3)
                # tp's two curves sit close: label them at different rolls, on the side away from the other
                i = 4 if arm == 'ao' else (2 if nm == 'S1p' else 6)
                other = o['tp|Gmean' if nm == 'S1p' else 'tp|S1p'][key][i]
                dy = 6 if arm == 'ao' else (6 if y[i] >= other else -12)
                ax.annotate(f'{LAB[arm].split(" ")[0]} {"S1′" if nm == "S1p" else "zGmean"}', (k[i], y[i]),
                            xytext=(0, dy), textcoords='offset points', ha='center',
                            fontsize=7.5, color=INK if arm == 'tp' else INK_2)
            if col == 0:
                ax.set_ylabel(ylab)
            if row == 1:
                ax.set_xlabel('circular roll (% of the clip)')
            ax.set_title(f'{ds} ToyCar')
    t = {ds: D[ds]['tp|S1p']['headline_penalty_sd'] for ds in ('dev', 'eval')}
    a = {ds: D[ds]['ao|S1p']['headline_penalty_sd'] for ds in ('dev', 'eval')}
    fig.suptitle(f'ToyCar under circular rolls: S1′ penalty at 37.5–62.5% is {t["dev"]:.2f} / {t["eval"]:.2f} sd for tp '
                 f'vs {a["dev"]:.1f} / {a["eval"]:.1f} for ao_nogain (dev / eval)',
                 x=0.01, ha='left', fontsize=10.5, fontweight='bold', y=1.0)
    fig.tight_layout()
    save(fig, 'fig4_toycar_roll.png')


def fig5_probe(seeds):
    curves = {}
    for arm, fmt in (('ao', 'SCLTFSTgramMFN_ao_nogain_s{}'), ('tp', 'SCLTFSTgramMFN_tp_s{}')):
        got = []
        for s in seeds:
            try:
                p = os.path.join(scores_dir(fmt.format(s)), 'probe.csv')
            except FileNotFoundError:
                continue
            if os.path.exists(p):
                got.append({int(r['epoch']): r for r in csv.DictReader(open(p))})
        if got:
            curves[arm] = got
    if len(curves) < 2:
        print('skip fig5: probe logs missing for an arm')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.3), sharex=True)
    ends = {}
    for ax, key, title in zip(axes, ('AUC', 'ToyCar_AUC'), ('dev AUC, all machines', 'dev ToyCar AUC')):
        ax.grid(axis='y')
        for arm in ('ao', 'tp'):
            got = curves[arm]
            ep = sorted(set.intersection(*(set(r) for r in got)))
            y = np.array([[100 * float(r[e][key]) for e in ep] for r in got]).mean(0)
            ax.plot(np.array(ep) + 1, y, color=COL[arm], lw=1.6)
            ends[(key, arm)] = (ep[-1] + 1, y[-1])
        hi = max(('ao', 'tp'), key=lambda a: ends[(key, a)][1])
        for arm in ('ao', 'tp'):
            x, v = ends[(key, arm)]
            ax.annotate(f'{LAB[arm].split(" ")[0]} {v:.2f}', (x, v), xytext=(4, 6 if arm == hi else -6),
                        textcoords='offset points', va='center', fontsize=7.5, color=INK if arm == 'tp' else INK_2)
        ends = {k2: (v2 if not isinstance(v2, tuple) else v2[1]) for k2, v2 in ends.items()}
        ax.set_ylim(85, 100)
        ax.set_xlim(100, 345)
        ax.set_title(title)
        ax.set_xlabel('epoch')
    axes[0].set_ylabel('EMA-net R0 probe AUC (points), seed mean')
    fig.suptitle(f'Dev-test R0 probe: tp trails ao_nogain from epoch 130 on (final {ends[("AUC", "tp")]:.2f} vs '
                 f'{ends[("AUC", "ao")]:.2f}); ToyCar ends level ({ends[("ToyCar_AUC", "tp")]:.2f} vs '
                 f'{ends[("ToyCar_AUC", "ao")]:.2f}); diagnostic only', x=0.01, ha='left',
                 fontsize=10.5, fontweight='bold', y=1.03)
    save(fig, 'fig5_probe.png')


if __name__ == '__main__':
    S = summary()
    if S is None:
        print('skip fig1-3: no summary_*.json')
    else:
        fig1_rules(S)
        fig2_machines(S)
        fig3_toy_ids(S)
    fig4_roll()
    fig5_probe(S['seeds'] if S else (2040, 2041, 2042, 2043))
