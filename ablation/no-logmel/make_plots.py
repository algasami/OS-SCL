#!/usr/bin/env python
"""Figures for no-logmel.

Run from the repo root (matplotlib lives in base, not os-scl):
    /home/f74134118/anaconda3/bin/python ablation/no-logmel/make_plots.py

Reads scores/ only: this ablation's runs and bench.csv, and ../fast-recipe's
fr_f1/fr_f4 via scores_path. A figure whose inputs are missing is skipped.
Palette as in the repo's make_plots.py files: the three mel-free arms take the
three categorical slots (T+TF blue, TF orange, T green); the Log-Mel cells are
achromatic, told apart by marker and a direct label.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from scores_path import scores_dir  # noqa: E402

SCORES = os.path.join(HERE, 'scores')
PLOTS = os.path.join(HERE, 'plots')
os.makedirs(PLOTS, exist_ok=True)

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8983'
GRID = '#e4e3df'
BLUE, ORANGE, GREEN = '#2a78d6', '#eb6834', '#1baf7a'
MACH = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
PAPER = {'dev': 95.71, 'eval': 93.22}
SEEDS = (2040, 2041, 2042, 2043)

# (run prefix, label, uses Log-Mel, colour, marker, fussion)
CELLS = [
    ('fr_f1', 'LM+T+TF\n(paper)', True, INK, 'o', 1),
    ('fr_f4', 'LM+T', True, INK_2, 's', 4),
    ('nl_f2', 'LM', True, MUTED, 'D', 2),
    ('nl_f5', 'T+TF', False, BLUE, 'o', 5),
    ('nl_f6', 'TF', False, ORANGE, 'o', 6),
    ('nl_f7', 'T', False, GREEN, 'o', 7),
]
ARMS = [c for c in CELLS if not c[2]]

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'text.color': INK, 'axes.labelcolor': INK_2, 'axes.edgecolor': GRID,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False,
})


def short(label):
    return label.split('\n')[0]


def rows(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding='UTF-8') as f:
        return list(csv.DictReader(f))


def run_dir(run):
    local = os.path.join(SCORES, run)
    if os.path.isdir(local):
        return local
    try:
        return scores_dir(run)
    except FileNotFoundError:
        return None


def metric(prefix, seed, ds, machine=None, key='AUC'):
    """AUC/pAUC/mAUC (%) of one run, overall or for one machine type; None if unscored."""
    d = run_dir(f'SCLTFSTgramMFN_{prefix}_s{seed}')
    r = rows(os.path.join(d, f'eval_{ds}.csv')) if d else None
    if not r:
        return None
    for x in r:
        if machine is None and x['scope'] == 'overall':
            return 100 * float(x[key])
        if machine is not None and x['scope'] == 'machine' and x['machine'] == machine:
            return 100 * float(x[key])
    return None


def values(prefix, ds, machine=None, key='AUC'):
    v = [metric(prefix, s, ds, machine, key) for s in SEEDS]
    return np.array([x for x in v if x is not None])


def diff_se(a, b):
    """Mean difference a - b and its unpaired standard error (Welch)."""
    if len(a) < 2 or len(b) < 2:
        return a.mean() - b.mean(), np.nan
    return a.mean() - b.mean(), np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


def save(fig, name):
    fig.savefig(os.path.join(PLOTS, name), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


# ---------------------------------------------------------------- fig1
def fig1():
    data = {(c[0], ds): values(c[0], ds) for c in CELLS for ds in ('dev', 'eval')}
    if not len(data[('fr_f1', 'dev')]) or not any(len(data[(a[0], 'dev')]) for a in ARMS):
        print('skip fig1: needs fr_f1 and at least one mel-free arm scored')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    rng = np.random.default_rng(0)
    for ax, ds in zip(axes, ('dev', 'eval')):
        xs = [0, 1, 2, 3.6, 4.6, 5.6]
        for x, (p, label, mel, col, mk, _) in zip(xs, CELLS):
            v = data[(p, ds)]
            if not len(v):
                ax.text(x, 0.02, 'not\nscored', transform=ax.get_xaxis_transform(), ha='center',
                        va='bottom', color=MUTED, fontsize=7.5)
                continue
            jit = rng.uniform(-0.12, 0.12, len(v))
            ax.scatter(x + jit, v, s=34, marker=mk, color=col, edgecolor=SURFACE, linewidth=1.2, zorder=3)
            ax.plot([x - 0.28, x + 0.28], [v.mean()] * 2, color=col, lw=2, solid_capstyle='round', zorder=2)
            ax.annotate(f'{v.mean():.2f}' + (f'\nn={len(v)}' if len(v) != 4 else ''), (x + 0.32, v.mean()),
                        va='center', fontsize=7.5, color=INK_2)
        ax.axhline(PAPER[ds], color=INK, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.annotate(f'paper {PAPER[ds]:.2f}', (5.95, PAPER[ds]), xycoords='data', va='bottom', ha='right',
                    fontsize=7.5, color=INK_2)
        ax.axvline(2.8, color=GRID, lw=1)
        ax.set_xticks(xs, [c[1] for c in CELLS])
        ax.set_xlim(-0.5, 6.3)
        ax.text(1, 1.0, 'with Log-Mel', transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                color=INK_2, fontsize=8)
        ax.text(4.6, 1.0, 'without Log-Mel', transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                color=INK_2, fontsize=8)
        ax.grid(axis='y')
        ax.set_ylabel(f'{ds} AUC (%), R0, per seed and mean')
    f1, f5 = data[('fr_f1', 'dev')], data[('nl_f5', 'dev')]
    if len(f5):
        d = f5.mean() - f1.mean()
        verb = 'costs' if d < -0.2 else ('matches within 0.2' if d <= 0.2 else 'gains')
        title = f'Dropping Log-Mel from the paper model {verb}: T+TF {d:+.2f} dev AUC vs LM+T+TF'
    else:
        title = 'Input configurations, dev and eval AUC per seed'
    fig.suptitle(title, x=0.01, ha='left', fontsize=11, y=1.02)
    fig.text(0.01, -0.03, 'The fourth mel-free cell (neither gram) has no input: AUC 50 by construction, '
             'not trained. Dashed: the paper\'s single number (one run, no seeds).', fontsize=7.5, color=INK_2)
    save(fig, 'fig1_factorial_auc.png')


# ---------------------------------------------------------------- fig2
def fig2():
    if not len(values('fr_f1', 'dev')) or not any(len(values(a[0], 'dev')) for a in ARMS):
        print('skip fig2: needs fr_f1 and a mel-free arm')
        return
    cells = [c for c in CELLS if c[0] != 'fr_f1']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    ylab = MACH + ['overall']
    worst = None
    for ax, ds in zip(axes, ('dev', 'eval')):
        for i, (p, label, mel, col, mk, _) in enumerate(cells):
            off = (i - (len(cells) - 1) / 2) * 0.13
            for j, m in enumerate(ylab):
                a = values(p, ds, None if m == 'overall' else m)
                b = values('fr_f1', ds, None if m == 'overall' else m)
                if not len(a) or not len(b):
                    continue
                d, se = diff_se(a, b)
                y = j + off
                ax.plot([d - se, d + se], [y, y], color=col, lw=1.4, alpha=0.8, solid_capstyle='round')
                ax.scatter([d], [y], s=26, marker=mk, color=col if not mel else SURFACE,
                           edgecolor=col, linewidth=1.3, zorder=3,
                           label=short(label) if j == 0 and ds == 'dev' else None)
                if ds == 'dev' and not mel and m != 'overall' and (worst is None or d < worst[0]):
                    worst = (d, m, short(label))
        ax.axvline(0, color=INK, lw=0.9)
        ax.axhline(len(MACH) - 0.5, color=GRID, lw=1)
        ax.set_yticks(range(len(ylab)), ylab)
        ax.invert_yaxis()
        ax.grid(axis='x')
        ax.set_xlabel(f'{ds} AUC − LM+T+TF (paper model), points; bar = ±1 se of the 4-vs-4 difference')
    axes[0].legend(loc='lower left', fontsize=8, ncol=1, title='arm (hollow = with Log-Mel)', title_fontsize=7.5)
    t = 'Per machine type, change against the paper model'
    if worst:
        t = f'Without Log-Mel, the largest dev loss is {worst[2]} on {worst[1]} ({worst[0]:+.1f})'
    fig.suptitle(t, x=0.01, ha='left', fontsize=11, y=1.0)
    save(fig, 'fig2_per_machine_delta.png')


# ---------------------------------------------------------------- fig3
def bench_ms():
    r = rows(os.path.join(SCORES, 'bench.csv'))
    if not r:
        return None
    b = {}
    for x in r:
        b[(x['part'], x['what'], x['fussion'], x['device'], x['threads'], x['batch'], x['precision'])] = \
            float(x['ms_per_clip'])
    return b


def fig3():
    b = bench_ms()
    have = [c for c in CELLS if len(values(c[0], 'dev'))]
    if not b or not have:
        print('skip fig3: needs scores/bench.csv and scored runs')
        return
    dec = b[('feat', 'decode', '', 'cpu', '1', '1', 'fp32')]
    mel = {'gpu': b[('feat', 'logmel', '', 'gpu', '', '64', 'fp32')],
           'cpu1': b[('feat', 'logmel', '', 'cpu', '1', '1', 'fp32')],
           'cpu8': b[('feat', 'logmel', '', 'cpu', '8', '64', 'fp32')]}
    targets = [
        ('GPU server: network fp16, batch 64\nLog-Mel on the GPU (filled) or 8 CPU threads (hollow)',
         lambda f: b[('infer', 'net', str(f), 'gpu', '', '64', 'fp16')], 'gpu', 'cpu8'),
        ('CPU only: 1 thread, batch 1\nLog-Mel on the same thread', lambda f: b[('infer', 'net', str(f), 'cpu', '1', '1', 'fp32')],
         'cpu1', None),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, (name, net, m1, m2) in zip(axes, targets):
        for p, label, uses_mel, col, mk, f in have:
            y = values(p, 'dev').mean()
            x = dec + net(f) + (mel[m1] if uses_mel else 0)
            ax.scatter([x], [y], s=46, marker=mk, color=col, edgecolor=SURFACE, linewidth=1.2, zorder=3)
            ax.annotate(short(label), (x, y), xytext=(6, 4), textcoords='offset points', fontsize=8, color=INK_2)
            if uses_mel and m2:
                x2 = dec + net(f) + mel[m2]
                ax.scatter([x2], [y], s=46, marker=mk, color=SURFACE, edgecolor=col, linewidth=1.3, zorder=3)
                ax.plot([x, x2], [y, y], color=col, lw=0.8, ls=':', zorder=2)
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.tick_params(axis='x', which='minor', labelsize=7.5)
        ax.grid(True, which='both', axis='x')
        ax.grid(True, axis='y')
        ax.set_title(name, fontsize=8.5, color=INK_2, loc='left')
    axes[0].set_ylabel('dev AUC (%), mean over seeds')
    fig.supxlabel('end-to-end ms per clip: decode + Log-Mel (if used) + network; log scale', fontsize=9,
                  color=INK_2, y=-0.02)
    e2e = {f: dec + b[('infer', 'net', str(f), 'cpu', '1', '1', 'fp32')] + (mel['cpu1'] if f == 2 else 0)
           for f in (2, 5)}
    fig.suptitle(f'On one CPU thread, mel-free T+TF costs {e2e[5] / e2e[2]:.1f}× Log-Mel only end to end; '
                 f'Log-Mel itself is {mel["cpu1"]:.1f} ms', x=0.01, ha='left', fontsize=11, y=1.04)
    save(fig, 'fig3_cost_vs_auc.png')


# ---------------------------------------------------------------- fig4
def fig4():
    series = []
    for p, label, mel, col, mk, _ in CELLS:
        curves = []
        for s in SEEDS:
            d = run_dir(f'SCLTFSTgramMFN_{p}_s{s}')
            r = rows(os.path.join(d, 'probe.csv')) if d else None
            if r:
                curves.append({int(x['epoch']): 100 * float(x['AUC']) for x in r if x.get('AUC')})
        if not curves:
            continue
        ep = sorted(set.intersection(*[set(c) for c in curves]))
        if ep:
            series.append((short(label), mel, col, ep, np.array([[c[e] for e in ep] for c in curves]).mean(0)))
    if not series:
        print('skip fig4: no probe.csv')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={'width_ratios': [1, 1.15]})
    for ax, lo in zip(axes, (0, 150)):
        ends = []
        for label, mel, col, ep, y in series:
            k = [i for i, e in enumerate(ep) if e >= lo]
            ax.plot([ep[i] for i in k], y[k], color=col, lw=2 if not mel else 1.4,
                    ls='-' if not mel else (0, (4, 2)))
            ends.append([y[-1], label, ep[-1]])
        if lo:   # direct labels, dodged to at least 3% of the axis apart
            ends.sort()
            lo_y, hi_y = ax.get_ylim()
            gap = 0.045 * (hi_y - lo_y)
            for i in range(1, len(ends)):
                ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
            for yl, label, e in ends:
                ax.annotate(label, (e, yl), xytext=(6, 0), textcoords='offset points', va='center',
                            fontsize=8, color=INK_2, annotation_clip=False)
        ax.grid(True)
        ax.set_xlabel('epoch')
    axes[0].set_ylim(48, None)
    axes[0].set_ylabel('dev AUC (%) of the EMA net, R0, mean over seeds')
    axes[0].set_title('whole run', fontsize=8.5, color=INK_2, loc='left')
    axes[1].set_title('epochs 150-300 (zoom)', fontsize=8.5, color=INK_2, loc='left')
    fig.suptitle('Dev AUC during training (probe: diagnostic, never used for selection); dashed = with Log-Mel',
                 x=0.01, ha='left', fontsize=11, y=1.02)
    save(fig, 'fig4_probe_curves.png')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4):
        try:
            f()
        except Exception as e:   # one broken figure must not stop the rest
            print(f'{f.__name__} failed: {e!r}')
