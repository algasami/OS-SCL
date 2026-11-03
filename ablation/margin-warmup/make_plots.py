#!/usr/bin/env python
"""Figures and a text summary for margin-warmup.

Run from the repo root (matplotlib lives in base, not os-scl):
    /home/f74134118/anaconda3/bin/python ablation/margin-warmup/make_plots.py

Reads scores/ only. This ablation's mw_* runs come from ./scores. The registered
no-logmel runs (nl_*) are found by name with ../scores_path.py, or, before the
no-logmel branch is merged, in $MW_EXTRA_SCORES (run.sh points it at the
no-logmel worktree's scores/, read-only). Any figure whose inputs are missing is
skipped. $MW_SCORES / $MW_PLOTS override the directories (tests only).

Palette as in ../no-logmel: TF (f6) orange, T (f7) green, T+TF (f5) blue, the
Log-Mel control (f2) achromatic. An arm keeps its colour; the fix is drawn
solid, the registered run dashed and lighter.
"""
import csv
import glob
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from scores_path import scores_dir  # noqa: E402

SCORES = os.environ.get('MW_SCORES') or os.path.join(HERE, 'scores')
PLOTS = os.environ.get('MW_PLOTS') or os.path.join(HERE, 'plots')
EXTRA = [p for p in os.environ.get('MW_EXTRA_SCORES', '').split(':') if p]
os.makedirs(PLOTS, exist_ok=True)

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8983'
GRID = '#e4e3df'
BAND = '#efeeea'
BLUE, ORANGE, GREEN = '#2a78d6', '#eb6834', '#1baf7a'
SEEDS = (2040, 2041, 2042, 2043)
HOLD, RAMP_END, M_FINAL = 60, 120, 0.4      # the registered schedule, --m_warmup 60,120

# (arm, label, colour)
ARMS = [('f6', 'TF only', ORANGE), ('f7', 'T only', GREEN), ('f2', 'Log-Mel only (control)', INK_2),
        ('f5', 'T+TF', BLUE)]

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'text.color': INK, 'axes.labelcolor': INK_2, 'axes.edgecolor': GRID,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False,
})


def rows(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding='UTF-8') as f:
        return list(csv.DictReader(f))


def run_dir(run):
    for base in [SCORES] + EXTRA:
        if os.path.isdir(os.path.join(base, run)):
            return os.path.join(base, run)
    try:
        return scores_dir(run)
    except FileNotFoundError:
        return None


def name(fix, arm, seed):
    return f'SCLTFSTgramMFN_{"mw" if fix else "nl"}_{arm}_s{seed}'


def table(fix, arm, seed, file):
    d = run_dir(name(fix, arm, seed))
    return rows(os.path.join(d, file)) if d else None


def overall(fix, arm, seed, ds):
    r = table(fix, arm, seed, f'eval_{ds}.csv')
    for x in r or []:
        if x['scope'] == 'overall':
            return 100 * float(x['AUC'])
    return None


def values(fix, arm, ds):
    v = [overall(fix, arm, s, ds) for s in SEEDS]
    return {s: x for s, x in zip(SEEDS, v) if x is not None}


def save(fig, fname):
    fig.savefig(os.path.join(PLOTS, fname), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('wrote', fname)


def schedule_bands(ax):
    ax.axvspan(0, HOLD, color=BAND, lw=0, zorder=0)
    ax.axvspan(HOLD, RAMP_END, color=BAND, alpha=0.45, lw=0, zorder=0)


def end_label(ax, x, y, text, col):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords='offset points', va='center', fontsize=7.5, color=col)


# ---------------------------------------------------------------- fig1: probe curves
def fig1():
    arms = [a for a in ARMS if any(table(True, a[0], s, 'probe.csv') for s in SEEDS)]
    if not arms:
        print('skip fig1: no mw_* probe.csv in scores/')
        return
    fig, axes = plt.subplots(1, len(arms), figsize=(4.6 * len(arms), 3.8), sharey=True, squeeze=False)
    fig.subplots_adjust(wspace=0.38)
    ends = {}
    for ax, (arm, label, col) in zip(axes[0], arms):
        schedule_bands(ax)
        for fix in (False, True):
            for s in SEEDS:
                r = table(fix, arm, s, 'probe.csv')
                if not r:
                    continue
                e = [int(x['epoch']) + 1 for x in r]
                a = [100 * float(x['AUC']) for x in r]
                ax.plot(e, a, color=col, lw=1.6 if fix else 1.1, alpha=1 if fix else 0.55,
                        ls='-' if fix else (0, (4, 2)), zorder=3 if fix else 2)
                ends.setdefault((arm, fix), []).append(a[-1])
        ys = {fix: np.mean(ends[(arm, fix)]) for fix in (False, True) if (arm, fix) in ends}
        if len(ys) == 2 and abs(ys[True] - ys[False]) < 5:     # keep the two (2-line) end labels apart
            hi = max(ys, key=ys.get)
            ys[hi] += 2.5
            ys[not hi] -= 2.5
        for fix, y in ys.items():
            v = ends[(arm, fix)]
            end_label(ax, 300, y, f'{"warm-up" if fix else "reg."} {np.mean(v):.1f}'
                      + (f'\nn={len(v)}' if len(v) > 1 else ''), col)
        ax.set_title(label, loc='left', fontsize=9.5, color=col if col != INK_2 else INK)
        ax.set_xlim(0, 300)
        ax.set_xlabel('epoch')
        ax.grid(axis='y')
        ax.text(HOLD / 2, 0.02, 'm = 0', transform=ax.get_xaxis_transform(), ha='center', fontsize=7, color=INK_2)
        ax.text((HOLD + RAMP_END) / 2, 0.02, 'ramp', transform=ax.get_xaxis_transform(), ha='center',
                fontsize=7, color=INK_2)
    axes[0][0].set_ylabel('probe dev AUC (%), EMA net, scored at m = 0.4')
    deltas = [(lab, np.mean(ends[(a, True)]) - np.mean(ends[(a, False)]))
              for a, lab, _ in arms if (a, True) in ends and (a, False) in ends]
    if deltas:
        t = 'Final probe AUC, warm-up − registered: ' + ', '.join(f'{lab.split(" (")[0]} {d:+.1f}' for lab, d in deltas)
    else:
        t = 'Probe dev AUC under the margin warm-up (registered runs not found)'
    fig.suptitle(t, x=0.01, ha='left', fontsize=11, y=1.03)
    fig.text(0.01, -0.06, 'Solid: --m_warmup 60,120. Dashed (reg.): the registered ../no-logmel run (fixed m = 0.4). '
             'Shaded: student margin 0 (dark), ramping to 0.4 (light). One line per seed.',
             fontsize=7.5, color=INK_2)
    save(fig, 'fig1_probe_curves.png')


# ---------------------------------------------------------------- fig2: train accuracy at m = 0.4
def gap_break(e, a, gap=10):
    """NaN between EPOCH lines more than `gap` epochs apart: no line drawn where nothing was logged."""
    E, A = [], []
    for i, (x, y) in enumerate(zip(e, a)):
        if i and x - e[i - 1] > gap:
            E.append(np.nan)
            A.append(np.nan)
        E.append(x)
        A.append(y)
    return E, A


def fig2():
    arms = [a for a in ARMS if any(table(True, a[0], s, 'curve_mw.csv') for s in SEEDS)]
    if not arms:
        print('skip fig2: no mw_* curve_mw.csv in scores/')
        return
    fig, axes = plt.subplots(1, len(arms), figsize=(4.2 * len(arms), 3.6), sharey=True, squeeze=False)
    trapped = []
    for ax, (arm, label, col) in zip(axes[0], arms):
        schedule_bands(ax)
        for s in SEEDS:
            r = table(False, arm, s, 'curve.csv')       # registered: Train_accuracy is already at m = 0.4
            if r:
                ax.plot(*gap_break([int(x['epoch']) for x in r], [float(x['train_acc']) for x in r]), color=col,
                        lw=1.2, alpha=0.75, ls=(0, (3, 1.5)))
            r = table(True, arm, s, 'curve_mw.csv')
            if r:
                e = [int(x['epoch']) for x in r]
                a = [float(x['train_acc_mfinal']) for x in r]
                ax.plot(*gap_break(e, a), color=col, lw=1.5, marker='.', ms=2.5, zorder=3)
                late = [v for ep, v in zip(e, a) if ep >= RAMP_END]
                if late and max(late) == 0:
                    trapped.append(label)
        ep = np.linspace(0, 300, 601)     # the margin shares the axis: both run 0 to 0.4
        ax.plot(ep, np.clip((ep - HOLD) / (RAMP_END - HOLD), 0, 1) * M_FINAL, color=MUTED, lw=1.1, ls=':')
        ax.annotate('student m', (HOLD + 0.5 * (RAMP_END - HOLD), M_FINAL / 2), xytext=(6, -2),
                    textcoords='offset points', fontsize=7, color=INK_2, ha='left', va='top')
        ax.set_title(label, loc='left', fontsize=9.5, color=col if col != INK_2 else INK)
        ax.set_xlim(0, 300)
        ax.set_xlabel('epoch (printed only when valid loss improves)')
        ax.grid(axis='y')
    axes[0][0].set_ylabel('train accuracy at m = 0.4 (mixup batches); margin m')
    t = ('Warm-up runs still collapse to 0 margin accuracy: ' + ', '.join(trapped)) if trapped else \
        'Train accuracy measured at the final margin: registered runs hit 0 at ~epoch 55'
    fig.suptitle(t, x=0.01, ha='left', fontsize=11, y=1.03)
    fig.text(0.01, -0.07, 'Solid: warm-up run, its logged Train_acc_mfinal. Dashed: registered run\'s Train_accuracy '
             '(margin 0.4). Both count cos(θ_y + 0.4) > every other cos. The ~0.4 ceiling is mixup. Gaps: no line logged for > 10 epochs.',
             fontsize=7.5, color=INK_2)
    save(fig, 'fig2_train_margin_acc.png')


# ---------------------------------------------------------------- fig3: geometry (diag_margin)
DIAG = re.compile(r'^(SCLTFSTgramMFN_\S+?)(?:@(e\d+):(\w+))?\s+acc\(no margin\) ([\d.]+)\s+acc\(margin [\d.]+\) '
                  r'([\d.]+)\s+cos tgt ([+-][\d.]+)')


def diag_rows():
    files = glob.glob(os.path.join(SCORES, 'diag_margin_*.txt'))
    for base in EXTRA + [os.path.join(HERE, '..', 'no-logmel', 'scores')]:
        files += glob.glob(os.path.join(base, 'diag_margin_*.txt'))
    out = {}
    for f in sorted(set(os.path.realpath(p) for p in files)):
        for line in open(f):
            m = DIAG.match(line.strip())
            if m:   # later files (by name) overwrite earlier rows of the same target
                run, ep, who, acc, marg, cos = m.groups()
                out[(run, ep or 'final', who or 'ema_net')] = (float(acc), float(marg), float(cos))
    return out


def fig3():
    d = diag_rows()
    mw = {k: v for k, v in d.items() if '_mw_' in k[0]}
    if not mw:
        print('skip fig3: no diag_margin rows for mw_* runs')
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.axvspan(0, 1.05, ymin=0.5 / 1.1 + 0.05 / 1.1, color=BAND, lw=0, zorder=0)
    ax.text(0.03, 0.52, 'pass region', ha='left', va='bottom', fontsize=7.5, color=INK_2)
    colour = {a: c for a, _, c in ARMS}
    labelled = set()
    for (run, ep, who), (acc, marg, cos) in sorted(d.items()):
        arm = re.search(r'_(f\d)_', run).group(1)
        col = colour.get(arm, MUTED)
        fix = '_mw_' in run
        if not fix and arm not in ('f6', 'f7'):
            continue
        mk = {'e60': 's', 'e100': 'D', 'final': 'o'}[ep]
        ax.scatter(cos, marg, s=44 if who == 'ema_net' else 30, marker=mk,
                   color=col if fix else SURFACE, edgecolor=col, linewidth=1.3, zorder=3)
        # Label only the points that stand apart; the e60 / EMA points pile up near the
        # origin and the footnote explains their markers.
        lab_col = col if col != GREEN else INK_2
        key = (arm, ep, who, fix)
        if key in labelled:
            continue
        labelled.add(key)
        if fix and ep == 'final':
            dy = {'f2': 6, 'f6': -2, 'f7': -10}.get(arm, 0)
            ax.annotate(f'{arm} final', (cos, marg), xytext=(-8, dy), ha='right',
                        textcoords='offset points', fontsize=6.5, color=lab_col)
        elif fix and ep == 'e100' and who != 'ema_net':
            dx, dy, ha = {'f2': (6, 4, 'left'), 'f6': (6, -10, 'left'), 'f7': (-6, 8, 'right')}.get(arm, (6, 0, 'left'))
            ax.annotate(f'{arm} e100 student', (cos, marg), xytext=(dx, dy), ha=ha,
                        textcoords='offset points', fontsize=6.5, color=lab_col)
        elif not fix:
            dy = {'f6': 6, 'f7': -12}.get(arm, 0)
            ax.annotate(f'registered {arm}', (cos, marg), xytext=(0, dy), ha='center',
                        textcoords='offset points', fontsize=6.5, color=lab_col)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('mean cos(embedding, own class weight)')
    ax.set_ylabel('train top-1 with margin 0.4 (1024 clips)')
    ax.grid()
    fin = {k[0]: v for k, v in mw.items() if k[1] == 'final'}
    passed = [r for r, v in fin.items() if v[1] >= 0.5 and v[2] > 0]
    t = (f'{len(passed)} of {len(fin)} warm-up runs end outside the margin trap' if fin
         else 'Margin-trap geometry during the warm-up')
    ax.set_title(t, loc='left', fontsize=11)
    fig.text(0.01, -0.04, 'Filled: warm-up runs (square e60 end of hold, diamond e100 mid-ramp, circle final '
             'best-EMA; small = student). Hollow: registered nl_f6/nl_f7 (../no-logmel diag).',
             fontsize=7.5, color=INK_2)
    save(fig, 'fig3_geometry.png')


# ---------------------------------------------------------------- fig4: AUC per seed
def fig4():
    have = [a for a in ARMS if values(True, a[0], 'dev')]
    if not have:
        print('skip fig4: no mw_* eval CSVs')
        return
    sets = [ds for ds in ('dev', 'eval') if any(values(True, a[0], ds) for a in have)]
    fig, axes = plt.subplots(1, len(sets), figsize=(4.9 * len(sets), 3.9), squeeze=False)
    rng = np.random.default_rng(0)
    short = {'f6': 'TF', 'f7': 'T', 'f2': 'LM', 'f5': 'T+TF'}
    for ax, ds in zip(axes[0], sets):
        for i, (arm, label, col) in enumerate(have):
            for j, fix in enumerate((False, True)):
                v = values(fix, arm, ds)
                if not v:
                    continue
                x = i * 2.6 + j
                a = np.array(list(v.values()))
                ax.scatter(x + rng.uniform(-0.1, 0.1, len(a)), a, s=30, color=col if fix else SURFACE,
                           edgecolor=col, linewidth=1.3, zorder=3)
                ax.plot([x - 0.25, x + 0.25], [a.mean()] * 2, color=col, lw=2, zorder=2)
                ax.annotate(f'{a.mean():.1f}', (x + 0.28, a.mean()), va='center', fontsize=7.5, color=INK_2)
        ax.set_xticks([i * 2.6 + j for i in range(len(have)) for j in (0, 1)],
                      [f'{short[a[0]]}\n{"reg." if j == 0 else "warm-up"}'
                       for a in have for j in (0, 1)], fontsize=7.5)
        ax.grid(axis='y')
        ax.set_ylabel(f'{ds} AUC (%), eval.py R0, m = 0.4')
    parts = []
    for arm, label, _ in have:
        a, b = values(True, arm, 'dev'), values(False, arm, 'dev')
        if a and b:
            parts.append(f'{label.split(" (")[0]} {np.mean(list(a.values())) - np.mean(list(b.values())):+.1f}')
    fig.suptitle('Dev AUC, warm-up − registered: ' + ', '.join(parts) if parts else 'AUC per seed',
                 x=0.01, ha='left', fontsize=11, y=1.03)
    fig.text(0.01, -0.06, 'Hollow: registered ../no-logmel run (fixed m = 0.4). Filled: --m_warmup 60,120. '
             'Bar: mean over the seeds present. LM = Log-Mel only, the control.', fontsize=7.5, color=INK_2)
    save(fig, 'fig4_auc.png')


# ---------------------------------------------------------------- summary
def summary():
    print('\n== dev / eval AUC (eval.py R0), mean over seeds present')
    print(f'{"arm":24s} {"ds":4s} {"registered":>16s} {"warm-up":>16s} {"delta":>7s}')
    delta = {}
    for arm, label, _ in ARMS:
        for ds in ('dev', 'eval'):
            a, b = values(True, arm, ds), values(False, arm, ds)
            if not a and not b:
                continue
            ma = np.mean(list(a.values())) if a else np.nan
            mb = np.mean(list(b.values())) if b else np.nan
            delta[(arm, ds)] = ma - mb
            print(f'{label:24s} {ds:4s} {mb:10.2f} (n={len(b)}) {ma:10.2f} (n={len(a)}) {ma - mb:+7.2f}')
    for arm in ('f6', 'f7', 'f5'):
        for ds in ('dev', 'eval'):
            if (arm, ds) in delta and ('f2', ds) in delta:
                print(f'W({arm}) - W(f2) {ds}: {delta[(arm, ds)] - delta[("f2", ds)]:+.2f}  '
                      '(decision: >= +1.0 dev, > 0 eval)')
    d = diag_rows()
    fin = sorted((k[0], v) for k, v in d.items() if '_mw_' in k[0] and k[1] == 'final')
    if fin:
        print('\n== mechanism (final best-EMA; pass = acc(margin 0.4) >= 0.5 and cos tgt > 0)')
        for run, (acc, marg, cos) in fin:
            print(f'{run:34s} acc {acc:.3f} acc@0.4 {marg:.3f} cos tgt {cos:+.3f} -> '
                  f'{"PASS" if marg >= 0.5 and cos > 0 else ("TRAPPED" if marg < 0.05 or cos < 0 else "partial")}')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4):
        f()
    summary()
