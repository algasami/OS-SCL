"""Figures and the summary table for ablation/lit-arms.

Reads scores/ only: each la_<arm>_s<seed>/{eval_dev,eval_eval,probe}.csv here, and the
paired baseline fr_f1_s<seed> from ../fast-recipe/scores via scores_path. Written
before any 300-epoch score existed (README addendum); missing inputs skip a figure.

Run with /home/f74134118/anaconda3/bin/python (matplotlib lives in base).
Prints the summary that run.sh tees to scores/summary.txt.
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

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8983'
GRID = '#e4e3df'
BLUE = '#2a78d6'
ORANGE = '#eb6834'
GREEN = '#1baf7a'
MACH = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
SEEDS = (2040, 2041, 2042, 2043)
BASE = ('fr_f1', 'baseline fr_f1', INK_2)
ARMS = [('la_aug', 'aug (wave augmentation)', BLUE),
        ('la_gate', 'gate (TriStat)', ORANGE),
        ('la_eca', 'eca (channel attention)', GREEN)]
D1_DEV, D1_EVAL = 0.20, 0.0
RNG = np.random.default_rng(0)

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'text.color': INK, 'axes.labelcolor': INK_2, 'axes.edgecolor': GRID,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False,
})


def run_name(arm, seed):
    return f'SCLTFSTgramMFN_{arm}_s{seed}'


def rows(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding='UTF-8') as f:
        return list(csv.DictReader(f))


def eval_rows(arm, seed, ds):
    run = run_name(arm, seed)
    local = os.path.join(SCORES, run, f'eval_{ds}.csv')
    if os.path.exists(local):
        return rows(local)
    try:
        return rows(scores_csv(run, ds))
    except Exception:
        return None


def overall(arm, seed, ds, key='AUC'):
    r = eval_rows(arm, seed, ds)
    if not r:
        return None
    for x in r:
        if x['scope'] == 'overall':
            return 100 * float(x[key])
    return None


def machine(arm, seed, ds):
    r = eval_rows(arm, seed, ds)
    return {x['machine']: 100 * float(x['AUC']) for x in r if x['scope'] == 'machine'} if r else None


def by_seed(arm, ds, key='AUC'):
    """{seed: value} for the seeds that have a score."""
    out = {}
    for s in SEEDS:
        v = overall(arm, s, ds, key)
        if v is not None:
            out[s] = v
    return out


def contrast(arm, ds, key='AUC'):
    """Arm − baseline: mean diff, Welch se, bootstrap 95% interval, paired Δs, n."""
    a, b = by_seed(arm, ds, key), by_seed(BASE[0], ds, key)
    if len(a) < 2 or len(b) < 2:
        return None
    A, B = np.array(list(a.values())), np.array(list(b.values()))
    d = A.mean() - B.mean()
    se = np.sqrt(A.var(ddof=1) / len(A) + B.var(ddof=1) / len(B))
    boot = [RNG.choice(A, len(A)).mean() - RNG.choice(B, len(B)).mean() for _ in range(20000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    paired = {s: a[s] - b[s] for s in a if s in b}
    return dict(d=d, se=se, lo=lo, hi=hi, paired=paired, n=len(A), mean=A.mean(), base=B.mean())


def verdict(dev, ev):
    if dev is None or ev is None:
        return 'incomplete'
    if dev['n'] < 4 or ev['n'] < 4:
        return f'incomplete (n={min(dev["n"], ev["n"])})'
    if dev['d'] >= D1_DEV and ev['d'] >= D1_EVAL:
        return 'PASS' + (' (weak: dev interval includes 0)' if dev['lo'] <= 0 else '')
    return 'fail'


def save(fig, name):
    out = os.path.join(PLOTS, name)
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote', os.path.relpath(out, HERE))


# ---------------------------------------------------------------- fig1: per-seed AUC
def fig1_auc():
    series = [BASE] + ARMS
    have = {(a, ds): by_seed(a, ds) for a, _, _ in series for ds in ('dev', 'eval')}
    if not any(have[(a, 'dev')] for a, _, _ in ARMS):
        print('skip fig1: no la_* eval CSVs in scores/')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for ax, ds in zip(axes, ('dev', 'eval')):
        for x, (arm, label, col) in enumerate(series):
            v = list(have[(arm, ds)].values())
            if not v:
                continue
            jit = np.linspace(-0.12, 0.12, len(v))
            base = arm == BASE[0]
            ax.scatter(x + jit, v, s=40, color=SURFACE if base else col, edgecolor=col, lw=1.4, zorder=3)
            m = np.mean(v)
            ax.plot([x - 0.25, x + 0.25], [m, m], color=col, lw=2, zorder=2)
            ax.annotate(f'{m:.2f}', (x + 0.27, m), va='center', fontsize=8, color=INK)
        ax.set_xticks(range(len(series)), [f'{s[1].split(" (")[0]}\n(n={len(have[(s[0], ds)])})' for s in series])
        ax.set_xlim(-0.5, len(series) - 0.3)
        ax.grid(axis='y')
        ax.set_title(f'{ds} set', loc='left', fontsize=9.5)
    axes[0].set_ylabel("overall AUC (%), eval.py's score (R0)")
    fig.suptitle('Each arm against the fast-recipe paper model (bar = mean, dots = seeds 2040–2043)',
                 x=0.06, ha='left', fontsize=10.5)
    fig.text(0.06, -0.03, 'Hollow = baseline fr_f1 (../fast-recipe). Every arm uses the same seeds and recipe.',
             fontsize=7.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig1_auc.png')


# ---------------------------------------------------------------- fig2: contrasts vs the D1 bar
def fig2_contrasts():
    c = {(a, ds): contrast(a, ds) for a, _, _ in ARMS for ds in ('dev', 'eval')}
    if not any(c.values()):
        print('skip fig2: no arm has 2+ scored seeds')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.2), sharey=True)
    passed = []
    for ax, ds, bar in zip(axes, ('dev', 'eval'), (D1_DEV, D1_EVAL)):
        for y, (arm, label, col) in enumerate(reversed(ARMS)):
            r = c[(arm, ds)]
            if not r:
                continue
            ax.plot([r['lo'], r['hi']], [y, y], color=col, lw=1.2, zorder=2)
            ax.plot([r['d'] - r['se'], r['d'] + r['se']], [y, y], color=col, lw=4, alpha=0.45, zorder=2,
                    solid_capstyle='butt')
            ax.scatter([r['d']], [y], s=46, color=col, edgecolor=SURFACE, lw=1.2, zorder=3)
            ax.scatter(list(r['paired'].values()), [y + 0.22] * len(r['paired']), s=12, color=col, alpha=0.7,
                       marker='|', zorder=3)
            ax.annotate(f'{r["d"]:+.2f}', (r['hi'], y), xytext=(6, 0), textcoords='offset points',
                        va='center', fontsize=8, color=INK)
        ax.axvline(0, color=INK_2, lw=0.8)
        ax.axvline(bar, color=INK, lw=1.0, ls=(0, (4, 3)))
        ax.set_ylim(-0.5, len(ARMS) - 0.2)          # headroom above the top arm's paired ticks
        ax.annotate(f'D1 bar {bar:+.2f}', (bar, len(ARMS) - 0.3), xytext=(4, 0), textcoords='offset points',
                    va='center', fontsize=7.5, color=INK)
        ax.set_yticks(range(len(ARMS)), [a[1] for a in reversed(ARMS)])
        ax.grid(axis='x')
        ax.set_xlabel(f'{ds} AUC − fr_f1 (points)')
        ax.set_title(f'{ds} set', loc='left', fontsize=9.5)
    for arm, label, _ in ARMS:
        if verdict(c[(arm, 'dev')], c[(arm, 'eval')]).startswith('PASS'):
            passed.append(label.split(' (')[0])
    n = min((r['n'] for r in c.values() if r), default=0)
    t = (f'{len(passed)} of 3 arms pass D1' + (f': {", ".join(passed)}' if passed else '')) if n == 4 \
        else f'Arm − baseline so far (n={n}; D1 needs all 4 seeds)'
    fig.suptitle(t, x=0.06, ha='left', fontsize=10.5, y=1.04)
    fig.text(0.06, -0.08, 'Dot = mean difference; thick bar = ±1 Welch se; thin line = bootstrap 95%; '
             'ticks above = the four seed-paired Δs. Dashed = D1 bar (dev ≥ +0.20 and eval ≥ 0).',
             fontsize=7.5, color=INK_2)
    save(fig, 'fig2_contrasts.png')


# ---------------------------------------------------------------- fig3: per-machine Δ
def fig3_machines():
    deltas = {}
    for arm, _, _ in ARMS:
        for ds in ('dev', 'eval'):
            a = [machine(arm, s, ds) for s in SEEDS]
            b = [machine(BASE[0], s, ds) for s in SEEDS]
            a, b = [x for x in a if x], [x for x in b if x]
            if len(a) >= 2 and b:
                deltas[(arm, ds)] = [np.mean([x[m] for x in a]) - np.mean([x[m] for x in b]) for m in MACH]
    if not deltas:
        print('skip fig3: per-machine inputs missing')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), sharey=True)
    w = 0.26
    for ax, ds in zip(axes, ('dev', 'eval')):
        for k, (arm, label, col) in enumerate(ARMS):
            if (arm, ds) not in deltas:
                continue
            x = np.arange(len(MACH)) + (k - 1) * w
            ax.bar(x, deltas[(arm, ds)], width=w - 0.03, color=col, label=label, zorder=2)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_xticks(range(len(MACH)), MACH, fontsize=8)
        ax.grid(axis='y')
        ax.set_title(f'{ds} set', loc='left', fontsize=9.5)
    axes[0].set_ylabel('machine AUC, arm − fr_f1 (points, mean of seeds)')
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper right', ncol=3, fontsize=7.5, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle('Where each arm moves the score, by machine type', x=0.06, ha='left', fontsize=10.5)
    fig.text(0.06, -0.03, 'ToyConveyor seed noise is > 4 points (CLAUDE.md); a per-machine bar under ~1 point '
             'is not resolvable at n=4.', fontsize=7.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig3_machines.png')


# ---------------------------------------------------------------- fig4: probe curves
def probe(arm, seed):
    run = run_name(arm, seed)
    local = os.path.join(SCORES, run, 'probe.csv')
    if not os.path.exists(local):
        try:
            local = os.path.join(os.path.dirname(scores_csv(run, 'dev')), 'probe.csv')
        except Exception:
            return None
    r = rows(local)
    return [x for x in r if x.get('AUC')] if r else None


def fig4_probe():
    have = {a: [p for p in (probe(a, s) for s in SEEDS) if p] for a, _, _ in [BASE] + ARMS}
    if not any(have[a] for a, _, _ in ARMS):
        print('skip fig4: no la_* probe.csv')
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharey=True)
    for ax, (arm, label, col) in zip(axes, ARMS):
        for p in have[BASE[0]]:
            ax.plot([int(x['epoch']) + 1 for x in p], [100 * float(x['AUC']) for x in p], color=MUTED, lw=1.0,
                    ls=(0, (4, 2)), zorder=2)
        for p in have[arm]:
            ax.plot([int(x['epoch']) + 1 for x in p], [100 * float(x['AUC']) for x in p], color=col, lw=1.4,
                    zorder=3)
        ax.set_ylim(90, 96.6)
        ax.set_xlim(100, 305)
        ax.set_xlabel('epoch')
        ax.grid(axis='y')
        ax.set_title(f'{label} (n={len(have[arm])})', loc='left', fontsize=9.5, color=col if col != GREEN else INK)
    axes[0].set_ylabel('dev AUC (%) of the EMA net, probed every 5 epochs')
    fig.suptitle('Training curves: arm (solid) vs baseline fr_f1 (dashed grey)', x=0.06, ha='left', fontsize=10.5)
    fig.text(0.06, -0.03, 'Diagnostic only: the probe scores the dev test set and is never used for selection.',
             fontsize=7.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig4_probe.png')


# ---------------------------------------------------------------- summary
def summary():
    print('\n== R0 overall, arm − fr_f1 (mean diff, Welch se, bootstrap 95%, paired by seed)')
    for key in ('AUC', 'pAUC'):
        for arm, label, _ in ARMS:
            for ds in ('dev', 'eval'):
                r = contrast(arm, ds, key)
                if not r:
                    print(f'{key:4s} {label:28s} {ds:4s} -- not enough seeds')
                    continue
                pd = ' '.join(f'{v:+.2f}' for v in r['paired'].values())
                print(f'{key:4s} {label:28s} {ds:4s} {r["mean"]:.2f} vs {r["base"]:.2f}  '
                      f'{r["d"]:+.2f} (se {r["se"]:.2f}) [{r["lo"]:+.2f}, {r["hi"]:+.2f}]  paired {pd}  n={r["n"]}')
    print('\n== D1 (dev ≥ +0.20 and eval ≥ 0, AUC, 4-seed means)')
    for arm, label, _ in ARMS:
        print(f'{label:28s} {verdict(contrast(arm, "dev"), contrast(arm, "eval"))}')


# ---------------------------------------------------------------- D2: aug on fresh seeds
REP_SEEDS = (2044, 2045, 2046, 2047)
R1_DEV, R2_EVAL = 0.20, 0.0


def seeds_of(arm, seeds, ds, key='AUC'):
    return [v for v in (overall(arm, s, ds, key) for s in seeds) if v is not None]


def boot_ci(a, b, n=20000):
    a, b = np.array(a), np.array(b)
    d = [RNG.choice(a, len(a)).mean() - RNG.choice(b, len(b)).mean() for _ in range(n)]
    return np.percentile(d, [2.5, 97.5])


def fig5_replication():
    base = {ds: seeds_of(BASE[0], SEEDS, ds) for ds in ('dev', 'eval')}
    st1 = {ds: seeds_of('la_aug', SEEDS, ds) for ds in ('dev', 'eval')}
    rep = {ds: seeds_of('la_aug', REP_SEEDS, ds) for ds in ('dev', 'eval')}
    if any(len(v[ds]) < 4 for v in (base, st1, rep) for ds in ('dev', 'eval')):
        print('skip fig5_replication: D2 scores incomplete')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    groups = [('baseline fr_f1\nseeds 2040–43', base, INK_2, True),
              ('aug, stage 1\nseeds 2040–43', st1, BLUE, False),
              ('aug, replication\nseeds 2044–47', rep, BLUE, False)]
    for ax, ds in zip(axes, ('dev', 'eval')):
        b = np.mean(base[ds])
        bar, name = (b + R1_DEV, 'R1 bar (base +0.20)') if ds == 'dev' else (b + R2_EVAL, 'R2 bar (base +0.00)')
        ax.axhline(bar, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.annotate(name, (2.55, bar), va='bottom', ha='right', fontsize=7.5, color=INK,
                    xytext=(0, 2), textcoords='offset points')
        if ds == 'dev':
            ax.axhline(95.71, color=MUTED, lw=0.9, ls=(0, (1, 2)), zorder=1)
            ax.annotate('paper 95.71', (-0.45, 95.71), va='bottom', ha='left', fontsize=7.5, color=INK_2,
                        xytext=(0, 2), textcoords='offset points')
        for x, (label, g, col, hollow) in enumerate(groups):
            v = np.array(g[ds])
            jit = np.linspace(-0.12, 0.12, len(v))
            ax.scatter(x + jit, v, s=40, color=SURFACE if hollow else col, edgecolor=col, lw=1.4, zorder=3)
            m = v.mean()
            ax.plot([x - 0.25, x + 0.25], [m, m], color=col, lw=2, zorder=2)
            txt = f'{m:.2f}' if hollow else f'{m:.2f} ({m - b:+.2f})'
            # the base mean sits on the R2 line (base +0.00): label it just below
            ax.annotate(txt, (x + 0.3, m), va='top' if hollow else 'center', fontsize=8, color=INK,
                        xytext=(0, -3 if hollow else 0), textcoords='offset points',
                        bbox=dict(boxstyle='square,pad=0.1', fc=SURFACE, ec='none'), zorder=4)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g[0] for g in groups])
        ax.set_xlim(-0.5, 2.6)
        ax.grid(axis='y')
        ax.set_title(f'{ds} set', loc='left', fontsize=9.5)
    axes[0].set_ylabel("overall AUC (%), eval.py's score (R0)")
    d_dev = np.mean(rep['dev']) - np.mean(base['dev'])
    d_ev = np.mean(rep['eval']) - np.mean(base['eval'])
    fig.suptitle(f"Fresh seeds repeat aug's dev gain ({d_dev:+.2f}) but not its eval gain ({d_ev:+.2f})",
                 x=0.06, ha='left', fontsize=10.5)
    fig.text(0.06, -0.04, 'Bar = mean of 4 seeds; labels give mean and Δ vs fr_f1. Dashed = the pre-registered '
             'replication bars (R1 dev Δ ≥ +0.20, R2 eval Δ ≥ 0).', fontsize=7.5, color=INK_2)
    fig.tight_layout()
    save(fig, 'fig5_replication.png')


def summary_d2():
    b_dev, b_ev = seeds_of(BASE[0], SEEDS, 'dev'), seeds_of(BASE[0], SEEDS, 'eval')
    r_dev, r_ev = seeds_of('la_aug', REP_SEEDS, 'dev'), seeds_of('la_aug', REP_SEEDS, 'eval')
    if min(len(b_dev), len(b_ev), len(r_dev), len(r_ev)) < 4:
        print('\n== D2: replication incomplete')
        return
    print('\n== D2: aug on fresh seeds 2044–2047 vs fr_f1 2040–2043 (R0, unpaired)')
    for ds in ('dev', 'eval'):
        for key in ('AUC', 'pAUC'):
            b = seeds_of(BASE[0], SEEDS, ds, key)
            r = seeds_of('la_aug', REP_SEEDS, ds, key)
            p = seeds_of('la_aug', SEEDS, ds, key) + r
            se = np.sqrt(np.var(r, ddof=1) / len(r) + np.var(b, ddof=1) / len(b))
            lo, hi = boot_ci(r, b)
            plo, phi = boot_ci(p, b)
            print(f'{key:4s} {ds:4s} rep {np.mean(r):.2f} vs {np.mean(b):.2f}  {np.mean(r) - np.mean(b):+.2f} '
                  f'(se {se:.2f}) [{lo:+.2f}, {hi:+.2f}]  seeds {" ".join(f"{x:.2f}" for x in r)}  |  '
                  f'pooled n=8 {np.mean(p) - np.mean(b):+.2f} [{plo:+.2f}, {phi:+.2f}]')
    p_dev = seeds_of('la_aug', SEEDS, 'dev') + r_dev
    r1 = np.mean(r_dev) - np.mean(b_dev) >= R1_DEV
    r2 = np.mean(r_ev) - np.mean(b_ev) >= R2_EVAL
    r3 = boot_ci(p_dev, b_dev)[0] > 0
    print(f'R1 dev Δ ≥ +0.20: {"holds" if r1 else "fails"}   R2 eval Δ ≥ 0: {"holds" if r2 else "fails"}   '
          f'R3 pooled dev interval excludes 0: {"holds" if r3 else "fails"}')
    print(f'adopt aug (R1 and R2): {"YES" if r1 and r2 else "NO"}')


if __name__ == '__main__':
    fig1_auc()
    fig2_contrasts()
    fig3_machines()
    fig4_probe()
    fig5_replication()
    summary()
    summary_d2()
