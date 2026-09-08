"""Plot this repo's evaluation matrix against the values reported in the paper.

Reads an eval CSV written by `eval.py --csv` and renders two figures into figures/:
    auc_pauc_by_machine.<png|pdf>   per-machine AUC and pAUC, reported vs. this run
    averages.<png|pdf>              average AUC / pAUC / mAUC

Paper values are Tables II and IV of arXiv:2509.13853 (development set).
Needs matplotlib, which is not in the `os-scl` env:
    /home/f74134118/anaconda3/bin/python plot_comparison.py
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAME_LIST = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']

# arXiv:2509.13853, Table II (per machine) and Table IV (mAUC), development set.
PAPER = {
    'fan': (98.41, 94.50), 'pump': (96.25, 88.74), 'slider': (99.63, 98.04),
    'ToyCar': (96.55, 91.17), 'ToyConveyor': (83.56, 69.87), 'valve': (99.85, 99.23),
}
PAPER_AVG = {'AUC': 95.71, 'pAUC': 90.23, 'mAUC': 91.23}
PAPER_LOGMEL = {'AUC': 94.64, 'pAUC': 88.42, 'mAUC': 89.24}   # Log-Mel only ablation

C_PAPER, C_OURS, C_ABL = 'C0', 'C1', 'C2'


def read_eval_csv(path):
    """-> ({machine: (AUC, pAUC)}, {'AUC': .., 'pAUC': .., 'mAUC': ..}), in percent."""
    per_machine, avg = {}, {}
    with open(path, newline='', encoding='UTF-8') as f:
        for r in csv.DictReader(f):
            if r['scope'] == 'machine':
                per_machine[r['machine']] = (float(r['AUC']) * 100, float(r['pAUC']) * 100)
            elif r['scope'] == 'overall':
                avg = {'AUC': float(r['AUC']) * 100,
                       'pAUC': float(r['pAUC']) * 100,
                       'mAUC': float(r['mAUC']) * 100}
    return per_machine, avg


def style(ax, xlim, ticks, labels):
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.grid(axis='x')
    ax.set_axisbelow(True)


def caption(fig, text):
    fig.text(0.5, -0.02, text, ha='center', va='top', fontsize=8, color='dimgray')


def figure_machines(ours, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, i, title in zip(axes, (0, 1), ('AUC (%)', 'pAUC (%)')):
        for y, m in enumerate(NAME_LIST):
            # offset the pair so coinciding values stay visible
            paper, mine = PAPER[m][i], ours[m][i]
            ax.plot([paper, mine], [y - 0.13, y + 0.13], color='lightgray', lw=2, zorder=1)
            ax.scatter(paper, y - 0.13, s=45, color=C_PAPER, zorder=3,
                       label='Paper' if y == 0 else None)
            ax.scatter(mine, y + 0.13, s=45, color=C_OURS, zorder=3,
                       label='This run' if y == 0 else None)
        style(ax, (65, 101), [70, 80, 90, 100], NAME_LIST)
        ax.set_title(title)
    axes[0].legend(loc='lower left')
    fig.suptitle('Per-machine AUC and pAUC: paper vs. this run')
    fig.tight_layout()
    caption(fig, 'DCASE 2020 Task 2, development set. Paper values: arXiv:2509.13853, Table II.')
    save(fig, out)


def figure_averages(avg, out):
    metrics = ['AUC', 'pAUC', 'mAUC']
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for y, k in enumerate(metrics):
        vals = [PAPER_AVG[k], PAPER_LOGMEL[k], avg[k]]
        ax.plot([min(vals), max(vals)], [y, y], color='lightgray', lw=2, zorder=1)
        # the ablation is a ring, so this run landing on it stays visible
        ax.scatter(PAPER_LOGMEL[k], y, s=160, facecolor='none', edgecolor=C_ABL, lw=1.8,
                   zorder=2, label='Paper, Log-Mel only' if y == 0 else None)
        ax.scatter(PAPER_AVG[k], y, s=45, color=C_PAPER, zorder=3,
                   label='Paper' if y == 0 else None)
        ax.scatter(avg[k], y, s=45, color=C_OURS, zorder=3,
                   label='This run' if y == 0 else None)
    style(ax, (88, 96.5), [88, 90, 92, 94, 96], metrics)
    ax.set_xlabel('%')
    ax.legend(loc='lower right')
    ax.set_title('Average AUC, pAUC and mAUC')
    fig.tight_layout()
    caption(fig, 'DCASE 2020 Task 2, development set. Paper values: arXiv:2509.13853, '
                 'Tables II and IV.\nmAUC is the mean over machine types of the minimum '
                 'per-ID AUC.')
    save(fig, out)


def save(fig, out):
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}.png / {out}.pdf')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', default='check_points/SCLTFSTgramMFN_test/eval_dev.csv',
                   help='eval matrix written by eval.py --csv')
    p.add_argument('--out_dir', default='figures')
    args = p.parse_args()

    ours, avg = read_eval_csv(args.csv)
    os.makedirs(args.out_dir, exist_ok=True)
    figure_machines(ours, os.path.join(args.out_dir, 'auc_pauc_by_machine'))
    figure_averages(avg, os.path.join(args.out_dir, 'averages'))


if __name__ == '__main__':
    main()
