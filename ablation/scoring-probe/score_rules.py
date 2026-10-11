#!/usr/bin/env python
"""Score the dumped cosines under R0-R4 and write the committed record.

    python ablation/scoring-probe/score_rules.py ablation/scoring-probe/results/cosines

Reads every <run>_<dev|eval>.npz in the cosine directory. Writes to scores/:

  rule_metrics.csv   per run x dataset x rule: id / machine / overall rows,
                     eval.py's metrics (AUC, pAUC pooled per machine; mAUC = min
                     per-ID AUC) plus idAUC, the mean of per-ID AUCs
  cross_type.csv     per run x dataset x machine x normal/anomaly: how much of
                     R0's softmax mass, and how often the nearest rival centre,
                     belongs to another machine type
  reproduction.json  R0 against eval.py: the stored float32 score, and the
                     run's committed eval_<dataset>.csv

All rules are computed in float64. A confident clip's CE is
log(1 + sum exp(z_j - z_y)) with sum ~ e^-30, which float32 rounds to exactly 0
and turns into ties; log1p in float64 keeps the ordering.
"""
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np
from sklearn import metrics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from scores_path import scores_csv    # noqa: E402

SCORES = os.path.join(HERE, 'scores')
NAME_LIST = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
N_IDS = {'fan': 7, 'pump': 7, 'slider': 7, 'ToyCar': 7, 'ToyConveyor': 6, 'valve': 7}
S = 30.0   # ArcMarginProduct scale (model/net.py default)

# class index -> machine-type index, from the same ordering dataloader.py uses
TYPE_OF = np.concatenate([np.full(N_IDS[m], i) for i, m in enumerate(NAME_LIST)])
assert len(TYPE_OF) == 41

RULES = {
    'R0': 'CE, 41 classes, margin on (eval.py)',
    'R1': 'CE, 41 classes, margin off',
    'R2': 'CE, own machine type, margin on',
    'R3': 'CE, own machine type, margin off',
    'R4': '1 - cos(own centre)',
}


def margin_logit(cos_y, m):
    """ArcMarginProduct's target logit (easy_margin=False), in float64."""
    th, mm = math.cos(math.pi - m), math.sin(math.pi - m) * m
    sine = np.sqrt(np.clip(1.0 - cos_y ** 2, 0.0, None))
    phi = cos_y * math.cos(m) - sine * math.sin(m)
    return S * np.where(cos_y - th > 0, phi, cos_y - mm)


def ce(z_y, z_rivals, rival_mask):
    """log(1 + sum_j exp(z_j - z_y)) over the masked rivals, stably."""
    t = np.where(rival_mask, z_rivals - z_y[:, None], -np.inf)
    top = t.max(axis=1)
    small = top <= 0
    out = np.empty_like(z_y)
    out[small] = np.log1p(np.exp(t[small]).sum(axis=1))
    big = ~small
    tb = top[big]
    out[big] = tb + np.log(np.exp(-tb) + np.exp(t[big] - tb[:, None]).sum(axis=1))
    return out


def rule_scores(d, m):
    cos = d['cos'].astype(np.float64)
    y_cls = d['label']
    n = len(y_cls)
    cos_y = cos[np.arange(n), y_cls]
    z = S * cos
    z_plain, z_marg = S * cos_y, margin_logit(cos_y, m)

    not_self = np.ones_like(cos, dtype=bool)
    not_self[np.arange(n), y_cls] = False
    own_type = TYPE_OF[None, :] == TYPE_OF[y_cls][:, None]

    return {
        'R0': ce(z_marg, z, not_self),
        'R1': ce(z_plain, z, not_self),
        'R2': ce(z_marg, z, not_self & own_type),
        'R3': ce(z_plain, z, not_self & own_type),
        'R4': 1.0 - cos_y,
    }, cos, z, z_marg, not_self, own_type


def auc_pauc(y, s):
    return metrics.roc_auc_score(y, s), metrics.roc_auc_score(y, s, max_fpr=0.1)


def evaluate(d, score):
    """eval.py's metric set, plus idAUC (mean per-ID AUC). Returns csv rows."""
    rows, mach_rows = [], []
    for machine in NAME_LIST:
        sel = d['machine'] == machine
        ids = sorted(set(d['id'][sel]))
        id_auc = {}
        for i in ids:
            s2 = sel & (d['id'] == i)
            a, p = auc_pauc(d['y'][s2], score[s2])
            id_auc[i] = a
            rows.append((machine, i, 'id', a, p, '', ''))
        a, p = auc_pauc(d['y'][sel], score[sel])
        mach = (machine, 'ALL', 'machine', a, p, min(id_auc.values()), float(np.mean(list(id_auc.values()))))
        rows.append(mach)
        mach_rows.append(mach)
    rows.append(('ALL', 'ALL', 'overall', *[float(np.mean([r[k] for r in mach_rows])) for k in (3, 4, 5, 6)]))
    return rows


def logsumexp(x, mask):
    x = np.where(mask, x, -np.inf)
    top = x.max(axis=1, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    with np.errstate(divide='ignore'):
        return (top + np.log(np.exp(x - top).sum(axis=1, keepdims=True)))[:, 0]


def cross_type(d, cos, z, z_marg, not_self, own_type):
    """R0's softmax mass on other machine types, and where the nearest rival lives."""
    n = len(d['label'])
    z_full = z.copy()
    z_full[np.arange(n), d['label']] = z_marg
    log_p_cross = logsumexp(z_full, ~own_type) - logsumexp(z_full, np.ones_like(own_type))
    p_cross = np.exp(log_p_cross)
    rival = np.where(not_self, cos, -np.inf).argmax(axis=1)
    rival_foreign = TYPE_OF[rival] != TYPE_OF[d['label']]
    rows = []
    for machine in NAME_LIST:
        for y, cond in ((0, 'normal'), (1, 'anomaly')):
            sel = (d['machine'] == machine) & (d['y'] == y)
            pc = p_cross[sel]
            rows.append((machine, cond, int(sel.sum()), float(np.median(pc)), float(np.quantile(pc, 0.9)),
                         float((pc > 0.01).mean()), float(rival_foreign[sel].mean())))
    return rows


def committed_r0(run, ds):
    """{(machine, id): AUC} from the run's committed eval CSV, or None."""
    try:
        path = scores_csv(run, ds)
    except FileNotFoundError:
        return None
    if not os.path.exists(path):
        return None
    with open(path, encoding='UTF-8') as f:
        return {(r['machine'], r['id']): float(r['AUC']) for r in csv.DictReader(f) if r['scope'] == 'id'}


def main():
    cos_dir = sys.argv[1]
    os.makedirs(SCORES, exist_ok=True)
    metric_rows, cross_rows, repro = [], [], {}

    for path in sorted(glob.glob(os.path.join(cos_dir, '*.npz'))):
        run, ds = re.match(r'(.+)_(dev|eval)\.npz$', os.path.basename(path)).groups()
        d = dict(np.load(path))
        with open(os.path.join(os.path.dirname(scores_csv(run, ds)), 'config.json'), encoding='UTF-8') as f:
            m = json.load(f)['m']
        scores, cos, z, z_marg, not_self, own_type = rule_scores(d, m)

        for rule, score in scores.items():
            for row in evaluate(d, score):
                metric_rows.append((run, ds, rule, *row))
        for row in cross_type(d, cos, z, z_marg, not_self, own_type):
            cross_rows.append((run, ds, *row))

        # R0 must be eval.py's score: same per-ID ranking as the stored float32 CE,
        # and the same per-ID AUCs as the committed CSV.
        mine = {(r[0], r[1]): r[3] for r in evaluate(d, scores['R0']) if r[2] == 'id'}
        f32 = {(r[0], r[1]): r[3] for r in evaluate(d, d['ce_eval'].astype(np.float64)) if r[2] == 'id'}
        ref = committed_r0(run, ds)
        repro[f'{run}/{ds}'] = {
            'n_clips': int(len(d['y'])),
            'max_abs_id_auc_diff_vs_float32_score': max(abs(mine[k] - f32[k]) for k in mine),
            'max_abs_id_auc_diff_vs_committed_csv': (max(abs(mine[k] - ref[k]) for k in mine)
                                                     if ref else None),
            'max_abs_score_diff_vs_float32': float(np.abs(scores['R0'] - d['ce_eval']).max()),
            # float32 CE collapses confident clips to 0 / a few ulps: ties R0 in float64 breaks
            'frac_float32_score_zero': {c: float((d['ce_eval'][d['y'] == y] == 0).mean())
                                        for y, c in ((0, 'normal'), (1, 'anomaly'))},
            'frac_distinct_float32': float(len(np.unique(d['ce_eval'])) / len(d['ce_eval'])),
        }
        print(f'{run} {ds}: {repro[f"{run}/{ds}"]}')

    with open(os.path.join(SCORES, 'rule_metrics.csv'), 'w', newline='', encoding='UTF-8') as f:
        w = csv.writer(f)
        w.writerow(['model', 'dataset', 'rule', 'machine', 'id', 'scope', 'AUC', 'pAUC', 'mAUC', 'idAUC'])
        for r in metric_rows:
            w.writerow([v if isinstance(v, str) else f'{v:.5f}' for v in r])
    with open(os.path.join(SCORES, 'cross_type.csv'), 'w', newline='', encoding='UTF-8') as f:
        w = csv.writer(f)
        w.writerow(['model', 'dataset', 'machine', 'condition', 'n', 'p_cross_median', 'p_cross_q90',
                    'frac_p_cross_gt_1pct', 'frac_nearest_rival_foreign'])
        for r in cross_rows:
            w.writerow([v if isinstance(v, (str, int)) else f'{v:.6g}' for v in r])
    with open(os.path.join(SCORES, 'reproduction.json'), 'w', encoding='UTF-8') as f:
        json.dump({'rules': RULES, 'runs': repro}, f, indent=2)
    print('wrote scores/rule_metrics.csv, cross_type.csv, reproduction.json')


if __name__ == '__main__':
    main()
