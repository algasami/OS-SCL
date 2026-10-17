#!/usr/bin/env python
"""G1: compare a beta=0 KD run's student_e10.pth with the baseline's, parameter by parameter.

    python ablation/distill/gate.py <gate_ckpt_dir> <baseline_ckpt_dir> <gate_log> <baseline_log> <out.json>

Bit-exact means max |delta| == 0 over every tensor in the state dict (BatchNorm buffers
included). G2 reads the it/s that tqdm printed in both logs over epochs 2-10.
"""
import json
import re
import sys

import numpy as np
import torch

gate_dir, base_dir, gate_log, base_log, out = sys.argv[1:6]
a = torch.load(f'{gate_dir}/student_e10.pth', map_location='cpu')
b = torch.load(f'{base_dir}/student_e10.pth', map_location='cpu')
assert a.keys() == b.keys(), 'state dict keys differ'
diffs = {k: float((a[k].double() - b[k].double()).abs().max()) for k in a if a[k].numel()}
worst = max(diffs, key=diffs.get)


def its(log):
    # inner-loop tqdm lines look like "123/539 [00:54<02:33,  2.27it/s]"; skip epoch 1 (worker spin-up)
    txt = open(log, errors='ignore').read().replace('\r', '\n')
    v = [float(x) for x in re.findall(r'\d+/539 \[[^\]]*?([\d.]+)it/s\]', txt)]
    return float(np.median(v[len(v) // 10:])) if v else None


res = {'G1_bit_exact': diffs[worst] == 0.0, 'G1_max_abs_diff': diffs[worst], 'G1_worst_tensor': worst,
       'G1_tensors_compared': len(diffs), 'G2_its_gate': its(gate_log), 'G2_its_baseline': its(base_log)}
if res['G2_its_gate'] and res['G2_its_baseline']:
    res['G2_slowdown'] = res['G2_its_baseline'] / res['G2_its_gate']
json.dump(res, open(out, 'w'), indent=2)
print(json.dumps(res, indent=2))
