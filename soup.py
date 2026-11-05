"""Average several checkpoints' weights into one network (a uniform "model soup").

    python soup.py --out SCLTFSTgramMFN_<name> check_points/<fork1> check_points/<fork2> ...

Averages each run's model.pth (the EMA net, what eval.py scores) element-wise,
BatchNorm running statistics included; integer buffers (num_batches_tracked)
are taken from the first run. Writes check_points/<out>/model.pth and a
config.json copied from the first run, with `soup_of` listing the members. The
result is a single network: same architecture, same inference cost.

Averaging only makes sense for runs that share a trunk (forks of one
state_e<N>.pth). Independent seeds sit in different basins and average to
noise; the script refuses unless the members' configs agree on the trunk
(same --resume_state), or --force is given.
"""
import argparse
import json
import os

import torch

ARCH_KEYS = ('fussion', 'ht', 'm', 'lean', 'mel_gate', 'eca', 'tf_act', 'tf_pool')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='checkpoint directories holding model.pth + config.json')
    ap.add_argument('--out', required=True, help='name of the new check_points/ directory')
    ap.add_argument('--force', action='store_true', help='average runs that are not forks of one trunk')
    a = ap.parse_args()

    cfgs = [json.load(open(os.path.join(r, 'config.json'), encoding='UTF-8')) for r in a.runs]
    for k in ARCH_KEYS:
        vals = {json.dumps(c.get(k)) for c in cfgs}
        if len(vals) > 1:
            raise SystemExit(f'members disagree on {k}: {vals}')
    trunks = {c.get('resume_state') for c in cfgs}
    if (len(trunks) != 1 or None in trunks) and not a.force:
        raise SystemExit(f'members are not forks of one trunk: resume_state = {trunks} (use --force)')

    states = [torch.load(os.path.join(r, 'model.pth'), map_location='cpu') for r in a.runs]
    soup = {}
    for k, v in states[0].items():
        if v.is_floating_point():
            soup[k] = torch.stack([s[k].float() for s in states]).mean(0).to(v.dtype)
        else:
            soup[k] = v.clone()

    out_dir = os.path.join('check_points', a.out)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(soup, os.path.join(out_dir, 'model.pth'))
    cfg = dict(cfgs[0])
    cfg.update(model_name=a.out, desc=a.out.split('_', 1)[-1], soup_of=[os.path.basename(r.rstrip('/')) for r in a.runs])
    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)
    print(f'soup of {len(states)} -> {out_dir}/model.pth')


if __name__ == '__main__':
    main()
