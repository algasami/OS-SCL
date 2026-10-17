import random
import torch
import numpy as np
from dataloader import train_dataset
from torch.utils.data import DataLoader
from utils import dataset_split
from trainer import Trainer
import yaml
import os
import json
from model.net import SCLTFSTgramMFN, freeze_tfgram
import argparse


def load_config(file_path):
    with open(file_path, 'r', encoding='UTF-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def save_config(cfg, file_path):
    with open(file_path, 'w', encoding='UTF-8') as f:
        yaml.dump(cfg, f)


def update_config(cfg, args):
    for key, value in vars(args).items():
        if value is not None:
            cfg[key] = value
    return cfg


def parse_arguments():
    parser = argparse.ArgumentParser(description='Modify YAML configuration.')
    parser.add_argument('--desc', type=str, default=None, help='Description to append to model name')
    parser.add_argument('--m', type=float, help='m value')
    parser.add_argument('--gpu_num', type=int, help='GPU number')
    parser.add_argument('--epoch', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--fussion', type=int, help='Fussion')
    parser.add_argument('--ht', type=str, help=' basic or leaky_relu')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--num_workers', type=int, help='DataLoader workers (0 = original behaviour)')
    parser.add_argument('--tf_act', type=str, choices=['relu', 'leaky', 'linear'],
                        help="TFgram nonlinearity: 'relu' (paper), 'leaky' (LeakyReLU 0.2, as in Tgram) "
                             "or 'linear' (identity)")
    parser.add_argument('--tf_pool', type=str, choices=['max', 'avg'],
                        help="TFgram downsampling: 'max' (paper) or 'avg' (linear, sign-preserving)")
    parser.add_argument('--freeze_tfgram', type=str, choices=['none', 'strict', 'bn_adapt'],
                        help="Freeze the TFgram branch at its random init: 'strict' also pins its "
                             "BatchNorm to eval(), 'bn_adapt' lets the running stats track the data")
    parser.add_argument('--student_snapshots', type=str,
                        help="Comma-separated epoch counts, e.g. '10,30,90,300': also save the student "
                             "net (not the EMA) as student_e<N>.pth after that many epochs")
    parser.add_argument('--kd_teachers', type=str,
                        help="Comma-separated model names to distil from (ablation/distill); each is "
                             "loaded from <kd_teacher_dir>/<name>/{model.pth,config.json}")
    parser.add_argument('--amp', type=str, choices=['none', 'fp16', 'bf16'],
                        help="Autocast precision for branches + MobileFaceNet (default none = fp32)")
    parser.add_argument('--cudnn_bench', type=int, choices=[0, 1],
                        help='1 = cudnn.benchmark on, deterministic off (faster, not bit-reproducible)')
    parser.add_argument('--lean', type=int, choices=[0, 1],
                        help='1 = build only the branches the fussion mode uses, no dead TFgram modules')
    parser.add_argument('--cache_dir', type=str, help='Read train/devtest arrays from cache_data.py output')
    parser.add_argument('--probe_every', type=int,
                        help='Log dev-test R0 metrics of the EMA net every N epochs (diagnostic only)')
    parser.add_argument('--kd_teacher_dir', type=str, help="Directory holding the teachers (default check_points)")
    parser.add_argument('--kd_beta', type=float, help="Weight of the KD term (default 1.0)")
    parser.add_argument('--kd_tau', type=float, help="KD temperature on 30*cos (default 2.0)")

    return parser.parse_args()


def main():
    args = parse_arguments()
    cfg = load_config('config.yaml')
    print('Original Configuration...')
    print(cfg)

    cfg = update_config(cfg, args)
    print('Updated Configuration...')
    print(cfg)

    random_seed(seed=cfg.get('seed', 2024))
    if cfg.get('cudnn_bench'):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    if cfg.get('probe_every') and not cfg.get('cache_dir'):
        raise SystemExit('--probe_every needs --cache_dir (it reads devtest_*.npy)')

    name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']

    root_path = 'data/dataset'

    device = torch.device(f'cuda:{cfg["gpu_num"]}')

    print('training dataset loading...')
    dataset = train_dataset(root_path, name_list, cfg)

    train_ds, valid_ds = dataset_split(dataset)

    nw = cfg.get('num_workers', 0)
    train_dataloader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=nw,
                                  persistent_workers=nw > 0)
    valid_dataloader = DataLoader(valid_ds, batch_size=cfg['batch_size'], num_workers=nw,
                                  persistent_workers=nw > 0)

    model = SCLTFSTgramMFN(num_classes=41, m=cfg['m'], cfg=cfg).to(device)

    model_ema = SCLTFSTgramMFN(num_classes=41, m=cfg['m'], cfg=cfg).to(device)

    for param in model_ema.parameters():
        param.detach_()

    # frozen branches must be excluded from WeightEMA, whose per-step
    # `param.mul_(1 - wd)` would otherwise shrink them despite requires_grad=False
    freeze_mode = cfg.get('freeze_tfgram', 'none')
    skip_keys = freeze_tfgram(model, freeze_mode) | freeze_tfgram(model_ema, freeze_mode)
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'freeze_tfgram={freeze_mode} | params {n_all} | trainable {n_train} | frozen tensors {len(skip_keys)}')

    teachers = load_teachers(cfg, device)

    trainer = Trainer(device=device, net=model, ema_net=model_ema, epochs=cfg['epoch'], cfg=cfg,
                      ema_skip_keys=skip_keys, teachers=teachers)

    trainer.train(train_dataloader, valid_dataloader)


def load_teachers(cfg, device):
    """Frozen eval-mode teachers for --kd_teachers, or [] when distillation is off.

    Building a net draws its random init from the torch RNG, which would shift every
    later draw (batch order, mixup) and break the pairing with the same-seed baseline,
    so the teachers are built inside fork_rng: the stream is restored afterwards.
    """
    names = [n for n in (cfg.get('kd_teachers') or '').split(',') if n.strip()]
    teachers = []
    with torch.random.fork_rng(devices=[device]):
        for name in names:
            path = os.path.join(cfg.get('kd_teacher_dir') or 'check_points', name.strip())
            with open(os.path.join(path, 'config.json'), encoding='UTF-8') as f:
                tcfg = json.load(f)
            net = SCLTFSTgramMFN(num_classes=41, m=tcfg['m'], cfg=tcfg).to(device)
            net.load_state_dict(torch.load(os.path.join(path, 'model.pth'), map_location=device), strict=True)
            for p in net.parameters():
                p.requires_grad_(False)
            teachers.append(net.eval())
    print(f'kd teachers: {len(teachers)}')
    return teachers


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    os.makedirs('check_points', exist_ok=True)
    main()
