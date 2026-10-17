"""Diagnose the G1 failure: which part of the KD plumbing moves the student at beta = 0.

    python ablation/distill/diag_g1.py <mode> <gpu> <out.pt>   # one mode per process
    python ablation/distill/diag_g1.py compare <dir>           # reads <dir>/diag_*.pt

Runs train.main() unchanged (same seed, split, init, DataLoader) with Trainer.train
swapped for the first N_STEPS steps of its loop, then saves the per-step losses and
the student's weights. Modes:

    none    no teachers: the baseline recipe
    none2   the same again, in a fresh process: is training deterministic on this host?
    loaded  teachers built (fork_rng) and on the GPU, never run
    side    teachers run and KD is computed every step, but not added to the loss
    add0    loss + 0 * KD: what the gate ran
"""
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
import train as T  # noqa: E402
from trainer import Trainer  # noqa: E402
from utils import get_accuracy, mixup_data, os_scl  # noqa: E402

N_STEPS = 20
TDIR = '/home/f74134118/OS-SCL/check_points'
TEACHERS = ','.join(f'SCLTFSTgramMFN_tfl_relu_max_s{s}' for s in (2025, 2026, 2027))
# the TFgram modules forward() never uses (CLAUDE.md); they never change
UNUSED = ('TFgramNet.conv_block', 'TFgramNet.fc', 'TFgramNet.bn0.', 'TFgramNet.pre_block4')
MODES = ('none', 'none2', 'loaded', 'side', 'add0')


def short_train(mode, out):
    def train(self, train_loader, valid_loader):
        losses = []
        for i, (x_wavs, x_mels, labels) in enumerate(train_loader):
            if i == N_STEPS:
                break
            self.net.train()
            x_wavs, x_mels, labels = x_wavs.to(self.device), x_mels.to(self.device), labels.to(self.device)
            mixed_x_wavs, mixed_x_mels, y_a, y_b, lam, mixed_y = mixup_data(x_wavs, x_mels, labels, self.device)
            logits, feature = self.net(mixed_x_wavs, mixed_x_mels, labels)
            loss = os_scl(self.criterion, logits, y_a, y_b, lam, self.suploss, feature)
            if mode in ('side', 'add0'):
                kd = self.kd_loss(mixed_x_wavs, mixed_x_mels, labels, feature)
                if mode == 'add0':
                    loss = loss + 0.0 * kd
                kd.item()
            get_accuracy(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema_optimizer.step()
            losses.append((lam, loss.item()))
        state = {k: v.detach().cpu() for k, v in self.net.state_dict().items() if not k.startswith(UNUSED)}
        torch.save({'mode': mode, 'losses': losses, 'state': state}, out)
        print(f'{mode}: step-1 loss {losses[0][1]!r}, step-{N_STEPS} loss {losses[-1][1]!r}', flush=True)
        os._exit(0)  # don't wait on the persistent DataLoader workers
    return train


def run(mode, gpu, out):
    assert mode in MODES, mode
    sys.argv = ['train.py', '--m', '0.4', '--fussion', '1', '--ht', 'basic', '--gpu_num', gpu,
                '--num_workers', '8', '--seed', '2024', '--epoch', '300', '--tf_act', 'relu',
                '--tf_pool', 'max', '--desc', f'kddiag_{mode}', '--student_snapshots', '10,30,90,300']
    if mode not in ('none', 'none2'):
        sys.argv += ['--kd_teachers', TEACHERS, '--kd_teacher_dir', TDIR, '--kd_beta', '0', '--kd_tau', '2']
    Trainer.train = short_train(mode, out)
    T.main()


def compare(d):
    runs = {m: torch.load(os.path.join(d, f'diag_{m}.pt'), weights_only=False)
            for m in MODES if os.path.exists(os.path.join(d, f'diag_{m}.pt'))}
    ref = runs['none']
    for m, r in runs.items():
        first = next((i + 1 for i, (a, b) in enumerate(zip(ref['losses'], r['losses'])) if a != b), None)
        lam_same = [a[0] for a in ref['losses']] == [b[0] for b in r['losses']]
        diffs = [(v.float() - r['state'][k].float()).abs().max().item() for k, v in ref['state'].items()]
        n_diff = sum(x > 0 for x in diffs)
        print(f'{m:7s} mixup lambdas same: {lam_same} | first differing loss: step {first} | '
              f'tensors differing: {n_diff}/{len(diffs)} | max |dW| {max(diffs):.3e}')


if __name__ == '__main__':
    if sys.argv[1] == 'compare':
        compare(sys.argv[2])
    else:
        run(*sys.argv[1:4])
