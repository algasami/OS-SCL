import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from utils import get_accuracy, mixup_data, os_scl
from losses import ASDLoss, SupConLoss, WeightEMA
from probe import append_row, devtest_metrics, load_devtest


class Trainer:
    def __init__(self, device, net, ema_net, epochs=300, lr=0.0001, cfg=None, ema_skip_keys=(), teachers=()):
        self.device = device
        self.epochs = epochs
        self.net = net
        self.ema_net = ema_net
        self.cfg = cfg

        self.suploss = SupConLoss()

        model_name = type(self.net).__name__

        if cfg["desc"] != 'None':
            model_name += f'_{cfg["desc"]}'
        self.cfg['model_name'] = model_name
        os.makedirs(f'./check_points/{model_name}', exist_ok=True)

        self.model_path = f'./check_points/{model_name}/model.pth'
        with open(
                f'./check_points/{model_name}/config.json',
                'w',
                encoding='utf-8') as json_file:
            json.dump(cfg, json_file, ensure_ascii=False, indent=4)

        self.optimizer = torch.optim.AdamW(
            [p for p in self.net.parameters() if p.requires_grad], lr=lr)

        self.ema_optimizer = WeightEMA(self.net, self.ema_net, skip_keys=ema_skip_keys)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs,
                                                                    eta_min=0.1 * float(lr))
        # the EMA (alpha 0.9999) is still mostly init for the first ~90 epochs,
        # so probing training needs the student's own weights
        snaps = cfg.get('student_snapshots') or ''
        self.student_snapshots = {int(e) for e in str(snaps).split(',') if e.strip()}
        # full training state for forks (ablation/trunk-soup)
        snaps = cfg.get('state_snapshots') or ''
        self.state_snapshots = {int(e) for e in str(snaps).split(',') if e.strip()}
        # --state_every N: overwrite state_last.pth every N epochs, for --crash_resume
        self.state_every = int(cfg.get('state_every') or 0)

        # distillation (ablation/distill): frozen teachers score the same mixed batch
        self.teachers = list(teachers)
        self.kd_beta = float(cfg.get('kd_beta', 1.0))
        self.kd_tau = float(cfg.get('kd_tau', 2.0))

        # --m_warmup START,END (ablation/margin-warmup): only the student's margin moves.
        # ema_net keeps cfg['m'], so validation, checkpoint selection, the probe and
        # model.pth all score with the final margin, as eval.py does. Off = original.
        self.m_final = float(cfg['m'])
        self.m_warmup = None
        if cfg.get('m_warmup'):
            start, end = (float(e) for e in str(cfg['m_warmup']).split(','))
            if not 0 <= start <= end:
                raise ValueError(f"--m_warmup needs 0 <= START <= END, got {cfg['m_warmup']!r}")
            self.m_warmup = (start, end)

        self.criterion = ASDLoss().to(self.device)
        self.test_criterion = ASDLoss(reduction=False).to(self.device)

        # --amp (ablation/fast-recipe): autocast the branches and MobileFaceNet; the head,
        # ArcFace and both losses stay fp32 (see SCLTFSTgramMFN.forward). Off = original.
        amp = cfg.get('amp') or 'none'
        self.amp = amp != 'none'
        self.amp_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16}.get(amp, torch.float32)
        self.scaler = torch.amp.GradScaler('cuda', enabled=amp == 'fp16')

        # --probe_every N: log the EMA net's dev-test R0 metrics every N epochs to
        # probe.csv. A diagnostic only; never read by checkpoint selection.
        self.probe_every = int(cfg.get('probe_every') or 0)
        self.probe_path = os.path.join(os.path.dirname(self.model_path), 'probe.csv')
        self._probe_data = None
        self.skipped = 0

        # --wave_aug (ablation/lit-arms): the Sensors 26:3214 recipe, applied to the
        # batch on the GPU before mixup; augmented clips get their mel recomputed there
        self.wave_aug = bool(cfg.get('wave_aug'))
        if self.wave_aug:
            from dataloader import Wave2Mel
            w2m = Wave2Mel()
            w2m.mel_transform.to(self.device)
            w2m.amplitude_to_db.to(self.device)
            self.gpu_mel = w2m
        # --wave_aug_ops (ablation/aug-ops): which of the three ops run; default all
        ops = cfg.get('wave_aug_ops') or 'noise,shift,gain'
        self.wave_ops = set(ops.split(','))
        assert self.wave_ops and self.wave_ops <= {'noise', 'shift', 'gain'}, f'bad --wave_aug_ops {ops}'

    def train(self, train_loader, valid_loader, start_epoch=0):
        num_steps = len(train_loader)
        # a fork selects its own best epoch from the fork point on, not against the trunk's;
        # a crash-resume picks up the interrupted run's own best
        min_val_loss = getattr(self, 'crash_min_val_loss', 1e10)

        for epoch in tqdm(range(start_epoch, self.epochs), total=self.epochs - start_epoch, dynamic_ncols=True):
            sum_loss = 0.
            sum_accuracy = 0.
            sum_kd = 0.
            sum_acc_final = 0.

            for step, (x_wavs, x_mels, labels) in tqdm(enumerate(train_loader), total=num_steps, dynamic_ncols=True):
                self.net.train()
                if self.m_warmup:
                    self.net.arcface.set_margin(self.margin_at(epoch + step / num_steps))

                x_wavs, x_mels, labels = x_wavs.to(self.device), x_mels.to(self.device), labels.to(self.device)

                if self.wave_aug:
                    x_wavs, x_mels = self.augment(x_wavs, x_mels)

                mixed_x_wavs, mixed_x_mels, y_a, y_b, lam, mixed_y = mixup_data(x_wavs, x_mels, labels,
                                                                                self.device)
                with torch.autocast('cuda', dtype=self.amp_dtype, enabled=self.amp):
                    logits, feature = self.net(mixed_x_wavs, mixed_x_mels, labels)
                loss = os_scl(self.criterion, logits, y_a, y_b, lam,
                              self.suploss,
                              feature)
                if self.teachers:
                    kd = self.kd_loss(mixed_x_wavs, mixed_x_mels, labels, feature)
                    loss = loss + self.kd_beta * kd
                    sum_kd += kd.item()

                sum_accuracy += get_accuracy(logits, labels)
                if self.m_warmup:
                    sum_acc_final += self.acc_at_final_margin(feature, labels)

                if self.amp and not torch.isfinite(loss):
                    # GradScaler only guards the gradients: BN running stats were already
                    # updated by this forward, so a non-finite loss means a bad batch. Skip it.
                    self.skipped += 1
                    print(f'skip | EPOCH: {epoch} | non-finite loss, {self.skipped} skipped so far', flush=True)
                    self.optimizer.zero_grad()
                    continue

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.ema_optimizer.step()

                sum_loss += loss.item()
            self.scheduler.step()

            avg_loss = sum_loss / num_steps
            avg_accuracy = sum_accuracy / num_steps

            valid_loss, valid_accuracy = self.valid(valid_loader)
            saved = min_val_loss > valid_loss
            if min_val_loss > valid_loss:
                min_val_loss = valid_loss
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f'lr: {lr:.7f} | EPOCH: {epoch} | Train_loss: {avg_loss:.5f} | Train_accuracy: {avg_accuracy:.5f} | Valid_loss: {valid_loss:.5f} | Valid_accuracy: {valid_accuracy:.5f}'
                    + (f' | KD: {sum_kd / num_steps:.5f}' if self.teachers else '')
                    + (f' | m: {self.net.arcface.m:.4f} | Train_acc_mfinal: {sum_acc_final / num_steps:.5f}'
                       if self.m_warmup else ''))
                torch.save(self.ema_net.state_dict(), self.model_path)
            if self.probe_every and ((epoch + 1) % self.probe_every == 0 or epoch + 1 == self.epochs):
                self.probe(epoch, valid_loss, saved)
            if epoch + 1 in self.state_snapshots:
                self.save_state(epoch, min_val_loss, valid_loss)
            if self.state_every and (epoch + 1) % self.state_every == 0 and epoch + 1 < self.epochs:
                self.save_state(epoch, min_val_loss, valid_loss, name='state_last.pth')
            if epoch + 1 in self.student_snapshots:
                torch.save(self.net.state_dict(),
                           os.path.join(os.path.dirname(self.model_path), f'student_e{epoch + 1}.pth'))

    def margin_at(self, t):
        """The student's margin at fractional epoch t under --m_warmup START,END."""
        start, end = self.m_warmup
        if t >= end:
            return self.m_final
        if t < start:
            return 0.0
        return self.m_final * (t - start) / (end - start)

    @torch.no_grad()
    def acc_at_final_margin(self, feature, labels):
        """Train accuracy as the original run measures it (margin m_final on the target):
        cos(theta_y + m_final) beats every other cosine. The trap signal, logged while the
        student's own margin is still warming up."""
        cos = F.linear(feature.float(), F.normalize(self.net.arcface.weight.float()))
        tgt = cos.gather(1, labels[:, None])[:, 0]
        other = cos.scatter(1, labels[:, None], -2.).max(1).values
        return (torch.cos(torch.acos(tgt.clamp(-1, 1)) + self.m_final) > other).float().mean().item()

    def save_state(self, epoch, min_val_loss, valid_loss, name=None):
        """Everything needed to continue this run after `epoch` (0-based): a fork
        (--resume_state) or a crash-resume (--crash_resume). Written atomically."""
        path = os.path.join(os.path.dirname(self.model_path), name or f'state_e{epoch + 1}.pth')
        rng = {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(self.device),
               'numpy': np.random.get_state(), 'python': random.getstate()}
        torch.save({'epoch': epoch, 'min_val_loss': min_val_loss, 'valid_loss': valid_loss,
                    'net': self.net.state_dict(), 'ema_net': self.ema_net.state_dict(),
                    'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
                    'scaler': self.scaler.state_dict(), 'rng': rng}, path + '.tmp')
        os.replace(path + '.tmp', path)

    def load_state(self, path):
        """Load a save_state() file in place; returns the epoch to continue from.

        load_state_dict copies into the existing tensors, so WeightEMA's references to
        the student's and the EMA's tensors stay valid.
        """
        st = torch.load(path, map_location=self.device, weights_only=False)  # our own file: RNG state is not tensors
        self.net.load_state_dict(st['net'])
        self.ema_net.load_state_dict(st['ema_net'])
        self.optimizer.load_state_dict(st['optimizer'])
        self.scheduler.load_state_dict(st['scheduler'])
        self.scaler.load_state_dict(st['scaler'])
        self.resumed_valid_loss = st['valid_loss']
        self.resumed_state = st
        return st['epoch'] + 1

    def crash_resume(self, path):
        """Continue an interrupted run from its state_last.pth: weights, optimiser,
        best-so-far validation loss and RNG streams (not bit-exact on this host)."""
        start = self.load_state(path)
        st = self.resumed_state
        self.crash_min_val_loss = st['min_val_loss']
        torch.set_rng_state(st['rng']['torch'].cpu())
        torch.cuda.set_rng_state(st['rng']['cuda'].cpu(), self.device)
        np.random.set_state(st['rng']['numpy'])
        random.setstate(st['rng']['python'])
        return start

    @torch.no_grad()
    def augment(self, wavs, mels, p=0.3):
        """Each op independently with probability p, per clip (Sensors 26:3214, Sec. 3):
        Gaussian noise with std ~ U[0.001, 0.005]; a circular time shift of
        ~ U[0.03, 0.08] of the length, random sign (direction is not stated in the
        paper); gain ~ U[0.9, 1.1]. Mels of the touched clips are recomputed on the
        GPU, which differs from the cached CPU mel by ~1e-5 dB on untouched audio.
        """
        b, dev = wavs.shape[0], wavs.device
        u = lambda lo, hi: lo + (hi - lo) * torch.rand(b, 1, 1, device=dev)   # noqa: E731
        noise, shift, gain = (torch.rand(b, device=dev) < p for _ in range(3))
        # a disabled op still draws its mask, so the other ops' draws keep their place in the stream
        ops = getattr(self, 'wave_ops', {'noise', 'shift', 'gain'})
        noise, shift, gain = (m & (o in ops) for m, o in ((noise, 'noise'), (shift, 'shift'), (gain, 'gain')))
        wavs = wavs.clone()
        if noise.any():
            wavs[noise] += (torch.randn_like(wavs) * u(0.001, 0.005))[noise]
        if shift.any():
            n = wavs.shape[-1]
            k = (u(0.03, 0.08).view(b) * n).long() * (torch.randint(0, 2, (b,), device=dev) * 2 - 1)
            for i in torch.nonzero(shift).flatten().tolist():
                wavs[i] = torch.roll(wavs[i], int(k[i]), dims=-1)
        if gain.any():
            wavs[gain] *= u(0.9, 1.1)[gain]
        touched = noise | shift | gain
        if touched.any():
            mels = mels.clone()
            mels[touched] = self.gpu_mel(wavs[touched])
        return wavs, mels

    def kd_loss(self, wavs, mels, labels, feature):
        """tau^2 * KL(p_T || p_S) on margin-free 30*cos, p_T the teachers' mean softmax.

        `feature` is the student's L2-normalised head output, so its cosines need only the
        normalised ArcFace weights. The teachers run in eval mode under fp16 autocast and
        draw no random numbers.
        """
        s, tau = self.net.arcface.s, self.kd_tau
        cos_s = F.linear(feature, F.normalize(self.net.arcface.weight))
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            p_t = torch.stack([F.softmax(s * teacher_cos(t, wavs, mels).float() / tau, dim=1)
                               for t in self.teachers]).mean(0)
        log_p_s = F.log_softmax(s * cos_s / tau, dim=1)
        kl = (p_t * (p_t.clamp_min(1e-12).log() - log_p_s)).sum(1).mean()
        return tau ** 2 * kl

    @torch.no_grad()
    def valid(self, valid_loader):

        net = self.ema_net.eval()

        num_steps = len(valid_loader)
        sum_loss = 0.
        sum_accuracy = 0.

        for (x_wavs, x_mels, labels) in valid_loader:
            x_wavs, x_mels, labels = x_wavs.to(self.device), x_mels.to(self.device), labels.to(self.device)

            # fp32 even under --amp: the early EMA net's BN stats are still near init,
            # its activations overflow fp16, and a NaN loss would silently stop saving
            logits, _ = net(x_wavs, x_mels, labels, train=True)

            sum_accuracy += get_accuracy(logits, labels)
            loss = self.criterion(logits, labels)

            sum_loss += loss.item()

        avg_loss = sum_loss / num_steps
        avg_accuracy = sum_accuracy / num_steps
        return avg_loss, avg_accuracy


    def probe(self, epoch, valid_loss, saved):
        """Dev-test R0 metrics of the EMA net, appended to probe.csv (diagnostic only)."""
        if self._probe_data is None:
            self._probe_data = load_devtest(self.cfg['cache_dir'])
        row = {'epoch': epoch, 'valid_loss': f'{valid_loss:.5f}', 'saved': int(saved)}
        if self.m_warmup:   # the student's margin at the end of this epoch; the probe itself scores at m_final
            row['m'] = f'{self.net.arcface.m:.4f}'
        try:
            metrics_row = devtest_metrics(self.ema_net, self._probe_data, self.device)  # fp32, as valid()
        except Exception as e:  # a diagnostic must never kill the run
            print(f'probe | EPOCH: {epoch} | failed: {e!r}', flush=True)
            return
        row.update({k: f'{v:.5f}' for k, v in metrics_row.items()})
        append_row(self.probe_path, row)
        print(f"probe | EPOCH: {epoch} | dev AUC {row['AUC']} pAUC {row['pAUC']} mAUC {row['mAUC']}")


def teacher_cos(net, wavs, mels):
    """Margin-free cosines to the 41 class centres; net.forward's path without the margin."""
    _, feature = net.mobilefacenet(net.branch_input(wavs, mels, train=wavs.shape[0] > 1))
    feature = F.normalize(net.head(feature), dim=1)
    return F.linear(feature, F.normalize(net.arcface.weight))
