"""BEATs encoder + attentive statistics pooling + ArcFace over the 41 machine IDs
(ablation/beats; recipe after AnoPatch, arXiv 2406.11364).

    wave (B, 160000) -> Kaldi fbank (B, 998, 128), BEATs normalisation [fp32]
      -> SpecAugment (training only: one freq mask <= 40 bins, one time mask <= 80 frames)
      -> BEATs ViT (16x16 patches: 62 time x 8 freq = 496 tokens, 768-d)
      -> ASP (ECAPA-TDNN attentive statistics pooling, 1536-d) -> Linear -> `pre` (128-d)
      -> ArcMarginProduct(s 30, m) -> logits

`pre` plays the role of the MFN's pre-head embedding (kNN back-end K), `gmean` -- the
mean of the 496 final tokens, 768-d -- the role of pooled-knn's Gmean (C1).
"""
import hashlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as ta_kaldi

from losses import ArcMarginProduct
from model.beats import BEATs, BEATsConfig

FBANK_MEAN, FBANK_STD = 15.41663, 6.55582      # BEATs.preprocess defaults
CKPT_SHA256 = {   # https://huggingface.co/lpepino/beats_ckpts (LFS oids); see ablation/beats/README.md
    'BEATs_iter3.pt': '8d1b234032a9ccff353612dc6c20982346dc2968b205b79d97303eb5e77bfb34',
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


class ASP(nn.Module):
    """Attentive statistics pooling with global context (ECAPA-TDNN). x: (B, C, T) -> (B, 2C)."""

    def __init__(self, c, bottleneck=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(3 * c, bottleneck, 1), nn.ReLU(), nn.BatchNorm1d(bottleneck), nn.Tanh(),
            nn.Conv1d(bottleneck, c, 1), nn.Softmax(dim=2))

    def forward(self, x):
        t = x.shape[-1]
        mu = x.mean(2, keepdim=True)
        sd = torch.sqrt(x.var(2, keepdim=True, unbiased=False).clamp(min=1e-4))
        w = self.attention(torch.cat([x, mu.expand(-1, -1, t), sd.expand(-1, -1, t)], 1))
        m = (x * w).sum(2)
        s = torch.sqrt((((x ** 2) * w).sum(2) - m ** 2).clamp(min=1e-4))
        return torch.cat([m, s], 1)


class BEATsASD(nn.Module):
    def __init__(self, beats_cfg, num_classes=41, emb_dim=128, m=0.4, s=30.0,
                 freq_mask=40, time_mask=80, grad_ckpt=False, sdpa=False):
        super().__init__()
        cfg = BEATsConfig(beats_cfg)
        self.beats = BEATs(cfg)
        self.beats.encoder.grad_ckpt = grad_ckpt
        for layer in self.beats.encoder.layers:
            layer.self_attn.sdpa = sdpa
        d = cfg.encoder_embed_dim
        self.pool = ASP(d)
        self.fc = nn.Linear(2 * d, emb_dim)
        self.arcface = ArcMarginProduct(emb_dim, num_classes, s=s, m=m)
        self.fmask = torchaudio.transforms.FrequencyMasking(freq_mask, iid_masks=True) if freq_mask else None
        self.tmask = torchaudio.transforms.TimeMasking(time_mask, iid_masks=True) if time_mask else None

    @staticmethod
    def fbank(wav):
        """(B, N) wave in [-1, 1] -> (B, frames, 128), exactly BEATs.preprocess (always fp32)."""
        with torch.autocast(wav.device.type, enabled=False):
            fb = torch.stack([ta_kaldi.fbank(w.float().unsqueeze(0) * 2 ** 15, num_mel_bins=128,
                                             sample_frequency=16000, frame_length=25, frame_shift=10)
                              for w in wav])
        return (fb - FBANK_MEAN) / (2 * FBANK_STD)

    def specaug(self, fb):
        x = fb.transpose(1, 2).unsqueeze(1)          # (B, 1, F, T): torchaudio masks (..., freq, time)
        if self.fmask is not None:
            x = self.fmask(x)
        if self.tmask is not None:
            x = self.tmask(x)
        return x.squeeze(1).transpose(1, 2)

    def tokens(self, fb):
        """BEATs.extract_features after preprocess, no padding mask: (B, 496, 768)."""
        b = self.beats
        x = b.patch_embedding(fb.unsqueeze(1))
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
        x = b.layer_norm(x)
        if b.post_extract_proj is not None:
            x = b.post_extract_proj(x)
        x = b.dropout_input(x)
        x, _ = b.encoder(x, padding_mask=None)
        return x

    def forward(self, wav, label, augment=False):
        """-> logits (ArcFace with margin vs `label`), pre (B, 128), gmean (B, 768)."""
        fb = self.fbank(wav)
        if augment:
            fb = self.specaug(fb)
        tok = self.tokens(fb)
        pre = self.fc(self.pool(tok.transpose(1, 2)))
        logits = self.arcface(pre.float(), label)
        return logits, pre, tok.mean(1)


def load_pretrained(path, check_hash=True):
    """The upstream checkpoint dict {'cfg', 'model'}; weights_only (the file is downloaded data)."""
    name = os.path.basename(path)
    if check_hash and name in CKPT_SHA256:
        got = sha256(path)
        if got != CKPT_SHA256[name]:
            raise SystemExit(f'{path}: sha256 {got} != published {CKPT_SHA256[name]}')
    ck = torch.load(path, map_location='cpu', weights_only=True)
    return ck['cfg'], ck['model']


def build(cfg, pretrained=True):
    """BEATsASD from a run config. `pretrained`: load the upstream BEATs weights (training);
    otherwise the architecture comes from cfg['beats_cfg'] and the caller loads model.pth."""
    if pretrained:
        bcfg, state = load_pretrained(cfg['beats_ckpt'])
    else:
        bcfg, state = cfg['beats_cfg'], None
    net = BEATsASD(bcfg, m=cfg['m'], s=cfg.get('arc_s', 30.0), emb_dim=cfg.get('emb_dim', 128),
                   freq_mask=cfg.get('freq_mask', 40), time_mask=cfg.get('time_mask', 80),
                   grad_ckpt=bool(cfg.get('grad_ckpt', 0)), sdpa=bool(cfg.get('sdpa', 0)))
    if state is not None:
        missing, unexpected = net.beats.load_state_dict(state, strict=False)
        assert not unexpected and not missing, (missing, unexpected)
    return net, bcfg
