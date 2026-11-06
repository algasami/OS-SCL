"""
modification made on the basis of link:https://github.com/liuyoude/STgram-MFN
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import ArcMarginProduct
from model.waveq import TFgram


class BasicHead(nn.Module):
    def __init__(self, in_dim=128):
        super(BasicHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
        )

    def forward(self, x):
        return self.head(x)


class LeakyReLUHead(nn.Module):
    def __init__(self, in_dim=128):
        super(LeakyReLUHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.BatchNorm1d(64),

            nn.Linear(64, 128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.BatchNorm1d(128),
        )

    def forward(self, x):
        return self.head(x)


class ECA(nn.Module):
    """Efficient channel attention (Wang et al., CVPR 2020): GAP -> 1-D conv over
    channels -> sigmoid -> channel-wise scale. Kernel from the paper's adaptive
    rule k = |log2(C)/2 + 1/2|_odd, which gives 5 for both of MobileFaceNet's
    expanded widths here (128 and 512)."""

    def __init__(self, channels):
        super().__init__()
        k = int(abs(math.log2(channels) / 2 + 0.5))
        k = k if k % 2 else k + 1
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        w = self.conv(x.mean((2, 3)).unsqueeze(1)).squeeze(1)
        return x * torch.sigmoid(w)[:, :, None, None]


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion, eca=False):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        dw = [
            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
        ]
        if eca:  # after the depthwise conv, as in the Sensors 26:3214 ECA-MobileFaceNet
            dw.append(ECA(inp * expansion))
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            *dw,

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


# https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Liu_8_t2.pdf
Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 128, 2, 2],
    [4, 128, 2, 2],
    [4, 128, 2, 2],

]

# Input branches per fussion mode, in concat order. 1 and 2 are the paper's
# settings; 3 and 4 isolate a single learned gram against the Log-Mel baseline.
FUSSION_BRANCHES = {
    1: ('mel', 'tgram', 'tfgram'),   # TFSTgram (paper)
    2: ('mel',),                     # Log-Mel only (paper)
    3: ('mel', 'tfgram'),            # Log-Mel + TFgram
    4: ('mel', 'tgram'),             # Log-Mel + Tgram (STgram)
    # no Log-Mel: raw-waveform branches only (ablation/no-logmel)
    5: ('tgram', 'tfgram'),          # Tgram + TFgram
    6: ('tfgram',),                  # TFgram only
    7: ('tgram',),                   # Tgram only
}
FUSSION_CHANNELS = {k: len(v) for k, v in FUSSION_BRANCHES.items()}


def input_branches(cfg):
    """The fussion mode's branches, plus the gated Log-Mel channel right after the
    Log-Mel one when --mel_gate is set (ESTM's ESTgram order: Mel, EMel, T)."""
    branches = FUSSION_BRANCHES[cfg['fussion']]
    if (cfg.get('mel_gate') or 'none') == 'none':
        return branches
    if 'mel' not in branches:
        raise ValueError(f"--mel_gate needs a Log-Mel branch; fussion {cfg['fussion']} has none")
    i = branches.index('mel') + 1
    return branches[:i] + ('emel',) + branches[i:]


def tristat_gate(x_mel, alpha=2.0):
    """ESTM's TriStat gating (arXiv 2509.02471, Eq. 1-2), parameter-free.

    Per frame, over the 128 mel bins: s = median + RMS + variance; the gate is
    sigmoid(alpha * (s - mu)) broadcast over frequency. The paper calls mu "the
    mean used for de-centering". Here s is z-scored over the clip's 313 frames:
    on raw dB the variance term (~1e2) dominates and the gate saturates to a
    0/1 mask (ablation/lit-arms README). Computed in fp32.
    """
    with torch.autocast('cuda', enabled=False):
        x = x_mel.float()
        s = x.median(dim=2).values + x.pow(2).mean(2).sqrt() + x.var(2)       # (B, 1, T)
        s = (s - s.mean(-1, keepdim=True)) / (s.std(-1, keepdim=True) + 1e-5)
        return torch.sigmoid(alpha * s).unsqueeze(2) * x                      # (B, 1, F, T)


class MobileFaceNet(nn.Module):
    def __init__(self,
                 num_class,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 cfg=None):
        super(MobileFaceNet, self).__init__()

        self.conv1 = ConvBlock(len(input_branches(cfg)), 64, 3, 2, 1)
        self.eca = bool(cfg.get('eca', False))

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)
        # 20(10), 4(2), 8(4)
        self.linear7 = ConvBlock(512, 512, (8, 20), 1, 0, dw=True, linear=True)
        self.tpool = bool(cfg.get('tpool', False))

        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        self.fc_out = nn.Linear(128, num_class)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.tpool:
            self._tie_linear7_over_time()

    def _tie_linear7_over_time(self):
        """--tpool 1 (ablation/timepool-head): linear7 with its weights tied across time.

        The (8, 20) depthwise kernel is one weight per (frequency, time) cell, so linear7 is
        where absolute time position enters (ablation/pooled-knn diagnostic). Tied over time,
        it becomes a (8, 1) depthwise kernel applied to the map's time mean: per channel,
        sum_f w[c, f] * mean_t x[c, f, t], then BN as before. That is the only linear
        functional of the map that is invariant to circular time shifts while keeping a
        free weight per channel and frequency band. Output stays 512 x 1 x 1.

        Init: the (8, 20) layer above is built and drawn as usual, so every other module and
        the RNG stream after the net are exactly those of the flag-off net at the same seed.
        The tied kernel is then set deterministically to the time-sum of that draw (the
        (8, 20) kernel's response to a time-constant map); its construction draws are
        discarded via fork_rng. BN that follows makes the kernel's overall scale irrelevant.
        """
        old = self.linear7
        with torch.random.fork_rng(devices=[]):
            self.linear7 = ConvBlock(512, 512, (8, 1), 1, 0, dw=True, linear=True)
        with torch.no_grad():
            self.linear7.conv.weight.copy_(old.conv.weight.sum(3, keepdim=True))
        self.linear7.bn.load_state_dict(old.bn.state_dict())

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t, eca=self.eca))
                else:
                    layers.append(block(self.inplanes, c, 1, t, eca=self.eca))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        if self.tpool:
            x = x.mean(3, keepdim=True)          # (B, 512, 8, 20) -> (B, 512, 8, 1)
        x = self.linear7(x)
        x = self.linear1(x)
        feature = x.view(x.size(0), -1)
        out = self.fc_out(feature)
        return out, feature


class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512):
        super(TgramNet, self).__init__()
        # if "center=True" of stft, padding = win_len / 2
        self.conv_extrctor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                # 313(10) , 63(2), 126(4)
                nn.LayerNorm(313),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
            ) for _ in range(num_layer)])

    def forward(self, x):
        out = self.conv_extrctor(x)
        out = self.conv_encoder(out)
        return out


class SpecNet(nn.Module):
    """Whole-clip magnitude-spectrum branch (ablation/spectrum-branch, --spec_branch 1).

    A port of the "FFT" sub-network of Wilkinghoff's DCASE2023 Task 2 system
    (github.com/wilkinghoff/DCASE2023_task2, main.py, model_emb_cnn): |FFT| of the
    whole clip, bins 0-8 kHz, no log and no input normalisation; three strided 1-D
    convs (128 ch; k/s 256/64, 64/32, 16/4; no bias; ReLU); flatten; four
    Linear(128) + BN + ReLU; a final Linear(128) embedding. Keras 'same' padding is
    reproduced exactly for a 160000-sample clip: 80000 -> 1250 -> 40 -> 10 frames.
    Differences from the original: PyTorch default init (not Glorot), and no
    per-layer L2 (AdamW's decay applies instead).

    The FFT and the whole branch run in fp32 even under --amp: raw spectrum
    magnitudes reach ~1e3 and the first conv sums 256 of them.
    """

    def __init__(self, n_samples=160000, emb=128, ch=128):
        super().__init__()
        self.n_bins = n_samples // 2
        layers, length = [], self.n_bins
        for i, (k, s) in enumerate(((256, 64), (64, 32), (16, 4))):
            out = -(-length // s)                                   # ceil: Keras 'same'
            pad = max((out - 1) * s + k - length, 0)
            assert pad % 2 == 0, (length, k, s, pad)                # symmetric for these sizes
            layers += [nn.Conv1d(1 if i == 0 else ch, ch, k, s, pad // 2, bias=False), nn.ReLU(inplace=True)]
            length = out
        self.conv = nn.Sequential(*layers)
        dense, width = [], ch * length
        for _ in range(4):
            dense += [nn.Linear(width, emb, bias=False), nn.BatchNorm1d(emb), nn.ReLU(inplace=True)]
            width = emb
        self.dense = nn.Sequential(*dense, nn.Linear(emb, emb, bias=False))

    def forward(self, x_wav):
        with torch.autocast('cuda', enabled=False):
            x = x_wav.float().reshape(x_wav.shape[0], -1)
            spec = torch.fft.rfft(x).abs()[:, :self.n_bins].unsqueeze(1)   # (B, 1, 80000)
            return self.dense(self.conv(spec).flatten(1))


class SCLTFSTgramMFN(nn.Module):

    def __init__(self, cfg,
                 num_classes=41,
                 c_dim=128,
                 win_len=1024,
                 hop_len=512,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting, m=0.7, s=30,
                 ):
        super().__init__()

        self.cfg = cfg

        self.arcface = ArcMarginProduct(in_features=c_dim, out_features=num_classes,
                                        m=m, s=s)

        # lean: build only the branches this fussion mode consumes, and no dead TFgram
        # modules. Changes the state_dict keys and the init draw order (ablation/fast-recipe).
        lean = bool(cfg.get('lean', False))
        branches = FUSSION_BRANCHES[cfg['fussion']]

        if not lean or 'tgram' in branches:
            self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)

        self.mobilefacenet = MobileFaceNet(num_class=num_classes,
                                           bottleneck_setting=bottleneck_setting,
                                           cfg=cfg)

        if not lean or 'tfgram' in branches:
            self.TFgramNet = TFgram(classes_num=41, act=cfg.get('tf_act', 'relu'),
                                    pool=cfg.get('tf_pool', 'max'), lean=lean)
        # --spec_branch 1: the spectrum embedding is concatenated with MobileFaceNet's
        # before the head, so the head's first layer widens to 256. Built last, so the
        # modules above draw exactly the init they draw without it.
        self.spec_branch = bool(cfg.get('spec_branch', False))
        emb_dim = c_dim * 2 if self.spec_branch else c_dim
        head_type = cfg["ht"]
        if head_type == 'basic':
            self.head = BasicHead(emb_dim)
        elif head_type == 'leaky_relu':
            self.head = LeakyReLUHead(emb_dim)
        if self.spec_branch:
            self.specnet = SpecNet(emb=c_dim)

    def branch_input(self, x_wav, x_mel, train=True):
        """The fussion mode's branches stacked as channels: the MobileFaceNet input."""
        branches = input_branches(self.cfg)

        # only run the branches this fussion mode actually consumes
        feats = []
        for name in branches:
            if name == 'mel':
                feats.append(x_mel)
            elif name == 'emel':
                feats.append(tristat_gate(x_mel))
            elif name == 'tgram':
                feats.append(self.tgramnet(x_wav).unsqueeze(1))
            elif name == 'tfgram':
                feats.append(self.TFgramNet(x_wav, train).unsqueeze(1))

        return feats[0] if len(feats) == 1 else torch.cat(feats, dim=1)

    def embed(self, x_wav, x_mel, train=True):
        """The pre-head embedding: MobileFaceNet's 128-d output, with the spectrum
        branch's 128-d embedding appended under --spec_branch (fp32)."""
        _, feature = self.mobilefacenet(self.branch_input(x_wav, x_mel, train))
        if self.spec_branch:
            feature = torch.cat([feature.float(), self.specnet(x_wav)], dim=1)
        return feature

    def forward(self, x_wav, x_mel, label, train=True):

        feature = self.embed(x_wav, x_mel, train)

        # head, normalisation and the ArcFace margin stay fp32 under --amp: acos-free
        # margin math needs sqrt(1 - cos^2) near |cos| = 1, and SupCon's 1/0.02 logits
        # would overflow fp16. A no-op when autocast is off.
        with torch.autocast('cuda', enabled=False):
            feature = F.normalize(self.head(feature.float()), dim=1)
            out = self.arcface(feature, label)
        return out, feature


def freeze_tfgram(net, mode):
    """Freeze the TFgram branch at its random init (ablation/frozen-tfgram).

    'strict'   - no gradients, and the branch is pinned to eval() so its
                 BatchNorm running stats stay at the init values (mean 0, var 1,
                 weight 1, bias 0), which makes BN an identity. A pure fixed
                 random projection.
    'bn_adapt' - no gradients on the weights, but BatchNorm keeps tracking the
                 data, so the random features stay sanely scaled.

    Returns the state_dict keys WeightEMA must leave alone. Its per-step
    `param.mul_(1 - wd)` runs over the whole state_dict, so without this the
    branch is decayed to 0.72x over a 300-epoch run despite requires_grad=False
    -- and that silently applies to BatchNorm buffers, not just the weights.

    'strict' skips the branch's buffers as well, so nothing under TFgramNet ever
    changes. 'bn_adapt' skips only the parameters, leaving the running stats to
    be tracked and EMA'd exactly as they are for every other branch.
    """
    if mode in (None, 'none'):
        return set()
    if mode not in ('strict', 'bn_adapt'):
        raise ValueError(f'unknown freeze_tfgram mode: {mode!r}')

    tf = net.TFgramNet
    for p in tf.parameters():
        p.requires_grad_(False)

    if mode == 'strict':
        tf.eval()
        # trainer.py calls net.train() once per iteration, which would put the
        # branch back in train mode. nn.Module.train() recurses through
        # children, so stubbing this one out stops the recursion here.
        tf.train = lambda mode=True: tf
        return {f'TFgramNet.{k}' for k in tf.state_dict()}

    return {f'TFgramNet.{n}' for n, _ in tf.named_parameters()}
