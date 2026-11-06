import numpy as np
import torch
from torch.utils.data import random_split


def get_accuracy(logits, y):
    pred_label = torch.argmax(logits, dim=-1)
    return torch.sum(pred_label == y) / len(pred_label)


def dataset_split(dataset, split_ratio=0.95):
    data_len = len(dataset)

    train_len = int(data_len * split_ratio)
    valid_len = data_len - train_len

    train_dataset, valid_dataset = random_split(dataset, [train_len, valid_len])

    return train_dataset, valid_dataset


# Mixup is referred to https://github.com/facebookresearch/mixup-cifar10

def mixup_data(x_wavs, x_mels, y, device, alpha=0.5):
    '''Returns mixed inputs, pairs of targets, and lambda,
    x_wav(32,1,160000),
    x_mel(32,1,128,313),
    y_a(32)'''

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    # lam = max(lam, 1 - lam)
    batch_size = x_mels.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x_wavs = lam * x_wavs + (1 - lam) * x_wavs[index, :]
    mixed_x_mels = lam * x_mels + (1 - lam) * x_mels[index, :]

    y_a, y_b = y, y[index]
    mixed_y = lam * y_a + (1 - lam) * y_b
    return mixed_x_wavs, mixed_x_mels, y_a, y_b, lam, mixed_y


def os_scl(criterion, pred, y_a, y_b, lam, sup, feature):
    loss1 = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    loss2 = sup(feature.unsqueeze(1), y_a)
    loss = loss2 + loss1
    return loss


# PerMix (Choi & Choi, arXiv 2610.31434; ablation/permix). Off by default: trainer.py
# calls these only under --permix 1, and they draw nothing from any RNG otherwise.

def permix_data(x_wavs, x_mels, y, a=0.5, r_max=3):
    '''Each anchor i is mixed with R_i ~ U{1..r_max} other clips of the batch, drawn without
    replacement from B \\ {i}, with weights ~ Dir(a); the anchor takes the largest weight
    (paper Eq. 1). Returns the mixed wave and mel and the dominant-class ratio lambda_i, the
    summed weight of every mixed clip whose label equals the anchor's (Eq. 2). The anchor's
    own label `y` is the dominant class, so it is the target for both losses.

    Draws come from numpy's global RNG (saved by --crash_resume, as mixup's are). Wave and
    mel are both mixed, as mixup_data does (OS-SCL's code mixes the dB mel linearly too).'''
    b = y.shape[0]
    r_max = min(r_max, b - 1)                                 # a short last batch
    idx = np.tile(np.arange(b)[:, None], (1, r_max + 1))   # unused slots: anchor, weight 0
    w = np.zeros((b, r_max + 1))
    for i in range(b):
        r = np.random.randint(1, r_max + 1)
        others = np.random.choice(b - 1, r, replace=False)
        idx[i, 1:r + 1] = others + (others >= i)             # skip i itself
        alpha = np.random.dirichlet(np.full(r + 1, a))
        k = int(np.argmax(alpha))                             # largest -> anchor; the rest keep
        w[i, 0] = alpha[k]                                    # their (exchangeable) draw order
        w[i, 1:r + 1] = np.delete(alpha, k)
    dev = x_wavs.device
    idx_t = torch.from_numpy(idx).to(dev)
    w_t = torch.from_numpy(w).float().to(dev)
    mixed_wavs = sum(w_t[:, s].view(-1, *[1] * (x_wavs.dim() - 1)) * x_wavs[idx_t[:, s]]
                     for s in range(r_max + 1))
    mixed_mels = sum(w_t[:, s].view(-1, *[1] * (x_mels.dim() - 1)) * x_mels[idx_t[:, s]]
                     for s in range(r_max + 1))
    lam = (w_t * (y[idx_t] == y[:, None]).float()).sum(1).clamp(0., 1.)
    return mixed_wavs, mixed_mels, lam


def permix_loss(logits, y, lam, sup, feature):
    '''L_per + L_dom (paper Eq. 6-8, equal weights). `logits` are the lambda-scaled ArcFace
    logits (ArcMarginProduct.forward_permix); L_per is the BCE between lambda and the softmax
    probability of the dominant class, with log(1 - p) computed stably as the log-mass of
    the other classes. L_dom is our SupConLoss on the mixed embeddings with the dominant
    class as label: the same call os_scl makes with y_a, because y_a is the anchor label.'''
    logits = logits.float()
    lse = torch.logsumexp(logits, dim=1)
    log_p = logits.gather(1, y[:, None])[:, 0] - lse
    log_1mp = torch.logsumexp(logits.scatter(1, y[:, None], float('-inf')), dim=1) - lse
    l_per = -(lam * log_p + (1 - lam) * log_1mp).mean()
    l_dom = sup(feature.unsqueeze(1), y)
    return l_per + l_dom, l_per, l_dom
