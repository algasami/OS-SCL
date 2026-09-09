import sys, torch, numpy as np
sys.path.insert(0, '.')
from model.net import SCLTFSTgramMFN
from dataloader import train_dataset

DEV = torch.device(f'cuda:{int(sys.argv[1]) if len(sys.argv) > 1 else 0}')
CKPT = 'check_points/SCLTFSTgramMFN_test/model.pth'
cfg = {'m': 0.4, 'gpu_num': 0, 'epoch': 300, 'batch_size': 64, 'fussion': 1, 'ht': 'basic', 'desc': 'test'}

net = SCLTFSTgramMFN(cfg=cfg, num_classes=41, m=0.4).to(DEV)
net.load_state_dict(torch.load(CKPT, map_location=DEV), strict=True)
net.eval()

name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
ds = train_dataset('data/dataset', name_list, cfg)
labels = ds.labels.numpy()

# stratified subsample: ~16 clips per machine ID
rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(np.where(labels == c)[0], 16, replace=False)
                      for c in np.unique(labels)])
rng.shuffle(idx)
print(f'{len(idx)} clips across {len(np.unique(labels))} machine IDs')

MEL, TG, TF = [], [], []
with torch.no_grad():
    for s in range(0, len(idx), 32):
        b = idx[s:s+32]
        wav = torch.stack([ds[i][0] for i in b]).to(DEV)
        mel = torch.stack([ds[i][1] for i in b]).to(DEV)
        TG.append(net.tgramnet(wav).cpu())
        TF.append(net.TFgramNet(wav, True).cpu())
        MEL.append(mel.squeeze(1).cpu())
mel = torch.cat(MEL).double(); tg = torch.cat(TG).double(); tf = torch.cat(TF).double()
print('shapes:', tuple(mel.shape), tuple(tg.shape), tuple(tf.shape))

def cka(X, Y):
    """linear CKA, X/Y are N x D (rows = examples)"""
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    xty = (X.T @ Y).norm()**2
    return (xty / ((X.T @ X).norm() * (Y.T @ Y).norm())).item()

reps = {'Log-Mel': mel, 'Tgram': tg, 'TFgram': tf}
print('\n=== linear CKA, clip level (N x 128*313) ===')
flat = {k: v.reshape(v.shape[0], -1) for k, v in reps.items()}
print('=== linear CKA, frame level (N*313 x 128) ===')
frame = {k: v.permute(0, 2, 1).reshape(-1, 128) for k, v in reps.items()}
for lvl, R in (('clip ', flat), ('frame', frame)):
    for a, b in (('Tgram','TFgram'), ('Log-Mel','Tgram'), ('Log-Mel','TFgram')):
        print(f'  {lvl}  CKA({a:7s}, {b:7s}) = {cka(R[a], R[b]):.4f}')

# how much of TFgram is a *linear* function of Tgram, frame-wise?
def lin_r2(X, Y):
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    X1 = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype)], 1)
    W = torch.linalg.lstsq(X1, Y).solution
    return (1 - ((Y - X1 @ W)**2).sum() / (Y**2).sum()).item()

print('\n=== frame-level linear predictability (R^2) ===')
for a, b in (('Tgram','TFgram'), ('TFgram','Tgram'), ('Log-Mel','TFgram'), ('Log-Mel','Tgram')):
    print(f'  {a:7s} -> {b:7s} : R^2 = {lin_r2(frame[a], frame[b]):.4f}')
