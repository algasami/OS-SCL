import glob
import itertools
import os

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


class Wave2Mel(object):
    def __init__(self, sr=16000,
                 n_fft=2048,
                 n_mels=128,
                 win_length=2048,
                 hop_length=512,
                 power=2.0
                 ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power,
                                                                  pad_mode='constant',
                                                                  norm='slaney',
                                                                  mel_scale='slaney'
                                                                  )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        return self.amplitude_to_db(self.mel_transform(x))


def machine_ids(name):
    """The machine's IDs, in class-index order. Changing this invalidates every checkpoint."""
    if name == 'ToyConveyor':
        return ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06']
    if name == 'ToyCar':
        return ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07']
    return ['id_00', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06']


def load_wav(file_name):
    """(1, 160000) float waveform: the first 10 s at 16 kHz."""
    y, sr = torchaudio.load(file_name)
    return y[..., :sr * 10], sr


def test_index(root_path, name_list):
    """test_dataset's clips for every machine, in its order, without loading audio.

    Rows are (file, machine, id, class label, y_true), y_true 1 for anomalies.
    """
    rows = []
    for name in name_list:
        d = os.path.join(root_path, name, 'test')
        files = [(f, 0) for f in sorted(glob.glob(f'{d}/normal_*'))] + \
                [(f, 1) for f in sorted(glob.glob(f'{d}/anomaly_*'))]
        init = sum(len(machine_ids(n)) for n in name_list[:name_list.index(name)])
        for f, y in files:
            hits = [i for i, lab in enumerate(machine_ids(name)) if lab in f]
            assert len(hits) == 1, f
            rows.append((f, name, machine_ids(name)[hits[0]], init + hits[0], y))
    return rows


def cached_item_ok(wav, mel):
    """Range check for a cached item: 16-bit PCM lies in [-1, 1] and the dB mel
    (amin 1e-10, ref 1) in about [-100, 100]. Catches the exponent and sign-bit
    flips that smartlab1's faulty RAM produces (ablation/fast-recipe)."""
    return bool(np.isfinite(mel).all() and np.abs(mel).max() <= 200 and np.abs(wav).max() <= 1.0)


class test_dataset(torch.utils.data.Dataset):
    def __init__(self, root_path, test_name, name_list, cfg):
        dataset_dir = os.path.join(root_path, test_name, 'test')
        normal_files = sorted(glob.glob('{dir}/normal_*'.format(dir=dataset_dir)))
        anomaly_files = sorted(glob.glob('{dir}/anomaly_*'.format(dir=dataset_dir)))

        self.test_files = np.concatenate((normal_files, anomaly_files), axis=0)
        self.cfg = cfg

        self.Logmel = Wave2Mel()

        normal_labels = np.zeros(len(normal_files))
        anomaly_labels = np.ones(len(anomaly_files))
        self.y_true = torch.LongTensor(np.concatenate((normal_labels, anomaly_labels), axis=0))

        target_idx = name_list.index(test_name)

        label_init_num = 0
        for i, name in enumerate(name_list):
            if i == target_idx:
                break
            label_init_num += len(self._get_label_list(name))

        self.labels = []
        self.id_labels = []
        label_list = self._get_label_list(test_name)
        for file_name in self.test_files:
            for idx, label_idx in enumerate(label_list):
                if label_idx in file_name:
                    self.labels.append(idx + label_init_num)
                    self.id_labels.append(label_idx)

        self.labels = torch.LongTensor(self.labels)

        self.y_list = []
        self.y_spec_list = []

        for i in tqdm(range(len(self.test_files)), desc=f'{test_name} Log-Mel Convetor', dynamic_ncols=True):
            y, sr = self._file_load(self.test_files[i])
            y_specgram = self.Logmel(y)
            self.y_list.append(y)
            self.y_spec_list.append(y_specgram)

    def __getitem__(self, idx):
        anomal_label = self.y_true[idx]
        label = self.labels[idx]
        id_label = self.id_labels[idx]
        return self.y_list[idx], self.y_spec_list[idx], label, anomal_label, id_label

    def __len__(self):
        return len(self.test_files)

    def _file_load(self, file_name):
        try:
            return load_wav(file_name)
        except:
            print("file_broken or not exists!! : {}".format(file_name))

    def _get_label_list(self, name):
        return machine_ids(name)


class train_dataset(torch.utils.data.Dataset):
    def __init__(self, root_path, name_list, cfg):
        data_path = [os.path.join(root_path, name, 'train') for name in name_list]
        files_list = [self._file_list_generator(target_path) for target_path in data_path]

        self.labels = []
        self.cfg = cfg

        self.Logmel = Wave2Mel()

        maximum = 0
        for i, files in enumerate(files_list):
            label_list = self._get_label_list(name_list[i])
            for file_name in files:
                for idx, label_idx in enumerate(label_list):
                    if label_idx in file_name:
                        self.labels.append(idx + maximum)
            maximum = max(self.labels) + 1

        self.unrolled_files_list = list(itertools.chain.from_iterable(files_list))

        unique_labels = set(self.labels)

        num_unique_labels = len(unique_labels)

        print(f"class :{num_unique_labels}")

        self.labels = torch.LongTensor(self.labels)

        # --cache_dir: read the arrays cache_data.py wrote (the same values) instead of
        # decoding each wav and computing its mel per item. Opened lazily per worker.
        self.cache_dir = cfg.get('cache_dir') if cfg else None
        self._cache = None
        if self.cache_dir:
            with open(os.path.join(self.cache_dir, 'train_files.txt')) as f:
                cached = f.read().split()
            if cached != self.unrolled_files_list:
                raise SystemExit(f'{self.cache_dir}/train_files.txt does not match the training file list')

    def __getitem__(self, idx):
        if self.cache_dir:
            if self._cache is None:
                self._cache = tuple(np.load(os.path.join(self.cache_dir, f'train_{k}.npy'), mmap_mode='r')
                                    for k in ('wav', 'mel'))
            wav, mel = self._cache
            w, m = np.array(wav[idx]), np.array(mel[idx])
            if not cached_item_ok(w, m):
                # an exponent flip in a cached float32 mel reached 1e11 and NaN'd a run
                print(f'cache: item {idx} out of range (|mel| max {np.abs(m).max():.3g}); recomputed', flush=True)
                y, _ = self._file_load(self.unrolled_files_list[idx])
                return y, self.Logmel(y), self.labels[idx]
            return torch.from_numpy(w), torch.from_numpy(m), self.labels[idx]

        y, sr = self._file_load(self.unrolled_files_list[idx])
        y_specgram = self.Logmel(y)
        return y, y_specgram, self.labels[idx]

    def __len__(self):
        return len(self.unrolled_files_list)

    def _get_label_list(self, name):
        return machine_ids(name)

    def _file_list_generator(self, target_dir):
        training_list_path = os.path.abspath('{dir}/*.wav'.format(dir=target_dir))
        files = sorted(glob.glob(training_list_path))
        if len(files) == 0:
            print('no_wav_file!!')
        return files

    def _file_load(self, file_name):
        try:
            return load_wav(file_name)
        except:
            print("file_broken or not exists!! : {}".format(file_name))
