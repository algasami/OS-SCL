import csv
import os
import sys
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from model.net import SCLTFSTgramMFN
from losses import ASDLoss
from dataloader import test_dataset
import yaml
from tqdm import tqdm
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
    parser.add_argument('--m', type=float, help='m value')
    parser.add_argument('--gpu_num', type=int, help='GPU number')
    parser.add_argument('--fussion', type=int, help='Fussion')
    parser.add_argument('--ht', type=str, help='Action function')
    parser.add_argument('--model_path', type=str, help='Action function')
    parser.add_argument('--d', action='store_true', help='Use development dataset for evaluation.')
    parser.add_argument('--e', action='store_true', help='Use evaluation dataset for evaluation.')
    parser.add_argument('--csv', type=str, nargs='?', const='auto',
                        help='Export the evaluation matrix as a .csv file. Give a path, or pass the '
                             'flag alone to write eval_<dev|eval>.csv next to the checkpoint.')
    parser.add_argument('--strict', dest='strict', action='store_true', default=None,
                        help='Load the checkpoint with strict=True (default). Use --no-strict for pruned models.')
    parser.add_argument('--no-strict', dest='strict', action='store_false',
                        help='Tolerate missing/unexpected keys, e.g. for pruned_model.pth.')
    return parser.parse_args()


CSV_HEADER = ['model', 'dataset', 'machine', 'id', 'scope', 'AUC', 'pAUC', 'mAUC']


def default_csv_path(model_path, dataset):
    return os.path.join(os.path.dirname(model_path) or '.', f'eval_{dataset}.csv')


def write_csv(csv_path, rows):
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    with open(csv_path, 'w', encoding='UTF-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in rows:
            writer.writerow([v if isinstance(v, str) else f'{v:.5f}' for v in row])
    print(f'Evaluation matrix written to {csv_path}')


def evaluator(net, test_loader, criterion, device, machine, cfg):
    net.eval()

    y_truet = []
    y_predt = []
    id_auc_pauc = {}
    progress_bar = tqdm(test_loader, desc=f'{machine} evaling', leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for x_wavs, x_mels, labels, AN_N_labels, id_label in progress_bar:
            x_wavs, x_mels, labels, AN_N_labels = x_wavs.to(device), x_mels.to(device), labels.to(
                device), AN_N_labels.to(device)
            id_label = ''.join(id_label)

            logits, _ = net(x_wavs, x_mels, labels, train=False)

            score = criterion(logits, labels)
            y_predt.extend(score.tolist())

            y_truet.extend(AN_N_labels.tolist())

            if id_label not in id_auc_pauc:
                id_auc_pauc[id_label] = {'y_true': [], 'y_pred': []}
            id_auc_pauc[id_label]['y_true'].extend(AN_N_labels.tolist())
            id_auc_pauc[id_label]['y_pred'].extend(score.tolist())

    id_results = {}
    for id_label, data in id_auc_pauc.items():
        y_true = data['y_true']
        y_pred = data['y_pred']
        auc = metrics.roc_auc_score(y_true, y_pred)
        pauc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
        id_results[id_label] = (auc, pauc)

    min_id = min(id_results, key=lambda k: id_results[k][0])
    min_auc = id_results[min_id][0]

    auc = metrics.roc_auc_score(y_truet, y_predt)
    pauc = metrics.roc_auc_score(y_truet, y_predt, max_fpr=0.1)
    print(f"{machine} | AUC: {auc:.4f} | pAUC: {pauc:.4f} | Min AUC: {min_auc:.4f}")
    return auc, pauc, min_auc, id_results


def main():
    args = parse_arguments()
    cfg = load_config('config.yaml')
    print('Original Configuration...')
    print(cfg)

    cfg = update_config(cfg, args)
    print('Updated Configuration...')
    print(cfg)

    device_num = cfg['gpu_num']
    device = torch.device(f'cuda:{device_num}')

    net = SCLTFSTgramMFN(m=cfg['m'], cfg=cfg).to(device)
    strict = cfg.get('strict', True)
    net.load_state_dict(torch.load(cfg["model_path"], map_location=device), strict=strict)
    net.eval()

    criterion = ASDLoss(reduction=False).to(device)

    data_loaders = {}
    name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']

    if args.d:
        root_path = 'data/dataset'
        dataset_name = 'dev'
        print("Using development dataset for evaluation.")
    elif args.e:
        root_path = 'data/eval_dataset'
        dataset_name = 'eval'
        print("Using evaluation dataset for evaluation.")
    else:
        print("Error: You must provide either --d for development or --e for evaluation.")
        sys.exit(1)

    model_name = cfg.get('model_name', os.path.basename(os.path.dirname(cfg['model_path'])))
    csv_rows = []

    avg_AUC = 0.
    avg_pAUC = 0.
    mini_auc = 0.
    for machine_type in name_list:
        test_ds = test_dataset(root_path, machine_type, name_list, cfg)
        test_dataloader = DataLoader(test_ds, batch_size=1, pin_memory=True, num_workers=8)
        data_loaders[machine_type] = test_dataloader

    for machine_type, test_dataloader in data_loaders.items():
        AUC, PAUC, mauc, id_results = evaluator(net, test_dataloader, criterion, device, machine_type, cfg)
        for id_label in sorted(id_results):
            id_auc, id_pauc = id_results[id_label]
            csv_rows.append([model_name, dataset_name, machine_type, id_label, 'id', id_auc, id_pauc, ''])
        csv_rows.append([model_name, dataset_name, machine_type, 'ALL', 'machine', AUC, PAUC, mauc])
        avg_AUC += AUC
        avg_pAUC += PAUC
        mini_auc += mauc

    avg_AUC /= len(name_list)
    avg_pAUC /= len(name_list)
    mini_auc /= len(name_list)
    csv_rows.append([model_name, dataset_name, 'ALL', 'ALL', 'overall', avg_AUC, avg_pAUC, mini_auc])
    print(f"Average AUC: {avg_AUC:.5f},  Average pAUC: {avg_pAUC:.5f} ,Average mAUC: {mini_auc:.5f}")

    if cfg.get('csv'):
        csv_path = cfg['csv']
        if csv_path == 'auto':
            csv_path = default_csv_path(cfg['model_path'], dataset_name)
        write_csv(csv_path, csv_rows)


if __name__ == '__main__':
    main()
