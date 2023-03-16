from argparse import ArgumentParser
import os
import numpy as np

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import torchmetrics

from dataset import LitAlignedDM
from model import LitAddaUnet

def parse_arguments():
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default='./predictions')
    parser.add_argument("--out_imsize", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_output", action='store_true')

    return parser.parse_args()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_arguments()

    os.makedirs(os.path.join(args.save_dir), exist_ok=True)

    model = LitAddaUnet.load_from_checkpoint(args.ckpt_path)
    model.eval().to(device)

    dm_test = LitAlignedDM(data_dir=args.data_dir,
                           out_imsize=args.out_imsize, bsize=1, 
                           num_workers=args.num_workers)
    dl_test = dm_test.test_dataloader()

    pearson_metric = torchmetrics.PearsonCorrCoef().to(device)
    mask = torch.tensor([False, False, True]).to(device)

    all_vanilla_ys = []
    all_pred_ys, all_ys = [], []

    ps = []
    with torch.no_grad():
        for i, (xs, ys) in enumerate(dl_test):
            xs = xs.to(device)
            ys = ys.to(device)

            pred_ys = model.G(xs)[:, mask, :, :]
            vanilla_ys = model.G_A(xs)[:, mask, :, :]
            ys = ys[:, mask, :, :]

            p = pearson_metric(pred_ys.flatten(), ys.flatten())
            ps.append(p.cpu().item())
            all_pred_ys.append(pred_ys.cpu().numpy())
            all_vanilla_ys.append(vanilla_ys.cpu().numpy())
            all_ys.append(ys.cpu().numpy())

    if args.save_name is None:
        print(np.mean(ps))
    else:
        all_pred_ys = np.concatenate(all_pred_ys, axis=0)
        all_ys = np.concatenate(all_ys, axis=0)
        save_path = os.path.join(args.save_dir, f'{args.save_name}.npy')
        np.save(save_path, all_pred_ys)

        if args.save_output:
            save_path = os.path.join(args.save_dir, 'gt.npy')
            np.save(save_path, all_ys)

            all_vanilla_ys = np.concatenate(all_vanilla_ys, axis=0)
            save_path = os.path.join(args.save_dir, 'vanilla.npy')
            np.save(save_path, all_vanilla_ys)

if __name__ == '__main__':
    main()

