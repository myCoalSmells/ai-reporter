import numpy as np
from argparse import ArgumentParser
from tqdm.auto import tqdm
import os
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
import torchmetrics

from captum.attr import LayerActivation

from dataset import LitAlignedDM
from model import LitAddaUnet

def get_named_module(model, name):
    for module_name, module in model.named_modules():
        if module_name == name:
            return module
    raise ValueError(f"{name} not found in model.")

def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--load_npy", type=str)
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--save_dir", type=str, default='./predictions')
    parser.add_argument("--bsize", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--proj_layer_name", type=str, default='model.11')

    return parser.parse_args()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_arguments()

    os.makedirs(os.path.join(args.save_dir), exist_ok=True)

    model = LitAddaUnet.load_from_checkpoint(args.ckpt_path)
    model.eval().to(device)

    proj_layer = get_named_module(model.D, args.proj_layer_name)
    layer_act_fn = LayerActivation(model.D, proj_layer)

    data = np.load(args.load_npy)
    data = np.concatenate([np.zeros_like(data), np.zeros_like(data), data], axis=1)
    dset = TensorDataset(torch.from_numpy(data))
    dl = DataLoader(dset, batch_size=args.bsize, num_workers=args.num_workers)

    acts = []
    with torch.no_grad():
        for xs_, in tqdm(dl, leave=False):
            xs_ = xs_.to(device)
            acts_ = layer_act_fn.attribute(xs_, attribute_to_layer_input=True)
            acts_ = acts_.cpu().numpy()
            acts.append(acts_)

    acts = np.concatenate(acts, axis=0)
    np.save(os.path.join(args.save_dir, args.save_name), acts)

if __name__ == '__main__':
    main()

