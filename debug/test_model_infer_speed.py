import os
import pdb
import sys
import time
import torch
import pytorch_lightning as pl

from mmcv import Config
from argparse import ArgumentParser
from tools.pl_model import pl_model
from tools.dataset_dm import DataModule
from debug.utils import *


def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='./configs/semantic_kitti.py')
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--seed', type=int, default=7240, help='random seed point')
    parser.add_argument('--log_folder', default='semantic_kitti')
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--test_mapping', action='store_true')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--log_every_n_steps', type=int, default=1000)

    args = parser.parse_args()
    cfg = Config.fromfile(args.config_path)

    cfg.update(vars(args))
    return args, cfg


def tocuda(batch):

    for k in batch.keys():
        if type(batch[k]) == torch.Tensor:
            batch[k] = batch[k].to('cuda')
        elif type(batch[k]) == dict:
            for kk in batch[k].keys():
                if type(batch[k][kk]) == torch.Tensor:
                    batch[k][kk] = batch[k][kk].to('cuda')
        elif type(batch[k]) == list:
            for i in range(len(batch[k])):
                if type(batch[k][i]) == torch.Tensor:
                    batch[k][i] = batch[k][i].to('cuda')


def main():
    args, config = parse_config()
    seed = config.seed
    pl.seed_everything(seed)

    model = pl_model(config)
    if torch.cuda.is_available():
        model = model.cuda()
    model = model.eval()

    data_dm = DataModule(config)
    data_dm.setup()
    dl = data_dm.train_dataloader()
    data_iter = iter(dl)

    i_start = 20
    i_end = 120
    sum_t = 0
    for i in range(i_end):
        print(f"Batch {i}", end='')
        batch = next(data_iter)
        if torch.cuda.is_available():
            bath = tocuda(batch)
        t_start = time.time()
        model_out = model.model.forward_test(batch)
        t_end = time.time()
        if i > i_start:
            sum_t += t_end - t_start
    print()
    print(f"Average time: {sum_t / (i_end - i_start)}")


if __name__ == '__main__':
    main()
