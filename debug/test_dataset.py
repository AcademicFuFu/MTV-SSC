import os
import pdb
import sys
import torch

from mmcv import Config
from argparse import ArgumentParser
from tools.pl_model import pl_model
from tools.dataset_dm import DataModule
from debug.utils import *


def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='./configs/semantic_kitti.py')
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--seed',
                        type=int,
                        default=7240,
                        help='random seed point')
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


def main():

    args, config = parse_config()
    data_dm = DataModule(config)
    data_dm.setup()

    # train
    print('---------------------train----------------------')
    dl = data_dm.train_dataloader()
    data_iter = iter(dl)
    desired_index = 0
    for _ in range(desired_index + 1):
        batch = next(data_iter)
    print('num_samples: ', len(dl))
    print_detail(batch, 'train')

    # val
    print('--------------------- val ----------------------')
    dl = data_dm.val_dataloader()
    data_iter = iter(dl)
    desired_index = 0
    for _ in range(desired_index + 1):
        batch = next(data_iter)
    print('num_samples: ', len(dl))
    print_detail(batch, 'val')

    # test
    print('--------------------- test ---------------------')
    dl = data_dm.test_dataloader()
    data_iter = iter(dl)
    desired_index = 0
    for _ in range(desired_index + 1):
        batch = next(data_iter)
    print('num_samples: ', len(dl))
    print_detail(batch, 'test')


if __name__ == '__main__':
    main()
