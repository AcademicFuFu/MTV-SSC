import os
import pdb
import sys
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
    parser.add_argument('--train_or_test', default='train')
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
    data_dm = DataModule(config)
    data_dm.setup()

    seed = config.seed
    pl.seed_everything(seed)
    model = pl_model(config)

    if args.ckpt_path is not None:
        print('load ckpt: ', args.ckpt_path)
        model.load_state_dict(torch.load(args.ckpt_path)['state_dict'], strict=False)

    # 计算模型的所有可训练参数总量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params/np.power(10, 6):.2f}M")

    if args.train_or_test == 'train':
        print('---------------------train----------------------')
        dl = data_dm.train_dataloader()
        data_iter = iter(dl)
        desired_index = 0
        for _ in range(desired_index + 1):
            batch = next(data_iter)

        if torch.cuda.is_available():
            model = model.cuda().eval()
            tocuda(batch)
            print_detail(batch)
        model_out = model.model.forward_train(batch)
        mem()
        print_detail(model_out)

    elif args.train_or_test == 'val':
        print('--------------------- val ----------------------')
        dl = data_dm.val_dataloader()
        data_iter = iter(dl)
        desired_index = 0
        for _ in range(desired_index + 1):
            batch = next(data_iter)

        if torch.cuda.is_available():
            model = model.cuda().eval()
            tocuda(batch)
            print_detail(batch)

        model_out = model.model.forward_test(batch)
        mem()
        print_detail(model_out)

    elif args.train_or_test == 'test':
        print('--------------------- test ---------------------')
        dl = data_dm.test_dataloader()
        data_iter = iter(dl)
        desired_index = 0
        for _ in range(desired_index + 1):
            batch = next(data_iter)

        if torch.cuda.is_available():
            model = model.cuda().eval()
            tocuda(batch)
            print_detail(batch)

        model_out = model.model.forward_test(batch)
        mem()
        print_detail(model_out)


if __name__ == '__main__':
    main()
