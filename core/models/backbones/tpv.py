from mmdet3d.models.builder import BACKBONES
import torch
from mmcv.runner import BaseModule
from mmdet3d.models import builder
import torch.nn as nn
import torch.nn.functional as F
import pdb
from debug.utils import print_detail as pd, mem, save_feature_map_as_image


class TPVMaxPooler(BaseModule):

    def __init__(
        self,
        embed_dims=128,
        split=[8, 8, 8],
        grid_size=[128, 128, 16],
    ):
        super().__init__()
        self.pool_xy = nn.MaxPool3d(kernel_size=[1, 1, grid_size[2] // split[2]],
                                    stride=[1, 1, grid_size[2] // split[2]],
                                    padding=0)

        self.pool_yz = nn.MaxPool3d(kernel_size=[grid_size[0] // split[0], 1, 1],
                                    stride=[grid_size[0] // split[0], 1, 1],
                                    padding=0)

        self.pool_zx = nn.MaxPool3d(kernel_size=[1, grid_size[1] // split[1], 1],
                                    stride=[1, grid_size[1] // split[1], 1],
                                    padding=0)

        in_channels = [int(embed_dims * s) for s in split]
        out_channels = [int(embed_dims) for s in split]

        self.mlp_xy = nn.Sequential(nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1), nn.ReLU(),
                                    nn.Conv2d(out_channels[2], out_channels[2], kernel_size=1, stride=1))

        self.mlp_yz = nn.Sequential(nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1), nn.ReLU(),
                                    nn.Conv2d(out_channels[0], out_channels[0], kernel_size=1, stride=1))

        self.mlp_zx = nn.Sequential(nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, stride=1), nn.ReLU(),
                                    nn.Conv2d(out_channels[1], out_channels[1], kernel_size=1, stride=1))

    def forward(self, x):
        tpv_xy = self.mlp_xy(self.pool_xy(x).permute(0, 4, 1, 2, 3).flatten(start_dim=1, end_dim=2))
        tpv_yz = self.mlp_yz(self.pool_yz(x).permute(0, 2, 1, 3, 4).flatten(start_dim=1, end_dim=2))
        tpv_zx = self.mlp_zx(self.pool_zx(x).permute(0, 3, 1, 2, 4).flatten(start_dim=1, end_dim=2))

        tpv_list = [tpv_xy, tpv_yz, tpv_zx]

        return tpv_list


class GlobalWeightedAvgPool3D(nn.Module):

    def __init__(self, dim, split, grid_size):
        super(GlobalWeightedAvgPool3D, self).__init__()
        self.dim = dim
        self.split = split
        self.grid_size = grid_size

    def forward(self, x, weights):
        # x : b, c, h, w, z
        # weights : b, 3, h, w, z
        if self.dim == 'xy':
            weight = F.softmax(weights[:, 0:1, :, :, :], dim=-1)
            feat = (x * weight).sum(dim=-1)
        elif self.dim == 'yz':
            weight = F.softmax(weights[:, 1:2, :, :, :], dim=-3)
            feat = (x * weight).sum(dim=-3)
        elif self.dim == 'zx':
            weight = F.softmax(weights[:, 2:3, :, :, :], dim=-2)
            feat = (x * weight).sum(dim=-2)

        return feat


class LocalWeightedAvgPool3D(nn.Module):

    def __init__(self, dim, split, grid_size):
        super(LocalWeightedAvgPool3D, self).__init__()
        self.dim = dim
        self.split = split
        self.grid_size = grid_size

    def forward(self, x, weights):
        # x : b, c, h, w, z
        # weights : b, 3, h, w, z

        b, c, h, w, z = x.size()
        if self.dim == 'xy':
            weight = weights[:, 0:1, :, :, :].reshape(b, 1, h, w, self.split[2], z // self.split[2]).softmax(dim=-1)
            x = x.reshape(b, c, h, w, self.split[2], z // self.split[2])
            feat = (x * weight).sum(dim=-1).permute(0, 1, 4, 2, 3).flatten(start_dim=1, end_dim=2)
        elif self.dim == 'yz':
            weight = weights[:, 1:2, :, :, :].reshape(b, 1, self.split[0], h // self.split[0], w, z).softmax(dim=-3)
            x = x.reshape(b, c, self.split[0], h // self.split[0], w, z)
            feat = (x * weight).sum(dim=-3).permute(0, 1, 2, 3, 4).flatten(start_dim=1, end_dim=2)
        elif self.dim == 'zx':
            weight = weights[:, 2:3, :, :, :].reshape(b, 1, h, self.split[1], w // self.split[1], z).softmax(dim=-2)
            x = x.reshape(b, c, h, self.split[1], w // self.split[1], z)
            feat = (x * weight).sum(dim=-2).permute(0, 1, 3, 2, 4).flatten(start_dim=1, end_dim=2)

        return feat


class TPVWeightedAvgPooler(BaseModule):

    def __init__(
        self,
        embed_dims=128,
        split=[8, 8, 8],
        grid_size=[128, 128, 16],
        pool_type='global',
    ):
        super().__init__()
        self.weights_conv = nn.Sequential(nn.Conv3d(embed_dims, 3, kernel_size=1, bias=False))
        if pool_type == 'global':
            self.pool_xy = GlobalWeightedAvgPool3D(dim='xy', split=split, grid_size=grid_size)
            self.pool_yz = GlobalWeightedAvgPool3D(dim='yz', split=split, grid_size=grid_size)
            self.pool_zx = GlobalWeightedAvgPool3D(dim='zx', split=split, grid_size=grid_size)
            in_channels = [embed_dims for _ in split]
            out_channels = [embed_dims for _ in split]

        elif pool_type == 'local':
            self.pool_xy = LocalWeightedAvgPool3D(dim='xy', split=split, grid_size=grid_size)
            self.pool_yz = LocalWeightedAvgPool3D(dim='yz', split=split, grid_size=grid_size)
            self.pool_zx = LocalWeightedAvgPool3D(dim='zx', split=split, grid_size=grid_size)
            in_channels = [int(embed_dims * s) for s in split]
            out_channels = [int(embed_dims) for s in split]

        self.mlp_xy = nn.Sequential(nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1), nn.ReLU(),
                                    nn.Conv2d(out_channels[2], out_channels[2], kernel_size=1, stride=1))

        self.mlp_yz = nn.Sequential(nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1), nn.ReLU(),
                                    nn.Conv2d(out_channels[0], out_channels[0], kernel_size=1, stride=1))

        self.mlp_zx = nn.Sequential(nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, stride=1), nn.ReLU(),
                                    nn.Conv2d(out_channels[1], out_channels[1], kernel_size=1, stride=1))

    def forward(self, x):
        weights = self.weights_conv(x)
        tpv_xy = self.mlp_xy(self.pool_xy(x, weights))
        tpv_yz = self.mlp_yz(self.pool_yz(x, weights))
        tpv_zx = self.mlp_zx(self.pool_zx(x, weights))

        tpv_list = [tpv_xy, tpv_yz, tpv_zx]

        return tpv_list


@BACKBONES.register_module()
class TPVGenerator(BaseModule):

    def __init__(
        self,
        embed_dims=128,
        split=[8, 8, 8],
        grid_size=[128, 128, 16],
        pooler='avg',
        pool_type='global',
        global_encoder_backbone=None,
        global_encoder_neck=None,
    ):
        super().__init__()

        # pooling
        if pooler == 'avg':
            self.tpv_pooler = TPVWeightedAvgPooler(embed_dims=embed_dims, split=split, grid_size=grid_size, pool_type=pool_type)
        elif pooler == 'max':
            self.tpv_pooler = TPVMaxPooler(embed_dims=embed_dims, split=split, grid_size=grid_size)

        self.global_encoder_backbone = builder.build_backbone(global_encoder_backbone)
        self.global_encoder_neck = builder.build_neck(global_encoder_neck)

    def forward(self, x):
        """
        xy: [b, c, h, w, z] -> [b, c, h, w]
        yz: [b, c, h, w, z] -> [b, c, w, z]
        zx: [b, c, h, w, z] -> [b, c, h, z]
        """

        x_tpv = self.tpv_pooler(x)

        x_tpv = self.global_encoder_backbone(x_tpv)

        tpv_list = []
        neck_out = []
        for view in x_tpv:
            view = self.global_encoder_neck(view)
            neck_out.append(view)
            if not isinstance(view, torch.Tensor):
                view = view[0]
            tpv_list.append(view)

        feats_all = dict()
        feats_all['feats3d'] = x
        feats_all['tpv_backbone'] = x_tpv
        feats_all['tpv_neck'] = neck_out

        # xy
        tpv_list[0] = F.interpolate(tpv_list[0], size=(128, 128), mode='bilinear', align_corners=False).unsqueeze(-1)
        # yz
        tpv_list[1] = F.interpolate(tpv_list[1], size=(128, 16), mode='bilinear', align_corners=False).unsqueeze(2)
        # zx
        tpv_list[2] = F.interpolate(tpv_list[2], size=(128, 16), mode='bilinear', align_corners=False).unsqueeze(3)

        return tpv_list, feats_all


@BACKBONES.register_module()
class TPVAggregator(BaseModule):

    def __init__(self, embed_dims=128):
        super().__init__()
        # self.combine_coeff = nn.Sequential(nn.Conv3d(embed_dims, sum(num_views), kernel_size=1, bias=False))
        self.combine_coeff = nn.Conv3d(embed_dims, 3, kernel_size=1, bias=False)

    def forward(self, tpv_list, x3d):
        weights = self.combine_coeff(x3d)
        out_feats = self.weighted_sum(tpv_list, F.softmax(weights, dim=1))

        return [out_feats], weights

    def weighted_sum(self, global_feats, weights):
        out_feats = global_feats[0] * weights[:, 0:1, ...]
        for i in range(1, len(global_feats)):
            out_feats += global_feats[i] * weights[:, i:i + 1, ...]
        return out_feats


@BACKBONES.register_module()
class TPVAggregatorV1(BaseModule):

    def __init__(
        self,
        embed_dims=128,
    ):
        super().__init__()
        self.combine_coeff = nn.Conv3d(embed_dims, 3, kernel_size=1, bias=False)

    def forward(self, tpv_list, x3d):
        b, c, h, w, z = x3d.size()
        weights = torch.ones([b, 4, h, w, z], device=x3d.device)
        x3d_ = self.weighted_sum([*tpv_list, x3d], weights)
        weights = self.combine_coeff(x3d_)
        out_feats = self.weighted_sum(tpv_list, F.softmax(weights, dim=1))

        return [out_feats], weights

    def weighted_sum(self, global_feats, weights):
        out_feats = global_feats[0] * weights[:, 0:1, ...]
        for i in range(1, len(global_feats)):
            out_feats += global_feats[i] * weights[:, i:i + 1, ...]
        return out_feats


@BACKBONES.register_module()
class TPVAggregatorV2(BaseModule):

    def __init__(
        self,
        embed_dims=128,
    ):
        super().__init__()
        self.combine_coeff = nn.Conv3d(embed_dims, 4, kernel_size=1, bias=False)

    def forward(self, tpv_list, x3d):
        b, c, h, w, z = x3d.size()
        weights = torch.ones([b, 4, h, w, z], device=x3d.device)
        x3d_ = self.weighted_sum([*tpv_list, x3d], weights)
        weights = self.combine_coeff(x3d_)
        out_feats = self.weighted_sum([*tpv_list, x3d], F.softmax(weights, dim=1))

        return [out_feats], weights

    def weighted_sum(self, global_feats, weights):
        out_feats = global_feats[0] * weights[:, 0:1, ...]
        for i in range(1, len(global_feats)):
            out_feats += global_feats[i] * weights[:, i:i + 1, ...]
        return out_feats
