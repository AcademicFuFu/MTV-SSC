from mmdet3d.models.builder import BACKBONES
import pdb
import torch
from mmcv.runner import BaseModule
from mmdet3d.models import builder
import torch.nn as nn
import torch.nn.functional as F

from debug.utils import print_detail as pd, mem, save_feature_map_as_image, count_trainable_parameters as param


class WeightedAvgPool3D(nn.Module):

    def __init__(self, dim):
        super(WeightedAvgPool3D, self).__init__()
        self.dim = dim

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


class TPVPooler(BaseModule):

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


class TPVPoolerV1(BaseModule):

    def __init__(
        self,
        embed_dims=128,
        split=[8, 8, 8],
        grid_size=[128, 128, 16],
    ):
        super().__init__()
        self.weights_conv = nn.Sequential(nn.Conv3d(embed_dims, 3, kernel_size=1, bias=False))
        self.pool_xy = WeightedAvgPool3D(dim='xy')

        self.pool_yz = WeightedAvgPool3D(dim='yz')

        self.pool_zx = WeightedAvgPool3D(dim='zx')

        in_channels = embed_dims
        out_channels = embed_dims

        self.mlp_xy = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1), nn.ReLU(),
                                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1))

        self.mlp_yz = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1), nn.ReLU(),
                                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1))

        self.mlp_zx = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1), nn.ReLU(),
                                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1))

    def forward(self, x):
        weights = self.weights_conv(x)
        tpv_xy = self.mlp_xy(self.pool_xy(x, weights))
        tpv_yz = self.mlp_yz(self.pool_yz(x, weights))
        tpv_zx = self.mlp_zx(self.pool_zx(x, weights))

        tpv_list = [tpv_xy, tpv_yz, tpv_zx]

        return tpv_list


@BACKBONES.register_module()
class TPVTransformer_Cam_V0(BaseModule):

    def __init__(
        self,
        embed_dims=128,
        split=[8, 8, 8],
        grid_size=[128, 128, 16],
        global_encoder_backbone=None,
        global_encoder_neck=None,
    ):
        super().__init__()

        # max pooling
        self.tpv_pooler = TPVPooler(embed_dims=embed_dims, split=split, grid_size=grid_size)

        self.global_encoder_backbone = builder.build_backbone(global_encoder_backbone)
        self.global_encoder_neck = builder.build_neck(global_encoder_neck)

    def forward(self, x):
        """
        xy: [b, c, h, w, z] -> [b, c, h, w]
        yz: [b, c, h, w, z] -> [b, c, w, z]
        zx: [b, c, h, w, z] -> [b, c, h, z]
        """
        x_3view = self.tpv_pooler(x)
        x_3view = self.global_encoder_backbone(x_3view)

        tpv_list = []
        for x_tpv in x_3view:
            x_tpv = self.global_encoder_neck(x_tpv)
            if not isinstance(x_tpv, torch.Tensor):
                x_tpv = x_tpv[0]
            tpv_list.append(x_tpv)
        tpv_list[0] = F.interpolate(tpv_list[0], size=(128, 128), mode='bilinear').unsqueeze(-1)
        tpv_list[1] = F.interpolate(tpv_list[1], size=(128, 16), mode='bilinear').unsqueeze(2)
        tpv_list[2] = F.interpolate(tpv_list[2], size=(128, 16), mode='bilinear').unsqueeze(3)

        return tpv_list


@BACKBONES.register_module()
class TPVTransformer_Cam_V1(BaseModule):

    def __init__(
        self,
        embed_dims=128,
        split=[8, 8, 8],
        grid_size=[128, 128, 16],
        global_encoder_backbone=None,
    ):
        super().__init__()

        # max pooling
        self.tpv_pooler = TPVPooler(embed_dims=embed_dims, split=split, grid_size=grid_size)

        self.global_encoder_backbone = builder.build_backbone(global_encoder_backbone)

    def forward(self, x):
        """
        xy: [b, c, h, w, z] -> [b, c, h, w]
        yz: [b, c, h, w, z] -> [b, c, w, z]
        zx: [b, c, h, w, z] -> [b, c, h, z]
        """
        x_3view = self.tpv_pooler(x)
        tpv_list, out_tpv_feature_lists, out_tpv_residue_lists = self.global_encoder_backbone(x_3view)

        tpv_list[0] = F.interpolate(tpv_list[0], size=(128, 128), mode='bilinear').unsqueeze(-1)
        tpv_list[1] = F.interpolate(tpv_list[1], size=(128, 16), mode='bilinear').unsqueeze(2)
        tpv_list[2] = F.interpolate(tpv_list[2], size=(128, 16), mode='bilinear').unsqueeze(3)

        return tpv_list


@BACKBONES.register_module()
class TPVAggregator_Cam_V0(BaseModule):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, tpv_list):
        feats_xy, feats_yz, feats_zx = tpv_list
        x, y, z = feats_xy.shape[2], feats_xy.shape[3], feats_yz.shape[4]
        feats_xy = feats_xy.repeat(1, 1, 1, 1, z)
        feats_yz = feats_yz.repeat(1, 1, x, 1, 1)
        feats_zx = feats_zx.repeat(1, 1, 1, y, 1)
        out_feats = feats_xy + feats_yz + feats_zx
        return [out_feats]


@BACKBONES.register_module()
class TPVAggregator_Cam_V1(BaseModule):

    def __init__(self, embed_dims=128, **kwargs):
        super().__init__()
        self.combine_coeff = nn.Sequential(nn.Conv3d(embed_dims, 3, kernel_size=1, bias=False), nn.Softmax(dim=1))

    def forward(self, tpv_list, x3d):
        weights = self.combine_coeff(x3d)
        out_feats = self.weighted_sum(tpv_list, weights)
        return [out_feats], weights

    def weighted_sum(self, global_feats, weights):
        out_feats = global_feats[0] * weights[:, 0:1, ...]
        for i in range(1, len(global_feats)):
            out_feats += global_feats[i] * weights[:, i:i + 1, ...]
        return out_feats
