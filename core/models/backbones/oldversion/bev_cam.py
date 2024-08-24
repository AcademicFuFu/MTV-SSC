from mmdet3d.models.builder import BACKBONES
import pdb
import torch
from mmcv.runner import BaseModule
from mmdet3d.models import builder
import torch.nn as nn
import torch.nn.functional as F

from debug.utils import print_detail as pd, mem, save_feature_map_as_image, count_trainable_parameters as param


def normal_cdf(x, mean=0.0, std=1.0):
    # 将 x 标准化
    z = (x - mean) / std
    # 使用 0.5 * (1 + erf(z / sqrt(2))) 公式计算 CDF
    return 0.5 * (1 + torch.erf(z / torch.sqrt(torch.tensor(2.0))))


class MultiViewNormalDistWeightedPool3D(nn.Module):

    def __init__(self, num_views, dim='xy'):
        super().__init__()
        assert num_views > 1
        assert dim in ['xy', 'yz', 'zx']

        self.num_views = num_views
        self.sigmas = nn.Parameter(torch.ones(num_views), requires_grad=True)
        self.dim = dim

    def forward(self, x):
        if self.dim == 'yz':
            x = x.permute(0, 1, 3, 4, 2)
        elif self.dim == 'zx':
            x = x.permute(0, 1, 2, 4, 3)
        x = x.permute(0, 2, 3, 4, 1)

        device = x.device
        b, h, w, z, c = x.shape
        num_views = self.num_views
        sigmas = self.sigmas * torch.sqrt(torch.tensor(z, device=device))
        mus = [int(i / (num_views - 1) * z) for i in range(0, num_views)]

        mtv_out = []
        for i in range(num_views):
            mu = mus[i]
            sigma = sigmas[i]
            weights = torch.empty(z, dtype=torch.float32, device=device)
            for j in range(z):
                Z_l = normal_cdf(j, mu, sigma)
                Z_h = normal_cdf(j + 1, mu, sigma)
                weights[j] = Z_h - Z_l
            # pdb.set_trace()
            weights = weights / weights.sum()
            out_i = torch.sum(x * weights.view(1, 1, 1, z, 1), dim=3).permute(0, 3, 1, 2)
            mtv_out.append(out_i)

        return mtv_out


class BEVPooler(BaseModule):

    def __init__(
        self,
        embed_dims=128,
        split=8,
        grid_size=[128, 128, 16],
        num_feats=1,
    ):
        super().__init__()

        in_channel = int(embed_dims * split)
        out_channel = int(embed_dims)

        self.num_feats = num_feats
        if num_feats == 1:
            self.pool_xy = nn.MaxPool3d(kernel_size=[1, 1, grid_size[2] // split],
                                        stride=[1, 1, grid_size[2] // split],
                                        padding=0)
            self.mlp_xy = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1), nn.ReLU(),
                                        nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1))
        else:
            self.pool_xy = MultiViewNormalDistWeightedPool3D(num_views=self.num_feats, dim='xy')

    def forward(self, x):
        if self.num_feats == 1:
            bev_feats = [self.mlp_xy(self.pool_xy(x).permute(0, 4, 1, 2, 3).flatten(start_dim=1, end_dim=2))]
        else:
            bev_feats = self.pool_xy(x)

        return bev_feats


@BACKBONES.register_module()
class BEVTransformer_Cam_V0(BaseModule):

    def __init__(
        self,
        embed_dims=128,
        split=8,
        grid_size=[128, 128],
        num_feats=1,
        global_encoder_backbone=None,
        global_encoder_neck=None,
    ):
        super().__init__()

        self.bev_pooler = BEVPooler(embed_dims=embed_dims, split=split, grid_size=grid_size, num_feats=num_feats)
        self.global_encoder_backbone = builder.build_backbone(global_encoder_backbone)
        self.global_encoder_neck = builder.build_neck(global_encoder_neck)

    def forward(self, x):
        """
        xy: [b, c, h, w, z] -> [b, c, h, w]
        """
        x_bev = self.bev_pooler(x)
        x_bev = self.global_encoder_backbone(x_bev)[0]

        bev_list = []
        feat = self.global_encoder_neck(x_bev)
        if not isinstance(feat, torch.Tensor):
            feat = feat[0]
        bev_list.append(feat)

        for i in range(len(bev_list)):
            bev_list[i] = F.interpolate(bev_list[i], size=(128, 128), mode='bilinear')

        return bev_list


@BACKBONES.register_module()
class BEVAggregator_Cam_V0(BaseModule):

    def __init__(self, embed_dims=128, height=16, num_feats=1, **kwargs):
        super().__init__()
        self.height = height
        self.num_feats = num_feats
        if self.num_feats > 1:
            self.combine_coeff = nn.Sequential(nn.Conv3d(embed_dims, num_feats, kernel_size=1, bias=False), nn.Softmax(dim=1))

        self.out_conv = nn.Conv3d(embed_dims // height, embed_dims, kernel_size=1, stride=1)

    def forward(self, bev_list, x3d):
        for i in range(len(bev_list)):
            b, c, h, w = bev_list[i].shape
            bev_list[i] = self.out_conv(bev_list[i].view(b, c // self.height, self.height, h, w).permute(0, 1, 3, 4, 2))

        if self.num_feats > 1:
            weights = self.combine_coeff(x3d)
            out_feats = self.weighted_sum(bev_list, weights)
            return [out_feats], weights
        else:
            out_feats = bev_list[0]
            return [out_feats], None

    def weighted_sum(self, global_feats, weights):
        out_feats = global_feats[0] * weights[:, 0:1, ...]
        for i in range(1, len(global_feats)):
            out_feats += global_feats[i] * weights[:, i:i + 1, ...]
        return out_feats
