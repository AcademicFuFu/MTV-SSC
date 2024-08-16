from mmdet3d.models.builder import BACKBONES
import torch
from mmcv.runner import BaseModule
from mmdet3d.models import builder
import pdb
import torch.nn as nn
import torch.nn.functional as F
import pdb

from debug.utils import print_detail as pd, mem, save_feature_map_as_image


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


class MPVPooler(BaseModule):

    def __init__(
        self,
        embed_dims=128,
        split=[8, 8, 8],
        grid_size=[128, 128, 16],
        num_views_xyz=[1, 1, 1],
    ):
        super().__init__()
        self.num_views = num_views_xyz
        in_channels = [int(embed_dims * s) for s in split]
        out_channels = [int(embed_dims) for s in split]

        # xy
        if self.num_views[0] == 1:
            self.pool_xy = nn.MaxPool3d(kernel_size=[1, 1, grid_size[2] // split[2]],
                                        stride=[1, 1, grid_size[2] // split[2]],
                                        padding=0)
            self.mlp_xy = nn.Sequential(nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1), nn.ReLU(),
                                        nn.Conv2d(out_channels[2], out_channels[2], kernel_size=1, stride=1))
        else:
            self.pool_xy = MultiViewNormalDistWeightedPool3D(num_views=self.num_views[0], dim='xy')

        # yz
        if self.num_views[1] == 1:
            self.pool_yz = nn.MaxPool3d(kernel_size=[grid_size[0] // split[0], 1, 1],
                                        stride=[grid_size[0] // split[0], 1, 1],
                                        padding=0)
            self.mlp_yz = nn.Sequential(nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1), nn.ReLU(),
                                        nn.Conv2d(out_channels[0], out_channels[0], kernel_size=1, stride=1))
        else:
            self.pool_yz = MultiViewNormalDistWeightedPool3D(num_views=self.num_views[1], dim='yz')

        # zx
        if self.num_views[2] == 1:
            self.pool_zx = nn.MaxPool3d(kernel_size=[1, grid_size[1] // split[1], 1],
                                        stride=[1, grid_size[1] // split[1], 1],
                                        padding=0)
            self.mlp_zx = nn.Sequential(nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, stride=1), nn.ReLU(),
                                        nn.Conv2d(out_channels[1], out_channels[1], kernel_size=1, stride=1))
        else:
            self.pool_zx = MultiViewNormalDistWeightedPool3D(num_views=self.num_views[2], dim='zx')

    def forward(self, x):
        if self.num_views[0] == 1:
            mpv_xy = [self.mlp_xy(self.pool_xy(x).permute(0, 4, 1, 2, 3).flatten(start_dim=1, end_dim=2))]
        else:
            mpv_xy = self.pool_xy(x)

        if self.num_views[1] == 1:
            mpv_yz = [self.mlp_yz(self.pool_yz(x).permute(0, 2, 1, 3, 4).flatten(start_dim=1, end_dim=2))]
        else:
            mpv_yz = self.pool_yz(x)

        if self.num_views[2] == 1:
            mpv_zx = [self.mlp_zx(self.pool_zx(x).permute(0, 3, 1, 2, 4).flatten(start_dim=1, end_dim=2))]
        else:
            mpv_zx = self.pool_zx(x)

        mpv_list = [mpv_xy, mpv_yz, mpv_zx]

        return mpv_list


@BACKBONES.register_module()
class MTVTransformer_Cam_V0(BaseModule):

    def __init__(
        self,
        embed_dims=128,
        num_views=[1, 1, 1],
        split=[8, 8, 8],
        grid_size=[128, 128, 16],
        global_encoder_backbone=None,
        global_encoder_neck=None,
    ):
        super().__init__()

        # weighted pooling
        self.num_views = num_views
        self.mpv_pooler = MPVPooler(embed_dims=embed_dims, split=split, grid_size=grid_size, num_views_xyz=num_views)

        self.global_encoder_backbone = builder.build_backbone(global_encoder_backbone)
        self.global_encoder_neck = builder.build_neck(global_encoder_neck)

    def save_mpv(self, mpv_list):
        xy_list = [view for view in mpv_list if view.shape[-1] == 1]
        yz_list = [view for view in mpv_list if view.shape[-3] == 1]
        zx_list = [view for view in mpv_list if view.shape[-2] == 1]

        # format to b,n,c,h,w
        xy_list = [view.squeeze(-1).unsqueeze(1).permute(0, 1, 2, 3, 4) for view in xy_list]
        yz_list = [torch.flip(view.squeeze(-3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1]) for view in yz_list]
        zx_list = [torch.flip(view.squeeze(-2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1]) for view in zx_list]

        feats_xy = torch.cat(xy_list, dim=1)
        feats_yz = torch.cat(yz_list, dim=1)
        feats_zx = torch.cat(zx_list, dim=1)

        save_feature_map_as_image(feats_xy.detach(), 'save/mpv/pca', 'xy', method='pca')
        save_feature_map_as_image(feats_yz.detach(), 'save/mpv/pca', 'yz', method='pca')
        save_feature_map_as_image(feats_zx.detach(), 'save/mpv/pca', 'zx', method='pca')
        # save_feature_map_as_image(feats_xy.detach(), 'save/mpv/avg', 'xy', method='average')
        # save_feature_map_as_image(feats_yz.detach(), 'save/mpv/avg', 'yz', method='average')
        # save_feature_map_as_image(feats_zx.detach(), 'save/mpv/avg', 'zx', method='average')

        # remind to comment this function while training
        pdb.set_trace()
        return

    def forward(self, x):
        """
        xy: [b, c, h, w, z] -> [b, c, h, w]
        yz: [b, c, h, w, z] -> [b, c, w, z]
        zx: [b, c, h, w, z] -> [b, c, h, z]
        """

        # pdb.set_trace()
        x_multi_view = self.mpv_pooler(x)
        x_multi_view = self.global_encoder_backbone([*x_multi_view[0], *x_multi_view[1], *x_multi_view[2]])

        mpv_list = []
        for x_mpv in x_multi_view:
            x_mpv = self.global_encoder_neck(x_mpv)
            if not isinstance(x_mpv, torch.Tensor):
                x_mpv = x_mpv[0]
            mpv_list.append(x_mpv)

        # xy
        for i in range(0, self.num_views[0]):
            mpv_list[i] = F.interpolate(mpv_list[i], size=(128, 128), mode='bilinear').unsqueeze(-1)
        # yz
        for i in range(self.num_views[0], self.num_views[0] + self.num_views[1]):
            mpv_list[i] = F.interpolate(mpv_list[i], size=(128, 16), mode='bilinear').unsqueeze(2)
        # zx
        for i in range(self.num_views[0] + self.num_views[1], self.num_views[0] + self.num_views[1] + self.num_views[2]):
            mpv_list[i] = F.interpolate(mpv_list[i], size=(128, 16), mode='bilinear').unsqueeze(3)

        # self.save_mpv(mpv_list)
        return mpv_list


@BACKBONES.register_module()
class MTVAggregator_Cam_V0(BaseModule):

    def __init__(self, embed_dims=128, num_views=[1, 1, 1]):
        super().__init__()
        self.combine_coeff = nn.Sequential(nn.Conv3d(embed_dims, sum(num_views), kernel_size=1, bias=False), nn.Softmax(dim=1))

    def forward(self, mtv_list, x3d):
        weights = self.combine_coeff(x3d)
        out_feats = self.weighted_sum(mtv_list, weights)

        return [out_feats], weights

    def weighted_sum(self, global_feats, weights):
        out_feats = global_feats[0] * weights[:, 0:1, ...]
        for i in range(1, len(global_feats)):
            out_feats += global_feats[i] * weights[:, i:i + 1, ...]
        return out_feats
