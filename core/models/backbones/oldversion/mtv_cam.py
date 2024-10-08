from mmdet3d.models.builder import BACKBONES
import torch
from mmcv.runner import BaseModule
from mmdet3d.models import builder
import pdb
import torch.nn as nn
import torch.nn.functional as F
import pdb

from .tpv_cam import TPVPooler, TPVPoolerV1
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
        weights_out = []
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
            weights_out.append(weights)

        return mtv_out, weights_out


class SingleViewNormalDistWeightedPool3D(nn.Module):

    def __init__(self, dim='xy'):
        super().__init__()
        assert dim in ['xy', 'yz', 'zx']

        self.sigma = nn.Parameter(torch.ones(1), requires_grad=True)
        self.mu = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
        self.dim = dim

    def forward(self, x):
        if self.dim == 'yz':
            x = x.permute(0, 1, 3, 4, 2)
        elif self.dim == 'zx':
            x = x.permute(0, 1, 2, 4, 3)
        x = x.permute(0, 2, 3, 4, 1)

        device = x.device
        b, h, w, z, c = x.shape
        sigma = self.sigma * torch.sqrt(torch.tensor(z, device=device))
        mu = self.mu * z

        weights = torch.empty(z, dtype=torch.float32, device=device)
        for j in range(z):
            Z_l = normal_cdf(j, mu, sigma)
            Z_h = normal_cdf(j + 1, mu, sigma)
            weights[j] = Z_h - Z_l
        # pdb.set_trace()
        weights = weights / weights.sum()
        out = torch.sum(x * weights.view(1, 1, 1, z, 1), dim=3).permute(0, 3, 1, 2)

        return out, weights


class MTVPooler(BaseModule):

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
            mtv_xy = [self.mlp_xy(self.pool_xy(x).permute(0, 4, 1, 2, 3).flatten(start_dim=1, end_dim=2))]
            weights_xy = []
        else:
            mtv_xy, weights_xy = self.pool_xy(x)

        if self.num_views[1] == 1:
            mtv_yz = [self.mlp_yz(self.pool_yz(x).permute(0, 2, 1, 3, 4).flatten(start_dim=1, end_dim=2))]
            weights_yz = []
        else:
            mtv_yz, weights_yz = self.pool_yz(x)

        if self.num_views[2] == 1:
            mtv_zx = [self.mlp_zx(self.pool_zx(x).permute(0, 3, 1, 2, 4).flatten(start_dim=1, end_dim=2))]
            weights_zx = []
        else:
            mtv_zx, weights_zx = self.pool_zx(x)

        mtv_list = [mtv_xy, mtv_yz, mtv_zx]
        weights_list = [weights_xy, weights_yz, weights_zx]

        return mtv_list, weights_list


@BACKBONES.register_module()
class MTVTransformer_V0(BaseModule):

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
        self.mtv_pooler = MTVPooler(embed_dims=embed_dims, split=split, grid_size=grid_size, num_views_xyz=num_views)

        self.global_encoder_backbone = builder.build_backbone(global_encoder_backbone)
        self.global_encoder_neck = builder.build_neck(global_encoder_neck)

    def save_mtv(self, mtv_list):
        xy_list = [view for view in mtv_list if view.shape[-1] == 1]
        yz_list = [view for view in mtv_list if view.shape[-3] == 1]
        zx_list = [view for view in mtv_list if view.shape[-2] == 1]

        # format to b,n,c,h,w
        xy_list = [view.squeeze(-1).unsqueeze(1).permute(0, 1, 2, 3, 4) for view in xy_list]
        yz_list = [torch.flip(view.squeeze(-3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1]) for view in yz_list]
        zx_list = [torch.flip(view.squeeze(-2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1]) for view in zx_list]

        feats_xy = torch.cat(xy_list, dim=1)
        feats_yz = torch.cat(yz_list, dim=1)
        feats_zx = torch.cat(zx_list, dim=1)

        save_feature_map_as_image(feats_xy.detach(), 'save/mtv/pca', 'xy', method='pca')
        save_feature_map_as_image(feats_yz.detach(), 'save/mtv/pca', 'yz', method='pca')
        save_feature_map_as_image(feats_zx.detach(), 'save/mtv/pca', 'zx', method='pca')

        # remind to comment this function while training
        pdb.set_trace()
        return

    def forward(self, x):
        """
        xy: [b, c, h, w, z] -> [b, c, h, w]
        yz: [b, c, h, w, z] -> [b, c, w, z]
        zx: [b, c, h, w, z] -> [b, c, h, z]
        """

        x_multi_view, weights = self.mtv_pooler(x)
        x_multi_view = self.global_encoder_backbone([*x_multi_view[0], *x_multi_view[1], *x_multi_view[2]])

        mtv_list = []
        neck_out = []
        for x_mtv in x_multi_view:
            x_mtv = self.global_encoder_neck(x_mtv)
            neck_out.append(x_mtv)
            if not isinstance(x_mtv, torch.Tensor):
                x_mtv = x_mtv[0]
            mtv_list.append(x_mtv)

        feats_all = dict()
        feats_all['feats3d'] = x
        feats_all['mtv_backbone'] = x_multi_view
        feats_all['mtv_neck'] = neck_out

        # xy
        for i in range(0, self.num_views[0]):
            mtv_list[i] = F.interpolate(mtv_list[i], size=(128, 128), mode='bilinear', align_corners=False).unsqueeze(-1)
        # yz
        for i in range(self.num_views[0], self.num_views[0] + self.num_views[1]):
            mtv_list[i] = F.interpolate(mtv_list[i], size=(128, 16), mode='bilinear', align_corners=False).unsqueeze(2)
        # zx
        for i in range(self.num_views[0] + self.num_views[1], self.num_views[0] + self.num_views[1] + self.num_views[2]):
            mtv_list[i] = F.interpolate(mtv_list[i], size=(128, 16), mode='bilinear', align_corners=False).unsqueeze(3)

        # self.save_mtv(mtv_list)
        return mtv_list, weights, feats_all


@BACKBONES.register_module()
class MTVTransformer_V1(BaseModule):

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
        for i in range(len(num_views)):
            assert num_views[i] == 1 or num_views[i] == 2

        self.tpv_pooler = TPVPooler(embed_dims=embed_dims, split=split, grid_size=grid_size)
        if num_views[0] == 2:
            self.pool_xy = SingleViewNormalDistWeightedPool3D(dim='xy')
        if num_views[1] == 2:
            self.pool_yz = SingleViewNormalDistWeightedPool3D(dim='yz')
        if num_views[2] == 2:
            self.pool_zx = SingleViewNormalDistWeightedPool3D(dim='zx')

        self.global_encoder_backbone = builder.build_backbone(global_encoder_backbone)
        self.global_encoder_neck = builder.build_neck(global_encoder_neck)

    def forward(self, x):
        """
        xy: [b, c, h, w, z] -> [b, c, h, w]
        yz: [b, c, h, w, z] -> [b, c, w, z]
        zx: [b, c, h, w, z] -> [b, c, h, z]
        """

        x_tpv = self.tpv_pooler(x)

        feats_xy = [x_tpv[0]]
        if self.num_views[0] == 2:
            x_xy, weights_xy = self.pool_xy(x)
            feats_xy.append(x_xy)

        feats_yz = [x_tpv[1]]
        if self.num_views[1] == 2:
            x_yz, weights_yz = self.pool_yz(x)
            feats_yz.append(x_yz)

        feats_zx = [x_tpv[2]]
        if self.num_views[2] == 2:
            x_zx, weights_zx = self.pool_zx(x)
            feats_zx.append(x_zx)

        x_multi_view = self.global_encoder_backbone([*feats_xy, *feats_yz, *feats_zx])

        mtv_list = []
        neck_out = []
        for x_mtv in x_multi_view:
            x_mtv = self.global_encoder_neck(x_mtv)
            neck_out.append(x_mtv)
            if not isinstance(x_mtv, torch.Tensor):
                x_mtv = x_mtv[0]
            mtv_list.append(x_mtv)

        feats_all = dict()
        feats_all['feats3d'] = x
        feats_all['mtv_backbone'] = x_multi_view
        feats_all['mtv_neck'] = neck_out

        # xy
        for i in range(self.num_views[0]):
            mtv_list[i] = F.interpolate(mtv_list[i], size=(128, 128), mode='bilinear', align_corners=False).unsqueeze(-1)
        # yz
        for i in range(self.num_views[0], self.num_views[0] + self.num_views[1]):
            mtv_list[i] = F.interpolate(mtv_list[i], size=(128, 16), mode='bilinear', align_corners=False).unsqueeze(2)
        # zx
        for i in range(self.num_views[0] + self.num_views[1], self.num_views[0] + self.num_views[1] + self.num_views[2]):
            mtv_list[i] = F.interpolate(mtv_list[i], size=(128, 16), mode='bilinear', align_corners=False).unsqueeze(3)

        return mtv_list, None, feats_all


@BACKBONES.register_module()
class MTVTransformer_V2(BaseModule):

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
        for i in range(len(num_views)):
            assert num_views[i] == 1

        self.tpv_pooler = TPVPooler(embed_dims=embed_dims, split=split, grid_size=grid_size)
        self.pool_xy = SingleViewNormalDistWeightedPool3D(dim='xy')
        self.mlp_xy = nn.Sequential(nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1), nn.ReLU(),
                                    nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1))
        self.pool_yz = SingleViewNormalDistWeightedPool3D(dim='yz')
        self.mlp_yz = nn.Sequential(nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1), nn.ReLU(),
                                    nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1))
        self.pool_zx = SingleViewNormalDistWeightedPool3D(dim='zx')
        self.mlp_zx = nn.Sequential(nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1), nn.ReLU(),
                                    nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1))

        self.global_encoder_backbone = builder.build_backbone(global_encoder_backbone)
        self.global_encoder_neck = builder.build_neck(global_encoder_neck)

    def forward(self, x):
        """
        xy: [b, c, h, w, z] -> [b, c, h, w]
        yz: [b, c, h, w, z] -> [b, c, w, z]
        zx: [b, c, h, w, z] -> [b, c, h, z]
        """

        x_tpv = self.tpv_pooler(x)

        x_xy, weights_xy = self.pool_xy(x)
        x_xy = self.mlp_xy(x_xy)
        feats_xy = x_tpv[0] + x_xy

        x_yz, weights_yz = self.pool_yz(x)
        x_yz = self.mlp_yz(x_yz)
        feats_yz = x_tpv[1] + x_yz

        x_zx, weights_zx = self.pool_zx(x)
        x_zx = self.mlp_zx(x_zx)
        feats_zx = x_tpv[2] + x_zx

        x_multi_view = self.global_encoder_backbone([feats_xy, feats_yz, feats_zx])

        mtv_list = []
        neck_out = []
        for x_mtv in x_multi_view:
            x_mtv = self.global_encoder_neck(x_mtv)
            neck_out.append(x_mtv)
            if not isinstance(x_mtv, torch.Tensor):
                x_mtv = x_mtv[0]
            mtv_list.append(x_mtv)

        feats_all = dict()
        feats_all['feats3d'] = x
        feats_all['mtv_backbone'] = x_multi_view
        feats_all['mtv_neck'] = neck_out

        # xy
        for i in range(self.num_views[0]):
            mtv_list[i] = F.interpolate(mtv_list[i], size=(128, 128), mode='bilinear', align_corners=False).unsqueeze(-1)
        # yz
        for i in range(self.num_views[0], self.num_views[0] + self.num_views[1]):
            mtv_list[i] = F.interpolate(mtv_list[i], size=(128, 16), mode='bilinear', align_corners=False).unsqueeze(2)
        # zx
        for i in range(self.num_views[0] + self.num_views[1], self.num_views[0] + self.num_views[1] + self.num_views[2]):
            mtv_list[i] = F.interpolate(mtv_list[i], size=(128, 16), mode='bilinear', align_corners=False).unsqueeze(3)

        return mtv_list, None, feats_all


@BACKBONES.register_module()
class MTVTransformer_V3(BaseModule):

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
        for i in range(len(num_views)):
            assert num_views[i] == 1

        self.tpv_pooler = TPVPooler(embed_dims=embed_dims, split=split, grid_size=grid_size)
        self.tpv_pooler_ = TPVPoolerV1(embed_dims=embed_dims, split=split, grid_size=grid_size)
        self.global_encoder_backbone = builder.build_backbone(global_encoder_backbone)
        self.global_encoder_neck = builder.build_neck(global_encoder_neck)

    def forward(self, x):
        """
        xy: [b, c, h, w, z] -> [b, c, h, w]
        yz: [b, c, h, w, z] -> [b, c, w, z]
        zx: [b, c, h, w, z] -> [b, c, h, z]
        """

        x_tpv = self.tpv_pooler(x)
        x_tpv_ = self.tpv_pooler_(x)
        for i in range(len(x_tpv)):
            x_tpv[i] = x_tpv[i] + x_tpv_[i]

        x_multi_view = self.global_encoder_backbone(x_tpv)

        mtv_list = []
        neck_out = []
        for x_mtv in x_multi_view:
            x_mtv = self.global_encoder_neck(x_mtv)
            neck_out.append(x_mtv)
            if not isinstance(x_mtv, torch.Tensor):
                x_mtv = x_mtv[0]
            mtv_list.append(x_mtv)

        feats_all = dict()
        feats_all['feats3d'] = x
        feats_all['mtv_backbone'] = x_multi_view
        feats_all['mtv_neck'] = neck_out

        # xy
        for i in range(self.num_views[0]):
            mtv_list[i] = F.interpolate(mtv_list[i], size=(128, 128), mode='bilinear', align_corners=False).unsqueeze(-1)
        # yz
        for i in range(self.num_views[0], self.num_views[0] + self.num_views[1]):
            mtv_list[i] = F.interpolate(mtv_list[i], size=(128, 16), mode='bilinear', align_corners=False).unsqueeze(2)
        # zx
        for i in range(self.num_views[0] + self.num_views[1], self.num_views[0] + self.num_views[1] + self.num_views[2]):
            mtv_list[i] = F.interpolate(mtv_list[i], size=(128, 16), mode='bilinear', align_corners=False).unsqueeze(3)

        return mtv_list, None, feats_all


@BACKBONES.register_module()
class MTVTransformer_V4(BaseModule):

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
        for i in range(len(num_views)):
            assert num_views[i] == 1

        self.tpv_pooler = TPVPoolerV1(embed_dims=embed_dims, split=split, grid_size=grid_size)
        self.global_encoder_backbone = builder.build_backbone(global_encoder_backbone)
        self.global_encoder_neck = builder.build_neck(global_encoder_neck)

    def forward(self, x):
        """
        xy: [b, c, h, w, z] -> [b, c, h, w]
        yz: [b, c, h, w, z] -> [b, c, w, z]
        zx: [b, c, h, w, z] -> [b, c, h, z]
        """

        x_tpv = self.tpv_pooler(x)

        x_multi_view = self.global_encoder_backbone(x_tpv)

        mtv_list = []
        neck_out = []
        for x_mtv in x_multi_view:
            x_mtv = self.global_encoder_neck(x_mtv)
            neck_out.append(x_mtv)
            if not isinstance(x_mtv, torch.Tensor):
                x_mtv = x_mtv[0]
            mtv_list.append(x_mtv)

        feats_all = dict()
        feats_all['feats3d'] = x
        feats_all['mtv_backbone'] = x_multi_view
        feats_all['mtv_neck'] = neck_out

        # xy
        for i in range(self.num_views[0]):
            mtv_list[i] = F.interpolate(mtv_list[i], size=(128, 128), mode='bilinear', align_corners=False).unsqueeze(-1)
        # yz
        for i in range(self.num_views[0], self.num_views[0] + self.num_views[1]):
            mtv_list[i] = F.interpolate(mtv_list[i], size=(128, 16), mode='bilinear', align_corners=False).unsqueeze(2)
        # zx
        for i in range(self.num_views[0] + self.num_views[1], self.num_views[0] + self.num_views[1] + self.num_views[2]):
            mtv_list[i] = F.interpolate(mtv_list[i], size=(128, 16), mode='bilinear', align_corners=False).unsqueeze(3)

        return mtv_list, None, feats_all


@BACKBONES.register_module()
class MTVAggregator_V0(BaseModule):

    def __init__(self, embed_dims=128, num_views=[1, 1, 1]):
        super().__init__()
        self.combine_coeff = nn.Sequential(nn.Conv3d(embed_dims, sum(num_views), kernel_size=1, bias=False), nn.Softmax(dim=1))

    def forward(self, mtv_list, mtv_weights, x3d):
        weights = self.combine_coeff(x3d)
        out_feats = self.weighted_sum(mtv_list, weights)

        return [out_feats], weights

    def weighted_sum(self, global_feats, weights):
        out_feats = global_feats[0] * weights[:, 0:1, ...]
        for i in range(1, len(global_feats)):
            out_feats += global_feats[i] * weights[:, i:i + 1, ...]
        return out_feats


@BACKBONES.register_module()
class MTVAggregator_V1(BaseModule):

    def __init__(self, embed_dims=128, num_views=[1, 1, 1]):
        super().__init__()
        self.combine_coeff = nn.Sequential(nn.Conv3d(embed_dims, 3, kernel_size=1, bias=False), nn.Softmax(dim=1))
        self.num_views = num_views
        self.grid_size = [128, 128, 16]

    def forward(self, mtv_list, weights_mtv_normal_dist, x3d):
        # weights_tpv_3d: weight for tpv 3d feature
        # weights_mtv_normal_dist: normal distribution weight generating mtv feature

        mtv_3d = self.mtv23d(mtv_list, weights_mtv_normal_dist)
        weights = self.combine_coeff(x3d)
        out_feats = self.weighted_sum(mtv_3d, weights)

        return [out_feats], weights

    def mtv23d(self, mtv_list, mtv_weights):
        mtv_xy = mtv_list[:self.num_views[0]]
        mtv_yz = mtv_list[self.num_views[0]:self.num_views[0] + self.num_views[1]]
        mtv_zx = mtv_list[self.num_views[0] + self.num_views[1]:]

        weights_xy, weights_yz, weights_zx = mtv_weights

        mtv_3d_xy = self.to3d(mtv_xy, weights_xy, dim='xy')
        mtv_3d_yz = self.to3d(mtv_yz, weights_yz, dim='yz')
        mtv_3d_zx = self.to3d(mtv_zx, weights_zx, dim='zx')

        return [mtv_3d_xy, mtv_3d_yz, mtv_3d_zx]

    def to3d(self, feats, weights, dim):
        channal = feats[0].shape[1]

        if dim == 'xy':
            height = self.grid_size[2]
        elif dim == 'yz':
            height = self.grid_size[0]
        elif dim == 'zx':
            height = self.grid_size[1]

        if len(feats) == 1:
            if dim == 'xy':
                return feats[0].repeat(1, 1, 1, 1, height)
            elif dim == 'yz':
                return feats[0].repeat(1, 1, height, 1, 1)
            elif dim == 'zx':
                return feats[0].repeat(1, 1, 1, height, 1)

        weights = torch.stack(weights, dim=1)
        weights = weights / weights.sum(dim=1, keepdim=True)

        if dim == 'xy':
            x3d = torch.zeros_like(feats[0]).repeat(1, 1, 1, 1, height)
            for i in range(len(feats)):
                feat = feats[i].repeat(1, 1, 1, 1, height).permute(0, 2, 3, 4, 1)
                weight = weights[:, i].unsqueeze(-1).repeat(1, channal)
                x3d += (feat * weight).permute(0, 4, 1, 2, 3)
        return x3d

    def weighted_sum(self, global_feats, weights):
        out_feats = global_feats[0] * weights[:, 0:1, ...]
        for i in range(1, len(global_feats)):
            out_feats += global_feats[i] * weights[:, i:i + 1, ...]
        return out_feats


@BACKBONES.register_module()
class MTVAggregator_V2(BaseModule):

    def __init__(self, embed_dims=128, num_views=[1, 1, 1]):
        super().__init__()
        self.combine_coeff = nn.Sequential(nn.Conv3d(embed_dims, 3, kernel_size=1, bias=False), nn.Softmax(dim=1))
        self.num_views = num_views
        self.grid_size = [128, 128, 16]

    def forward(self, mtv_list, weights_mtv_normal_dist, x3d):
        # -----------------------------------------------
        # weights_tpv_3d: weight for tpv 3d feature
        # weights_mtv_normal_dist: normal distribution weight generating mtv feature

        weights_tpv_3d = self.combine_coeff(x3d)
        out_feats = self.weighted_sum(mtv_list, weights_tpv_3d, weights_mtv_normal_dist)
        return [out_feats], weights_tpv_3d
        # -----------------------------------------------

        # mtv_3d = self.mtv23d(mtv_list, weights_mtv_normal_dist)
        # weights = self.combine_coeff(x3d)
        # out_feats = self.weighted_sum(mtv_3d, weights)

        # return [out_feats], weights

    def mtv23d(self, mtv_list, mtv_weights):
        mtv_xy = mtv_list[:self.num_views[0]]
        mtv_yz = mtv_list[self.num_views[0]:self.num_views[0] + self.num_views[1]]
        mtv_zx = mtv_list[self.num_views[0] + self.num_views[1]:]

        weights_xy, weights_yz, weights_zx = mtv_weights

        mtv_3d_xy = self.to3d(mtv_xy, weights_xy, dim='xy')
        mtv_3d_yz = self.to3d(mtv_yz, weights_yz, dim='yz')
        mtv_3d_zx = self.to3d(mtv_zx, weights_zx, dim='zx')

        return [mtv_3d_xy, mtv_3d_yz, mtv_3d_zx]

    def to3d(self, feats, weights, dim):
        channal = feats[0].shape[1]

        if dim == 'xy':
            height = self.grid_size[2]
        elif dim == 'yz':
            height = self.grid_size[0]
        elif dim == 'zx':
            height = self.grid_size[1]

        if len(feats) == 1:
            if dim == 'xy':
                return feats[0].repeat(1, 1, 1, 1, height)
            elif dim == 'yz':
                return feats[0].repeat(1, 1, height, 1, 1)
            elif dim == 'zx':
                return feats[0].repeat(1, 1, 1, height, 1)

        weights = torch.stack(weights, dim=1)
        weights = weights / weights.sum(dim=1, keepdim=True)

        if dim == 'xy':
            x3d = torch.zeros_like(feats[0]).repeat(1, 1, 1, 1, height)
            for i in range(len(feats)):
                feat = feats[i].repeat(1, 1, 1, 1, height).permute(0, 2, 3, 4, 1)
                weight = weights[:, i].unsqueeze(-1).repeat(1, channal)
                x3d += (feat * weight).permute(0, 4, 1, 2, 3)
        return x3d

    def weighted_sum(self, global_feats, weights_tpv_3d, weights_mtv_normal_dist):

        # combine weights
        weights = []
        for i in range(len(weights_mtv_normal_dist)):
            w_tpv = weights_tpv_3d[:, i:i + 1, ...]

            if len(weights_mtv_normal_dist[i]) == 0:
                weights.append(w_tpv)
                continue

            # normalize mtv weights
            weights_mtv = torch.stack(weights_mtv_normal_dist[i], dim=1)
            weights_mtv = weights_mtv / weights_mtv.sum(dim=1, keepdim=True)

            for j in range(len(weights_mtv_normal_dist[i])):
                # xy
                if i == 0:
                    w_mtv = weights_mtv[:, j].unsqueeze(0).unsqueeze(0).repeat(self.grid_size[0], self.grid_size[1], 1)
                else:
                    raise NotImplementedError

                w_mtv = w_mtv.unsqueeze(0).unsqueeze(0).expand_as(w_tpv)
                w = w_tpv * w_mtv
                weights.append(w)

        out_feats = global_feats[0] * weights[0]
        for i in range(1, len(global_feats)):
            out_feats += global_feats[i] * weights[i]
        return out_feats


@BACKBONES.register_module()
class MTVAggregator_V3(BaseModule):

    def __init__(self, embed_dims=128, num_views=[1, 1, 1], grid_size=[128, 128, 16]):
        super().__init__()
        self.combine_coeff = nn.Sequential(nn.Conv3d(embed_dims, sum(num_views), kernel_size=1, bias=False), nn.Softmax(dim=1))
        self.grid_size = grid_size
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dims, grid_size[0], grid_size[1], grid_size[2]), requires_grad=True)

    def forward(self, mtv_list, mtv_weights, x3d):
        weights = self.combine_coeff(x3d + self.pos_embed)
        out_feats = self.weighted_sum(mtv_list, weights)

        return [out_feats], weights

    def weighted_sum(self, global_feats, weights):
        out_feats = global_feats[0] * weights[:, 0:1, ...]
        for i in range(1, len(global_feats)):
            out_feats += global_feats[i] * weights[:, i:i + 1, ...]
        return out_feats
