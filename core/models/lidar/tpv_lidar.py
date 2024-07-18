import torch, torch.nn as nn, torch.nn.functional as F
from mmdet.models import BACKBONES
from mmcv.runner import BaseModule
from copy import deepcopy
from core.utils.lovasz_softmax import lovasz_softmax
from core.utils.semkitti import geo_scal_loss, sem_scal_loss
import numpy as np

import pdb
from debug.utils import print_detail as pd, mem, save_feature_map_as_image


def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.

    this function don't contain except handle code. so use this carefully when
    indice repeats, don't support repeat add which is supported in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret


@BACKBONES.register_module()
class TPVTransformer_Lidar_V0(BaseModule):

    def __init__(self, tpv_h, tpv_w, tpv_z, grid_size_occ, coarse_ratio, scale_h=2, scale_w=2, scale_z=2, use_checkpoint=False):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        self.grid_size_occ = np.asarray(grid_size_occ).astype(np.int32)
        self.coarse_ratio = coarse_ratio

    def save_tpv(self, tpv_list):
        # format to b,n,c,h,w
        feat_xy = tpv_list[0].squeeze(-1).unsqueeze(1).permute(0, 1, 2, 3, 4)
        feat_yz = torch.flip(tpv_list[1].squeeze(-3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
        feat_zx = torch.flip(tpv_list[2].squeeze(-2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])

        save_feature_map_as_image(feat_xy.detach(), 'save/lidar/tpv/pca', 'xy', method='pca')
        save_feature_map_as_image(feat_yz.detach(), 'save/lidar/tpv/pca', 'yz', method='pca')
        save_feature_map_as_image(feat_zx.detach(), 'save/lidar/tpv/pca', 'zx', method='pca')

        # remind to comment while training
        pdb.set_trace()
        return

    def tpv_polar2cart(self, tpv_list_polar, voxels_coarse):
        tpv_hw, tpv_zh, tpv_wz = tpv_list_polar
        # voxel_coarse: bs, (vox_w*vox_h*vox_z)/coarse_ratio**3, 3
        bs = 1
        _, n, _ = voxels_coarse.shape
        voxels_coarse = voxels_coarse.reshape(bs, 1, n, 3)
        voxels_coarse[..., 0] = voxels_coarse[..., 0] / (self.tpv_w * self.scale_w) * 2 - 1
        voxels_coarse[..., 1] = voxels_coarse[..., 1] / (self.tpv_h * self.scale_h) * 2 - 1
        voxels_coarse[..., 2] = voxels_coarse[..., 2] / (self.tpv_z * self.scale_z) * 2 - 1

        B, C, H, W, D = 1, 192, 128, 128, 16
        sample_loc_vox = voxels_coarse[:, :, :, [0, 1]]
        tpv_hw_vox = F.grid_sample(tpv_hw, sample_loc_vox, padding_mode="border").squeeze(2).view(B, C, H, W, D)[:, :, :, :, 1]
        sample_loc_vox = voxels_coarse[:, :, :, [1, 2]]
        tpv_zh_vox = F.grid_sample(tpv_zh, sample_loc_vox, padding_mode="border").squeeze(2).view(B, C, H, W, D)[:, :, 0, :, :]
        sample_loc_vox = voxels_coarse[:, :, :, [2, 0]]
        tpv_wz_vox = F.grid_sample(tpv_wz, sample_loc_vox, padding_mode="border").squeeze(2).view(B, C, H, W, D)[:, :, :, 0, :]
        pdb.set_trace()
        return [tpv_hw_vox, tpv_zh_vox, tpv_wz_vox]

    def forward(self, tpv_list, voxels_coarse=None):
        """
        x y z -> w h z
        tpv_list[0]: bs, c, w, h
        tpv_list[1]: bs, c, h, z
        tpv_list[2]: bs, c, z, w
        """
        tpv_xy, tpv_yz, tpv_zx = tpv_list[0], tpv_list[1], tpv_list[2]
        tpv_hw = tpv_xy.permute(0, 1, 3, 2)
        tpv_wz = tpv_zx.permute(0, 1, 3, 2)
        tpv_zh = tpv_yz.permute(0, 1, 3, 2)
        B, C, _, _ = tpv_hw.shape
        W, H, D = int(self.grid_size_occ[0] / self.coarse_ratio), int(self.grid_size_occ[1] / self.coarse_ratio), int(
            self.grid_size_occ[2] / self.coarse_ratio)

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(tpv_hw, size=(int(self.tpv_h * self.scale_h), int(self.tpv_w * self.scale_w)), mode='bilinear')
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(tpv_zh, size=(int(self.tpv_z * self.scale_z), int(self.tpv_h * self.scale_h)), mode='bilinear')
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(tpv_wz, size=(int(self.tpv_w * self.scale_w), int(self.tpv_z * self.scale_z)), mode='bilinear')

        # voxel_coarse: bs, (vox_w*vox_h*vox_z)/coarse_ratio**3, 3
        _, n, _ = voxels_coarse.shape
        voxels_coarse = voxels_coarse.reshape(B, 1, n, 3)
        voxels_coarse[..., 0] = voxels_coarse[..., 0] / (self.tpv_w * self.scale_w) * 2 - 1
        voxels_coarse[..., 1] = voxels_coarse[..., 1] / (self.tpv_h * self.scale_h) * 2 - 1
        voxels_coarse[..., 2] = voxels_coarse[..., 2] / (self.tpv_z * self.scale_z) * 2 - 1

        sample_loc_vox = voxels_coarse[:, :, :, [0, 1]]
        tpv_hw_vox = F.grid_sample(tpv_hw, sample_loc_vox, padding_mode="border").squeeze(2).view(B, C, H, W, D)[:, :, :, :, 1]
        sample_loc_vox = voxels_coarse[:, :, :, [1, 2]]
        tpv_zh_vox = F.grid_sample(tpv_zh, sample_loc_vox, padding_mode="border").squeeze(2).view(B, C, H, W, D)[:, :, 0, :, :]
        sample_loc_vox = voxels_coarse[:, :, :, [2, 0]]
        tpv_wz_vox = F.grid_sample(tpv_wz, sample_loc_vox, padding_mode="border").squeeze(2).view(B, C, H, W, D)[:, :, :, 0, :]

        tpv_list = [tpv_hw_vox.unsqueeze(4), tpv_zh_vox.unsqueeze(2), tpv_wz_vox.unsqueeze(3)]
        # self.save_tpv(tpv_list)
        return tpv_list


@BACKBONES.register_module()
class TPVAggregator_Lidar_V0(BaseModule):

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
