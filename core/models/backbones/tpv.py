from mmdet3d.models.builder import BACKBONES
import torch
from mmcv.runner import BaseModule
from mmdet3d.models import builder
import torch.nn as nn
import torch.nn.functional as F
import pdb
from debug.utils import print_detail as pd, mem, save_feature_map_as_image


@BACKBONES.register_module()
class TPVV0(BaseModule):

    def __init__(self, embed_dims=128, global_aggregator=None):
        super().__init__()
        self.global_aggregator = builder.build_backbone(global_aggregator)
        self.combine_coeff = nn.Sequential(nn.Conv3d(embed_dims, 3, kernel_size=1, bias=False), nn.Softmax(dim=1))

    def forward(self, x):
        global_feats = self.global_aggregator(x)
        weights = self.combine_coeff(x)
        out_feats = global_feats[0] * weights[:, 0:1, ...] + global_feats[1] * weights[:, 1:2,
                                                                                       ...] + global_feats[2] * weights[:, 2:3,
                                                                                                                        ...]
        return out_feats


@BACKBONES.register_module()
class TPVV1(BaseModule):

    def __init__(self, global_aggregator=None, **kwargs):
        super().__init__()
        self.global_aggregator = builder.build_backbone(global_aggregator)

    def forward(self, x):
        global_feats = self.global_aggregator(x)
        feats_xy, feats_yz, feats_zx = global_feats
        x, y, z = feats_xy.shape[2], feats_xy.shape[3], feats_yz.shape[4]
        feats_xy = feats_xy.repeat(1, 1, 1, 1, z)
        feats_yz = feats_yz.repeat(1, 1, x, 1, 1)
        feats_zx = feats_zx.repeat(1, 1, 1, y, 1)
        out_feats = feats_xy + feats_yz + feats_zx
        return out_feats
