from mmdet3d.models.builder import BACKBONES
import torch
from mmcv.runner import BaseModule
from mmdet3d.models import builder
import pdb
import torch.nn as nn
import torch.nn.functional as F


@BACKBONES.register_module()
class MTVV1(BaseModule):

    def __init__(self, embed_dims=128, global_aggregator=None, num_views=[1, 1, 1]):
        super().__init__()
        self.global_aggregator = builder.build_backbone(global_aggregator)
        self.combine_coeff = nn.Sequential(nn.Conv3d(embed_dims, sum(num_views), kernel_size=1, bias=False), nn.Softmax(dim=1))

    def forward(self, x):
        global_feats = self.global_aggregator(x)
        weights = self.combine_coeff(x)
        out_feats = torch.stack([global_feats[i] * weights[:, i:i + 1, ...] for i in range(len(global_feats))]).sum(dim=0)

        return out_feats
