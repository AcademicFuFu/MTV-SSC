# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import os
import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding

from debug.utils import print_detail as pd, mem


@HEADS.register_module()
class VoxFormerHead_Tiny(nn.Module):

    def __init__(self, *args, volume_h, volume_w, volume_z, embed_dims, **kwargs):
        super().__init__()
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z
        self.embed_dims = embed_dims

        self.mlp_prior = nn.Sequential(nn.Linear(self.embed_dims, self.embed_dims // 2), nn.LayerNorm(self.embed_dims // 2),
                                       nn.LeakyReLU(), nn.Linear(self.embed_dims // 2, self.embed_dims))

    def forward(self, mlvl_feats, proposal, cam_params, lss_volume=None, img_metas=None, **kwargs):
        """ Forward funtion.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            img_metas: Meta information
            depth: Pre-estimated depth map, (B, 1, H_d, W_d)
            cam_params: Transformation matrix, (rots, trans, intrins, post_rots, post_trans, bda)
        """

        lss_volume_flatten = lss_volume.flatten(2).squeeze(0).permute(1, 0)

        if proposal.sum() < 2:
            proposal = torch.ones_like(proposal)

        vox_feats_flatten = self.mlp_prior(lss_volume_flatten)

        vox_feats_diff = vox_feats_flatten.reshape(self.volume_h, self.volume_w, self.volume_z, self.embed_dims)
        vox_feats_diff = vox_feats_diff.permute(3, 0, 1, 2).unsqueeze(0)

        return vox_feats_diff
