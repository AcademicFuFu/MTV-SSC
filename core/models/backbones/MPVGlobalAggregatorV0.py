from mmdet3d.models.builder import BACKBONES
import torch
from mmcv.runner import BaseModule
from mmdet3d.models import builder
import torch.nn as nn
import torch.nn.functional as F
import pdb

from mmcv.ops import MultiScaleDeformableAttention

from debug.utils import print_detail as pd, mem


class DeformableTransformer2D(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 num_levels=3,
                 num_points=4,
                 mlp_ratio=4,
                 attn_layer=MultiScaleDeformableAttention,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.norm1 = norm_layer(embed_dims)
        self.attn = attn_layer(embed_dims, num_heads, num_levels, num_points, batch_first=True, **kwargs)

        if mlp_ratio == 0:
            return
        self.norm2 = norm_layer(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dims * mlp_ratio, embed_dims),
        )

    def forward(self, query, value=None, query_pos=None):
        b, c, h, w = query.shape
        query = query.view(b, c, h * w).permute(0, 2, 1)
        value = value.view(b, c, h * w).permute(0, 2, 1)
        rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w))
        rows = rows / float(h)
        cols = cols / float(w)
        ref_pts = torch.stack([rows, cols], dim=-1).to(query.device).view(b, -1, 1, 2)
        spatial_shapes = torch.tensor([[h, w]], dtype=torch.long).to(query.device)
        level_start_index = torch.tensor([0], dtype=torch.long).to(query.device)

        query = query + self.attn(self.norm1(query),
                                  value=value,
                                  query_pos=query_pos,
                                  reference_points=ref_pts,
                                  spatial_shapes=spatial_shapes,
                                  level_start_index=level_start_index)

        if not hasattr(self, 'ffn'):
            return query
        query = (query + self.ffn(self.norm2(query))).view(b, h, w, c).permute(0, 3, 1, 2)
        return query


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

        self.transformer_xy = DeformableTransformer2D(embed_dims, num_heads=8, num_levels=3, num_points=4)
        self.transformer_yz = DeformableTransformer2D(embed_dims, num_heads=8, num_levels=3, num_points=4)
        self.transformer_zx = DeformableTransformer2D(embed_dims, num_heads=8, num_levels=3, num_points=4)

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

        # pdb.set_trace()
        tpv_xy = self.transformer_xy(tpv_xy, tpv_xy)
        tpv_yz = self.transformer_yz(tpv_yz, tpv_yz)
        tpv_zx = self.transformer_zx(tpv_zx, tpv_yz)

        tpv_list = [tpv_xy, tpv_yz, tpv_zx]

        return tpv_list


@BACKBONES.register_module()
class MPVGlobalAggregatorV0(BaseModule):

    def __init__(
        self,
        embed_dims=128,
        split=[8, 8, 8],
        grid_size=[128, 128, 16],
        global_encoder_backbone=None,
        global_encoder_neck=None,
        num_views_x=[1, 1, 1],
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

        # pdb.set_trace()
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
