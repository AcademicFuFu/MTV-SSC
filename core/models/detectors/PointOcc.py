import torch
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS, HEADS
from mmdet3d.models import builder
from mmcv.runner import force_fp32
import os
import pdb
from debug.utils import print_detail as pd, mem, save_feature_map_as_image


@DETECTORS.register_module()
class PointTPV_Occ(BaseModule):

    def __init__(
        self,
        lidar_tokenizer=None,
        lidar_backbone=None,
        lidar_neck=None,
        tpv_aggregator=None,
        pts_bbox_head=None,
        **kwargs,
    ):

        super().__init__()

        if lidar_tokenizer:
            self.lidar_tokenizer = builder.build_backbone(lidar_tokenizer)
        if lidar_backbone:
            self.lidar_backbone = builder.build_backbone(lidar_backbone)
        if lidar_neck:
            self.lidar_neck = builder.build_neck(lidar_neck)
        if tpv_aggregator:
            self.tpv_aggregator = builder.build_backbone(tpv_aggregator)
        if pts_bbox_head:
            self.pts_bbox_head = builder.build_head(pts_bbox_head)

        self.fp16_enabled = False

    def extract_lidar_feat(self, points, grid_ind):
        """Extract features of points."""
        x_3view = self.lidar_tokenizer(points, grid_ind)
        tpv_list = []
        x_tpv = self.lidar_backbone(x_3view)
        for x in x_tpv:
            x = self.lidar_neck(x)
            if not isinstance(x, torch.Tensor):
                x = x[0]
            tpv_list.append(x)
        return tpv_list

    def save_tpv(self, tpv_list):
        # format to b,n,c,h,w
        feat_xy = tpv_list[0].squeeze(-1).unsqueeze(1).permute(0, 1, 2, 3, 4)
        feat_yz = torch.flip(tpv_list[1].squeeze(-3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
        feat_zx = torch.flip(tpv_list[2].squeeze(-2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])

        save_feature_map_as_image(feat_xy.detach(), 'save/lidar/tpv_neck/pca', 'xy', method='pca')
        save_feature_map_as_image(feat_yz.detach(), 'save/lidar/tpv_neck/pca', 'yz', method='pca')
        save_feature_map_as_image(feat_zx.detach(), 'save/lidar/tpv_neck/pca', 'zx', method='pca')

        # remind to comment while training
        pdb.set_trace()
        return

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)

    def forward_train(self, data_dict):
        points = data_dict['points'][0]
        grid_ind = data_dict['grid_ind'][0]
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']
        voxel_pos_grid_coarse = data_dict['voxel_position_grid_coarse'][0]

        x_lidar_tpv = self.extract_lidar_feat(points=points, grid_ind=grid_ind)
        # self.save_tpv(x_lidar_tpv)
        x_3d = self.tpv_aggregator(x_lidar_tpv, voxel_pos_grid_coarse)
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        losses = dict()
        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output['output_voxels'],
            target_voxels=gt_occ,
        )
        losses.update(losses_occupancy)

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        train_output = {'losses': losses, 'pred': pred, 'gt_occ': gt_occ}
        return train_output

    def forward_test(self, data_dict):
        points = data_dict['points'][0]
        grid_ind = data_dict['grid_ind'][0]
        img_metas = data_dict['img_metas']
        if 'gt_occ' in data_dict:
            gt_occ = data_dict['gt_occ']
        else:
            gt_occ = None
        voxel_pos_grid_coarse = data_dict['voxel_position_grid_coarse'][0]

        x_lidar_tpv = self.extract_lidar_feat(points=points, grid_ind=grid_ind)
        x_3d = self.tpv_aggregator(x_lidar_tpv, voxels_coarse=voxel_pos_grid_coarse)
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {'pred': pred, 'gt_occ': gt_occ}
        return test_output
