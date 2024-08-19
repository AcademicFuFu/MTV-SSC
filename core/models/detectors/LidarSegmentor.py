import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from mmdet.models import DETECTORS, HEADS
from mmdet3d.models import builder
from mmcv.runner import force_fp32
import os
import pdb
from debug.utils import print_detail as pd, mem, save_feature_map_as_image


@DETECTORS.register_module()
class LidarSegmentorPointOcc(BaseModule):

    def __init__(
        self,
        lidar_tokenizer=None,
        lidar_backbone=None,
        lidar_neck=None,
        tpv_transformer=None,
        tpv_conv=None,
        tpv_aggregator=None,
        pts_bbox_head=None,
        **kwargs,
    ):

        super().__init__()

        self.lidar_tokenizer = builder.build_backbone(lidar_tokenizer)
        self.lidar_backbone = builder.build_backbone(lidar_backbone)
        self.lidar_neck = builder.build_neck(lidar_neck)
        self.tpv_transformer = builder.build_backbone(tpv_transformer)
        self.tpv_aggregator = builder.build_backbone(tpv_aggregator)
        self.pts_bbox_head = builder.build_head(pts_bbox_head)

        if tpv_conv is not None:
            self.tpv_conv = nn.Conv3d(tpv_conv.dim, tpv_conv.dim, kernel_size=1)

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

        save_feature_map_as_image(feat_xy.detach(), 'save/lidar/tpv', 'xy', method='pca')
        save_feature_map_as_image(feat_yz.detach(), 'save/lidar/tpv', 'yz', method='pca')
        save_feature_map_as_image(feat_zx.detach(), 'save/lidar/tpv', 'zx', method='pca')

        # remind to comment while training
        pdb.set_trace()
        return

    def save_logits_map(self, logits):
        # format to b,n,c,h,w
        feat_xy = logits.mean(dim=4).unsqueeze(1).permute(0, 1, 2, 3, 4)
        feat_yz = torch.flip(logits.mean(dim=2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
        feat_zx = torch.flip(logits.mean(dim=3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])

        save_feature_map_as_image(feat_xy.detach(), 'save/lidar/logits', 'xy', method='pca')
        save_feature_map_as_image(feat_yz.detach(), 'save/lidar/logits', 'yz', method='pca')
        save_feature_map_as_image(feat_zx.detach(), 'save/lidar/logits', 'zx', method='pca')

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
        tpv_lists = self.tpv_transformer(x_lidar_tpv, voxel_pos_grid_coarse)
        if hasattr(self, 'tpv_conv'):
            tpv_lists = [self.tpv_conv(view) for view in tpv_lists]

        x_3d, _ = self.tpv_aggregator(tpv_lists)
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        # self.save_tpv(tpv_lists)
        # self.save_logits_map(output['output_voxels'])

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
        tpv_lists = self.tpv_transformer(x_lidar_tpv, voxel_pos_grid_coarse)
        if hasattr(self, 'tpv_conv'):
            tpv_lists = [self.tpv_conv(view) for view in tpv_lists]
        x_3d, _ = self.tpv_aggregator(tpv_lists)
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        # for name, param in self.named_parameters():
        #     print(f"Parameter name: {name}, Size: {param.size()}")
        # self.save_tpv(tpv_lists)
        # self.save_logits_map(output['output_voxels'])

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {'pred': pred, 'gt_occ': gt_occ}
        return test_output


@DETECTORS.register_module()
class LidarSegmentorPointOccV1(BaseModule):

    def __init__(
        self,
        lidar_tokenizer=None,
        lidar_backbone=None,
        tpv_transformer=None,
        tpv_aggregator=None,
        pts_bbox_head=None,
        **kwargs,
    ):

        super().__init__()

        self.lidar_tokenizer = builder.build_backbone(lidar_tokenizer)
        self.lidar_backbone = builder.build_backbone(lidar_backbone)
        self.tpv_transformer = builder.build_backbone(tpv_transformer)
        self.tpv_aggregator = builder.build_backbone(tpv_aggregator)
        self.pts_bbox_head = builder.build_head(pts_bbox_head)

        self.fp16_enabled = False

    def extract_lidar_feat(self, points, grid_ind):
        """Extract features of points."""
        x_3view = self.lidar_tokenizer(points, grid_ind)
        tpv_list, tpv_feature_lists, _ = self.lidar_backbone(x_3view)

        return tpv_list

    def save_tpv(self, tpv_list):
        # format to b,n,c,h,w
        feat_xy = tpv_list[0].squeeze(-1).unsqueeze(1).permute(0, 1, 2, 3, 4)
        feat_yz = torch.flip(tpv_list[1].squeeze(-3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
        feat_zx = torch.flip(tpv_list[2].squeeze(-2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])

        save_feature_map_as_image(feat_xy.detach(), 'save/lidar/tpv', 'xy', method='pca')
        save_feature_map_as_image(feat_yz.detach(), 'save/lidar/tpv', 'yz', method='pca')
        save_feature_map_as_image(feat_zx.detach(), 'save/lidar/tpv', 'zx', method='pca')

        # remind to comment while training
        pdb.set_trace()
        return

    def save_logits_map(self, logits):
        # format to b,n,c,h,w
        feat_xy = logits.mean(dim=4).unsqueeze(1).permute(0, 1, 2, 3, 4)
        feat_yz = torch.flip(logits.mean(dim=2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
        feat_zx = torch.flip(logits.mean(dim=3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])

        save_feature_map_as_image(feat_xy.detach(), 'save/lidar/logits', 'xy', method='pca')
        save_feature_map_as_image(feat_yz.detach(), 'save/lidar/logits', 'yz', method='pca')
        save_feature_map_as_image(feat_zx.detach(), 'save/lidar/logits', 'zx', method='pca')

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
        tpv_lists = self.tpv_transformer(x_lidar_tpv, voxel_pos_grid_coarse)

        x_3d, _ = self.tpv_aggregator(tpv_lists)
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        # self.save_tpv(tpv_lists)
        # self.save_logits_map(output['output_voxels'])

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
        tpv_lists = self.tpv_transformer(x_lidar_tpv, voxel_pos_grid_coarse)
        x_3d, _ = self.tpv_aggregator(tpv_lists)
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        # for name, param in self.named_parameters():
        #     print(f"Parameter name: {name}, Size: {param.size()}")
        # self.save_tpv(tpv_lists)
        # self.save_logits_map(output['output_voxels'])

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {'pred': pred, 'gt_occ': gt_occ}
        return test_output


@DETECTORS.register_module()
class LidarSegmentorV0(BaseModule):

    def __init__(
        self,
        lidar_tokenizer=None,
        lidar_backbone=None,
        lidar_neck=None,
        mtv_transformer=None,
        mtv_aggregator=None,
        pts_bbox_head=None,
        **kwargs,
    ):

        super().__init__()

        self.lidar_tokenizer = builder.build_backbone(lidar_tokenizer)
        self.lidar_backbone = builder.build_backbone(lidar_backbone)
        self.lidar_neck = builder.build_neck(lidar_neck)
        self.mtv_transformer = builder.build_backbone(mtv_transformer)
        self.mtv_aggregator = builder.build_backbone(mtv_aggregator)
        self.pts_bbox_head = builder.build_head(pts_bbox_head)

        self.fp16_enabled = False

    def extract_lidar_feat(self, points, grid_ind):
        """Extract features of points."""
        x_3d = self.lidar_tokenizer(points, grid_ind)

        x_list = self.lidar_backbone(x_3d)
        output = self.lidar_neck(x_list)
        output = output[0]

        return output

    def save_logits_map(self, logits):
        # format to b,n,c,h,w
        feat_xy = logits.mean(dim=4).unsqueeze(1).permute(0, 1, 2, 3, 4)
        feat_yz = torch.flip(logits.mean(dim=2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
        feat_zx = torch.flip(logits.mean(dim=3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])

        save_feature_map_as_image(feat_xy.detach(), 'save/lidar/logits', 'xy', method='pca')
        save_feature_map_as_image(feat_yz.detach(), 'save/lidar/logits', 'yz', method='pca')
        save_feature_map_as_image(feat_zx.detach(), 'save/lidar/logits', 'zx', method='pca')

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

        # lidar encoder
        lidar_voxel_feats = self.extract_lidar_feat(points=points, grid_ind=grid_ind)

        # mtv transformer
        mtv_lists, mtv_weights, _ = self.mtv_transformer(lidar_voxel_feats)

        # mtv aggregator
        x_3d, aggregator_weights = self.mtv_aggregator(mtv_lists, mtv_weights, lidar_voxel_feats)

        # cls head
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        # loss
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

        # lidar encoder
        lidar_voxel_feats = self.extract_lidar_feat(points=points, grid_ind=grid_ind)

        # mtv transformer
        mtv_lists, mtv_weights, _ = self.mtv_transformer(lidar_voxel_feats)

        # mtv aggregator
        x_3d, aggregator_weights = self.mtv_aggregator(mtv_lists, mtv_weights, lidar_voxel_feats)

        # cls head
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {'pred': pred, 'gt_occ': gt_occ}
        return test_output
