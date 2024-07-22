import torch
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS, HEADS
from mmdet3d.models import builder
from mmcv.runner import force_fp32
import os
import pdb
from debug.utils import print_detail as pd, mem, save_feature_map_as_image, save_denormalized_images


@DETECTORS.register_module()
class CameraSegmentorEfficientSSC(BaseModule):

    def __init__(
        self,
        img_backbone,
        img_neck,
        img_view_transformer,
        depth_net=None,
        proposal_layer=None,
        VoxFormer_head=None,
        tpv_transformer=None,
        tpv_aggregator=None,
        pts_bbox_head=None,
        init_cfg=None,
        **kwargs,
    ):

        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)

        if depth_net is not None:
            self.depth_net = builder.build_neck(depth_net)
        else:
            self.depth_net = None
        if img_view_transformer is not None:
            self.img_view_transformer = builder.build_neck(img_view_transformer)
        else:
            self.img_view_transformer = None

        if proposal_layer is not None:
            self.proposal_layer = builder.build_head(proposal_layer)
        else:
            self.proposal_layer = None

        if VoxFormer_head is not None:
            self.VoxFormer_head = builder.build_head(VoxFormer_head)
        else:
            self.VoxFormer_head = None

        if tpv_transformer:
            self.tpv_transformer = builder.build_backbone(tpv_transformer)
        if tpv_aggregator:
            self.tpv_aggregator = builder.build_backbone(tpv_aggregator)
        if pts_bbox_head:
            self.pts_bbox_head = builder.build_head(pts_bbox_head)

        self.init_cfg = init_cfg
        self.init_weights()

    def save_img(self, img):
        save_denormalized_images(img.detach(), 'save/img')
        pdb.set_trace()

    def image_encoder(self, img):
        imgs = img
        # self.save_img(img)

        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)

        x = self.img_backbone(imgs)

        if self.img_neck is not None:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)

        return x

    def extract_img_feat(self, img_inputs, img_metas):
        img = img_inputs[0]
        img_enc_feats = self.image_encoder(img)

        rots, trans, intrins, post_rots, post_trans, bda = img_inputs[1:7]
        mlp_input = self.depth_net.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)

        geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]

        context, depth = self.depth_net([img_enc_feats] + geo_inputs, img_metas)
        view_trans_inputs = [
            rots[:, 0:1, ...], trans[:, 0:1, ...], intrins[:, 0:1, ...], post_rots[:, 0:1, ...], post_trans[:, 0:1, ...], bda
        ]

        if self.img_view_transformer is not None:
            lss_volume = self.img_view_transformer(context, depth, view_trans_inputs)
        else:
            lss_volume = None

        query_proposal = self.proposal_layer(view_trans_inputs, img_metas)

        if query_proposal.shape[1] == 2:
            proposal = torch.argmax(query_proposal, dim=1)
        else:
            proposal = query_proposal
        if depth is not None:
            mlvl_dpt_dists = [depth.unsqueeze(1)]
        else:
            mlvl_dpt_dists = None
        x = self.VoxFormer_head([context],
                                proposal,
                                cam_params=view_trans_inputs,
                                lss_volume=lss_volume,
                                img_metas=img_metas,
                                mlvl_dpt_dists=mlvl_dpt_dists)
        # pdb.set_trace()
        return x, query_proposal, depth

    def save_tpv(self, tpv_list):
        # format to b,n,c,h,w
        feat_xy = tpv_list[0].squeeze(-1).unsqueeze(1).permute(0, 1, 2, 3, 4)
        feat_yz = torch.flip(tpv_list[1].squeeze(-3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
        feat_zx = torch.flip(tpv_list[2].squeeze(-2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])

        save_feature_map_as_image(feat_xy.detach(), 'save/img/tpv_neck/pca', 'xy', method='pca')
        save_feature_map_as_image(feat_yz.detach(), 'save/img/tpv_neck/pca', 'yz', method='pca')
        save_feature_map_as_image(feat_zx.detach(), 'save/img/tpv_neck/pca', 'zx', method='pca')

        # remind to comment while training
        pdb.set_trace()
        return

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)

    def forward_train(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']

        img_voxel_feats, query_proposal, depth = self.extract_img_feat(img_inputs, img_metas)
        tpv_lists = self.tpv_transformer(img_voxel_feats)
        x_3d = self.tpv_aggregator(tpv_lists)
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        losses = dict()
        if depth is not None:
            losses['loss_depth'] = self.depth_net.get_depth_loss(img_inputs[-4][:, 0:1, ...], depth)
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
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']
        if 'gt_occ' in data_dict:
            gt_occ = data_dict['gt_occ']
        else:
            gt_occ = None

        img_voxel_feats, query_proposal, depth = self.extract_img_feat(img_inputs, img_metas)
        tpv_lists = self.tpv_transformer(img_voxel_feats)
        x_3d = self.tpv_aggregator(tpv_lists)
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {'pred': pred, 'gt_occ': gt_occ}
        return test_output
