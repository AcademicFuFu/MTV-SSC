import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS, HEADS
from mmdet3d.models import builder
from mmcv.runner import force_fp32
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import os
import pdb
from debug.utils import print_detail as pd, mem, count_trainable_parameters as param
from debug.utils import save_feature_map_as_image, save_denormalized_images, save_all_feats, save_mtv, save_weights


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


@DETECTORS.register_module()
class CameraSegmentorEfficientSSCV1(BaseModule):

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
        teacher=None,
        teacher_ckpt=None,
        normalize_loss=False,
        ratio_logit=10.0,
        ratio_tpv_feats=2,
        ratio_tpv_relation=10,
        **kwargs,
    ):

        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)
        self.depth_net = builder.build_neck(depth_net)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.proposal_layer = builder.build_head(proposal_layer)
        self.VoxFormer_head = builder.build_head(VoxFormer_head)
        self.tpv_transformer = builder.build_backbone(tpv_transformer)
        self.tpv_aggregator = builder.build_backbone(tpv_aggregator)
        self.pts_bbox_head = builder.build_head(pts_bbox_head)

        self.normalize_loss = normalize_loss

        # init before teacher to avoid overwriting teacher weights
        self.init_cfg = init_cfg
        self.init_weights()

        if teacher:
            self.teacher = builder.build_detector(teacher).eval()
            if os.path.exists(teacher_ckpt):
                ckpt = torch.load(teacher_ckpt)['state_dict']
                adjusted_ckpt = {key.replace('model.', ''): value for key, value in ckpt.items()}
                self.teacher.load_state_dict(adjusted_ckpt)
                print(f"Load teacher model from {teacher_ckpt}")
            self.freeze_model(self.teacher)
            self.ratio_logit = ratio_logit
            self.ratio_tpv_feats = ratio_tpv_feats
            self.ratio_tpv_relation = ratio_tpv_relation

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

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

    def save_tpv(self, tpv_cam, tpv_lidar=None):
        tpv_list = tpv_cam
        # format to b,n,c,h,w
        feat_xy = tpv_list[0].squeeze(-1).unsqueeze(1).permute(0, 1, 2, 3, 4)
        feat_yz = torch.flip(tpv_list[1].squeeze(-3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
        feat_zx = torch.flip(tpv_list[2].squeeze(-2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])

        save_feature_map_as_image(feat_xy.detach(), 'save/distill/tpv_cam', 'xy', method='pca')
        save_feature_map_as_image(feat_yz.detach(), 'save/distill/tpv_cam', 'yz', method='pca')
        save_feature_map_as_image(feat_zx.detach(), 'save/distill/tpv_cam', 'zx', method='pca')

        if tpv_lidar is not None:
            tpv_list = tpv_lidar
            feat_xy = tpv_list[0].squeeze(-1).unsqueeze(1).permute(0, 1, 2, 3, 4)
            feat_yz = torch.flip(tpv_list[1].squeeze(-3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
            feat_zx = torch.flip(tpv_list[2].squeeze(-2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])

            save_feature_map_as_image(feat_xy.detach(), 'save/distill/tpv_lidar', 'xy', method='pca')
            save_feature_map_as_image(feat_yz.detach(), 'save/distill/tpv_lidar', 'yz', method='pca')
            save_feature_map_as_image(feat_zx.detach(), 'save/distill/tpv_lidar', 'zx', method='pca')

        # remind to comment while training
        pdb.set_trace()
        return

    def save_logits_map(self, logits_cam, logits_lidar=None):
        # format to b,n,c,h,w
        feat_xy = logits_cam.mean(dim=4).unsqueeze(1).permute(0, 1, 2, 3, 4)
        feat_yz = torch.flip(logits_cam.mean(dim=2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
        feat_zx = torch.flip(logits_cam.mean(dim=3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])

        save_feature_map_as_image(feat_xy.detach(), 'save/distill/logits_cam', 'xy', method='pca')
        save_feature_map_as_image(feat_yz.detach(), 'save/distill/logits_cam', 'yz', method='pca')
        save_feature_map_as_image(feat_zx.detach(), 'save/distill/logits_cam', 'zx', method='pca')

        if logits_lidar is not None:
            feat_xy = logits_lidar.mean(dim=4).unsqueeze(1).permute(0, 1, 2, 3, 4)
            feat_yz = torch.flip(logits_lidar.mean(dim=2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
            feat_zx = torch.flip(logits_lidar.mean(dim=3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])

            save_feature_map_as_image(feat_xy.detach(), 'save/distill/logits_lidar', 'xy', method='pca')
            save_feature_map_as_image(feat_yz.detach(), 'save/distill/logits_lidar', 'yz', method='pca')
            save_feature_map_as_image(feat_zx.detach(), 'save/distill/logits_lidar', 'zx', method='pca')

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

        if self.normalize_loss:
            for key in losses:
                losses[key] = losses[key] / losses[key]

        losses_distill = {}
        if hasattr(self, 'teacher'):
            with torch.no_grad():
                tpv_lists_teacher, output_teacher = self.forward_teacher(data_dict)

            # self.save_tpv(tpv_lists, tpv_lists_teacher)
            # self.save_logits_map(output['output_voxels'], output_teacher['output_voxels'])

            if self.ratio_logit > 0:
                losses_distill_logit = self.distill_loss_logits(output_teacher['output_voxels'], output['output_voxels'], gt_occ,
                                                                self.ratio_logit)
                losses_distill.update(losses_distill_logit)

            if self.ratio_tpv_feats > 0:
                losses_distill_tpv_feature = self.distill_loss_tpv_feature(tpv_lists_teacher, tpv_lists, gt_occ,
                                                                           self.ratio_tpv_feats)
                losses_distill.update(losses_distill_tpv_feature)

            if self.ratio_tpv_relation > 0:
                losses_distill_tpv_relation = self.distill_loss_tpv_relation(tpv_lists_teacher, tpv_lists, gt_occ,
                                                                             self.ratio_tpv_relation)
                losses_distill.update(losses_distill_tpv_relation)

            if self.normalize_loss:
                for key in losses_distill:
                    losses_distill[key] = losses_distill[key] / losses_distill[key]
            losses.update(losses_distill)

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

        # self.save_tpv(tpv_lists)
        # self.save_logits_map(output['output_voxels'])

        # if hasattr(self, 'teacher'):
        #     with torch.no_grad():
        #         tpv_lists_teacher, output_teacher = self.forward_teacher(data_dict)

        #     self.save_tpv(tpv_lists, tpv_lists_teacher)
        #     self.save_logits_map(output['output_voxels'], output_teacher['output_voxels'])

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {'pred': pred, 'gt_occ': gt_occ}
        return test_output

    def forward_teacher(self, data_dict):
        points = data_dict['points'][0]
        grid_ind = data_dict['grid_ind'][0]
        img_metas = data_dict['img_metas']
        if 'gt_occ' in data_dict:
            gt_occ = data_dict['gt_occ']
        else:
            gt_occ = None
        voxel_pos_grid_coarse = data_dict['voxel_position_grid_coarse'][0]

        x_lidar_tpv = self.teacher.extract_lidar_feat(points=points, grid_ind=grid_ind)
        tpv_lists = self.teacher.tpv_transformer(x_lidar_tpv, voxel_pos_grid_coarse)
        x_3d = self.teacher.tpv_aggregator(tpv_lists)
        output = self.teacher.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        return tpv_lists, output

    def calculate_cosine_similarity(self, x, y):
        # 验证形状是否一致
        assert x.shape == y.shape, "输入特征的形状必须相同"
        B, C, H, W = x.shape

        # 扁平化特征的H和W维度，形状从 (B, C, H, W) 变为 (B, C, H*W)
        x_flat = x.reshape(B, C, -1)
        y_flat = y.reshape(B, C, -1)

        # # 计算归一化后的特征，使特征在C维度上具有单位长度
        x_norm = F.normalize(x_flat, p=2, dim=1)
        y_norm = F.normalize(y_flat, p=2, dim=1)

        # # 使用批量矩阵乘法计算cosine相似度 (B, H*W, H*W)
        cosine_similarity_flat = torch.bmm(x_norm.permute(0, 2, 1), y_norm)

        return cosine_similarity_flat

    def distill_loss_logits(self, logits_teacher, logits_student, target, ratio):
        b, c, h, w, z = logits_teacher.shape
        logits_student_softmax = F.log_softmax(logits_student, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        logits_teacher_softmax = F.softmax(logits_teacher, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        target = target.reshape(b, h * w * z)

        loss = 0
        for i in range(target.shape[0]):
            valid = (target[i] != 255)
            logits_student_i = logits_student_softmax[i][valid]
            logits_teacher_i = logits_teacher_softmax[i][valid]
            loss += nn.KLDivLoss(reduction="mean")(logits_student_i.unsqueeze(0), logits_teacher_i.unsqueeze(0))
        loss = loss / float(target.size(0)) * ratio
        return dict(loss_distill_logits=loss)

    # def distill_loss_tpv_feature(self, tpv_teacher, tpv_student, target, ratio):
    #     loss = 0
    #     for i in range(target.shape[0]):
    #         # target_i = target[i].to(torch.float32)
    #         tpv_xy_teacher_i = tpv_teacher[0][i]
    #         tpv_xy_student_i = tpv_student[0][i]
    #         cos_sim = cosine_similarity(tpv_xy_student_i, tpv_xy_teacher_i, dim=0)
    #         loss_xy = 1 - cos_sim.mean()

    #         tpv_yz_teacher_i = tpv_teacher[1][i]
    #         tpv_yz_student_i = tpv_student[1][i]
    #         cos_sim = cosine_similarity(tpv_yz_student_i, tpv_yz_teacher_i, dim=0)
    #         loss_yz = 1 - cos_sim.mean()

    #         tpv_zx_teacher_i = tpv_teacher[2][i]
    #         tpv_zx_student_i = tpv_student[2][i]
    #         cos_sim = cosine_similarity(tpv_zx_student_i, tpv_zx_teacher_i, dim=0)
    #         loss_zx = 1 - cos_sim.mean()

    #         loss += loss_xy + loss_yz + loss_zx
    #     loss = loss * ratio / (3 * target.shape[0])
    #     return dict(loss_distill_tpv_feature=loss)

    def distill_loss_tpv_feature(self, tpv_teacher, tpv_student, target, ratio):
        loss = 0
        for i in range(target.shape[0]):
            # target_i = target[i].to(torch.float32)
            tpv_xy_teacher_i = tpv_teacher[0][i]
            tpv_xy_student_i = tpv_student[0][i]
            loss_xy = F.l1_loss(tpv_xy_student_i, tpv_xy_teacher_i)

            tpv_yz_teacher_i = tpv_teacher[1][i]
            tpv_yz_student_i = tpv_student[1][i]
            loss_yz = F.l1_loss(tpv_yz_student_i, tpv_yz_teacher_i)

            tpv_zx_teacher_i = tpv_teacher[2][i]
            tpv_zx_student_i = tpv_student[2][i]
            loss_zx = F.l1_loss(tpv_zx_student_i, tpv_zx_teacher_i)

            loss += loss_xy + loss_yz + loss_zx
        loss = loss * ratio / (3 * target.shape[0])
        return dict(loss_distill_tpv_feature=loss)

    def distill_loss_tpv_relation(self, tpv_teacher, tpv_student, target, ratio):
        loss = 0
        tpv_teacher_i_xy = tpv_teacher[0].squeeze(4)
        tpv_student_i_xy = tpv_student[0].squeeze(4)
        cos_sim_student = self.calculate_cosine_similarity(tpv_student_i_xy, tpv_student_i_xy)
        cos_sim_teacher = self.calculate_cosine_similarity(tpv_student_i_xy, tpv_teacher_i_xy)
        diff_abs = torch.abs(cos_sim_student - cos_sim_teacher)
        l1_norm = torch.sum(diff_abs)
        loss_xy = l1_norm / float(diff_abs.size(1) * diff_abs.size(1))

        tpv_teacher_i_yz = tpv_teacher[1].squeeze(2)
        tpv_student_i_yz = tpv_student[1].squeeze(2)
        cos_sim_student = self.calculate_cosine_similarity(tpv_student_i_yz, tpv_student_i_yz)
        cos_sim_teacher = self.calculate_cosine_similarity(tpv_student_i_yz, tpv_teacher_i_yz)
        diff_abs = torch.abs(cos_sim_student - cos_sim_teacher)
        l1_norm = torch.sum(diff_abs)
        loss_yz = l1_norm / float(diff_abs.size(1) * diff_abs.size(1))

        tpv_teacher_i_zx = tpv_teacher[2].squeeze(3)
        tpv_student_i_zx = tpv_student[2].squeeze(3)
        cos_sim_student = self.calculate_cosine_similarity(tpv_student_i_zx, tpv_student_i_zx)
        cos_sim_teacher = self.calculate_cosine_similarity(tpv_student_i_zx, tpv_teacher_i_zx)
        diff_abs = torch.abs(cos_sim_student - cos_sim_teacher)
        l1_norm = torch.sum(diff_abs)
        loss_zx = l1_norm / float(diff_abs.size(1) * diff_abs.size(1))

        loss += (loss_xy + loss_yz + loss_zx) / 3
        loss = loss * ratio / target.shape[0]
        return dict(loss_distill_tpv_relation=loss)


@DETECTORS.register_module()
class CameraSegmentorEfficientSSCV2(BaseModule):

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
        teacher=None,
        teacher_ckpt=None,
        normalize_loss=False,
        feature_loss_type='l1',
        ratio_logit=10.0,
        ratio_tpv_feats=2,
        ratio_tpv_relation=10,
        ratio_tpv_weights=10,
        tpv_conv=None,
        **kwargs,
    ):

        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)
        self.depth_net = builder.build_neck(depth_net)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.proposal_layer = builder.build_head(proposal_layer)
        self.VoxFormer_head = builder.build_head(VoxFormer_head)
        self.tpv_transformer = builder.build_backbone(tpv_transformer)
        self.tpv_aggregator = builder.build_backbone(tpv_aggregator)
        self.pts_bbox_head = builder.build_head(pts_bbox_head)

        self.normalize_loss = normalize_loss

        if tpv_conv is not None:
            self.tpv_conv = nn.Conv3d(tpv_conv.dim, tpv_conv.dim, kernel_size=1)
        # init before teacher to avoid overwriting teacher weights
        self.init_cfg = init_cfg
        self.init_weights()

        if teacher:
            self.teacher = builder.build_detector(teacher).eval()
            if os.path.exists(teacher_ckpt):
                ckpt = torch.load(teacher_ckpt)['state_dict']
                adjusted_ckpt = {key.replace('model.', ''): value for key, value in ckpt.items()}
                self.teacher.load_state_dict(adjusted_ckpt)
                print(f"Load teacher model from {teacher_ckpt}")
            self.freeze_model(self.teacher)
            self.feature_loss_type = feature_loss_type
            self.ratio_logit = ratio_logit
            self.ratio_tpv_feats = ratio_tpv_feats
            self.ratio_tpv_relation = ratio_tpv_relation
            self.ratio_tpv_weights = ratio_tpv_weights

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

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

    def save_tpv(self, tpv_cam, tpv_lidar=None, gt_occ_1_2=None):
        tpv_list = tpv_cam
        if gt_occ_1_2 is not None:
            target = gt_occ_1_2.to(torch.float32)
            target[target == 255] = 0

            target_xy_mean = target.mean(dim=3)
            mask_xy = target_xy_mean == 0
            mask_xy = mask_xy.unsqueeze(1).unsqueeze(4).expand_as(tpv_list[0])

            target_yz_mean = target.mean(dim=1)
            mask_yz = target_yz_mean == 0
            mask_yz = mask_yz.unsqueeze(1).unsqueeze(2).expand_as(tpv_list[1])

            target_zx_mean = target.mean(dim=2)
            mask_zx = target_zx_mean == 0
            mask_zx = mask_zx.unsqueeze(1).unsqueeze(3).expand_as(tpv_list[2])

            tpv_list[0][mask_xy] = 0
            tpv_list[1][mask_yz] = 0
            tpv_list[2][mask_zx] = 0

        # format to b,n,c,h,w
        feat_xy = tpv_list[0].squeeze(-1).unsqueeze(1).permute(0, 1, 2, 3, 4)
        feat_yz = torch.flip(tpv_list[1].squeeze(-3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
        feat_zx = torch.flip(tpv_list[2].squeeze(-2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])

        save_feature_map_as_image(feat_xy.detach(), 'save/distill/tpv_cam', 'xy', method='pca')
        save_feature_map_as_image(feat_yz.detach(), 'save/distill/tpv_cam', 'yz', method='pca')
        save_feature_map_as_image(feat_zx.detach(), 'save/distill/tpv_cam', 'zx', method='pca')

        if tpv_lidar is not None:
            tpv_list = tpv_lidar
            if gt_occ_1_2 is not None:
                target = gt_occ_1_2.to(torch.float32)
                target[target == 255] = 0

                target_xy_mean = target.mean(dim=3)
                mask_xy = target_xy_mean == 0
                mask_xy = mask_xy.unsqueeze(1).unsqueeze(4).expand_as(tpv_list[0])

                target_yz_mean = target.mean(dim=1)
                mask_yz = target_yz_mean == 0
                mask_yz = mask_yz.unsqueeze(1).unsqueeze(2).expand_as(tpv_list[1])

                target_zx_mean = target.mean(dim=2)
                mask_zx = target_zx_mean == 0
                mask_zx = mask_zx.unsqueeze(1).unsqueeze(3).expand_as(tpv_list[2])

                tpv_list[0][mask_xy] = 0
                tpv_list[1][mask_yz] = 0
                tpv_list[2][mask_zx] = 0
                feat_xy = tpv_list[0].squeeze(-1).unsqueeze(1).permute(0, 1, 2, 3, 4)
                feat_yz = torch.flip(tpv_list[1].squeeze(-3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
                feat_zx = torch.flip(tpv_list[2].squeeze(-2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])

            save_feature_map_as_image(feat_xy.detach(), 'save/distill/tpv_lidar', 'xy', method='pca')
            save_feature_map_as_image(feat_yz.detach(), 'save/distill/tpv_lidar', 'yz', method='pca')
            save_feature_map_as_image(feat_zx.detach(), 'save/distill/tpv_lidar', 'zx', method='pca')

        # remind to comment while training
        pdb.set_trace()
        return

    def save_logits_map(self, logits_cam, logits_lidar=None):
        # format to b,n,c,h,w
        feat_xy = logits_cam.mean(dim=4).unsqueeze(1).permute(0, 1, 2, 3, 4)
        feat_yz = torch.flip(logits_cam.mean(dim=2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
        feat_zx = torch.flip(logits_cam.mean(dim=3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])

        save_feature_map_as_image(feat_xy.detach(), 'save/distill/logits_cam', 'xy', method='pca')
        save_feature_map_as_image(feat_yz.detach(), 'save/distill/logits_cam', 'yz', method='pca')
        save_feature_map_as_image(feat_zx.detach(), 'save/distill/logits_cam', 'zx', method='pca')

        if logits_lidar is not None:
            feat_xy = logits_lidar.mean(dim=4).unsqueeze(1).permute(0, 1, 2, 3, 4)
            feat_yz = torch.flip(logits_lidar.mean(dim=2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
            feat_zx = torch.flip(logits_lidar.mean(dim=3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])

            save_feature_map_as_image(feat_xy.detach(), 'save/distill/logits_lidar', 'xy', method='pca')
            save_feature_map_as_image(feat_yz.detach(), 'save/distill/logits_lidar', 'yz', method='pca')
            save_feature_map_as_image(feat_zx.detach(), 'save/distill/logits_lidar', 'zx', method='pca')

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
        gt_occ_1_2 = img_metas['gt_occ_1_2']

        img_voxel_feats, query_proposal, depth = self.extract_img_feat(img_inputs, img_metas)
        tpv_lists = self.tpv_transformer(img_voxel_feats)
        if hasattr(self, 'tpv_conv'):
            tpv_lists = [self.tpv_conv(view) for view in tpv_lists]
        x_3d, weights = self.tpv_aggregator(tpv_lists, img_voxel_feats)
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        losses = dict()
        if depth is not None:
            losses['loss_depth'] = self.depth_net.get_depth_loss(img_inputs[-4][:, 0:1, ...], depth)
        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output['output_voxels'],
            target_voxels=gt_occ,
        )
        losses.update(losses_occupancy)

        if self.normalize_loss:
            for key in losses:
                losses[key] = losses[key] / losses[key]

        losses_distill = {}
        if hasattr(self, 'teacher'):
            with torch.no_grad():
                tpv_lists_teacher, output_teacher, weights_teacher = self.forward_teacher(data_dict)

            # self.save_tpv(tpv_lists, tpv_lists_teacher, gt_occ_1_2)
            # self.save_logits_map(output['output_voxels'], output_teacher['output_voxels'])

            if self.ratio_logit > 0:
                losses_distill_logit = self.distill_loss_logits(output_teacher['output_voxels'], output['output_voxels'], gt_occ,
                                                                self.ratio_logit)
                losses_distill.update(losses_distill_logit)

            if self.ratio_tpv_feats > 0:
                losses_distill_tpv_feature = self.distill_loss_tpv_feature(tpv_lists_teacher, tpv_lists, gt_occ_1_2,
                                                                           self.ratio_tpv_feats)
                losses_distill.update(losses_distill_tpv_feature)

            if self.ratio_tpv_relation > 0:
                losses_distill_tpv_relation = self.distill_loss_tpv_relation(tpv_lists_teacher, tpv_lists, gt_occ_1_2,
                                                                             self.ratio_tpv_relation)
                losses_distill.update(losses_distill_tpv_relation)

            if self.ratio_tpv_weights > 0:
                losses_distill_tpv_weights = self.distill_loss_tpv_weights(weights_teacher, weights, gt_occ_1_2,
                                                                           self.ratio_tpv_weights)
                losses_distill.update(losses_distill_tpv_weights)

            losses.update(losses_distill)

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
        if hasattr(self, 'tpv_conv'):
            tpv_lists = [self.tpv_conv(view) for view in tpv_lists]
        x_3d, _ = self.tpv_aggregator(tpv_lists, img_voxel_feats)
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        # self.save_tpv(tpv_lists)
        # self.save_logits_map(output['output_voxels'])

        # if hasattr(self, 'teacher'):
        #     with torch.no_grad():
        #         tpv_lists_teacher, output_teacher = self.forward_teacher(data_dict)

        #     self.save_tpv(tpv_lists, tpv_lists_teacher)
        #     self.save_logits_map(output['output_voxels'], output_teacher['output_voxels'])

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {'pred': pred, 'gt_occ': gt_occ}
        return test_output

    def forward_teacher(self, data_dict):
        points = data_dict['points'][0]
        grid_ind = data_dict['grid_ind'][0]
        img_metas = data_dict['img_metas']
        if 'gt_occ' in data_dict:
            gt_occ = data_dict['gt_occ']
        else:
            gt_occ = None
        voxel_pos_grid_coarse = data_dict['voxel_position_grid_coarse'][0]

        x_lidar_tpv = self.teacher.extract_lidar_feat(points=points, grid_ind=grid_ind)
        tpv_lists = self.teacher.tpv_transformer(x_lidar_tpv, voxel_pos_grid_coarse)
        if hasattr(self.teacher, 'tpv_conv'):
            tpv_lists = [self.teacher.tpv_conv(view) for view in tpv_lists]
        x_3d, weights = self.teacher.tpv_aggregator(tpv_lists)
        output = self.teacher.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        return tpv_lists, output, weights

    def calculate_cosine_similarity(self, x, y):
        # 验证形状是否一致
        assert x.shape == y.shape, "输入特征的形状必须相同"
        B, C, H, W = x.shape

        # 扁平化特征的H和W维度，形状从 (B, C, H, W) 变为 (B, C, H*W)
        x_flat = x.reshape(B, C, -1)
        y_flat = y.reshape(B, C, -1)

        # # 计算归一化后的特征，使特征在C维度上具有单位长度
        x_norm = F.normalize(x_flat, p=2, dim=1)
        y_norm = F.normalize(y_flat, p=2, dim=1)

        # # 使用批量矩阵乘法计算cosine相似度 (B, H*W, H*W)
        cosine_similarity_flat = torch.bmm(x_norm.permute(0, 2, 1), y_norm)

        return cosine_similarity_flat

    def distill_loss_logits(self, logits_teacher, logits_student, target, ratio):
        b, c, h, w, z = logits_teacher.shape
        logits_student_softmax = F.log_softmax(logits_student, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        logits_teacher_softmax = F.softmax(logits_teacher, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        target = target.reshape(b, h * w * z)

        loss = 0
        for i in range(target.shape[0]):
            valid = (target[i] != 255)
            nonezero = (target[i] != 0)
            mask = valid * nonezero
            logits_student_i = logits_student_softmax[i][mask]
            logits_teacher_i = logits_teacher_softmax[i][mask]
            loss += nn.KLDivLoss(reduction="mean")(logits_student_i.unsqueeze(0), logits_teacher_i.unsqueeze(0))
        loss = loss / float(target.size(0)) * ratio
        return dict(loss_distill_logits=loss)

    def calc_feature_loss(self, feat1, feat2):
        loss_type = self.feature_loss_type
        if loss_type == 'l1':
            loss = F.l1_loss(feat1, feat2)
        elif loss_type == 'mse':
            loss = F.mse_loss(feat1, feat2)
        elif loss_type == 'cos_sim':
            cos_sim = cosine_similarity(feat1, feat2, dim=0)
            loss = 1 - cos_sim.mean()
        return loss

    def distill_loss_tpv_feature(self, tpv_teacher, tpv_student, target, ratio):
        loss = 0
        for i in range(target.shape[0]):
            target_i = target[i].to(torch.float32)
            target_i[target_i == 255] = 0

            tpv_xy_teacher_i = tpv_teacher[0][i].squeeze()
            tpv_xy_student_i = tpv_student[0][i].squeeze()
            target_xy_mean_i = target_i.mean(dim=2)
            mask = target_xy_mean_i != 0
            loss_xy = self.calc_feature_loss(tpv_xy_student_i[:, mask], tpv_xy_teacher_i[:, mask])

            tpv_yz_teacher_i = tpv_teacher[1][i].squeeze()
            tpv_yz_student_i = tpv_student[1][i].squeeze()
            target_yz_mean_i = target_i.mean(dim=0)
            mask = target_yz_mean_i != 0
            loss_yz = self.calc_feature_loss(tpv_yz_student_i[:, mask], tpv_yz_teacher_i[:, mask])

            tpv_zx_teacher_i = tpv_teacher[2][i].squeeze()
            tpv_zx_student_i = tpv_student[2][i].squeeze()
            target_zx_mean_i = target_i.mean(dim=1)
            mask = target_zx_mean_i != 0
            loss_zx = self.calc_feature_loss(tpv_zx_student_i[:, mask], tpv_zx_teacher_i[:, mask])

            loss += (loss_xy + loss_yz + loss_zx) / 3
        loss = loss / target.shape[0] * ratio
        return dict(loss_distill_tpv_feature=loss)

    def distill_loss_tpv_relation(self, tpv_teacher, tpv_student, target, ratio):
        loss = 0
        for i in range(target.shape[0]):
            # target_i = target[i].to(torch.float32)
            tpv_xy_teacher_i = tpv_teacher[0][i].unsqueeze(0).squeeze(4)
            tpv_xy_student_i = tpv_student[0][i].unsqueeze(0).squeeze(4)
            cos_sim_student = self.calculate_cosine_similarity(tpv_xy_student_i, tpv_xy_student_i)
            cos_sim_teacher = self.calculate_cosine_similarity(tpv_xy_teacher_i, tpv_xy_teacher_i)
            loss_xy = F.l1_loss(cos_sim_student, cos_sim_teacher)

            tpv_yz_teacher_i = tpv_teacher[1][i].unsqueeze(0).squeeze(2)
            tpv_yz_student_i = tpv_student[1][i].unsqueeze(0).squeeze(2)
            cos_sim_student = self.calculate_cosine_similarity(tpv_yz_student_i, tpv_yz_student_i)
            cos_sim_teacher = self.calculate_cosine_similarity(tpv_yz_teacher_i, tpv_yz_teacher_i)
            loss_yz = F.l1_loss(cos_sim_student, cos_sim_teacher)

            tpv_zx_teacher_i = tpv_teacher[2][i].unsqueeze(0).squeeze(3)
            tpv_zx_student_i = tpv_student[2][i].unsqueeze(0).squeeze(3)
            cos_sim_student = self.calculate_cosine_similarity(tpv_zx_student_i, tpv_zx_student_i)
            cos_sim_teacher = self.calculate_cosine_similarity(tpv_zx_teacher_i, tpv_zx_teacher_i)
            loss_zx = F.l1_loss(cos_sim_student, cos_sim_teacher)

            loss += (loss_xy + loss_yz + loss_zx) / 3
        loss = loss / target.shape[0] * ratio
        return dict(loss_distill_tpv_relation=loss)

    def distill_loss_tpv_weights(self, weights_teacher, weights_student, target, ratio):
        b, c, h, w, z = weights_teacher.shape
        weights_student_softmax = F.log_softmax(weights_student, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        weights_teacher_softmax = F.softmax(weights_teacher, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        target = target.reshape(b, h * w * z)

        loss = 0
        for i in range(target.shape[0]):
            valid = (target[i] != 255)
            nonezero = (target[i] != 0)
            mask = valid * nonezero
            weights_student_i = weights_student_softmax[i][mask]
            weights_teacher_i = weights_teacher_softmax[i][mask]
            loss += nn.KLDivLoss(reduction="mean")(weights_student_i.unsqueeze(0), weights_teacher_i.unsqueeze(0))
        loss = loss / float(target.size(0)) * ratio
        return dict(loss_distill_weights=loss)


@DETECTORS.register_module()
class CameraSegmentorEfficientSSCV3(BaseModule):

    def __init__(
        self,
        img_backbone,
        img_neck,
        img_view_transformer,
        depth_net=None,
        proposal_layer=None,
        VoxFormer_head=None,
        bev_transformer=None,
        bev_aggregator=None,
        pts_bbox_head=None,
        init_cfg=None,
        teacher=None,
        teacher_ckpt=None,
        feature_loss_type='l1',
        ratio_logit=10.0,
        ratio_feats_mse=2,
        ratio_feats_relation=10,
        **kwargs,
    ):

        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)
        self.depth_net = builder.build_neck(depth_net)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.proposal_layer = builder.build_head(proposal_layer)
        self.VoxFormer_head = builder.build_head(VoxFormer_head)
        self.bev_transformer = builder.build_backbone(bev_transformer)
        self.bev_aggregator = builder.build_backbone(bev_aggregator)
        self.pts_bbox_head = builder.build_head(pts_bbox_head)

        # init before teacher to avoid overwriting teacher weights

        self.init_cfg = init_cfg
        self.init_weights()

        if teacher:
            self.teacher = builder.build_detector(teacher).eval()
            if os.path.exists(teacher_ckpt):
                ckpt = torch.load(teacher_ckpt)['state_dict']
                adjusted_ckpt = {key.replace('model.', ''): value for key, value in ckpt.items()}
                self.teacher.load_state_dict(adjusted_ckpt)
                print(f"Load teacher model from {teacher_ckpt}")
            self.freeze_model(self.teacher)
            self.feature_loss_type = feature_loss_type
            self.ratio_logit = ratio_logit
            self.ratio_feats = ratio_feats_mse
            self.ratio_relation = ratio_feats_relation

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

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

        lss_volume = self.img_view_transformer(context, depth, view_trans_inputs)

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

    def save_bev(self, bev_cam, tpv_lidar=None, gt_occ_1_2=None):
        tpv_list = bev_cam
        if gt_occ_1_2 is not None:
            target = gt_occ_1_2.to(torch.float32)
            target[target == 255] = 0

            target_xy_mean = target.mean(dim=3)
            mask_xy = target_xy_mean == 0
            mask_xy = mask_xy.unsqueeze(1).unsqueeze(4).expand_as(tpv_list[0])

            target_yz_mean = target.mean(dim=1)
            mask_yz = target_yz_mean == 0
            mask_yz = mask_yz.unsqueeze(1).unsqueeze(2).expand_as(tpv_list[1])

            target_zx_mean = target.mean(dim=2)
            mask_zx = target_zx_mean == 0
            mask_zx = mask_zx.unsqueeze(1).unsqueeze(3).expand_as(tpv_list[2])

            tpv_list[0][mask_xy] = 0
            tpv_list[1][mask_yz] = 0
            tpv_list[2][mask_zx] = 0

        # format to b,n,c,h,w
        feat_xy = tpv_list[0].squeeze(-1).unsqueeze(1).permute(0, 1, 2, 3, 4)
        feat_yz = torch.flip(tpv_list[1].squeeze(-3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
        feat_zx = torch.flip(tpv_list[2].squeeze(-2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])

        save_feature_map_as_image(feat_xy.detach(), 'save/distill/tpv_cam', 'xy', method='pca')
        save_feature_map_as_image(feat_yz.detach(), 'save/distill/tpv_cam', 'yz', method='pca')
        save_feature_map_as_image(feat_zx.detach(), 'save/distill/tpv_cam', 'zx', method='pca')

        if tpv_lidar is not None:
            tpv_list = tpv_lidar
            if gt_occ_1_2 is not None:
                target = gt_occ_1_2.to(torch.float32)
                target[target == 255] = 0

                target_xy_mean = target.mean(dim=3)
                mask_xy = target_xy_mean == 0
                mask_xy = mask_xy.unsqueeze(1).unsqueeze(4).expand_as(tpv_list[0])

                target_yz_mean = target.mean(dim=1)
                mask_yz = target_yz_mean == 0
                mask_yz = mask_yz.unsqueeze(1).unsqueeze(2).expand_as(tpv_list[1])

                target_zx_mean = target.mean(dim=2)
                mask_zx = target_zx_mean == 0
                mask_zx = mask_zx.unsqueeze(1).unsqueeze(3).expand_as(tpv_list[2])

                tpv_list[0][mask_xy] = 0
                tpv_list[1][mask_yz] = 0
                tpv_list[2][mask_zx] = 0
                feat_xy = tpv_list[0].squeeze(-1).unsqueeze(1).permute(0, 1, 2, 3, 4)
                feat_yz = torch.flip(tpv_list[1].squeeze(-3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
                feat_zx = torch.flip(tpv_list[2].squeeze(-2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])

            save_feature_map_as_image(feat_xy.detach(), 'save/distill/tpv_lidar', 'xy', method='pca')
            save_feature_map_as_image(feat_yz.detach(), 'save/distill/tpv_lidar', 'yz', method='pca')
            save_feature_map_as_image(feat_zx.detach(), 'save/distill/tpv_lidar', 'zx', method='pca')

        # remind to comment while training
        pdb.set_trace()
        return

    def save_logits_map(self, logits_cam, logits_lidar=None):
        # format to b,n,c,h,w
        feat_xy = logits_cam.mean(dim=4).unsqueeze(1).permute(0, 1, 2, 3, 4)
        feat_yz = torch.flip(logits_cam.mean(dim=2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
        feat_zx = torch.flip(logits_cam.mean(dim=3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])

        save_feature_map_as_image(feat_xy.detach(), 'save/distill/logits_cam', 'xy', method='pca')
        save_feature_map_as_image(feat_yz.detach(), 'save/distill/logits_cam', 'yz', method='pca')
        save_feature_map_as_image(feat_zx.detach(), 'save/distill/logits_cam', 'zx', method='pca')

        if logits_lidar is not None:
            feat_xy = logits_lidar.mean(dim=4).unsqueeze(1).permute(0, 1, 2, 3, 4)
            feat_yz = torch.flip(logits_lidar.mean(dim=2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
            feat_zx = torch.flip(logits_lidar.mean(dim=3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])

            save_feature_map_as_image(feat_xy.detach(), 'save/distill/logits_lidar', 'xy', method='pca')
            save_feature_map_as_image(feat_yz.detach(), 'save/distill/logits_lidar', 'yz', method='pca')
            save_feature_map_as_image(feat_zx.detach(), 'save/distill/logits_lidar', 'zx', method='pca')

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
        gt_occ_1_2 = img_metas['gt_occ_1_2']

        img_voxel_feats, query_proposal, depth = self.extract_img_feat(img_inputs, img_metas)

        bev_list = self.bev_transformer(img_voxel_feats)

        x_3d, weights = self.bev_aggregator(bev_list, img_voxel_feats)
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        losses = dict()
        if depth is not None:
            losses['loss_depth'] = self.depth_net.get_depth_loss(img_inputs[-4][:, 0:1, ...], depth)
        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output['output_voxels'],
            target_voxels=gt_occ,
        )
        losses.update(losses_occupancy)

        losses_distill = {}
        if hasattr(self, 'teacher'):
            with torch.no_grad():
                tpv_lists_teacher, output_teacher, weights_teacher = self.forward_teacher(data_dict)

            # self.save_tpv(tpv_lists, tpv_lists_teacher, gt_occ_1_2)
            # self.save_logits_map(output['output_voxels'], output_teacher['output_voxels'])

            if self.ratio_logit > 0:
                losses_distill_logit = self.distill_loss_logits(output_teacher['output_voxels'], output['output_voxels'], gt_occ,
                                                                self.ratio_logit)
                losses_distill.update(losses_distill_logit)

            if self.ratio_tpv_feats > 0:
                losses_distill_tpv_feature = self.distill_loss_tpv_feature(tpv_lists_teacher, tpv_lists, gt_occ_1_2,
                                                                           self.ratio_tpv_feats)
                losses_distill.update(losses_distill_tpv_feature)

            if self.ratio_tpv_relation > 0:
                losses_distill_tpv_relation = self.distill_loss_tpv_relation(tpv_lists_teacher, tpv_lists, gt_occ_1_2,
                                                                             self.ratio_tpv_relation)
                losses_distill.update(losses_distill_tpv_relation)

            if self.ratio_tpv_weights > 0:
                losses_distill_tpv_weights = self.distill_loss_tpv_weights(weights_teacher, weights, gt_occ_1_2,
                                                                           self.ratio_tpv_weights)
                losses_distill.update(losses_distill_tpv_weights)

            losses.update(losses_distill)

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
        tpv_lists = self.bev_transformer(img_voxel_feats)
        x_3d, _ = self.bev_aggregator(tpv_lists, img_voxel_feats)
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        # self.save_tpv(tpv_lists)
        # self.save_logits_map(output['output_voxels'])

        # if hasattr(self, 'teacher'):
        #     with torch.no_grad():
        #         tpv_lists_teacher, output_teacher = self.forward_teacher(data_dict)

        #     self.save_tpv(tpv_lists, tpv_lists_teacher)
        #     self.save_logits_map(output['output_voxels'], output_teacher['output_voxels'])

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {'pred': pred, 'gt_occ': gt_occ}
        return test_output

    def forward_teacher(self, data_dict):
        points = data_dict['points'][0]
        grid_ind = data_dict['grid_ind'][0]
        img_metas = data_dict['img_metas']
        if 'gt_occ' in data_dict:
            gt_occ = data_dict['gt_occ']
        else:
            gt_occ = None
        voxel_pos_grid_coarse = data_dict['voxel_position_grid_coarse'][0]

        x_lidar_tpv = self.teacher.extract_lidar_feat(points=points, grid_ind=grid_ind)
        tpv_lists = self.teacher.tpv_transformer(x_lidar_tpv, voxel_pos_grid_coarse)
        if hasattr(self.teacher, 'tpv_conv'):
            tpv_lists = [self.teacher.tpv_conv(view) for view in tpv_lists]
        x_3d, weights = self.teacher.tpv_aggregator(tpv_lists)
        output = self.teacher.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        return tpv_lists, output, weights

    def calculate_cosine_similarity(self, x, y):
        # 验证形状是否一致
        assert x.shape == y.shape, "输入特征的形状必须相同"
        B, C, H, W = x.shape

        # 扁平化特征的H和W维度，形状从 (B, C, H, W) 变为 (B, C, H*W)
        x_flat = x.reshape(B, C, -1)
        y_flat = y.reshape(B, C, -1)

        # # 计算归一化后的特征，使特征在C维度上具有单位长度
        x_norm = F.normalize(x_flat, p=2, dim=1)
        y_norm = F.normalize(y_flat, p=2, dim=1)

        # # 使用批量矩阵乘法计算cosine相似度 (B, H*W, H*W)
        cosine_similarity_flat = torch.bmm(x_norm.permute(0, 2, 1), y_norm)

        return cosine_similarity_flat

    def distill_loss_logits(self, logits_teacher, logits_student, target, ratio):
        b, c, h, w, z = logits_teacher.shape
        logits_student_softmax = F.log_softmax(logits_student, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        logits_teacher_softmax = F.softmax(logits_teacher, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        target = target.reshape(b, h * w * z)

        loss = 0
        for i in range(target.shape[0]):
            valid = (target[i] != 255)
            nonezero = (target[i] != 0)
            mask = valid * nonezero
            logits_student_i = logits_student_softmax[i][mask]
            logits_teacher_i = logits_teacher_softmax[i][mask]
            loss += nn.KLDivLoss(reduction="mean")(logits_student_i.unsqueeze(0), logits_teacher_i.unsqueeze(0))
        loss = loss / float(target.size(0)) * ratio
        return dict(loss_distill_logits=loss)

    def calc_feature_loss(self, feat1, feat2):
        loss_type = self.feature_loss_type
        if loss_type == 'l1':
            loss = F.l1_loss(feat1, feat2)
        elif loss_type == 'mse':
            loss = F.mse_loss(feat1, feat2)
        elif loss_type == 'cos_sim':
            cos_sim = cosine_similarity(feat1, feat2, dim=0)
            loss = 1 - cos_sim.mean()
        return loss

    def distill_loss_tpv_feature(self, tpv_teacher, tpv_student, target, ratio):
        loss = 0
        for i in range(target.shape[0]):
            target_i = target[i].to(torch.float32)
            target_i[target_i == 255] = 0

            tpv_xy_teacher_i = tpv_teacher[0][i].squeeze()
            tpv_xy_student_i = tpv_student[0][i].squeeze()
            target_xy_mean_i = target_i.mean(dim=2)
            mask = target_xy_mean_i != 0
            loss_xy = self.calc_feature_loss(tpv_xy_student_i[:, mask], tpv_xy_teacher_i[:, mask])

            tpv_yz_teacher_i = tpv_teacher[1][i].squeeze()
            tpv_yz_student_i = tpv_student[1][i].squeeze()
            target_yz_mean_i = target_i.mean(dim=0)
            mask = target_yz_mean_i != 0
            loss_yz = self.calc_feature_loss(tpv_yz_student_i[:, mask], tpv_yz_teacher_i[:, mask])

            tpv_zx_teacher_i = tpv_teacher[2][i].squeeze()
            tpv_zx_student_i = tpv_student[2][i].squeeze()
            target_zx_mean_i = target_i.mean(dim=1)
            mask = target_zx_mean_i != 0
            loss_zx = self.calc_feature_loss(tpv_zx_student_i[:, mask], tpv_zx_teacher_i[:, mask])

            loss += (loss_xy + loss_yz + loss_zx) / 3
        loss = loss / target.shape[0] * ratio
        return dict(loss_distill_tpv_feature=loss)

    def distill_loss_tpv_relation(self, tpv_teacher, tpv_student, target, ratio):
        loss = 0
        for i in range(target.shape[0]):
            # target_i = target[i].to(torch.float32)
            tpv_xy_teacher_i = tpv_teacher[0][i].unsqueeze(0).squeeze(4)
            tpv_xy_student_i = tpv_student[0][i].unsqueeze(0).squeeze(4)
            cos_sim_student = self.calculate_cosine_similarity(tpv_xy_student_i, tpv_xy_student_i)
            cos_sim_teacher = self.calculate_cosine_similarity(tpv_xy_teacher_i, tpv_xy_teacher_i)
            loss_xy = F.l1_loss(cos_sim_student, cos_sim_teacher)

            tpv_yz_teacher_i = tpv_teacher[1][i].unsqueeze(0).squeeze(2)
            tpv_yz_student_i = tpv_student[1][i].unsqueeze(0).squeeze(2)
            cos_sim_student = self.calculate_cosine_similarity(tpv_yz_student_i, tpv_yz_student_i)
            cos_sim_teacher = self.calculate_cosine_similarity(tpv_yz_teacher_i, tpv_yz_teacher_i)
            loss_yz = F.l1_loss(cos_sim_student, cos_sim_teacher)

            tpv_zx_teacher_i = tpv_teacher[2][i].unsqueeze(0).squeeze(3)
            tpv_zx_student_i = tpv_student[2][i].unsqueeze(0).squeeze(3)
            cos_sim_student = self.calculate_cosine_similarity(tpv_zx_student_i, tpv_zx_student_i)
            cos_sim_teacher = self.calculate_cosine_similarity(tpv_zx_teacher_i, tpv_zx_teacher_i)
            loss_zx = F.l1_loss(cos_sim_student, cos_sim_teacher)

            loss += (loss_xy + loss_yz + loss_zx) / 3
        loss = loss / target.shape[0] * ratio
        return dict(loss_distill_tpv_relation=loss)

    def distill_loss_tpv_weights(self, weights_teacher, weights_student, target, ratio):
        b, c, h, w, z = weights_teacher.shape
        weights_student_softmax = F.log_softmax(weights_student, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        weights_teacher_softmax = F.softmax(weights_teacher, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        target = target.reshape(b, h * w * z)

        loss = 0
        for i in range(target.shape[0]):
            valid = (target[i] != 255)
            nonezero = (target[i] != 0)
            mask = valid * nonezero
            weights_student_i = weights_student_softmax[i][mask]
            weights_teacher_i = weights_teacher_softmax[i][mask]
            loss += nn.KLDivLoss(reduction="mean")(weights_student_i.unsqueeze(0), weights_teacher_i.unsqueeze(0))
        loss = loss / float(target.size(0)) * ratio
        return dict(loss_distill_weights=loss)


@DETECTORS.register_module()
class CameraSegmentorEfficientSSCV4(BaseModule):

    def __init__(
        self,
        img_backbone,
        img_neck,
        img_view_transformer,
        depth_net=None,
        proposal_layer=None,
        VoxFormer_head=None,
        mtv_transformer=None,
        mtv_aggregator=None,
        pts_bbox_head=None,
        init_cfg=None,
        teacher=None,
        teacher_ckpt=None,
        feature_loss_type='l1',
        ratio_logit_kl=10.0,
        ratio_feats_numeric=2,
        ratio_feats_relation=10,
        num_views=[1, 1, 1],
        grid_size=[128, 128, 16],
        **kwargs,
    ):

        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)
        self.depth_net = builder.build_neck(depth_net)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.proposal_layer = builder.build_head(proposal_layer)
        self.VoxFormer_head = builder.build_head(VoxFormer_head)
        self.mtv_transformer = builder.build_backbone(mtv_transformer)
        self.mtv_aggregator = builder.build_backbone(mtv_aggregator)
        self.pts_bbox_head = builder.build_head(pts_bbox_head)
        self.num_views = num_views
        self.grid_size = grid_size

        # init before teacher to avoid overwriting teacher weights
        self.init_cfg = init_cfg
        self.init_weights()

        if teacher:
            self.teacher = builder.build_detector(teacher).eval()
            if os.path.exists(teacher_ckpt):
                ckpt = torch.load(teacher_ckpt)['state_dict']
                adjusted_ckpt = {key.replace('model.', ''): value for key, value in ckpt.items()}
                self.teacher.load_state_dict(adjusted_ckpt)
                print(f"Load teacher model from {teacher_ckpt}")
            self.freeze_model(self.teacher)
            self.feature_loss_type = feature_loss_type
            self.ratio_logit_kl = ratio_logit_kl
            self.ratio_feats_numeric = ratio_feats_numeric
            self.ratio_feats_relation = ratio_feats_relation

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

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

        lss_volume = self.img_view_transformer(context, depth, view_trans_inputs)

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

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)

    def forward_train(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']
        gt_occ_1_2 = img_metas['gt_occ_1_2']

        # img encoder
        img_voxel_feats, query_proposal, depth = self.extract_img_feat(img_inputs, img_metas)

        # mtv transformer
        mtv_lists, mtv_weights, feats_all = self.mtv_transformer(img_voxel_feats)

        # mtv aggregator
        x_3d, aggregator_weights = self.mtv_aggregator(mtv_lists, mtv_weights, img_voxel_feats)

        # cls head
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        # loss
        losses = dict()
        losses['loss_depth'] = self.depth_net.get_depth_loss(img_inputs[-4][:, 0:1, ...], depth)
        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output['output_voxels'],
            target_voxels=gt_occ,
        )
        losses.update(losses_occupancy)

        # distillation
        losses_distill = {}
        if hasattr(self, 'teacher'):
            with torch.no_grad():
                mtv_lists_teacher, output_teacher, aggregator_weights_teacher, feats_all_teacher = self.forward_teacher(data_dict)

            # self.save_tpv(tpv_lists, tpv_lists_teacher, gt_occ_1_2)
            # self.save_logits_map(output['output_voxels'], output_teacher['output_voxels'])

            if self.ratio_logit_kl > 0:
                losses_distill_logit = self.distill_loss_logits(
                    logits_teacher=output_teacher['output_voxels'],
                    logits_student=output['output_voxels'],
                    target=gt_occ,
                    ratio=self.ratio_logit_kl,
                )
                losses_distill.update(losses_distill_logit)

            if self.ratio_feats_numeric > 0 or self.ratio_feats_relation > 0:
                losses_distill_feature, feats_student, feats_teacher, masks = self.distill_loss_feature(
                    feats_teacher=feats_all_teacher,
                    feats_student=feats_all,
                    target=gt_occ_1_2,
                    ratio_numeric=self.ratio_feats_numeric,
                    ration_relation=self.ratio_feats_relation,
                )
                losses_distill.update(losses_distill_feature)

            # save_mtv(mtv_lists, mtv_lists_teacher, self.num_views)
            # save_all_feats(feats_student, feats_teacher, masks)
            # save_weights(aggregator_weights, aggregator_weights_teacher)

            # if self.ratio_tpv_weights > 0:
            #     losses_distill_tpv_weights = self.distill_loss_tpv_weights(weights_teacher, weights, gt_occ_1_2,
            #                                                                self.ratio_tpv_weights)
            #     losses_distill.update(losses_distill_tpv_weights)

            losses.update(losses_distill)

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

        # img encoder
        img_voxel_feats, query_proposal, depth = self.extract_img_feat(img_inputs, img_metas)

        # mtv transformer
        mtv_lists, mtv_weights, _ = self.mtv_transformer(img_voxel_feats)

        # mtv aggregator
        x_3d, aggregator_weights = self.mtv_aggregator(mtv_lists, mtv_weights, img_voxel_feats)

        # cls head
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        # self.save_tpv(tpv_lists)
        # self.save_logits_map(output['output_voxels'])

        # if hasattr(self, 'teacher'):
        #     with torch.no_grad():
        #         tpv_lists_teacher, output_teacher = self.forward_teacher(data_dict)

        #     self.save_tpv(tpv_lists, tpv_lists_teacher)
        #     self.save_logits_map(output['output_voxels'], output_teacher['output_voxels'])

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {'pred': pred, 'gt_occ': gt_occ}
        return test_output

    def forward_teacher(self, data_dict):
        points = data_dict['points'][0]
        grid_ind = data_dict['grid_ind'][0]
        img_metas = data_dict['img_metas']
        if 'gt_occ' in data_dict:
            gt_occ = data_dict['gt_occ']
        else:
            gt_occ = None

        # lidar encoder
        lidar_voxel_feats = self.teacher.extract_lidar_feat(points=points, grid_ind=grid_ind)

        # mtv transformer
        mtv_lists, mtv_weights, feats_all = self.teacher.mtv_transformer(lidar_voxel_feats)

        # mtv aggregator
        x_3d, aggregator_weights = self.teacher.mtv_aggregator(mtv_lists, mtv_weights, lidar_voxel_feats)

        # cls head
        output = self.teacher.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        return mtv_lists, output, aggregator_weights, feats_all

    def calculate_cosine_similarity(self, x, y):
        assert x.shape == y.shape, "输入特征的形状必须相同"
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B, C, H * W)
        x_flat = x.reshape(B, C, -1)
        y_flat = y.reshape(B, C, -1)

        # normalize
        x_norm = F.normalize(x_flat, p=2, dim=1)
        y_norm = F.normalize(y_flat, p=2, dim=1)

        # bmm
        cosine_similarity_flat = torch.bmm(x_norm.permute(0, 2, 1), y_norm)

        return cosine_similarity_flat

    def distill_loss_logits(self, logits_teacher, logits_student, target, ratio):
        b, c, h, w, z = logits_teacher.shape
        logits_student_softmax = F.log_softmax(logits_student, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        logits_teacher_softmax = F.softmax(logits_teacher, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        target = target.reshape(b, h * w * z)

        loss = 0
        for i in range(target.shape[0]):
            valid = (target[i] != 255)
            nonezero = (target[i] != 0)
            mask = valid * nonezero
            logits_student_i = logits_student_softmax[i][mask]
            logits_teacher_i = logits_teacher_softmax[i][mask]
            loss += nn.KLDivLoss(reduction="mean")(logits_student_i.unsqueeze(0), logits_teacher_i.unsqueeze(0))
        loss = loss / float(target.size(0)) * ratio
        return dict(loss_distill_logits=loss)

    def distill_loss_feature(self, feats_teacher, feats_student, target, ratio_numeric, ration_relation):
        feats_teacher_list = []
        feats_student_list = []
        mask_list = []

        target = target.to(torch.float32)
        target[target == 255] = 0

        # feats 3d
        mask = (target != 0).unsqueeze(1).expand_as(feats_student['feats3d'])
        feats_student_list.append(feats_student['feats3d'])
        feats_teacher_list.append(feats_teacher['feats3d'])
        mask_list.append(mask)

        # feats 2d
        target_xy_mean = target.mean(dim=3)
        mask_xy = target_xy_mean != 0
        size_xy = (self.grid_size[0], self.grid_size[1])
        target_yz_mean = target.mean(dim=1)
        mask_yz = target_yz_mean != 0
        size_yz = (self.grid_size[1], self.grid_size[2])
        target_zx_mean = target.mean(dim=2)
        mask_zx = target_zx_mean != 0
        size_zx = (self.grid_size[0], self.grid_size[2])

        # mtv
        feats_backbone_teacher = feats_teacher['mtv_backbone']
        feats_backbone_student = feats_student['mtv_backbone']
        feats_neck_teacher = feats_teacher['mtv_neck']
        feats_neck_student = feats_student['mtv_neck']

        # xy plane
        for i in range(self.num_views[0]):
            for j in range(len(feats_backbone_teacher[i])):
                feat_student = feats_backbone_student[i][j]
                feat_teacher = feats_backbone_teacher[i][j]
                feat_student = F.interpolate(feat_student, size=size_xy, mode='bilinear', align_corners=False)
                feat_teacher = F.interpolate(feat_teacher, size=size_xy, mode='bilinear', align_corners=False)
                mask = mask_xy.unsqueeze(1).expand_as(feat_student)

                feats_student_list.append(feat_student)
                feats_teacher_list.append(feat_teacher)
                mask_list.append(mask)

            for j in range(len(feats_neck_teacher[i])):
                feat_student = feats_neck_student[i][j]
                feat_teacher = feats_neck_teacher[i][j]
                feat_student = F.interpolate(feat_student, size=size_xy, mode='bilinear', align_corners=False)
                feat_teacher = F.interpolate(feat_teacher, size=size_xy, mode='bilinear', align_corners=False)
                mask = mask_xy.unsqueeze(1).expand_as(feat_student)

                feats_student_list.append(feat_student)
                feats_teacher_list.append(feat_teacher)
                mask_list.append(mask)

        # yz plane
        for i in range(self.num_views[0], self.num_views[0] + self.num_views[1]):
            for j in range(len(feats_backbone_teacher[i])):
                feat_student = feats_backbone_student[i][j]
                feat_teacher = feats_backbone_teacher[i][j]
                feat_student = F.interpolate(feat_student, size=size_yz, mode='bilinear', align_corners=False)
                feat_teacher = F.interpolate(feat_teacher, size=size_yz, mode='bilinear', align_corners=False)
                mask = mask_yz.unsqueeze(1).expand_as(feat_student)

                feats_student_list.append(feat_student)
                feats_teacher_list.append(feat_teacher)
                mask_list.append(mask)

            for j in range(len(feats_neck_teacher[i])):
                feat_student = feats_neck_student[i][j]
                feat_teacher = feats_neck_teacher[i][j]
                feat_student = F.interpolate(feat_student, size=size_yz, mode='bilinear', align_corners=False)
                feat_teacher = F.interpolate(feat_teacher, size=size_yz, mode='bilinear', align_corners=False)
                mask = mask_yz.unsqueeze(1).expand_as(feat_student)

                feats_student_list.append(feat_student)
                feats_teacher_list.append(feat_teacher)
                mask_list.append(mask)

        # zx plane
        for i in range(self.num_views[0] + self.num_views[1], self.num_views[0] + self.num_views[1] + self.num_views[2]):
            for j in range(len(feats_backbone_teacher[i])):
                feat_student = feats_backbone_student[i][j]
                feat_teacher = feats_backbone_teacher[i][j]
                feat_student = F.interpolate(feat_student, size=size_zx, mode='bilinear', align_corners=False)
                feat_teacher = F.interpolate(feat_teacher, size=size_zx, mode='bilinear', align_corners=False)
                mask = mask_zx.unsqueeze(1).expand_as(feat_student)

                feats_student_list.append(feat_student)
                feats_teacher_list.append(feat_teacher)
                mask_list.append(mask)

            for j in range(len(feats_neck_teacher[i])):
                feat_student = feats_neck_student[i][j]
                feat_teacher = feats_neck_teacher[i][j]
                feat_student = F.interpolate(feat_student, size=size_zx, mode='bilinear', align_corners=False)
                feat_teacher = F.interpolate(feat_teacher, size=size_zx, mode='bilinear', align_corners=False)
                mask = mask_zx.unsqueeze(1).expand_as(feat_student)

                feats_student_list.append(feat_student)
                feats_teacher_list.append(feat_teacher)
                mask_list.append(mask)

        losses_feature = {}

        # numeric loss
        if ratio_numeric > 0:
            loss_numeric = 0
            for i in range(len(mask_list)):
                mask = mask_list[i]
                feat_student = feats_student_list[i][mask]
                feat_teacher = feats_teacher_list[i][mask]
                loss = (F.l1_loss(feat_student, feat_teacher) + F.mse_loss(feat_student, feat_teacher)) / 2
                loss_numeric += loss
            loss_numeric = loss_numeric / len(mask_list) * ratio_numeric
            losses_feature.update(dict(loss_distill_feature_numeric=loss_numeric))

        # relation loss
        if ration_relation > 0:
            # skip 3d feature
            for i in range(1, len(feats_student_list)):
                mask = mask_list[i]
                feat_student = feats_student_list[i] * mask
                feat_teacher = feats_teacher_list[i] * mask
                cos_sim_student = self.calculate_cosine_similarity(feat_student, feat_student)
                cos_sim_teacher = self.calculate_cosine_similarity(feat_teacher, feat_teacher)
                loss_relation = F.l1_loss(cos_sim_student, cos_sim_teacher)
                loss_relation += loss_relation
            loss_relation = loss_relation / len(feats_student_list) * ration_relation
            losses_feature.update(dict(loss_distill_feature_relation=loss_relation))

        return losses_feature, feats_student_list, feats_teacher_list, mask_list

    def distill_loss_tpv_weights(self, weights_teacher, weights_student, target, ratio):
        b, c, h, w, z = weights_teacher.shape
        weights_student_softmax = F.log_softmax(weights_student, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        weights_teacher_softmax = F.softmax(weights_teacher, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        target = target.reshape(b, h * w * z)

        loss = 0
        for i in range(target.shape[0]):
            valid = (target[i] != 255)
            nonezero = (target[i] != 0)
            mask = valid * nonezero
            weights_student_i = weights_student_softmax[i][mask]
            weights_teacher_i = weights_teacher_softmax[i][mask]
            loss += nn.KLDivLoss(reduction="mean")(weights_student_i.unsqueeze(0), weights_teacher_i.unsqueeze(0))
        loss = loss / float(target.size(0)) * ratio
        return dict(loss_distill_weights=loss)


@DETECTORS.register_module()
class CameraSegmentorEfficientSSCV5(BaseModule):

    def __init__(
        self,
        img_backbone,
        img_neck,
        img_view_transformer,
        depth_net=None,
        proposal_layer=None,
        VoxFormer_head=None,
        voxel_backbone=None,
        voxel_neck=None,
        mtv_transformer=None,
        mtv_aggregator=None,
        pts_bbox_head=None,
        init_cfg=None,
        teacher=None,
        teacher_ckpt=None,
        feature_loss_type='l1',
        ratio_logit_kl=10.0,
        ratio_feats_numeric=2,
        ratio_feats_relation=10,
        num_views=[1, 1, 1],
        grid_size=[128, 128, 16],
        **kwargs,
    ):

        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)
        self.depth_net = builder.build_neck(depth_net)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.proposal_layer = builder.build_head(proposal_layer)
        self.VoxFormer_head = builder.build_head(VoxFormer_head)
        self.voxel_backbone = builder.build_backbone(voxel_backbone)
        self.voxel_neck = builder.build_neck(voxel_neck)
        self.mtv_transformer = builder.build_backbone(mtv_transformer)
        self.mtv_aggregator = builder.build_backbone(mtv_aggregator)
        self.pts_bbox_head = builder.build_head(pts_bbox_head)
        self.num_views = num_views
        self.grid_size = grid_size

        # init before teacher to avoid overwriting teacher weights
        self.init_cfg = init_cfg
        self.init_weights()

        if teacher:
            self.teacher = builder.build_detector(teacher).eval()
            if os.path.exists(teacher_ckpt):
                ckpt = torch.load(teacher_ckpt)['state_dict']
                adjusted_ckpt = {key.replace('model.', ''): value for key, value in ckpt.items()}
                self.teacher.load_state_dict(adjusted_ckpt)
                print(f"Load teacher model from {teacher_ckpt}")
            self.freeze_model(self.teacher)
            self.feature_loss_type = feature_loss_type
            self.ratio_logit_kl = ratio_logit_kl
            self.ratio_feats_numeric = ratio_feats_numeric
            self.ratio_feats_relation = ratio_feats_relation

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

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

        lss_volume = self.img_view_transformer(context, depth, view_trans_inputs)

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
        x = self.voxel_backbone(x)
        x = self.voxel_neck(x)[0]
        # pdb.set_trace()
        return x, query_proposal, depth

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)

    def forward_train(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']
        gt_occ_1_2 = img_metas['gt_occ_1_2']

        # img encoder
        img_voxel_feats, query_proposal, depth = self.extract_img_feat(img_inputs, img_metas)

        # mtv transformer
        mtv_lists, mtv_weights, feats_all = self.mtv_transformer(img_voxel_feats)

        # mtv aggregator
        x_3d, aggregator_weights = self.mtv_aggregator(mtv_lists, mtv_weights, img_voxel_feats)

        # cls head
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        # loss
        losses = dict()
        losses['loss_depth'] = self.depth_net.get_depth_loss(img_inputs[-4][:, 0:1, ...], depth)
        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output['output_voxels'],
            target_voxels=gt_occ,
        )
        losses.update(losses_occupancy)

        # distillation
        losses_distill = {}
        if hasattr(self, 'teacher'):
            with torch.no_grad():
                mtv_lists_teacher, output_teacher, aggregator_weights_teacher, feats_all_teacher = self.forward_teacher(data_dict)

            # self.save_tpv(tpv_lists, tpv_lists_teacher, gt_occ_1_2)
            # self.save_logits_map(output['output_voxels'], output_teacher['output_voxels'])

            if self.ratio_logit_kl > 0:
                losses_distill_logit = self.distill_loss_logits(
                    logits_teacher=output_teacher['output_voxels'],
                    logits_student=output['output_voxels'],
                    target=gt_occ,
                    ratio=self.ratio_logit_kl,
                )
                losses_distill.update(losses_distill_logit)

            if self.ratio_feats_numeric > 0 or self.ratio_feats_relation > 0:
                losses_distill_feature, feats_student, feats_teacher, masks = self.distill_loss_feature(
                    feats_teacher=feats_all_teacher,
                    feats_student=feats_all,
                    target=gt_occ_1_2,
                    ratio_numeric=self.ratio_feats_numeric,
                    ration_relation=self.ratio_feats_relation,
                )
                losses_distill.update(losses_distill_feature)

            save_mtv(mtv_lists, mtv_lists_teacher, self.num_views)
            save_all_feats(feats_student, feats_teacher, masks)
            save_weights(aggregator_weights, aggregator_weights_teacher)

            # if self.ratio_tpv_weights > 0:
            #     losses_distill_tpv_weights = self.distill_loss_tpv_weights(weights_teacher, weights, gt_occ_1_2,
            #                                                                self.ratio_tpv_weights)
            #     losses_distill.update(losses_distill_tpv_weights)

            losses.update(losses_distill)

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

        # img encoder
        img_voxel_feats, query_proposal, depth = self.extract_img_feat(img_inputs, img_metas)

        # mtv transformer
        mtv_lists, mtv_weights, _ = self.mtv_transformer(img_voxel_feats)

        # mtv aggregator
        x_3d, aggregator_weights = self.mtv_aggregator(mtv_lists, mtv_weights, img_voxel_feats)

        # cls head
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        # self.save_tpv(tpv_lists)
        # self.save_logits_map(output['output_voxels'])

        # if hasattr(self, 'teacher'):
        #     with torch.no_grad():
        #         tpv_lists_teacher, output_teacher = self.forward_teacher(data_dict)

        #     self.save_tpv(tpv_lists, tpv_lists_teacher)
        #     self.save_logits_map(output['output_voxels'], output_teacher['output_voxels'])

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {'pred': pred, 'gt_occ': gt_occ}
        return test_output

    def forward_teacher(self, data_dict):
        points = data_dict['points'][0]
        grid_ind = data_dict['grid_ind'][0]
        img_metas = data_dict['img_metas']
        if 'gt_occ' in data_dict:
            gt_occ = data_dict['gt_occ']
        else:
            gt_occ = None

        # lidar encoder
        lidar_voxel_feats = self.teacher.extract_lidar_feat(points=points, grid_ind=grid_ind)

        # mtv transformer
        mtv_lists, mtv_weights, feats_all = self.teacher.mtv_transformer(lidar_voxel_feats)

        # mtv aggregator
        x_3d, aggregator_weights = self.teacher.mtv_aggregator(mtv_lists, mtv_weights, lidar_voxel_feats)

        # cls head
        output = self.teacher.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        return mtv_lists, output, aggregator_weights, feats_all

    def calculate_cosine_similarity(self, x, y):
        assert x.shape == y.shape, "输入特征的形状必须相同"
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B, C, H * W)
        x_flat = x.reshape(B, C, -1)
        y_flat = y.reshape(B, C, -1)

        # normalize
        x_norm = F.normalize(x_flat, p=2, dim=1)
        y_norm = F.normalize(y_flat, p=2, dim=1)

        # bmm
        cosine_similarity_flat = torch.bmm(x_norm.permute(0, 2, 1), y_norm)

        return cosine_similarity_flat

    def distill_loss_logits(self, logits_teacher, logits_student, target, ratio):
        b, c, h, w, z = logits_teacher.shape
        logits_student_softmax = F.log_softmax(logits_student, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        logits_teacher_softmax = F.softmax(logits_teacher, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        target = target.reshape(b, h * w * z)

        loss = 0
        for i in range(target.shape[0]):
            valid = (target[i] != 255)
            nonezero = (target[i] != 0)
            mask = valid * nonezero
            logits_student_i = logits_student_softmax[i][mask]
            logits_teacher_i = logits_teacher_softmax[i][mask]
            loss += nn.KLDivLoss(reduction="mean")(logits_student_i.unsqueeze(0), logits_teacher_i.unsqueeze(0))
        loss = loss / float(target.size(0)) * ratio
        return dict(loss_distill_logits=loss)

    def distill_loss_feature(self, feats_teacher, feats_student, target, ratio_numeric, ration_relation):
        feats_teacher_list = []
        feats_student_list = []
        mask_list = []

        target = target.to(torch.float32)
        target[target == 255] = 0

        # feats 3d
        mask = (target != 0).unsqueeze(1).expand_as(feats_student['feats3d'])
        feats_student_list.append(feats_student['feats3d'])
        feats_teacher_list.append(feats_teacher['feats3d'])
        mask_list.append(mask)

        # feats 2d
        target_xy_mean = target.mean(dim=3)
        mask_xy = target_xy_mean != 0
        size_xy = (self.grid_size[0], self.grid_size[1])
        target_yz_mean = target.mean(dim=1)
        mask_yz = target_yz_mean != 0
        size_yz = (self.grid_size[1], self.grid_size[2])
        target_zx_mean = target.mean(dim=2)
        mask_zx = target_zx_mean != 0
        size_zx = (self.grid_size[0], self.grid_size[2])

        # mtv
        feats_backbone_teacher = feats_teacher['mtv_backbone']
        feats_backbone_student = feats_student['mtv_backbone']
        feats_neck_teacher = feats_teacher['mtv_neck']
        feats_neck_student = feats_student['mtv_neck']

        # xy plane
        for i in range(self.num_views[0]):
            for j in range(len(feats_backbone_teacher[i])):
                feat_student = feats_backbone_student[i][j]
                feat_teacher = feats_backbone_teacher[i][j]
                feat_student = F.interpolate(feat_student, size=size_xy, mode='bilinear', align_corners=False)
                feat_teacher = F.interpolate(feat_teacher, size=size_xy, mode='bilinear', align_corners=False)
                mask = mask_xy.unsqueeze(1).expand_as(feat_student)

                feats_student_list.append(feat_student)
                feats_teacher_list.append(feat_teacher)
                mask_list.append(mask)

            for j in range(len(feats_neck_teacher[i])):
                feat_student = feats_neck_student[i][j]
                feat_teacher = feats_neck_teacher[i][j]
                feat_student = F.interpolate(feat_student, size=size_xy, mode='bilinear', align_corners=False)
                feat_teacher = F.interpolate(feat_teacher, size=size_xy, mode='bilinear', align_corners=False)
                mask = mask_xy.unsqueeze(1).expand_as(feat_student)

                feats_student_list.append(feat_student)
                feats_teacher_list.append(feat_teacher)
                mask_list.append(mask)

        # yz plane
        for i in range(self.num_views[0], self.num_views[0] + self.num_views[1]):
            for j in range(len(feats_backbone_teacher[i])):
                feat_student = feats_backbone_student[i][j]
                feat_teacher = feats_backbone_teacher[i][j]
                feat_student = F.interpolate(feat_student, size=size_yz, mode='bilinear', align_corners=False)
                feat_teacher = F.interpolate(feat_teacher, size=size_yz, mode='bilinear', align_corners=False)
                mask = mask_yz.unsqueeze(1).expand_as(feat_student)

                feats_student_list.append(feat_student)
                feats_teacher_list.append(feat_teacher)
                mask_list.append(mask)

            for j in range(len(feats_neck_teacher[i])):
                feat_student = feats_neck_student[i][j]
                feat_teacher = feats_neck_teacher[i][j]
                feat_student = F.interpolate(feat_student, size=size_yz, mode='bilinear', align_corners=False)
                feat_teacher = F.interpolate(feat_teacher, size=size_yz, mode='bilinear', align_corners=False)
                mask = mask_yz.unsqueeze(1).expand_as(feat_student)

                feats_student_list.append(feat_student)
                feats_teacher_list.append(feat_teacher)
                mask_list.append(mask)

        # zx plane
        for i in range(self.num_views[0] + self.num_views[1], self.num_views[0] + self.num_views[1] + self.num_views[2]):
            for j in range(len(feats_backbone_teacher[i])):
                feat_student = feats_backbone_student[i][j]
                feat_teacher = feats_backbone_teacher[i][j]
                feat_student = F.interpolate(feat_student, size=size_zx, mode='bilinear', align_corners=False)
                feat_teacher = F.interpolate(feat_teacher, size=size_zx, mode='bilinear', align_corners=False)
                mask = mask_zx.unsqueeze(1).expand_as(feat_student)

                feats_student_list.append(feat_student)
                feats_teacher_list.append(feat_teacher)
                mask_list.append(mask)

            for j in range(len(feats_neck_teacher[i])):
                feat_student = feats_neck_student[i][j]
                feat_teacher = feats_neck_teacher[i][j]
                feat_student = F.interpolate(feat_student, size=size_zx, mode='bilinear', align_corners=False)
                feat_teacher = F.interpolate(feat_teacher, size=size_zx, mode='bilinear', align_corners=False)
                mask = mask_zx.unsqueeze(1).expand_as(feat_student)

                feats_student_list.append(feat_student)
                feats_teacher_list.append(feat_teacher)
                mask_list.append(mask)

        losses_feature = {}

        # numeric loss
        if ratio_numeric > 0:
            loss_numeric = 0
            for i in range(len(mask_list)):
                mask = mask_list[i]
                feat_student = feats_student_list[i][mask]
                feat_teacher = feats_teacher_list[i][mask]
                loss = (F.l1_loss(feat_student, feat_teacher) + F.mse_loss(feat_student, feat_teacher)) / 2
                loss_numeric += loss
            loss_numeric = loss_numeric / len(mask_list) * ratio_numeric
            losses_feature.update(dict(loss_distill_feature_numeric=loss_numeric))

        # relation loss
        if ration_relation > 0:
            # skip 3d feature
            for i in range(1, len(feats_student_list)):
                feat_student = feats_student_list[i]
                feat_teacher = feats_teacher_list[i]
                cos_sim_student = self.calculate_cosine_similarity(feat_student, feat_student)
                cos_sim_teacher = self.calculate_cosine_similarity(feat_teacher, feat_teacher)
                loss_relation = F.l1_loss(cos_sim_student, cos_sim_teacher)
                loss_relation += loss_relation
            loss_relation = loss_relation / len(feats_student_list) * ration_relation
            losses_feature.update(dict(loss_distill_feature_relation=loss_relation))

        return losses_feature, feats_student_list, feats_teacher_list, mask_list

    def distill_loss_tpv_weights(self, weights_teacher, weights_student, target, ratio):
        b, c, h, w, z = weights_teacher.shape
        weights_student_softmax = F.log_softmax(weights_student, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        weights_teacher_softmax = F.softmax(weights_teacher, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        target = target.reshape(b, h * w * z)

        loss = 0
        for i in range(target.shape[0]):
            valid = (target[i] != 255)
            nonezero = (target[i] != 0)
            mask = valid * nonezero
            weights_student_i = weights_student_softmax[i][mask]
            weights_teacher_i = weights_teacher_softmax[i][mask]
            loss += nn.KLDivLoss(reduction="mean")(weights_student_i.unsqueeze(0), weights_teacher_i.unsqueeze(0))
        loss = loss / float(target.size(0)) * ratio
        return dict(loss_distill_weights=loss)


@DETECTORS.register_module()
class CameraSegmentor(BaseModule):

    def __init__(
        self,
        img_backbone,
        img_neck,
        img_view_transformer,
        depth_net=None,
        proposal_layer=None,
        VoxFormer_head=None,
        voxel_backbone=None,
        voxel_neck=None,
        tpv_generator=None,
        tpv_aggregator=None,
        pts_bbox_head=None,
        init_cfg=None,
        teacher=None,
        teacher_ckpt=None,
        ratio_logit_kl=10.0,
        ratio_feats_numeric=2,
        ratio_feats_relation=10,
        grid_size=[128, 128, 16],
        **kwargs,
    ):

        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)
        self.depth_net = builder.build_neck(depth_net)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.proposal_layer = builder.build_head(proposal_layer)
        self.VoxFormer_head = builder.build_head(VoxFormer_head)

        if voxel_backbone is not None:
            assert voxel_neck is not None
            self.voxel_backbone = builder.build_backbone(voxel_backbone)
            self.voxel_neck = builder.build_neck(voxel_neck)

        self.tpv_generator = builder.build_backbone(tpv_generator)
        self.tpv_aggregator = builder.build_backbone(tpv_aggregator)
        self.pts_bbox_head = builder.build_head(pts_bbox_head)

        self.grid_size = grid_size
        # init before teacher to avoid overwriting teacher weights
        self.init_cfg = init_cfg
        self.init_weights()

        if teacher:
            self.teacher = builder.build_detector(teacher).eval()
            if os.path.exists(teacher_ckpt):
                ckpt = torch.load(teacher_ckpt)['state_dict']
                adjusted_ckpt = {key.replace('model.', ''): value for key, value in ckpt.items()}
                self.teacher.load_state_dict(adjusted_ckpt)
                print(f"Load teacher model from {teacher_ckpt}")
            self.freeze_model(self.teacher)
            self.ratio_logit_kl = ratio_logit_kl
            self.ratio_feats_numeric = ratio_feats_numeric
            self.ratio_feats_relation = ratio_feats_relation

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

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

        lss_volume = self.img_view_transformer(context, depth, view_trans_inputs)

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

        if self.voxel_backbone is not None:
            x = self.voxel_backbone(x)
            x = self.voxel_neck(x)[0]

        return x, query_proposal, depth

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)

    def forward_train(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']
        gt_occ_1_2 = img_metas['gt_occ_1_2']

        # img encoder
        img_voxel_feats, query_proposal, depth = self.extract_img_feat(img_inputs, img_metas)

        # tpv generator
        tpv_list, feats_all = self.tpv_generator(img_voxel_feats)

        # tpv aggregator
        x_3d, aggregator_weights = self.tpv_aggregator(tpv_list, img_voxel_feats)

        # cls head
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        # loss
        losses = dict()
        losses['loss_depth'] = self.depth_net.get_depth_loss(img_inputs[-4][:, 0:1, ...], depth)
        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output['output_voxels'],
            target_voxels=gt_occ,
        )
        losses.update(losses_occupancy)

        # distillation
        losses_distill = {}
        if hasattr(self, 'teacher'):
            with torch.no_grad():
                tpv_list_teacher, output_teacher, aggregator_weights_teacher, feats_all_teacher = self.forward_teacher(data_dict)

            if self.ratio_logit_kl > 0:
                losses_distill_logit = self.distill_loss_logits(
                    logits_teacher=output_teacher['output_voxels'],
                    logits_student=output['output_voxels'],
                    target=gt_occ,
                    ratio=self.ratio_logit_kl,
                )
                losses_distill.update(losses_distill_logit)

            if self.ratio_feats_numeric > 0 or self.ratio_feats_relation > 0:
                losses_distill_feature, feats_student, feats_teacher, masks = self.distill_loss_feature(
                    feats_teacher=feats_all_teacher,
                    feats_student=feats_all,
                    target=gt_occ_1_2,
                    ratio_numeric=self.ratio_feats_numeric,
                    ration_relation=self.ratio_feats_relation,
                )
                losses_distill.update(losses_distill_feature)

            save_mtv(tpv_list, tpv_list_teacher)
            save_all_feats(feats_student, feats_teacher, masks)
            save_weights(aggregator_weights, aggregator_weights_teacher)

            # if self.ratio_tpv_weights > 0:
            #     losses_distill_tpv_weights = self.distill_loss_tpv_weights(weights_teacher, weights, gt_occ_1_2,
            #                                                                self.ratio_tpv_weights)
            #     losses_distill.update(losses_distill_tpv_weights)

            losses.update(losses_distill)

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

        # img encoder
        img_voxel_feats, query_proposal, depth = self.extract_img_feat(img_inputs, img_metas)

        # mtv transformer
        tpv_lists, _ = self.tpv_generator(img_voxel_feats)

        # mtv aggregator
        x_3d, aggregator_weights = self.tpv_aggregator(tpv_lists, img_voxel_feats)

        # cls head
        output = self.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        # self.save_tpv(tpv_lists)
        # self.save_logits_map(output['output_voxels'])

        # if hasattr(self, 'teacher'):
        #     with torch.no_grad():
        #         tpv_lists_teacher, output_teacher = self.forward_teacher(data_dict)

        #     self.save_tpv(tpv_lists, tpv_lists_teacher)
        #     self.save_logits_map(output['output_voxels'], output_teacher['output_voxels'])

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {'pred': pred, 'gt_occ': gt_occ}
        return test_output

    def forward_teacher(self, data_dict):
        points = data_dict['points'][0]
        grid_ind = data_dict['grid_ind'][0]
        img_metas = data_dict['img_metas']
        if 'gt_occ' in data_dict:
            gt_occ = data_dict['gt_occ']
        else:
            gt_occ = None

        # lidar encoder
        lidar_voxel_feats = self.teacher.extract_lidar_feat(points=points, grid_ind=grid_ind)

        # mtv transformer
        mtv_lists, feats_all = self.teacher.tpv_generator(lidar_voxel_feats)

        # mtv aggregator
        x_3d, aggregator_weights = self.teacher.tpv_aggregator(mtv_lists, lidar_voxel_feats)

        # cls head
        output = self.teacher.pts_bbox_head(voxel_feats=x_3d, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        return mtv_lists, output, aggregator_weights, feats_all

    def calculate_cosine_similarity(self, x, y):
        assert x.shape == y.shape, "输入特征的形状必须相同"
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B, C, H * W)
        x_flat = x.reshape(B, C, -1)
        y_flat = y.reshape(B, C, -1)

        # normalize
        x_norm = F.normalize(x_flat, p=2, dim=1)
        y_norm = F.normalize(y_flat, p=2, dim=1)

        # bmm
        cosine_similarity_flat = torch.bmm(x_norm.permute(0, 2, 1), y_norm)

        return cosine_similarity_flat

    def distill_loss_logits(self, logits_teacher, logits_student, target, ratio):
        b, c, h, w, z = logits_teacher.shape
        logits_student_softmax = F.log_softmax(logits_student, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        logits_teacher_softmax = F.softmax(logits_teacher, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        target = target.reshape(b, h * w * z)

        loss = 0
        for i in range(target.shape[0]):
            valid = (target[i] != 255)
            nonezero = (target[i] != 0)
            mask = valid * nonezero
            logits_student_i = logits_student_softmax[i][mask]
            logits_teacher_i = logits_teacher_softmax[i][mask]
            loss += nn.KLDivLoss(reduction="mean")(logits_student_i.unsqueeze(0), logits_teacher_i.unsqueeze(0))
        loss = loss / float(target.size(0)) * ratio
        return dict(loss_distill_logits=loss)

    def distill_loss_feature(self, feats_teacher, feats_student, target, ratio_numeric, ration_relation):
        feats_teacher_list = []
        feats_student_list = []
        mask_list = []

        target = target.to(torch.float32)
        target[target == 255] = 0

        # feats 3d
        mask = (target != 0).unsqueeze(1).expand_as(feats_student['feats3d'])
        feats_student_list.append(feats_student['feats3d'])
        feats_teacher_list.append(feats_teacher['feats3d'])
        mask_list.append(mask)

        # feats 2d
        target_xy_mean = target.mean(dim=3)
        mask_xy = target_xy_mean != 0
        size_xy = (self.grid_size[0], self.grid_size[1])
        target_yz_mean = target.mean(dim=1)
        mask_yz = target_yz_mean != 0
        size_yz = (self.grid_size[1], self.grid_size[2])
        target_zx_mean = target.mean(dim=2)
        mask_zx = target_zx_mean != 0
        size_zx = (self.grid_size[0], self.grid_size[2])

        # mtv
        feats_backbone_teacher = feats_teacher['mtv_backbone']
        feats_backbone_student = feats_student['mtv_backbone']
        feats_neck_teacher = feats_teacher['mtv_neck']
        feats_neck_student = feats_student['mtv_neck']

        # xy plane
        for i in range(1):
            for j in range(len(feats_backbone_teacher[i])):
                feat_student = feats_backbone_student[i][j]
                feat_teacher = feats_backbone_teacher[i][j]
                feat_student = F.interpolate(feat_student, size=size_xy, mode='bilinear', align_corners=False)
                feat_teacher = F.interpolate(feat_teacher, size=size_xy, mode='bilinear', align_corners=False)
                mask = mask_xy.unsqueeze(1).expand_as(feat_student)

                feats_student_list.append(feat_student)
                feats_teacher_list.append(feat_teacher)
                mask_list.append(mask)

            for j in range(len(feats_neck_teacher[i])):
                feat_student = feats_neck_student[i][j]
                feat_teacher = feats_neck_teacher[i][j]
                feat_student = F.interpolate(feat_student, size=size_xy, mode='bilinear', align_corners=False)
                feat_teacher = F.interpolate(feat_teacher, size=size_xy, mode='bilinear', align_corners=False)
                mask = mask_xy.unsqueeze(1).expand_as(feat_student)

                feats_student_list.append(feat_student)
                feats_teacher_list.append(feat_teacher)
                mask_list.append(mask)

        # yz plane
        for i in range(1, 2):
            for j in range(len(feats_backbone_teacher[i])):
                feat_student = feats_backbone_student[i][j]
                feat_teacher = feats_backbone_teacher[i][j]
                feat_student = F.interpolate(feat_student, size=size_yz, mode='bilinear', align_corners=False)
                feat_teacher = F.interpolate(feat_teacher, size=size_yz, mode='bilinear', align_corners=False)
                mask = mask_yz.unsqueeze(1).expand_as(feat_student)

                feats_student_list.append(feat_student)
                feats_teacher_list.append(feat_teacher)
                mask_list.append(mask)

            for j in range(len(feats_neck_teacher[i])):
                feat_student = feats_neck_student[i][j]
                feat_teacher = feats_neck_teacher[i][j]
                feat_student = F.interpolate(feat_student, size=size_yz, mode='bilinear', align_corners=False)
                feat_teacher = F.interpolate(feat_teacher, size=size_yz, mode='bilinear', align_corners=False)
                mask = mask_yz.unsqueeze(1).expand_as(feat_student)

                feats_student_list.append(feat_student)
                feats_teacher_list.append(feat_teacher)
                mask_list.append(mask)

        # zx plane
        for i in range(2, 3):
            for j in range(len(feats_backbone_teacher[i])):
                feat_student = feats_backbone_student[i][j]
                feat_teacher = feats_backbone_teacher[i][j]
                feat_student = F.interpolate(feat_student, size=size_zx, mode='bilinear', align_corners=False)
                feat_teacher = F.interpolate(feat_teacher, size=size_zx, mode='bilinear', align_corners=False)
                mask = mask_zx.unsqueeze(1).expand_as(feat_student)

                feats_student_list.append(feat_student)
                feats_teacher_list.append(feat_teacher)
                mask_list.append(mask)

            for j in range(len(feats_neck_teacher[i])):
                feat_student = feats_neck_student[i][j]
                feat_teacher = feats_neck_teacher[i][j]
                feat_student = F.interpolate(feat_student, size=size_zx, mode='bilinear', align_corners=False)
                feat_teacher = F.interpolate(feat_teacher, size=size_zx, mode='bilinear', align_corners=False)
                mask = mask_zx.unsqueeze(1).expand_as(feat_student)

                feats_student_list.append(feat_student)
                feats_teacher_list.append(feat_teacher)
                mask_list.append(mask)

        losses_feature = {}

        # numeric loss
        if ratio_numeric > 0:
            loss_numeric = 0
            for i in range(len(mask_list)):
                mask = mask_list[i]
                feat_student = feats_student_list[i][mask]
                feat_teacher = feats_teacher_list[i][mask]
                loss = (F.l1_loss(feat_student, feat_teacher) + F.mse_loss(feat_student, feat_teacher)) / 2
                loss_numeric += loss
            loss_numeric = loss_numeric / len(mask_list) * ratio_numeric
            losses_feature.update(dict(loss_distill_feature_numeric=loss_numeric))

        # relation loss
        if ration_relation > 0:
            # skip 3d feature
            for i in range(1, len(feats_student_list)):
                feat_student = feats_student_list[i]
                feat_teacher = feats_teacher_list[i]
                cos_sim_student = self.calculate_cosine_similarity(feat_student, feat_student)
                cos_sim_teacher = self.calculate_cosine_similarity(feat_teacher, feat_teacher)
                loss_relation = F.l1_loss(cos_sim_student, cos_sim_teacher)
                loss_relation += loss_relation
            loss_relation = loss_relation / len(feats_student_list) * ration_relation
            losses_feature.update(dict(loss_distill_feature_relation=loss_relation))

        return losses_feature, feats_student_list, feats_teacher_list, mask_list

    def distill_loss_tpv_weights(self, weights_teacher, weights_student, target, ratio):
        b, c, h, w, z = weights_teacher.shape
        weights_student_softmax = F.log_softmax(weights_student, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        weights_teacher_softmax = F.softmax(weights_teacher, dim=1).permute(0, 2, 3, 4, 1).reshape(b, h * w * z, c)
        target = target.reshape(b, h * w * z)

        loss = 0
        for i in range(target.shape[0]):
            valid = (target[i] != 255)
            nonezero = (target[i] != 0)
            mask = valid * nonezero
            weights_student_i = weights_student_softmax[i][mask]
            weights_teacher_i = weights_teacher_softmax[i][mask]
            loss += nn.KLDivLoss(reduction="mean")(weights_student_i.unsqueeze(0), weights_teacher_i.unsqueeze(0))
        loss = loss / float(target.size(0)) * ratio
        return dict(loss_distill_weights=loss)
