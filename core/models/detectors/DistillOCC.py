import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS, HEADS
from mmdet3d.models import builder
from mmcv.runner import force_fp32
import os
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import pdb
from debug.utils import print_detail as pd, mem, save_feature_map_as_image, save_denormalized_images


@DETECTORS.register_module()
class DistillOccV0(BaseModule):

    def __init__(
        self,
        teacher,
        teacher_ckpt,
        student,
        ratio_logit=10.0,
        ratio_tpv_feats=2,
        ratio_tpv_relation=10,
        init_cfg=None,
        **kwargs,
    ):

        super().__init__()

        self.teacher = builder.build_detector(teacher).eval()
        self.student = builder.build_detector(student)

        self.ratio_logit = ratio_logit
        self.ratio_tpv_feats = ratio_tpv_feats
        self.ratio_tpv_relation = ratio_tpv_relation

        if os.path.exists(teacher_ckpt):
            ckpt = torch.load(teacher_ckpt)['state_dict']
            adjusted_ckpt = {key.replace('model.', ''): value for key, value in ckpt.items()}
            self.teacher.load_state_dict(adjusted_ckpt)
            print(f"Load teacher model from {teacher_ckpt}")

        # self.freeze_model(self.teacher)
        self.init_cfg = init_cfg
        self.init_weights()

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

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

    def distill_loss_tpv_feature(self, tpv_teacher, tpv_student, target, ratio):
        loss = 0
        for i in range(target.shape[0]):
            # target_i = target[i].to(torch.float32)
            tpv_xy_teacher_i = tpv_teacher[0][i]
            tpv_xy_student_i = tpv_student[0][i]
            cos_sim = cosine_similarity(tpv_xy_student_i, tpv_xy_teacher_i, dim=0)
            loss_xy = 1 - cos_sim.mean()

            tpv_yz_teacher_i = tpv_teacher[1][i]
            tpv_yz_student_i = tpv_student[1][i]
            cos_sim = cosine_similarity(tpv_yz_student_i, tpv_yz_teacher_i, dim=0)
            loss_yz = 1 - cos_sim.mean()

            tpv_zx_teacher_i = tpv_teacher[2][i]
            tpv_zx_student_i = tpv_student[2][i]
            cos_sim = cosine_similarity(tpv_zx_student_i, tpv_zx_teacher_i, dim=0)
            loss_zx = 1 - cos_sim.mean()

            loss += loss_xy + loss_yz + loss_zx
        loss = loss * ratio / (3 * target.shape[0])
        return dict(loss_distill_tpv=loss)

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

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)

    def forward_train(self, data_dict):
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']
        img_inputs = data_dict['img_inputs']

        # lidar branch
        with torch.no_grad():
            points = data_dict['points'][0]
            grid_ind = data_dict['grid_ind'][0]
            voxel_pos_grid_coarse = data_dict['voxel_position_grid_coarse'][0]
            x_lidar_tpv = self.teacher.extract_lidar_feat(points=points, grid_ind=grid_ind)
            tpv_lists_lidar = self.teacher.tpv_transformer(x_lidar_tpv, voxel_pos_grid_coarse)
            x_3d_lidar = self.teacher.tpv_aggregator(tpv_lists_lidar)
            output_lidar = self.teacher.pts_bbox_head(voxel_feats=x_3d_lidar, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        # camera branch
        img_voxel_feats, query_proposal, depth = self.student.extract_img_feat(img_inputs, img_metas)
        tpv_lists_cam = self.student.tpv_transformer(img_voxel_feats)
        x_3d_cam = self.student.tpv_aggregator(tpv_lists_cam)
        output_cam = self.student.pts_bbox_head(voxel_feats=x_3d_cam, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        losses = dict()
        if depth is not None:
            losses['loss_depth'] = self.student.depth_net.get_depth_loss(img_inputs[-4][:, 0:1, ...], depth)
        losses_occupancy = self.student.pts_bbox_head.loss(
            output_voxels=output_cam['output_voxels'],
            target_voxels=gt_occ,
        )
        losses.update(losses_occupancy)

        # losses_distill_logit = self.distill_loss_logits(output_lidar['output_voxels'], output_cam['output_voxels'], gt_occ,
        #                                                 self.ratio_logit)
        # losses.update(losses_distill_logit)

        # losses_distill_tpv_feature = self.distill_loss_tpv_feature(tpv_lists_lidar, tpv_lists_cam, gt_occ, self.ratio_tpv_feats)
        # losses.update(losses_distill_tpv_feature)

        # losses_distill_tpv_relation = self.distill_loss_tpv_relation(tpv_lists_lidar, tpv_lists_cam, gt_occ,
        #                                                              self.ratio_tpv_relation)
        # losses.update(losses_distill_tpv_relation)

        pred = output_cam['output_voxels']
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

        img_voxel_feats, query_proposal, depth = self.student.extract_img_feat(img_inputs, img_metas)
        tpv_lists_cam = self.student.tpv_transformer(img_voxel_feats)
        x_3d_cam = self.student.tpv_aggregator(tpv_lists_cam)
        output_cam = self.student.pts_bbox_head(voxel_feats=x_3d_cam, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)

        pred = output_cam['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {'pred': pred, 'gt_occ': gt_occ}
        return test_output
