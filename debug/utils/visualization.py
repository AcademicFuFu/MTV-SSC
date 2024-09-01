import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import pdb

from debug.utils import print_detail as pd, mem, count_trainable_parameters as param


def denormalize(image, mean, std):
    """
    Denormalize a normalized image tensor.
    
    Args:
        image (torch.Tensor): Normalized image tensor of shape (C, H, W).
        mean (list or tuple): Mean used for normalization.
        std (list or tuple): Standard deviation used for normalization.
    
    Returns:
        torch.Tensor: Denormalized image tensor.
    """
    if isinstance(mean, (list, tuple)):
        mean = torch.tensor(mean).view(-1, 1, 1)
    if isinstance(std, (list, tuple)):
        std = torch.tensor(std).view(-1, 1, 1)

    denormalized_image = image * std + mean
    return denormalized_image


def save_denormalized_images(img_tensor, output_dir, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Save denormalized images from a normalized tensor.
    
    Args:
        img_tensor (torch.Tensor): Normalized image tensor of shape [B, N, C, H, W].
        output_dir (str): Directory to save the images.
        mean (list or tuple): Mean used for normalization.
        std (list or tuple): Standard deviation used for normalization.
    """
    os.makedirs(output_dir, exist_ok=True)

    B, N, C, H, W = img_tensor.shape
    for b in range(B):
        for n in range(N):
            img = img_tensor[b, n]
            denorm_img = denormalize(img.cpu(), mean, std)
            denorm_img_np = denorm_img.cpu().numpy().transpose(1, 2, 0)
            denorm_img_np = np.clip(denorm_img_np, 0, 1)

            img_path = os.path.join(output_dir, f'image_b{b}_n{n}.png')
            plt.imsave(img_path, denorm_img_np)
            print(f'Saved: {img_path}')


def save_feature_map_as_image(feature_map, output_dir, name='map', method='pca', n_components=3):
    """
    Save feature map as an image.

    Args:
        feature_map (torch.Tensor): Feature map tensor of shape [B, N, C, H, W].
        output_dir (str): Directory to save the images.
        method (str): Method for visualization ('single_channel', 'average', 'pca', 'max_activation').
        n_components (int): Number of components for PCA (default is 3 for RGB).
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Normalize feature map to [0, 1]
    def normalize(tensor):
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        return (tensor - tensor_min) / (tensor_max - tensor_min)

    B, N, C, H, W = feature_map.shape
    for b in range(B):
        for n in range(N):
            feature = feature_map[b, n]

            if method == 'single_channel':
                for c in range(C):
                    single_channel = feature[c:c + 1]
                    single_channel = normalize(single_channel)
                    single_channel_np = single_channel.cpu().numpy().transpose(1, 2, 0)
                    single_channel_np = np.flipud(single_channel_np)
                    single_channel_np = np.fliplr(single_channel_np)
                    img_path = os.path.join(output_dir, f'feature_{name}_b{b}_n{n}_c{c}.png')
                    plt.imsave(img_path, single_channel_np.squeeze(), cmap='gray')
                    print(f'Saved: {img_path}')

            elif method == 'average':
                avg_feature = torch.mean(feature, dim=0, keepdim=True)
                avg_feature = normalize(avg_feature)
                avg_feature_np = avg_feature.cpu().numpy().transpose(1, 2, 0)
                avg_feature_np = np.flipud(avg_feature_np)
                avg_feature_np = np.fliplr(avg_feature_np)
                img_path = os.path.join(output_dir, f'feature_{name}_b{b}_n{n}_avg.png')
                plt.imsave(img_path, avg_feature_np.squeeze(), cmap='gray')
                print(f'Saved: {img_path}')

            elif method == 'pca':
                feature_flattened = feature.reshape(C, H * W).cpu().numpy().T
                pca = PCA(n_components=n_components)
                pca_feature = pca.fit_transform(feature_flattened)
                pca_feature = pca_feature.T.reshape(n_components, H, W)
                pca_feature = normalize(torch.tensor(pca_feature))
                pca_feature_np = pca_feature.cpu().numpy().transpose(1, 2, 0)
                pca_feature_np = np.flipud(pca_feature_np)
                pca_feature_np = np.fliplr(pca_feature_np)
                img_path = os.path.join(output_dir, f'feature_{name}_b{b}_n{n}_pca.png')
                plt.imsave(img_path, pca_feature_np)
                print(f'Saved: {img_path}')

            elif method == 'max_activation':
                max_channel = torch.argmax(feature.mean(dim=(1, 2)))
                max_feature = feature[max_channel:max_channel + 1]
                max_feature = normalize(max_feature)
                max_feature_np = max_feature.cpu().numpy().transpose(1, 2, 0)
                max_feature_np = np.flipud(max_feature_np)
                max_feature_np = np.fliplr(max_feature_np)
                img_path = os.path.join(output_dir, f'feature_{name}_b{b}_n{n}_max.png')
                plt.imsave(img_path, max_feature_np.squeeze(), cmap='gray')
                print(f'Saved: {img_path}')


def get_pca_feature_map(feature_map, n_components=3):

    # Normalize feature map to [0, 1]
    def normalize(tensor):
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        return (tensor - tensor_min) / (tensor_max - tensor_min)

    B, C, H, W = feature_map.shape
    assert B == 1, 'Batch size must be 1 for PCA visualization.'
    feature = feature_map[0]
    feature_flattened = feature.reshape(C, H * W).cpu().numpy().T
    pca = PCA(n_components=n_components)
    pca_feature = pca.fit_transform(feature_flattened)
    pca_feature = pca_feature.T.reshape(n_components, H, W)
    pca_feature = normalize(torch.tensor(pca_feature))
    pca_feature_np = pca_feature.cpu().numpy().transpose(1, 2, 0)
    pca_feature_np = np.flipud(pca_feature_np)
    pca_feature_np = np.fliplr(pca_feature_np)

    return pca_feature_np


def save_all_feats(feats_student, feats_teacher, masks):
    os.makedirs('save/distill/feats_all', exist_ok=True)
    for i in range(len(feats_student)):
        feat_student = feats_student[i]
        feat_teacher = feats_teacher[i]
        if len(feat_student.shape) > 4:
            continue

        # mask = masks[i]
        # feat_student = feat_student * mask
        # feat_teacher = feat_teacher * mask

        b, c, h, w = feat_student.shape
        mask = masks[i][0, 0, :, :].unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()
        mask = np.flipud(mask)
        mask = np.fliplr(mask)
        if h > w:
            feat_student = feat_student.permute(0, 1, 3, 2)
            feat_teacher = feat_teacher.permute(0, 1, 3, 2)
            mask = mask.transpose(1, 0, 2)

        pca_student = get_pca_feature_map(feat_student.detach())
        pca_teacher = get_pca_feature_map(feat_teacher.detach())
        pca_student = pca_student * mask
        pca_teacher = pca_teacher * mask

        map = np.zeros((pca_student.shape[0], pca_student.shape[1] * 2, pca_student.shape[2]))
        map[:, :pca_student.shape[1], :] = pca_student
        map[:, pca_student.shape[1]:, :] = pca_teacher
        img_path = 'save/distill/feats_all/{}.png'.format(i)
        plt.imsave(img_path, map)
        print(f'Saved: {img_path}')
    pdb.set_trace()


def save_tpv(tpv_cam, tpv_lidar=None, num_views=[1, 1, 1]):
    tpv_list = tpv_cam

    # format to b,n,c,h,w
    feat_xy = tpv_list[0].squeeze(-1).unsqueeze(1).permute(0, 1, 2, 3, 4)
    save_feature_map_as_image(feat_xy.detach(), 'save/distill/tpv_cam', 'xy'.format(id), method='pca')

    feat_yz = torch.flip(tpv_list[1].squeeze(-3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
    save_feature_map_as_image(feat_yz.detach(), 'save/distill/tpv_cam', 'yz'.format(id), method='pca')

    feat_zx = torch.flip(tpv_list[2].squeeze(-2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
    save_feature_map_as_image(feat_zx.detach(), 'save/distill/tpv_cam', 'zx'.format(id), method='pca')

    if tpv_lidar is not None:
        tpv_list = tpv_lidar
        feat_xy = tpv_list[0].squeeze(-1).unsqueeze(1).permute(0, 1, 2, 3, 4)
        save_feature_map_as_image(feat_xy.detach(), 'save/distill/tpv_lidar', 'xy'.format(id), method='pca')

        feat_yz = torch.flip(tpv_list[1].squeeze(-3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
        save_feature_map_as_image(feat_yz.detach(), 'save/distill/tpv_lidar', 'yz'.format(id), method='pca')

        feat_zx = torch.flip(tpv_list[2].squeeze(-2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
        save_feature_map_as_image(feat_zx.detach(), 'save/distill/tpv_lidar', 'zx'.format(id), method='pca')

    # remind to comment while training
    pdb.set_trace()
    return


def save_mtv(mtv_cam, mtv_lidar=None, num_views=[1, 1, 1]):
    mtv_list = mtv_cam

    id = 0
    for i in range(num_views[0]):
        # format to b,n,c,h,w
        feat_xy = mtv_list[i].squeeze(-1).unsqueeze(1).permute(0, 1, 2, 3, 4)
        save_feature_map_as_image(feat_xy.detach(), 'save/distill/mtv_cam', 'xy_{}'.format(id), method='pca')
        id += 1

    id = 0
    for i in range(num_views[0], num_views[0] + num_views[1]):
        feat_yz = torch.flip(mtv_list[i].squeeze(-3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
        save_feature_map_as_image(feat_yz.detach(), 'save/distill/mtv_cam', 'yz_{}'.format(id), method='pca')
        id += 1

    id = 0
    for i in range(num_views[0] + num_views[1], num_views[0] + num_views[1] + num_views[2]):
        feat_zx = torch.flip(mtv_list[i].squeeze(-2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
        save_feature_map_as_image(feat_zx.detach(), 'save/distill/mtv_cam', 'zx_{}'.format(id), method='pca')
        id += 1

    if mtv_lidar is not None:
        mtv_list = mtv_lidar
        id = 0
        for i in range(num_views[0]):
            # format to b,n,c,h,w
            feat_xy = mtv_list[i].squeeze(-1).unsqueeze(1).permute(0, 1, 2, 3, 4)
            save_feature_map_as_image(feat_xy.detach(), 'save/distill/mtv_lidar', 'xy_{}'.format(id), method='pca')
            id += 1

        id = 0
        for i in range(num_views[0], num_views[0] + num_views[1]):
            feat_yz = torch.flip(mtv_list[i].squeeze(-3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
            save_feature_map_as_image(feat_yz.detach(), 'save/distill/mtv_lidar', 'yz_{}'.format(id), method='pca')
            id += 1

        id = 0
        for i in range(num_views[0] + num_views[1], num_views[0] + num_views[1] + num_views[2]):
            feat_zx = torch.flip(mtv_list[i].squeeze(-2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
            save_feature_map_as_image(feat_zx.detach(), 'save/distill/mtv_lidar', 'zx_{}'.format(id), method='pca')
            id += 1

    # remind to comment while training
    pdb.set_trace()
    return


def save_logits_map(logits_cam, logits_lidar=None):
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


# def save_weights(weights_cam, weights_lidar=None):
#     os.makedirs('save/distill/weights', exist_ok=True)
#     pdb.set_trace()
#     weights_cam = F.softmax(weights_cam, dim=1)
#     if weights_lidar is not None:
#         weights_lidar = F.softmax(weights_lidar, dim=1)

#     weights_cam_xy = weights_cam.mean(dim=4).squeeze(0).permute(1, 2, 0)
#     if weights_lidar is not None:
#         weights_lidar_xy = weights_lidar.mean(dim=4).squeeze(0).permute(1, 2, 0)
#         weights = torch.cat([weights_cam_xy, weights_lidar_xy], dim=1)
#     else:
#         weights = weights_cam_xy
#     weights = np.fliplr(np.flipud(weights.detach().cpu().numpy()))
#     img_path = 'save/distill/weights/weights_xy.png'
#     plt.imsave(img_path, weights)
#     print(f'Saved: {img_path}')

#     weights_cam_yz = torch.flip(weights_cam.mean(dim=2).squeeze(0).permute(2, 1, 0), dims=[-1])
#     if weights_lidar is not None:
#         weights_lidar_yz = torch.flip(weights_lidar.mean(dim=2).squeeze(0).permute(2, 1, 0), dims=[-1])
#         weights = torch.cat([weights_cam_yz, weights_lidar_yz], dim=1)
#     else:
#         weights = weights_cam_yz
#     weights = np.fliplr(np.flipud(weights.detach().cpu().numpy()))
#     img_path = 'save/distill/weights/weights_yz.png'
#     plt.imsave(img_path, weights)
#     print(f'Saved: {img_path}')

#     weights_cam_zx = torch.flip(weights_cam.mean(dim=3).squeeze(0).permute(2, 1, 0), dims=[-1])
#     if weights_lidar is not None:
#         weights_lidar_zx = torch.flip(weights_lidar.mean(dim=3).squeeze(0).permute(2, 1, 0), dims=[-1])
#         weights = torch.cat([weights_cam_zx, weights_lidar_zx], dim=1)
#     else:
#         weights = weights_cam_zx
#     weights = np.fliplr(np.flipud(weights.detach().cpu().numpy()))
#     img_path = 'save/distill/weights/weights_zx.png'
#     plt.imsave(img_path, weights)
#     print(f'Saved: {img_path}')

#     pdb.set_trace()
#     return


def save_weights(weights_cam, weights_lidar=None):
    os.makedirs('save/distill/weights', exist_ok=True)

    weights_cam = F.softmax(weights_cam, dim=1)
    if weights_lidar is not None:
        weights_lidar = F.softmax(weights_lidar, dim=1)

    def get_weights_23d(weights):
        weights_23d = torch.cat(
            [
                weights[:, :, 3:],
                weights[:, :, :3].sum(dim=2, keepdim=True),
                torch.zeros_like(weights[:, :, 3:]),
            ],
            dim=2,
        )
        return weights_23d

    def save_img(weights, img_path):
        weights = np.fliplr(np.flipud(weights.detach().cpu().numpy()))
        plt.imsave(img_path, weights)
        print(f'Saved: {img_path}')

    weights_cam_xy = weights_cam.mean(dim=4).squeeze(0).permute(1, 2, 0)
    weights_cam_xy_tpv = weights_cam_xy[:, :, :3]
    weights_cam_xy_23d = get_weights_23d(weights_cam_xy)
    if weights_lidar is not None:
        weights_lidar_xy = weights_lidar.mean(dim=4).squeeze(0).permute(1, 2, 0)
        weights_lidar_xy_tpv = weights_lidar_xy[:, :, :3]
        weights_lidar_xy_23d = get_weights_23d(weights_lidar_xy)
        weights_tpv = torch.cat([weights_cam_xy_tpv, weights_lidar_xy_tpv], dim=1)
        weights_23d = torch.cat([weights_cam_xy_23d, weights_lidar_xy_23d], dim=1)
    else:
        weights_tpv = weights_cam_xy_tpv
        weights_23d = weights_cam_xy_23d

    save_img(weights_tpv, 'save/distill/weights/weights_xy_tpv.png')
    save_img(weights_23d, 'save/distill/weights/weights_xy_23d.png')

    weights_cam_yz = torch.flip(weights_cam.mean(dim=2).squeeze(0).permute(2, 1, 0), dims=[-1])
    weights_cam_yz_tpv = weights_cam_yz[:, :, :3]
    weights_cam_yz_23d = get_weights_23d(weights_cam_yz)
    if weights_lidar is not None:
        weights_lidar_yz = torch.flip(weights_lidar.mean(dim=2).squeeze(0).permute(2, 1, 0), dims=[-1])
        weights_lidar_yz_tpv = weights_lidar_yz[:, :, :3]
        weights_lidar_yz_23d = get_weights_23d(weights_lidar_yz)
        weights_tpv = torch.cat([weights_cam_yz_tpv, weights_lidar_yz_tpv], dim=1)
        weights_23d = torch.cat([weights_cam_yz_23d, weights_lidar_yz_23d], dim=1)
    else:
        weights_tpv = weights_cam_yz_tpv
        weights_23d = weights_cam_yz_23d
    save_img(weights_tpv, 'save/distill/weights/weights_yz_tpv.png')
    save_img(weights_23d, 'save/distill/weights/weights_yz_23d.png')

    weights_cam_zx = torch.flip(weights_cam.mean(dim=3).squeeze(0).permute(2, 1, 0), dims=[-1])
    weights_cam_zx_tpv = weights_cam_zx[:, :, :3]
    weights_cam_zx_23d = get_weights_23d(weights_cam_zx)
    if weights_lidar is not None:
        weights_lidar_zx = torch.flip(weights_lidar.mean(dim=3).squeeze(0).permute(2, 1, 0), dims=[-1])
        weights_lidar_zx_tpv = weights_lidar_zx[:, :, :3]
        weights_lidar_zx_23d = get_weights_23d(weights_lidar_zx)
        weights_tpv = torch.cat([weights_cam_zx_tpv, weights_lidar_zx_tpv], dim=1)
        weights_23d = torch.cat([weights_cam_zx_23d, weights_lidar_zx_23d], dim=1)
    else:
        weights_tpv = weights_cam_zx_tpv
        weights_23d = weights_cam_zx_23d
    save_img(weights_tpv, 'save/distill/weights/weights_zx_tpv.png')
    save_img(weights_23d, 'save/distill/weights/weights_zx_23d.png')

    pdb.set_trace()
    return
