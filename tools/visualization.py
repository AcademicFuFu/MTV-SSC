import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


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


def save_tpv(tpv_list, save_folder, frame_id):

    # format to b,n,c,h,w
    feat_xy = tpv_list[0].squeeze(-1).unsqueeze(1).permute(0, 1, 2, 3, 4)
    save_feature_map_as_image(feat_xy.detach(), os.path.join(save_folder, 'xy'), frame_id, method='pca')

    feat_yz = torch.flip(tpv_list[1].squeeze(-3).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
    save_feature_map_as_image(feat_yz.detach(), os.path.join(save_folder, 'yz'), frame_id, method='pca')

    feat_zx = torch.flip(tpv_list[2].squeeze(-2).unsqueeze(1).permute(0, 1, 2, 4, 3), dims=[-1])
    save_feature_map_as_image(feat_zx.detach(), os.path.join(save_folder, 'zx'), frame_id, method='pca')

    return
