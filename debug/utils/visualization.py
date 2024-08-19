import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import pdb


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
