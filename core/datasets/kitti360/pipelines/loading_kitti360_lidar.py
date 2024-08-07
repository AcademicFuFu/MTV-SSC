import pdb
import mmcv
import torch
import numpy as np
from mmdet.datasets.builder import PIPELINES
from debug.utils import print_detail as pd, mem, save_feature_map_as_image


def cart2polar(input_xyz):
    rho = torch.sqrt(input_xyz[..., 0]**2 + input_xyz[..., 1]**2)
    phi = torch.atan2(input_xyz[..., 1], input_xyz[..., 0])
    return torch.stack((rho, phi, input_xyz[..., 2]), axis=-1)


@PIPELINES.register_module()
class LoadLidarPointsFromFiles_KITTI360(object):

    def __init__(self, data_config, is_train=False):
        super().__init__()

        self.is_train = is_train
        self.data_config = data_config

    def get_inputs(self, results):
        lidar_filenames = results['lidar_filename']
        data_lists = []

        for i in range(len(lidar_filenames)):
            lidar_filename = lidar_filenames[i]
            lidar_points = torch.tensor(np.fromfile(lidar_filename, dtype=np.float32).reshape(-1, 4))

            result = [lidar_points]
            result = [x[None] for x in result]

            data_lists.append(result)

        num = len(data_lists[0])
        result_list = []
        for i in range(num):
            result_list.append(torch.cat([x[i] for x in data_lists], dim=0))

        return result_list

    def __call__(self, results):
        results['lidar_inputs'] = self.get_inputs(results)

        return results


@PIPELINES.register_module()
class LidarPointsPreProcess_KITTI360(object):

    def __init__(
        self,
        data_config,
        point_cloud_range,
        grid_size,
        grid_size_vox,
        grid_size_occ,
        coarse_ratio=2,
        is_train=False,
    ):
        super().__init__()

        self.is_train = is_train
        self.point_cloud_range = torch.tensor(point_cloud_range)
        self.data_config = data_config

        self.grid_size = torch.tensor(grid_size, dtype=torch.int32)  # 256, 256, 32
        self.grid_size_vox = torch.tensor(grid_size_vox, dtype=torch.int32)  # 256, 256, 32
        self.grid_size_occ = torch.tensor(grid_size_occ, dtype=torch.int32)  # 256, 256, 32
        self.grid_size_occ_coarse = (self.grid_size_occ / coarse_ratio).to(torch.int32)  # 128, 128, 16

        self.min_volume_space = [point_cloud_range[0], -np.pi / 2, point_cloud_range[2]]
        self.max_volume_space = [point_cloud_range[3], np.pi / 2, point_cloud_range[5]]
        self.max_bound = torch.tensor(self.max_volume_space)
        self.min_bound = torch.tensor(self.min_volume_space)
        crop_range = self.max_bound - self.min_bound
        self.intervals = crop_range / (self.grid_size)
        intervals_vox = crop_range / (self.grid_size_vox)
        self.voxel_size = (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.grid_size_occ
        self.voxel_size_coarse = (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.grid_size_occ_coarse

        # get voxel_position_grid_coarse
        dim_array = np.ones(len(self.grid_size_occ_coarse) + 1, np.int32)
        dim_array[0] = -1
        voxel_position_coarse = torch.tensor((np.indices(self.grid_size_occ_coarse) + 0.5) *
                                             self.voxel_size_coarse.numpy().reshape(dim_array) +
                                             self.point_cloud_range[:3].numpy().reshape(dim_array)).reshape(3, -1).transpose(
                                                 1, 0).to(torch.float32)
        self.voxel_position_grid_coarse = (torch.clip(cart2polar(voxel_position_coarse), self.min_bound, self.max_bound - 1e-3) -
                                           self.min_bound) / intervals_vox

    def get_inputs(self, results):
        lidar_points = results['lidar_inputs']
        point_lists = []
        voxel_pos_lists = []
        grid_ind_lists = []

        for i in range(len(lidar_points)):
            points = lidar_points[i]
            xyz, feat = points[..., :3], points[..., 3:]
            xyz_pol = cart2polar(xyz)
            mask_x = (xyz_pol[:, :, 0] > self.min_volume_space[0]) & (xyz_pol[:, :, 0] < self.max_volume_space[0])
            mask_y = (xyz_pol[:, :, 1] > self.min_volume_space[1]) & (xyz_pol[:, :, 1] < self.max_volume_space[1])
            mask_z = (xyz_pol[:, :, 2] > self.min_volume_space[2]) & (xyz_pol[:, :, 2] < self.max_volume_space[2])
            mask = mask_x & mask_y & mask_z
            xyz = xyz[:, mask[0], :]
            feat = feat[:, mask[0], :]
            xyz_pol = xyz_pol[:, mask[0], :]
            grid_ind = torch.floor((xyz_pol - self.min_bound) / self.intervals).to(torch.int32)
            voxel_centers = (grid_ind.to(torch.float32) + 0.5) * self.intervals + self.min_bound
            return_xyz = xyz_pol - voxel_centers
            return_feat = torch.cat((return_xyz, xyz_pol, xyz[..., :2], feat), dim=-1)
            # pd(points, 'points')
            # pd(mask, 'mask')
            # pd(xyz, 'xyz')
            # pd(feat, 'feat')
            # pd(xyz_pol, 'xyz_pol')
            # pd(grid_ind, 'grid_ind')
            # pd(return_xyz, 'return_xyz')
            # pd(return_feat, 'return_feat')
            point_lists.append(return_feat)
            voxel_pos_lists.append(self.voxel_position_grid_coarse)
            grid_ind_lists.append(grid_ind)

        return point_lists, voxel_pos_lists, grid_ind_lists

    def __call__(self, results):
        results['points'], results['voxel_position_grid_coarse'], results['grid_ind'] = self.get_inputs(results)

        return results
