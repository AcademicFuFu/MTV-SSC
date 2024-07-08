import os
import torch
import importlib
import numpy as np
import platform

class_names = [
    "empty",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]


def init_plugin(cfg):
    plugin_dir = cfg.plugin_dir
    _module_dir = os.path.dirname(plugin_dir)
    _module_dir = _module_dir.split('/')
    _module_path = _module_dir[0]
    for m in _module_dir[1:]:
        _module_path = _module_path + '.' + m
    print(_module_path)
    plg_lib = importlib.import_module(_module_path)


def print_detail_core(name, value, value_check=False):
    if isinstance(value, torch.Tensor):
        print(name, type(value), value.shape, value.device, end=' ')
        if len(value.shape) == 0:
            print('value:', value.item())
        else:
            print()
        vcheck(value) if value_check else None
    elif isinstance(value, np.ndarray):
        print(name, type(value), value.shape)
        vcheck(value) if value_check else None
    elif isinstance(value, list):
        print(name, type(value), 'length: ', len(value))
        if len(value) < 10:
            for i in range(len(value)):
                print_detail_core(name + '[' + str(i) + ']', value[i])
    elif isinstance(value, tuple):
        print(name, type(value), 'length: ', len(value))
        if len(value) < 10:
            for i in range(len(value)):
                print_detail_core(name + '[' + str(i) + ']', value[i])
    elif isinstance(value, dict):
        print(name, type(value), 'keys:', value.keys())
        for key in value.keys():
            print_detail_core(name + '[' + str(key) + ']', value[key])
    elif isinstance(value, str):
        print(name, type(value), value)
    else:
        print(name, type(value))


def print_detail(value, name='None', vcheck=False):
    print_detail_core(name, value, vcheck)
    print()


def show_vox_info(vox: np.ndarray):
    print('vox.shape:', vox.shape)
    print('unique values:', np.unique(vox))
    for i in range(len(class_names)):
        mask = vox == i
        n = np.sum(mask)
        print('class {}: {}, {} voxels'.format(i, class_names[i], n))


def adjust_data_pathes(cfg):
    if platform.node() == "komorebi-yq-ubuntu":
        cfg.data.train[
            'data_root'] = '/home/komorebi/workspace/researches/01_datasets/semanticKITTI'
        cfg.data.train[
            'preprocess_root'] = '/home/komorebi/workspace/researches/01_datasets/semanticKITTI/dataset'
        cfg.data.train[
            'depth_gt_path'] = '/home/komorebi/workspace/researches/01_datasets/semanticKITTI/gt_renderocc/depth_gt_lidar'
        cfg.data.train[
            'semantic_gt_path_lidar'] = '/home/komorebi/workspace/researches/01_datasets/semanticKITTI/gt_renderocc/label_gt_lidar'
        cfg.data.train[
            'semantic_gt_path_segformer'] = '/home/komorebi/workspace/researches/01_datasets/semanticKITTI/gt_renderocc/kitti_semantics'


def mem(x=None):
    if x is not None:
        if type(x) == torch.Tensor:
            print('x  : ', x.shape, x.dtype)
            print('     ',
                  x.element_size() * x.numel() / 1024 / 1024 / 1024, 'GB')
        else:
            print('x  : ', type(x))
            # print('     ',
            #       sum(p.element_size() * p.numel() for p in x.parameters() if p.requires_grad) / 1024 / 1024 / 1024,
            #       'GB')
            print(
                '     ',
                sum(p.element_size() * p.numel()
                    for p in x.parameters()) / 1024 / 1024 / 1024, 'GB')

    print('all: ', torch.cuda.memory_allocated() / 1024 / 1024 / 1024, 'GB')
    print('reserved: ',
          torch.cuda.memory_reserved() / 1024 / 1024 / 1024, 'GB')


def vcheck(value):
    if isinstance(value, torch.Tensor):
        has_nan = torch.isnan(value).any().item()
        v_max = value.max().item()
        v_min = value.min().item()
    elif isinstance(value, np.ndarray):
        has_nan = np.isnan(value).any()
        v_max = value.max()
        v_min = value.min()
    else:
        raise ValueError('Unsupported type:', type(value))
    print(' has nan:', has_nan, ' max:', v_max, ' min:', v_min)
