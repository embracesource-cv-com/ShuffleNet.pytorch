# -*- coding: utf-8 -*-
"""
   File Name：     imagenet.py
   Description :
   Author :       mick.yi
   Date：          2019/8/12

"""
from torch.utils.data.dataset import Dataset
import os.path as osp
import codecs
from PIL import Image
import numpy as np
from utils import file_utils


class ImageNetTrain(Dataset):
    """
    训练集
    """

    def __init__(self, data_root, map_cls_file, transform=None, target_transform=None):
        """

        :param data_root: imagenet数据集根目录
        :param map_cls_file: 训练集类别id文件,内容如: n02110185 3 Siberian_husky; 代表:wnid class_id class_name
        """
        self.data_root = data_root
        self.data_dir = osp.join(data_root, 'train')
        self.image_path_list = file_utils.get_sub_files(self.data_dir)
        self.transform = transform
        self.target_transform = target_transform
        with codecs.open(map_cls_file, encoding='utf-8', mode='r') as f:
            lines = f.readlines()
        self.wnid_map_class_id = dict()
        self.id_map_name = dict()
        self.name_map_id = dict()

        for l in lines:
            wnid, class_id, class_name = l.split(' ')
            self.wnid_map_class_id[wnid] = int(class_id) - 1  # id从0开始
            self.id_map_name[int(class_id) - 1] = class_name
            self.name_map_id[class_name] = int(class_id) - 1

    def __getitem__(self, index):
        image_path = self.image_path_list[index]  # eg: /path/to/imagenet/n03032252_40790.JPEG
        image = Image.open(image_path).convert("RGB")
        wnid = osp.basename(image_path).split('_')[0]
        label = self.wnid_map_class_id[wnid]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(np.array(label))

        return image, label

    def __len__(self):
        return len(self.image_path_list)


class ImageNetVal(Dataset):
    def __init__(self, data_root, cls_file, transform=None, target_transform=None):
        """

        :param data_root:
        :param cls_file:
        """
        self.data_dir = osp.join(data_root, 'val')
        self.image_path_list = file_utils.get_sub_files(self.data_dir)
        self.image_path_list.sort()
        self.transform = transform
        self.target_transform = target_transform
        with codecs.open(cls_file, encoding='utf-8') as f:
            lines = f.readlines()
        self.class_ids = [int(class_id) - 1 for class_id in lines]  # 从0开始

    def __getitem__(self, index):
        image = Image.open(self.image_path_list[index]).convert("RGB")
        label = self.class_ids[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(np.array(label))
        return image, label

    def __len__(self):
        return len(self.image_path_list)


class ImageNetTest(Dataset):
    def __init__(self, data_root, transform=None):
        """

        :param data_root:
        """
        self.data_dir = osp.join(data_root, 'test')
        self.image_path_list = file_utils.get_sub_files(self.data_dir)
        self.image_path_list.sort()
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.image_path_list[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_path_list)
