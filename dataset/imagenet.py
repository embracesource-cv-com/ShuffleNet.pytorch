# -*- coding: utf-8 -*-
"""
   File Name：     imagenet.py
   Description :
   Author :       mick.yi
   Date：          2019/8/12

"""
from torch.utils.data.dataset import Dataset
import os.path as osp
import os
import codecs
from PIL import Image


class ImageNetTrain(Dataset):
    def __init__(self, data_root, map_cls_file):
        """

        :param data_root: imagenet数据集根目录
        :param map_cls_file: 训练集类别id文件,内容如: n02110185 3 Siberian_husky; 代表:wnid class_id class_name
        """
        self.data_root = data_root
        self.data_dir = osp.join(data_root, 'train')
        self.image_path_list = osp.join(self.data_dir, os.listdir(self.data_dir))
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
        image_path = self.image_path_list[index]
        image = Image.open(image_path)
        label = self.wnid_map_class_id(image_path.split('_')[0])
        return image, label

    def __len__(self):
        return len(self.image_path_list)
