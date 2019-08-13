# -*- coding: utf-8 -*-
"""
   File Name：     config.py
   Description :  
   Author :       mick.yi
   Date：          2019/8/13
"""
import torch
import os
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from dataset.imagenet import ImageNetTrain, ImageNetVal, ImageNetTest
from torch.utils.data import DataLoader
from net.shufflenetv2 import shufflenetv2_1x


class Config(object):
    DATA_ROOT_DIR = '/dataset/imagenet'

    # save
    save_path = '/tmp/{}-{}.pth'
    SNAPSHOT = 5
    log_dir = './logs.{}/'

    def __init__(self, data_set='imagenet', model_name='shufflenet_v2', batch_size=128):
        """

        :param data_set:
        :param model_name:
        """
        self.data_set = data_set
        self.model_name = model_name
        self.batch_size = batch_size
        self.save_path = self.save_path.format(self.MODEL_NAME, self.DATA_SET)
        self.log_dir = self.log_dir.format(self.MODEL_NAME)
        name, ext = os.path.splitext(self.save_path)
        self.snapshot_save_path = name + '.{:03d}' + ext  # eg: cifar10-vgg16.020.ckpt

    @property
    def num_classes(self):
        if 'imagenet' == self.data_set:
            return 1000
        return 10

    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    def train_loader(self, num_gpus=1):
        train_set = ImageNetTrain(self.DATA_ROOT_DIR, 'data/imagenet/map_clsloc.txt',
                                  transforms.Compose([
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
                                      transforms.ToTensor(),
                                      self.normalize
                                  ]),
                                  transforms.Lambda(lambda x: torch.from_numpy(x)))
        sample = DistributedSampler(train_set) if num_gpus > 1 else None
        data_loader = DataLoader(train_set,
                                 batch_size=self.batch_size,
                                 sampler=sample,
                                 shuffle=sample is None,
                                 num_workers=8)
        return data_loader

    def val_loader(self):
        val_set = ImageNetVal(self.DATA_ROOT_DIR,
                              'data/imagenet/ILSVRC2012_validation_ground_truth.txt',
                              transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  self.normalize
                              ]),
                              transforms.Lambda(lambda x: torch.from_numpy(x)))
        val_loader = DataLoader(val_set,
                                batch_size=self.batch_size,
                                num_workers=8,
                                pin_memory=False)
        return val_loader

    def test_loader(self):
        test_set = ImageNetTest(self.DATA_ROOT_DIR,
                                transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    self.normalize
                                ]))
        data_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size)
        return data_loader

    def model(self):
        if 'shufflenet_v2' == self.model_name:
            model = shufflenetv2_1x(num_classes=self.num_classes)
        return model

    def device(self, num_gpus=1):
        return torch.device('cuda' if torch.cuda.is_available() and num_gpus > 0 else 'cpu')
