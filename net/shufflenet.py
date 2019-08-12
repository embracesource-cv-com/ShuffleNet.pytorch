# -*- coding: utf-8 -*-
"""
Created on 2019/6/22 上午6:40

@author: mick.yi

ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile
Devices
https://arxiv.org/pdf/1707.01083.pdf

"""
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['ShuffleNet', 'shufflenet']


def conv_group_1x1(in_channels, out_channels, groups):
    """
    group convolution
    :param in_channels:
    :param out_channels:
    :param groups:
    :return:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1,
                  stride=1, padding=0, groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))


def conv_group_1x1_linear(in_channels, out_channels, groups):
    """
    linear group convolution;
    :param in_channels:
    :param out_channels:
    :param groups:
    :return:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1,
                  stride=1, padding=0, groups=groups),
        nn.BatchNorm2d(out_channels))


def conv_dw(in_channels, stride=1):
    """
    depthwise convolution
    :param in_channels:
    :param stride:
    :return:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3,
                  stride=stride, padding=1),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True))


class ShuffleChannel(nn.Module):
    def __init__(self, groups=2, **kwargs):
        self.groups = groups
        super(ShuffleChannel, self).__init__(**kwargs)

    def forward(self, x):
        """

        :param x: tensor [B,C,H,W]
        :return:
        """
        b, c, h, w = x.size()
        x = x.view(b, self.groups, c // self.groups, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, c, h, w)
        return x


class ShuffleBlock(nn.Module):
    """
    shuffle块，与resnet的bottleneck块相似，不同点如下
    a）1x1卷积使用分组卷积
    b）3x3卷积使用深度可分卷积
    c) 步长为2时使用concat合并; 且shutcut分支使用avgpool,而不是1x1卷积
    d) 3x3卷积之前通道做shuffle
    """

    def __init__(self, in_channels, out_channels, groups, stride=1, **kwargs):
        """

        :param in_channels:
        :param out_channels:
        :param groups:
        :param stride:
        :param kwargs:
        """
        super(ShuffleBlock, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride
        self.conv1 = conv_group_1x1(in_channels, out_channels // 4, groups=groups)
        self.conv2 = conv_dw(out_channels // 4, stride)
        self.conv3 = conv_group_1x1_linear(out_channels // 4, out_channels, groups=groups)
        self.relu = nn.ReLU(inplace=True)
        self.shuffle = ShuffleChannel(groups=2)

    def forward(self, x):
        """

        :param x: tensor [B,C,H,W]
        :return:
        """
        identity = x

        x = self.conv1(x)
        x = self.shuffle(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.stride == 1:
            x += identity
        else:
            identity = F.avg_pool2d(identity, 2, 2)

            x = torch.cat((identity, x), dim=1)

        x = self.relu(x)
        return x


class ShuffleNet(nn.Module):
    def __init__(self, num_classes=1000, groups=3, alpha=1.0, **kwargs):
        """

        :param num_classes:
        :param groups: 分组卷积的组数
        :param alpha: multipliers, 倍乘因子，用于增减通道数;
        :param kwargs:
        """
        super(ShuffleNet, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.groups = groups
        self.alpha = alpha
        self.out_channels_dict = {1: [144, 288, 576],
                                  2: [200, 400, 800],
                                  3: [240, 480, 960],
                                  4: [272, 544, 1088],
                                  8: [384, 768, 1536]}
        self.modify_channels()
        # 每个阶段输出通道数
        stage_channels = self.out_channels_dict[self.groups]

        self.conv_pool = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # stage 2,3,4
        self.stages = nn.Sequential(
            self._make_stage(24, stage_channels[0], 4, 2, 1),  # 不是分组卷积
            self._make_stage(stage_channels[0], stage_channels[1], 8, 2, groups),
            self._make_stage(stage_channels[1], stage_channels[2], 4, 2, groups))

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stage_channels[2], num_classes)

    def modify_channels(self):
        """
        根据alpha参数增减通道数
        :return:
        """
        if self.alpha != 1.:
            for groups in self.out_channels_dict.keys():
                out_channels = self.out_channels_dict[groups]
                out_channels = [int(c * self.alpha) for c in out_channels]
                self.out_channels_dict[groups] = out_channels

    def forward(self, x):
        """

        :param x: [B,3,H,W]
        :return: [B,C]
        """
        x = self.conv_pool(x)
        x = self.stages(x)
        x = self.global_pool(x)  # [B,C,1,1]
        x = x.view(x.size(0), x.size(1))  # [B,C]
        x = self.fc(x)
        return x

    @staticmethod
    def _make_stage(in_channels, out_channels, repeat_times,
                    stride=2, groups=3):
        """

        :param in_channels:
        :param out_channels:
        :param repeat_times: 重复次数
        :param stride: 步长
        :param groups: 分组卷积的组数
        :return:
        """
        blocks = list()
        blocks.append(ShuffleBlock(in_channels, out_channels - in_channels, groups, stride))
        # 添加其余block,stride都为1
        for _ in range(1, repeat_times):
            blocks.append(ShuffleBlock(out_channels, out_channels, groups, 1))
        return nn.Sequential(*blocks)


def shufflenet(num_classes=1000, **kwargs):
    return ShuffleNet(num_classes, **kwargs)
