# -*- coding: utf-8 -*-
"""
   File Name：     model_utils.py
   Description :   模型工具类
   Author :       mick.yi
   Date：          2019/7/11
"""
import numpy as np
import torch
from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from torch import save


def save_model(model, file_path):
    model = model.module if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel) else model
    save(model.state_dict(), file_path)


def init_weights(net):
    """the weights of conv layer and fully connected layers
    are both initilized with Xavier algorithm, In particular,
    we set the parameters to random values uniformly drawn from [-a, a]
    where a = sqrt(6 * (din + dout)), for batch normalization
    layers, y=1, b=0, all bias initialized to 0.
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    return net


def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias,
    bn weights, bn bias, linear bias)

    Args:
        net: network architecture

    Returns:
        a dictionary of params splite into to categlories
    """

    decay = [p for name, p in net.named_parameters() if ('bias' not in name and 'bn' not in name)]
    no_decay = [p for name, p in net.named_parameters() if ('bias' in name or 'bn' in name)]

    assert len(list(net.parameters())) == len(decay) + len(no_decay)
    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]


def mix_up_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, ...]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mix_up_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
