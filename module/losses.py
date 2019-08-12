# -*- coding: utf-8 -*-
"""
   File Name：     losses.py
   Description :  
   Author :       mick.yi
   Date：          2019/7/16
"""
import torch
from torch import nn
from torch.nn import functional as F


class SoftCrossEntropyLoss(nn.NLLLoss):
    def __init__(self, label_smoothing=0, num_classes=1000, **kwargs):
        assert 0 <= label_smoothing <= 1
        super(SoftCrossEntropyLoss, self).__init__(**kwargs)
        self.confidence = 1 - label_smoothing
        self.other = label_smoothing * 1.0 / (num_classes - 1)
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        print('using soft celoss!!!, label_smoothing = ', label_smoothing)

    def forward(self, input, target):
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.other)
        one_hot.scatter_(1, target.unsqueeze(1).long(), self.confidence)
        input = F.log_softmax(input, 1)
        return self.criterion(input, one_hot)
