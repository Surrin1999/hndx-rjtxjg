# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:02:01 2020

@author: S.Health
"""
import numpy as np

from torch import Tensor
import torch.nn as nn
import torch 
import torch.nn.functional as F


class MLP(nn.Module):
    # num_class类别数量
    def __init__(self, input_dim, hid1_dim, hid2_dim, num_class, dropout):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hid1_dim)        
        self.fc2 = nn.Linear(hid1_dim, hid2_dim)
        self.fc3 = nn.Linear(hid2_dim, num_class)
        # 防止过拟合的参数，提升模型泛化能力
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        # 随机初始化权重，xavier正态分布,gain是可选的缩放因子
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc3.weight, gain=1.414)
        # 初始化偏置，用0来填充
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

    #     前向传播
    def forward(self, x):
        x = F.relu(self.fc1(x))
        #随机将输入的某些元素归零，每个元素归零的概率默认为50%;self.training默认为True，表示应用dropout
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # 返回结果是一个与x维度相同的张量，每个元素的取值范围在（0,1）区间。
        return F.log_softmax(x, dim=1) # 按照行来做归一化；在softmax的结果上再做多一次log运算


