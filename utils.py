# -*- coding: UTF-8 -*-
"""
工具类，计算accuracy
@Project ：心脏病预测
@File ：utils.py
@IDE  ：PyCharm
@Author ：Sun Dewang
@Date ：2021/12/3 9:20
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(n_classes)[x]


def accuracy(output, labels):
    preds = np.argmax(output.detach().numpy(), 1)
    acc = accuracy_score(labels, preds)
    return acc


