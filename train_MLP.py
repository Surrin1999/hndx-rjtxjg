# -*- coding: UTF-8 -*-
"""
主函数，训练、保存模型
@Project ：心脏病预测
@File ：train_MLP.py
@IDE  ：PyCharm
@Author ：Sun Dewang
@Date ：2021/12/2 21:17
"""

from __future__ import division
from __future__ import print_function

import time
import numpy as np

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from model import MLP
from sklearn.model_selection import train_test_split
from utils import accuracy

data = pd.read_csv('heart.csv')
data = data[['trestbps', 'chol', 'fbs', 'ca', 'restecg', 'thalach', 'cp', 'exang', 'oldpeak', 'slope', 'target']]
# data = np.array(data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))).values)
data = np.array(data)

# Initial splitting
X = data[:, 0:10]
y = data[:, 10]

numRandom = 5  # number of random splits

Accuaracytrp = []

random_count = 0
random_state = 0

while random_count < numRandom:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    random_state = random_state + 1
    random_count = random_count + 1

    print('%d-th random split' % (random_count))
    print('train_index', X_train)

    # Model and optimizer
    model = MLP(input_dim=X_train.shape[1],
                hid1_dim=32,
                hid2_dim=16,
                num_class=2,
                dropout=0.5)

    loss_func = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

    train_loss_all = []
    val_ACC_all = []
    val_loss_all = []

    # Train model
    t_total = time.time()
    for epoch in range(500):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output_train = model(X_train)
        loss_train = loss_func(output_train, y_train)
        acc_train = accuracy(output_train, y_train)
        # F1_train = f1_score(y_train, output_train)
        loss_train.backward()
        optimizer.step()
        train_loss_all.append(loss_train.item())

        # Validation and Testing
        model.eval()  # deactivates dropout during validation run.
        with torch.no_grad():
            output_test = model(X_test)
            acc_val = accuracy(output_test, y_test)
            loss_val = loss_func(output_test, y_test)

            val_ACC_all.append(acc_val)
            val_loss_all.append(loss_val)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    ##find the epoch with highest validation accuaracy
    val = np.add(val_ACC_all, 0)  # 把验证的准确度和得分相加
    best_epoch = np.argmax(val)  # 求出最优的epoch
    # ##find the epoch with lowest validation loss
    best_epoch = np.argmin(val_loss_all)
    test_acc_best_val_epoch = val_ACC_all[best_epoch]

    print('the best validation epoch', best_epoch)
    print('best val acc %f' % (val_ACC_all[best_epoch]))
    Accuaracytrp.append(test_acc_best_val_epoch)  # 把每一轮最优的epoch的预测准确率加入到Accuaracytrp


avg_acc = np.mean(Accuaracytrp)
std_acc = np.std(Accuaracytrp)
print('avg testing Accuaracy over %d random splits: %f +/- %f' % (
    numRandom, avg_acc, std_acc))

torch.save(model, 'predict_model')

# test = torch.FloatTensor(data[200, 0:10].reshape(-1, 10))
# model(test)
# print(test)
