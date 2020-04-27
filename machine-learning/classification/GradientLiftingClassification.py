#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/12 21:38
#@Author: Kevin.Liu
#@File  : GradientLiftingClassification.py

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

Xi = np.arange(1, 11)
yi = np.array([0, 0, 0, 1, 1]*2)
# print(Xi, yi)

gdbt = GradientBoostingClassifier(n_estimators=3, max_depth=1)
fit = gdbt.fit(Xi.reshape(-1, 1), yi)
# print(fit)
# print(gdbt.estimators_.shape)

# print(((yi-yi.mean())**2).mean())
# print(np.var(yi))

# plt.figure(figsize=(9, 6))
# _ = tree.plot_tree(gdbt[0, 0], filled=True)
# plt.savefig('./GradientLiftingClassification.jpg')

F0 = np.log(4/6)
print(F0)
# 残差，概率，负梯度
yi_1 = yi - 1/(1 + np.exp(-F0))
print(yi_1)
# 计算每个裂分点mse
mse1 = []
for i in range(1, 11):
    if i == 10:
        mse1.append(np.var(yi_1))
    else:
        mse1.append((np.var(yi_1[:i])*i + np.var(yi_1[i:])*(10-i))/10)
print(np.round(mse1, 4))
mse1 = np.asarray(mse1)
print(mse1)

# 两个分支
# 左边这个分支的预测值
print(np.round(yi_1[:8].sum()/(((yi[:8] - yi_1[:8])*(1 - yi[:8] + yi_1[:8])).sum()), 3))
# 右边这个分支的预测值
print(np.round(yi_1[8:].sum()/(((yi[8:] - yi_1[8:])*(1 - yi[8:] + yi_1[8:])).sum()), 3))

# 第一颗数据预测的值
print(gdbt[0, 0].predict(Xi.reshape(-1, 1)))
y_1 = [-0.625]*8 + [2.5]*2
y_1 = np.asarray(y_1)
print(y_1)

# 学习率 learning_rate = 0.1
F1 = F0 + y_1*0.1
print(F1.round(4))
# 残差，概率，负梯度
yi_2 = yi - 1/(1 + np.exp(-F1))
print(yi_2)
# 计算每个裂分点mse
mse2 = []
for i in range(1, 11):
    if i == 10:
        mse2.append(np.var(yi_2))
    else:
        mse2.append((np.var(yi_2[:i])*i + np.var(yi_2[i:])*(10-i))/10)
print(np.round(mse2, 4))
mse2 = np.asarray(mse2)
print(mse2)

plt.figure(figsize=(9, 6))
_ = tree.plot_tree(gdbt[1, 0], filled=True)
# plt.savefig('./GradientLiftingClassification2.jpg')

# 两个分支
# 左边这个分支的预测值
print(np.round(yi_2[:8].sum()/(((yi[:8] - yi_2[:8])*(1 - yi[:8] + yi_2[:8])).sum()), 3))
# 右边这个分支的预测值
print(np.round(yi_2[8:].sum()/(((yi[8:] - yi_2[8:])*(1 - yi[8:] + yi_2[8:])).sum()), 3))