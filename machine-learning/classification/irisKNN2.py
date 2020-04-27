#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/9 14:14
#@Author: Kevin.Liu
#@File  : irisKNN2.py

# 鸢尾花分类 画图

from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import numpy as np
import matplotlib.pylab as pyb

X, y = datasets.load_iris(True)
# (150, 4) 150个样本 4个维度
# print(X.shape)

# 降维
X = X[:, :2]
# (150, 2)
# print(X.shape)

# 画图
# pyb.scatter(X[:, 0], X[:, 1], c=y)
# pyb.show()

knn = KNeighborsClassifier(n_neighbors=5)
# 训练数据
knn.fit(X, y)

# 横坐标 4~8
x1 = np.linspace(4, 8, 100)
# 纵坐标 2~4.5
y1 = np.linspace(2, 4.5, 80)
# print(x1.shape)
# print(y1.shape)

# 背景点，取出来 meshgrid
X1, Y1 = np.meshgrid(x1, y1)
# print(X1.shape)
# print(Y1.shape)

# 方式一
# X1 = X1.reshape(-1, 1)
# Y1 = Y1.reshape(-1, 1)
# X_test = np.concatenate([X1, Y1], axis=1)
# print(X_test.shape)

# 方式二
# 平铺 一维化reshape
X_test = np.c_[X1.ravel(), Y1.ravel()]
print(X_test.shape)

# 预测数据
y_ = knn.predict(X_test)

lc = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
lc2 = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

pyb.scatter(X_test[:, 0], X_test[:, 1], c=y_, cmap=lc)
pyb.scatter(X[:, 0], X[:, 1], c=y, cmap=lc2)
pyb.show()
