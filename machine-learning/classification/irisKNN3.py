#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/9 17:26
#@Author: Kevin.Liu
#@File  : irisKNN3.py

# 鸢尾花分类 调参

from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
# model_selection: 模型选择 / cross_val_score: cross 交叉，validation 验证(测试) 交叉验证
from sklearn.model_selection import cross_val_score

X, y = datasets.load_iris(True)
# print(X.shape)
# print(y.shape)

# 12.24744871391589 k选择1~13
# print(150**0.5)

# knn = KNeighborsClassifier()
# score = cross_val_score(knn, X, y, scoring='accuracy', cv=10)
# print(score.mean()) # 均值

# k选择越合适
# errors = []
# for k in range(1, 14):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     score = cross_val_score(knn, X, y, scoring='accuracy', cv=6).mean()
#     errors.append(1 - score) # 误差越小，说明k选择越合适
# # print(errors)
# plt.plot(np.arange(1, 14), errors)
# plt.show() # k=12时，误差最小

# k个近邻样本的权重
weights = ['uniform', 'distance']
# for w in weights:
#     knn = KNeighborsClassifier(n_neighbors=12, weights=w)
#     print(w, cross_val_score(knn, X, y, scoring='accuracy', cv=6).mean())

# 最优近邻和权重
# result = {}
# for k in range(1, 14):
#     for w in weights:
#         knn = KNeighborsClassifier(n_neighbors=k, weights=w)
#         sm = cross_val_score(knn, X, y, scoring='accuracy', cv=6).mean()
#         result[w + str(k)] = sm
# # print(result)
# print(list(result)[np.array(list(result.values())).argmax()])
