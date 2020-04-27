#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/12 22:22
#@Author: Kevin.Liu
#@File  : adaboost.py

from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

X = np.arange(10).reshape(-1, 1)
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
# print(X, y)

ada = AdaBoostClassifier(n_estimators=3)
ada.fit(X, y)
# print(ada.fit(X, y))
# print(ada.estimators_)

plt.figure(figsize=(9, 6))
_ = tree.plot_tree(ada[0])
# plt.savefig('./adaboost.png')

y_ = ada[0].predict(X)
print(y_)
# 误差率
e1 = 0.1*(y != y_).sum()
print(e1)
# 计算每一颗树权重
# 随机森林中每棵树的权重一样的
# adaboost提升树中每棵树的权重不同
a1 = np.round(1/2*np.log((1-e1)/e1), 4)
print(a1)
# 样本预测准确：更新权重
w2 = 0.1*np.e**(-a1*y*y_)
w2 = w2/w2.sum()
print(w2.round(4))

ada = AdaBoostClassifier(n_estimators=3, algorithm='SAMME')
ada.fit(X, y)
# print(ada.fit(X, y))
# print(ada.estimators_)
plt.figure(figsize=(9, 6))
_ = tree.plot_tree(ada[1])
# plt.savefig('./adaboost2.png')

y_ = ada[1].predict(X)
print(y_)
# 误差率
e2 = 0.0715*3
print(e2)
# 计算每二颗树权重
a2 = np.round(1/2*np.log((1-e2)/e2), 4)
print(a2)
# 样本预测准确：更新权重
w3 = w2*np.e**(-a2*y*y_)
w3 = w3/w3.sum()
print(w3.round(4))
plt.figure(figsize=(9, 6))
_ = tree.plot_tree(ada[2])
# plt.savefig('./adaboost3.png')

y_ = ada[2].predict(X)
print(y_)
# 误差率
e3 = (w3*(y_ != y)).sum()
print(e3)
# 计算每三树权重
a3 = 1/2*np.log((1-e3)/e3)
print(a3)
# 弱分类器合并成强分类器，加和

# 预测结果
y_predict = a1*ada[0].predict(X) + a2*ada[1].predict(X) + a3*ada[2].predict(X)
print(np.sign(y_predict))
# 集成树
print(ada.predict(X))