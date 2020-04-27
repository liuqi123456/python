#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/11 17:09
#@Author: Kevin.Liu
#@File  : gradientTree.py

# 梯度森林

from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt

# f(x) = 3*x**2
# 梯度 = 导数

# X, y = datasets.load_iris(True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# gbdt = GradientBoostingClassifier(n_estimators=10)
# gbdt.fit(X_train, y_train)
# print(gbdt.score(X_test, y_test))

# X 数据：购物金额和上网时间
X = np.array([[800, 3], [1200, 1], [1800, 4], [2500, 2]])
# y 目标：14(初一), 16(初三), 24(大学毕业), 26(工作两年)
y = np.array([14, 16, 24, 26])
# 使用回归
gbdt = GradientBoostingRegressor(n_estimators=10)
gbdt.fit(X, y)
print(gbdt.predict(X))

# mean-square-error 均方误差
# print(((y - y.mean())**2).mean())
# print(((y[:2] - y[:2].mean())**2).mean())

plt.rcParams['font.sans-serif'] = 'KaiTi' # 设置字体
plt.figure(figsize=(9, 6))
#   第一棵树，根据平均值，计算了残差
_ = tree.plot_tree(gbdt[0, 0], filled=True, feature_names=['消费', '上网'])
plt.savefig('./gradient1.jpg')

learning_rate = 0.1
gbdt1 = np.array([-6, -4, 6, 4])
# 梯度提升 学习率0.1
print(gbdt1 - gbdt1*0.1)
#   第二棵树，根据梯度提升，减少残差(残差越小，结果越好、越准确)
_ = tree.plot_tree(gbdt[1, 0], filled=True, feature_names=['消费', '上网'])
plt.savefig('./gradient2.jpg')

learning_rate = 0.1
gbdt2 = np.array([-5.4, -3.6, 5.4, 3.6])
# 梯度提升 学习率0.1
print(gbdt2 - gbdt2*0.1)
#  第三棵树，根据梯度提升，减少残差(残差越小，结果越好、越准确)
_ = tree.plot_tree(gbdt[2, 0], filled=True, feature_names=['消费', '上网'])
plt.savefig('./gradient3.jpg')

learning_rate = 0.1
gbdt3 = np.array([-4.86, -3.24, 4.86, 3.24])
# 梯度提升 学习率0.1
print(gbdt3 - gbdt3*0.1)
_ = tree.plot_tree(gbdt[3, 0], filled=True, feature_names=['消费', '上网'])
plt.savefig('./gradient4.jpg')

# 最后一棵树，预测的残差
nd2 = np.array([-1.395, -2.0925, 1.395, 2.0925])
# 最后一次的残差
residual = nd2 - nd2*0.1
print(residual)

# 根据最后一棵树，计算了算法最终的预测值
print((y[[0, 1, 3, 2]] - residual)[[0, 1, 3, 2]])
print(y - residual)

# 直接使用算法predict 返回的值，和手算一摸一样
print(gbdt.predict(X).round(4))