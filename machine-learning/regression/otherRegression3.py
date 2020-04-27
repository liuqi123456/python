#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/16 8:54
#@Author: Kevin.Liu
#@File  : otherrRegression3.py

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 2*np.pi, 50).reshape(-1, 1)
y = np.sin(X)

# 数据的范围(宽)
x = np.linspace(0, 2*np.pi, 150).reshape(-1, 1)

# lr = LinearRegression()
# lr.fit(X, y)
# y_ = lr.predict(x)
# plt.scatter(X, y)
# plt.plot(x, y_, c='green')
# plt.show()
# # coef_ 线性方程的系数
# print(lr.coef_, lr.intercept_)

# # KNN回归不是方程，更像是平均值。找5个邻居，计算5个邻居的平均值，然后穿过去
# knn = KNeighborsRegressor(n_neighbors=5)
# knn.fit(X, y)
# y_ = knn.predict(x)
# plt.scatter(X, y)
# plt.plot(x, y_, c='green')
# plt.show()

# dtr = DecisionTreeRegressor()
# dtr.fit(X, y)
# y_ = dtr.predict(x)
# plt.scatter(X, y)
# plt.plot(x, y_, c='green')
# plt.figure(figsize=(16, 12))
# _ = tree.plot_tree(dtr, filled=True)
# plt.show()

# 数据的范围(宽)
x = np.linspace(-np.pi, 3*np.pi, 200).reshape(-1, 1)

# lr = LinearRegression()
# lr.fit(X, y)
# y_ = lr.predict(x)
# plt.scatter(X, y)
# plt.plot(x, y_, c='green')
# plt.show()
# # coef_ 线性方程的系数
# print(lr.coef_, lr.intercept_)

# # KNN回归不是方程，更像是平均值。找5个邻居，计算5个邻居的平均值，然后穿过去
# knn = KNeighborsRegressor(n_neighbors=5)
# knn.fit(X, y)
# y_ = knn.predict(x)
# plt.scatter(X, y)
# plt.plot(x, y_, c='green')
# plt.show()

# dtr = DecisionTreeRegressor()
# dtr.fit(X, y)
# y_ = dtr.predict(x)
# plt.scatter(X, y)
# plt.plot(x, y_, c='green')
# plt.figure(figsize=(16, 12))
# _ = tree.plot_tree(dtr, filled=True)
# plt.show()

# # 测试数据进行数据清洗，研究规律，周期性2pi，还原数据
# dtr = DecisionTreeRegressor()
# dtr.fit(X, y)
# # 预处理
# pre_x = x.copy()
# cond = pre_x > 2*np.pi
# pre_x[cond] -= 2*np.pi
# cond2 = pre_x < 0
# pre_x[cond2] += 2*np.pi
# y_ = dtr.predict(pre_x)
# plt.scatter(X, y)
# plt.plot(x, y_, c='green')
# plt.figure(figsize=(16, 12))
# _ = tree.plot_tree(dtr, filled=True)
# plt.show()