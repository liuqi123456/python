#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/13 9:16
#@Author: Kevin.Liu
#@File  : tmallRegression.py

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

years = np.arange(2009, 2020)
sales = np.array([0.5, 9.36, 52, 191, 352, 571, 912, 1207, 1682.69, 2135, 2684])
# print(years)
# print(sales)
# plt.scatter(years, sales, c='red', marker='*', s=80)
# plt.show()

X = (years - 2008).reshape(-1, 1)
y = sales
# print(X, y)

# lr = LinearRegression(fit_intercept=True)
# lr.fit(X, y)
# # weight 权重
# w = lr.coef_[0]
# # bias 偏差
# b = lr.intercept_
# # print(w, b)
# plt.scatter(years - 2008, sales, c='red', marker='*', s=80)
# plt.show()
# x = np.linspace(1, 12, 50)
# plt.plot(x, w*x+b, c='green')
# plt.show()

# # 假定函数是一元二次f(x) = w1*x**2 + w2*x + b
# lr = LinearRegression(fit_intercept=True)
# X2 = np.concatenate([X**2, X], axis=1)
# lr.fit(X2, y)
# # weight 权重
# w1, w2 = lr.coef_
# # bias 偏差
# b = lr.intercept_
# # print(w1, w2, b)
# plt.scatter(years - 2008, sales, c='red', marker='*', s=80)
# # plt.show()
# x = np.linspace(1, 12, 50)
# # 一元二次
# f = lambda x: w1*x**2 + w2*x + b
# plt.plot(x, f(x), c='green')
# plt.show()
# print('2020年天猫双十一销量预测：', np.round(f(12), 2))

# 假定函数是一元三次f(x) =w1*x**3 + w2*x**2 + w3*x + b
lr = LinearRegression(fit_intercept=True)
X3 = np.concatenate([X**3, X**2, X], axis=1)
lr.fit(X3, y)
# weight 权重
w1, w2, w3 = lr.coef_
# bias 偏差
b = lr.intercept_
# print(w1, w2, b)
plt.scatter(years - 2008, sales, c='red', marker='*', s=80)
# plt.show()
x = np.linspace(1, 12, 50)
# 一元二次
f = lambda x: w1*x**3 + w2*x**2 + w3*x + b
plt.plot(x, f(x), c='green')
plt.show()
print('2020年天猫双十一销量预测：', np.round(f(12), 2))