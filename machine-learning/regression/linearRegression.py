#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/12 21:00
#@Author: Kevin.Liu
#@File  : linearRegression.py

from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(0, 10, 50).reshape(-1, 1)
# print(X)
y = np.random.randint(2, 8, size=1)*X
# print(y)
# print(y/X)

lr = LinearRegression()
lr.fit(X, y)
# coefficient 系数（斜率）个数：属性的个数决定
# w ---> weight 权重
# print(lr.coef_)

# 线性代数中矩阵运算
# print(np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y))

# 波士顿房价
boston = datasets.load_boston()
X = boston['data']
y = boston['target']
# print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# lr = LinearRegression(fit_intercept=False)
# lr.fit(X_train, y_train)
# w = lr.coef_
# # print(lr.coef_, lr.intercept_)
# # 自己计算预测的结果
# print(X_test.dot(w).round(2)[:25])
# # 算法预测的结果
# print(lr.predict(X_test).round(2)[:25])
# # 真实结果
# print(y_test[:25])

lr = LinearRegression(fit_intercept=True)
lr.fit(X_train, y_train)
w = lr.coef_
# print(lr.coef_, lr.intercept_)
# 自己计算预测的结果 --> 根据斜率和截距构造方程，进行求解的结果
print((X_test.dot(w) + lr.intercept_).round(2)[:25])
# 算法预测的结果
print(lr.predict(X_test).round(2)[:25])
# 真实结果
print(y_test[:25])