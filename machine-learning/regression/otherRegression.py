#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/15 10:07
#@Author: Kevin.Liu
#@File  : otherRegression.py

# CV cross validation 交叉验证
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

diabetes = datasets.load_diabetes()
X = diabetes['data']
y = diabetes['target']
X_train, X_Test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# 线性回归
lr = LinearRegression()
lr.fit(X_train, y_train)
# 回归问题的得分，不是准确率
# print(lr.score(X_Test, y_test))
y_ = lr.predict(X_Test)
# print(mean_squared_error(y_test, y_))
# print(r2_score(y_test, y_))

# 自己计算的结果
u = ((y_test - y_)**2).sum()
v = ((y_test - y_test.mean())**2).sum()
r2 = 1 - u / v
# print(r2)
# print(y_.round(0), y_test)
# print(np.linspace(0.00001, 1, 50))
# print(np.logspace(-5, 0, 50))

# 岭回归
ridge = Ridge(alpha=0.001)
ridge.fit(X_train, y_train)
# 回归问题的得分，不是准确率
# print(ridge.score(X_Test, y_test))
y_ = ridge.predict(X_Test)
# print(mean_squared_error(y_test, y_))
# print(r2_score(y_test, y_))

ridgeCV = RidgeCV(alphas=np.logspace(-5, 0, 50), scoring='r2', cv=6)
ridgeCV.fit(X_train, y_train)
y_ = ridgeCV.predict(X_Test)
print(mean_squared_error(y_test, y_))
print(r2_score(y_test, y_))

ridgeCV = RidgeCV(alphas=np.linspace(0.00001, 1, 50), scoring='r2', cv=6)
ridgeCV.fit(X_train, y_train)
y_ = ridgeCV.predict(X_Test)
print(mean_squared_error(y_test, y_))
print(r2_score(y_test, y_))