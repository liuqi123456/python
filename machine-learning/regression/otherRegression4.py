#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/16 9:43
#@Author: Kevin.Liu
#@File  : otherRegression4.py

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt

f = lambda x: (x - 3)**2 + 3.6*x + 2.718
X = np.linspace(-2, 4, 50).reshape(-1, 1)
y = f(X)
# plt.scatter(X, y)
# plt.show()

X = np.concatenate([X**2, X], axis=1)
print(X.shape)
X_test = np.linspace(-4, 8, 200).reshape(-1, 1)
X_test = np.concatenate([X_test**2, X_test], axis=1)
print(X_test.shape)

# lr = LinearRegression()
# lr.fit(X, y)
# y_ = lr.predict(X_test)
# plt.scatter(X[:, 1], y)
# plt.plot(X_test[:, 1], y_, c='g')
# plt.show()

# knn = KNeighborsRegressor()
# knn.fit(X, y)
# # knn模型，数据不是周期
# y_ = knn.predict(X_test)
# plt.scatter(X[:, 1], y)
# plt.plot(X_test[:, 1], y_, c='g')
# plt.show()

dtr = DecisionTreeRegressor()
dtr.fit(X, y)
y_ = dtr.predict(X_test)
plt.scatter(X[:, 1], y)
plt.plot(X_test[:, 1], y_, c='g')
plt.show()