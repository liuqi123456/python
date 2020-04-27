#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/15 11:19
#@Author: Kevin.Liu
#@File  : otherRegression2.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV

# 50个样本, 200个特征
X = np.random.randn(50, 200)
w = np.random.randn(200)
# print(X)
# print(w)

# 将其中的190个置为0
index = np.arange(0, 200)
np.random.shuffle(index)
# print(index)
w[index[0:190]] = 0
# print(w)
y = X.dot(w)
# print(y)

linear = LinearRegression(fit_intercept=False)
ridgeCV = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 2, 5, 10], cv=5, fit_intercept=False)
lassoCV = LassoCV(alphas=[0.001, 0.01, 0.1, 1, 2, 5, 10], cv=3, fit_intercept=False)
linear.fit(X, y)
ridgeCV.fit(X, y)
lassoCV.fit(X, y)
linear_w = linear.coef_
ridgeCV_w = ridgeCV.coef_
lassoCV_w = lassoCV.coef_

plt.figure(figsize=(12, 9))
axes = plt.subplot(2, 2, 1)
axes.plot(w)

axes = plt.subplot(2, 2, 2)
axes.plot(linear_w)
axes.set_title('linear')

axes = plt.subplot(2, 2, 3)
axes.plot(ridgeCV_w)
axes.set_title('ridgeCV')

axes = plt.subplot(2, 2, 4)
axes.plot(lassoCV_w)
axes.set_title('lassoCV')
plt.show()
