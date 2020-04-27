#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/15 12:51
#@Author: Kevin.Liu
#@File  : ridgeRegression.py

# 岭回归

from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np

X = 1/(np.arange(1, 11) + np.arange(0, 10).reshape(-1, 1))
y = np.ones(10)
# print(X)
# print(y)
ridge = Ridge(fit_intercept=False)
# logspace 等比数列
alphas = np.logspace(start=-10, stop=-2, num=200)
# print(alphas)
# linspace 等差数列
# alphas = np.linspace(1e-10, 1e-2, 200)
# print(alphas)
coefs = []
for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
plt.plot(alphas, coefs)
plt.xscale('log') # 横坐标已对数显示
plt.xlabel('alpha', fontsize=25)
plt.ylabel('coef', fontsize=25, c='red', rotation=0)
plt.show()