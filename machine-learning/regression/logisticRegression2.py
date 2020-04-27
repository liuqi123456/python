#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/15 14:31
#@Author: Kevin.Liu
#@File  : LogisticRegression2.py

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np
import warnings

warnings.filterwarnings('ignore')

X, y = datasets.load_iris(True)
print(X.shape, np.unique(y))
cond = y != 2
X = X[cond]
y = y[cond]
# X.shape 属性还是4个 np.unique(y) 二分类问题
# print(X.shape, np.unique(y))
result = train_test_split(X, y, test_size=0.2)
lr = LogisticRegression()
lr.fit(result[0], result[2])
# 几个方程就是几条线分开
# w = lr.coef_
# b = lr.intercept_
# print(w, b)
#
# proba_ = lr.predict_proba(result[1])
# print(proba_)
#
# # 收到计算概率
# h = result[1].dot(w[0].T) + b
# # 类别1的概率 p；另一类的概率是 1 - p
# # sigmoid 函数中计算概率
# p = 1/(1+np.e**(-h))
# print(np.c_[1-p, p])

# 多分类概率计算
X, y = datasets.load_iris(True)
X_train, X_test, y_train, t_test = train_test_split(X, y, test_size=0.2)
lr = LogisticRegression(multi_class='multinomial', solver='saga')
lr.fit(X_train, y_train)

proba_ = lr.predict_proba(X_test)
# print(proba_)

x = np.array([1, 3, -1, 10])
# softmax 软最大: 将数据转化成概率 比较
# p = np.e**(x)/((np.e**(x)).sum())
# print(p)
# print(p.sum())
# 三个分类(三个方程)，每个方程中4个系数
w = lr.coef_
b = lr.intercept_
# print(w, b)
h = X_test.dot(w.T) + b
# 根据softmax数学公式，计算了类别的概率
p = np.e**h/((np.e**h).sum(axis=1).reshape(-1, 1))
print(p[:10])
print(proba_[:10])