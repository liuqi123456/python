#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/15 13:57
#@Author: Kevin.Liu
#@File  : logisticRegression.py

# 逻辑斯蒂回归，用于分类而不是回归

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# print(LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1, 5, 10, 100]))

X, y = datasets.load_iris(True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lr = LogisticRegression()
lr.fit(X_train, y_train)
# 分类问题， 准确率96.6%
# print(lr.score(X_test, y_test))
# # 每个类别中4个属性
# print(X.shape)
# # 类别分成3类
# print(np.unique(y))
# # 截距
# print(lr.coef_)
# # 斜率
# print(lr.intercept_)
# 分类都是概率问题
# y_ = lr.predict(X_test)
# print(y_)

w = lr.coef_
b = lr.intercept_
# print(w, b)
proba_ = lr.predict_proba(X_test)
print(proba_)
# print(proba_.argmax(axis=1))

# # 计算概率，多分类问题
h = X_test.dot(w.T) + b
# y_ = X_test.dot(w.T) + b
# 逻辑斯蒂函数 = sigmoid函数
# 归一化操作和算法预测的概率完全一样
p = np.e**h/((np.e**h).sum(axis=1).reshape(-1, 1))
print(p)