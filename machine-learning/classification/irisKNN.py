#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/8 23:26
#@Author: Kevin.Liu
#@File  : irisKNN.py

# 鸢尾花分类

from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import pandas as pd
import numpy as np

iris = datasets.load_iris()
# print(iris)
X = iris['data']
y = iris['target']
# 150个样本 4个属性：花萼长、宽 花瓣长、宽
# print(X.shape)
# 鸢尾花的名称
# print(iris.target_names)

# 初始化数据
index = np.arange(150)
# print(index)

# 打乱顺序
np.random.shuffle(index)
# print(index)

# 将数据划分 一分为二 用于训练 用户测试
X_train, X_test = X[index[:100]], X[index[100:]]
y_train, y_test = y[index[:100]], y[index[-50:]]

#算法，训练
# n_neighbors ，邻居不能太多
# weights 权重
# p = 1, 采用曼哈顿距离
# p = 2, 采用欧氏距离
# n_jobs = -1，满进程运行
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', p=1, n_jobs=-1)
#训练学习算法，知道 数据和目标 什么关系
knn.fit(X_train, y_train)
#计算机根据规则返回结果
y_ = knn.predict(X_test)
print(y_)

# 命名规则 _结尾 表示算法返回的
proba_ = knn.predict_proba(X_test)
# print(proba_)
# 返回最大值得索引，鸢尾花的类别
# print(proba_.argmax(axis = 1))

# 准确率
# print(knn.score(X_test, y_test))

# 对比看算法的 预测和真实的结果 是否对应
# 对应，大部分正确
# 否则 说明算法的效果不好
# print(y_)
# print("---------------------")
# print(y_test)

# 准确率
# print((y_ == y_test).sum()/50)