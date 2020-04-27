#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/11 14:42
#@Author: Kevin.Liu
#@File  : DTree.py

# 决策树

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
# print(iris)
X = iris['data']
y = iris['target']
feature_names = iris.feature_names
# print(X)
# print(y)
# print(feature_names)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1024)
# 树的深度变浅了，树的剪裁
clf = DecisionTreeClassifier(criterion='entropy', max_depth=2) # max_depth 最大深度
clf.fit(X_train, y_train)
y_ = clf.predict(X_test)
plt.figure(figsize=(12, 9))
print(accuracy_score(y_test, y_))
_ = tree.plot_tree(clf, filled=True, feature_names=feature_names)
plt.savefig('./Dtree.jpg')

# entropy
print("entropy", 39/120*np.log2(120/39) + 42/120*np.log2(120/42) + 39/120*np.log2(120/39))
print("entropy", 42/81*np.log2(81/42) + 39/81*np.log2(81/39))
# 连续的(continuous 属性)，阈值 threshold
print(X_train)
# 波动程度，越大越离散（越容易分开）
print(X_train.std(axis=0)) # 标准差