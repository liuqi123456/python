#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/11 16:47
#@Author: Kevin.Liu
#@File  : extremeForest.py

# 极限森林

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# 葡萄酒
X, y = datasets.load_wine(True)

# 决策树
clf = DecisionTreeClassifier()
print(cross_val_score(clf, X, y, cv=6, scoring='accuracy').mean())

# 随机森林
forest = RandomForestClassifier(n_estimators=100)
print(cross_val_score(forest, X, y, cv=6, scoring='accuracy').mean())

# 极限森林
extra = ExtraTreesClassifier(n_estimators=100)
print(cross_val_score(extra, X, y, cv=6, scoring='accuracy').mean())

# 鸢尾花数据，特征只有4个，相对于葡萄酒 数据量简单
X, y = datasets.load_iris(True)

# 决策树
clf = DecisionTreeClassifier()
print(cross_val_score(clf, X, y, cv=6, scoring='accuracy').mean())

# 随机森林
forest = RandomForestClassifier(n_estimators=100)
print(cross_val_score(forest, X, y, cv=6, scoring='accuracy').mean())

# 极限森林
extra = ExtraTreesClassifier(n_estimators=100)
print(cross_val_score(extra, X, y, cv=6, scoring='accuracy').mean())