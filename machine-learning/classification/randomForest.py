#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/11 16:27
#@Author: Kevin.Liu
#@File  : randomForest.py

# 随机森林

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

wine = datasets.load_wine()
# print(wine)
X = wine['data']
y = wine['target']
# print(X.shape)
# print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1024)
# 随机森林
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_ = clf.predict(X_test)
print(accuracy_score(y_test, y_))

# 决策树
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
print(dt_clf.score(X_test, y_test))