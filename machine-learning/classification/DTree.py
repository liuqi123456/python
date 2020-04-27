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

X, y = datasets.load_iris(True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1024)
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)
y_ = clf.predict(X_test)
score = accuracy_score(y_test, y_)
print(score)