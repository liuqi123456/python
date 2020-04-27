#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/10 10:15
#@Author: Kevin.Liu
#@File  : salaryKNN2.py

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# cv 数据分成6份
from sklearn.model_selection import cross_val_score, GridSearchCV
# KFold StratifiedKFold 将数据分成多少份
from sklearn.model_selection import KFold, StratifiedKFold

data = pd.read_csv('./salary.txt')
# print(data.head())
# print(data.columns)
# print(data.shape)

data.drop(['fnlwgt', 'education', 'capital-gain', 'capital-loss'],
          axis=1, inplace=True)
# print(data.head())
# print(data.shape)

X = data.iloc[:, 0:-1]
y = data['salary']
# print(X)
# print(y)

# workclass = X['workclass'].unique()
# print(workclass)
# print(np.argwhere(workclass == 'Local-gov')[0, 0])

# def convert(x):
#     return np.argwhere(workclass == x)[0, 0]
# X['workclass'] = X['workclass'].map(convert)
# print(X.head())

columns = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
for col in columns:
    u = X[col].unique()
    def convert(x):
        return np.argwhere(u == x)[0, 0]
    X[col] = X[col].map(convert)
# print(X.head)

knn = KNeighborsClassifier()
kFold = KFold(10)
accuracy = 0
for train, test in kFold.split(X, y):
    knn.fit(X.loc[train], y[train])
    acc = knn.score(X.loc[test], y[test])
    accuracy += acc / 10
print(accuracy)