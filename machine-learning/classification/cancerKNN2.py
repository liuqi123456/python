#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/9 18:39
#@Author: Kevin.Liu
#@File  : cancerKNN.py

# 癌症分析

from sklearn.neighbors import KNeighborsClassifier
from pandas import Series, DataFrame
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# 搜索算法最合适的参数
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


cancer = pd.read_csv('./Cancer_Prostate_Cancer.csv')
# print(cancer)

cancer.drop('id', axis=1, inplace=True)
# print(cancer)

X = cancer.iloc[:, 1:]
# print(X.head())

y = cancer['diagnosis_result']
# print(y.head())

# 归一化操作
X_normal1 = (X - X.min())/(X.max() - X.min())
print(X_normal1.head())

# Z-Score 归一化，标准化
X_normal2 = (X - X.mean()) / X.std()
print(X_normal2.head())
# print(X_normal2.mean())
# print(X_normal2.std())

# AttributeError: 'DataFrame' object has no attribute 'get_value' TODO
# nd = X.get_values()
# print((nd - nd.mean(axis=0))/nd.std(axis=0))

# MinMaxScaler 和最大最小值 归一化效果一样
mms = MinMaxScaler()
mms.fit(X)
X2 = mms.transform(X)
print(X2.round(6))

# print(classification_report(y_test, y_, target_names=['B', 'M']))