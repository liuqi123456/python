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
# print(X_normal1.head())

# Z-Score 归一化，标准化
X_normal2 = (X - X.mean()) / X.std()
# print(X_normal2.head())
print(X_normal2.mean())
# print(X_normal2.std())

X_train, X_test, y_train, y_test = train_test_split(X_normal2, y, test_size=0.2)

# 网格搜索GridSearchCV 进行最佳参数的查找
knn = KNeighborsClassifier()
params = {'n_neighbors': [i for i in range(1, 30)],
          'weights': ['uniform', 'distance'],
          'p': [1, 2]}
gcv = GridSearchCV(knn, params, scoring='accuracy', cv=6) # cv 数据分成6份
gcv.fit(X_train, y_train)

# 查看了GridSearchCV 最佳的参数组合
# print(gcv.best_params_) # 最佳参数
# print(gcv.best_estimator_) # 最佳估计量
# print(gcv.best_score_) # 最佳得分

# 直接使用gcv进行预测，结果一样， 计算准确率
# y_ = gcv.predict(X_test)
# print((y_ == y_test).mean())
# print(gcv.score(X_test, y_test))
# print(accuracy_score(y_test, y_)) #

# 取出了最好的模型，进行预测
knn_best = gcv.best_estimator_
y_ = knn_best.predict(X_test)
# print(accuracy_score(y_test, y_)) # 最佳得分

# print(pd.crosstab(index=y_test, columns=y_, rownames=['True'], colnames=['Predict'], margins=True))
# print(y_test.value_counts()) # 真实的数据
# print(Series(y_).value_counts()) # 预测的数据
# print(confusion_matrix(y_test, y_))
# print(np.round(6/9, 2))
# precision    recall  f1-score
# 精确率       召回率     f1-score调和平均值
print(classification_report(y_test, y_, target_names=['B', 'M']))