#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/10 9:39
#@Author: Kevin.Liu
#@File  : salaryKNN.py

# train_test_split KFold StratifiedKFold 作用都是将数据拆分

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# cv 数据分成6份
from sklearn.model_selection import cross_val_score, GridSearchCV
# KFold StratifiedKFold 将数据分成多少份
from sklearn.model_selection import KFold, StratifiedKFold

data = np.random.randint(0, 10, size=(8, 2))
target = np.array([0, 0, 1, 0, 1, 1, 1, 0])
# print(data)
# print(target)

# print(train_test_split(data, target))

# 分成4份
# kFold = KFold(n_splits=4)
# # train test 是索引，只要有索引就可以获取数据
# for train, test in kFold.split(data, target):
#     print(target[train], target[test])

# 分成4份，每一份数据特征(数据样本)比例和原来一样的
skFold = StratifiedKFold(n_splits=4)
# train test 是索引，只要有索引就可以获取数据
for train, test in skFold.split(data, target):
    print(target[train], target[test])