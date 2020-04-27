#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/8 22:55
#@Author: Kevin.Liu
#@File  : knn1.py

# 电影识别

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# 分类：电影分类
# 动作 武打镜头：碟中谍6、杀死比尔
# 爱情 接吻镜头：泰坦尼克号
# 属性：武打镜头、接吻镜头
# 金融量化

movie = pd.read_excel('./movie.xlsx', sheet_name=0)
data = movie.iloc[:, 1:3]
print(data)
target = movie['分类情况']
print(target)

#算法，训练
knn = KNeighborsClassifier(n_neighbors=5)

#训练学习算法，知道 数据和目标 什么关系
knn.fit(data, target)

# 预测，使用
X_test = pd.DataFrame({'武打镜头':[100, 67, 1], '接吻镜头':[3, 2, 10]})

#计算机根据规则返回结果
print(knn.predict(X_test))