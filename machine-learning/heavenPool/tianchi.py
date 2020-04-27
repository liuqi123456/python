#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/22 15:57
#@Author: Kevin.Liu
#@File  : tianchi.py

# 天池项目
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('./zhengqi_train.txt', sep='\t')
test = pd.read_csv('./zhengqi_test.txt', sep='\t')
# print(train.shape, test.shape)
train['origin'] = 'train'
test['origin'] = 'test'
# print(train.shape, test.shape)
data_all = pd.concat([train, test])
# print(data_all.shape)
# print(data_all.head())
# print(data_all.tail())
# print(data_all.columns[:-2])

# 38个特征，将一些不重要的特征删除
# 例如：根据特征分布情况，训练和测试数据特征分布不均匀，删除
# plt.figure(figsize=(9, 38*6))
# for i, col in enumerate(data_all.columns[:-2]):
#     cond = data_all['origin'] == 'train'
#     train_col = data_all[col][cond] # 训练数据
#     cond = data_all['origin'] == 'test'
#     test_col = data_all[col][cond] # 测试数据
#     axes = plt.subplot(38, 1, i+1)
#     ax = sns.kdeplot(train_col, shade=True, ax=axes)
#     sns.kdeplot(test_col, shade=True, ax=ax)
#     plt.legend(['train', 'test'])
#     plt.xlabel(col)
# plt.show()

# 删除4列
drop_labels = ['V5', 'V11', 'V17', 'V22']
data_all.drop(drop_labels, axis=1, inplace=True)
# print(data_all.shape)

# 协方差
# 协方差是两个属性之间的关系，如果两个属性一样就是方差
# 方差
# 方差是协方差的一种特殊形式
# cov = train.cov()
# print(cov.head())

# 相关性系数，通过相关性系数找到了7个相关性不大的属性
corr = train.corr()
cond = corr.loc['target'].abs() < 0.1
# Index(['V14', 'V21', 'V25', 'V26', 'V32', 'V33', 'V34'], dtype='object')
drop_labels = corr.loc['target'].index[cond]
# 查看了属性的分布，分布不好的删除
drop_labels = ['V14', 'V21']
data_all.drop(drop_labels, axis=1, inplace=True)
# print(data_all.shape)

# 找出相关程度
# plt.figure(figsize=(20, 16)) # 指定绘图对象宽度和高度
# mcorr = train.corr() # 相关性系数矩阵，即给出了任意两个变量之间的相关系数
# mask = np.zeros_like(mcorr, dtype=np.bool) # 构建与mcorr同维度矩阵，为booble类型
# mask[np.triu_indices_from(mask)] = True # 角分线右侧为True（右三角形）
# cmap = sns.diverging_palette(220, 10, as_cmap=True) # 返回matplotlib colormap对象
# # 热力图(看两两相似度)
# g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
# plt.show()

# plt.figure(figsize=(9, 2*6))
# for col in data_all.columns[:2]:
#     g = sns.FacetGrid(data_all, col='origin')
#     g.map(sns.distplot, col)
# plt.show()