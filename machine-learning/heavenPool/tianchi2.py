#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/22 17:56
#@Author: Kevin.Liu
#@File  : tianchi2.py

from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('./zhengqi_train.txt', sep='\t')
test = pd.read_csv('./zhengqi_test.txt', sep='\t')
train['origin'] = 'train'
test['origin'] = 'test'
data_all = pd.concat([train, test])

# 删除4列
drop_labels = ['V5', 'V11', 'V17', 'V22']
data_all.drop(drop_labels, axis=1, inplace=True)

# 相关性系数-根据相关性系统再删除两列
drop_labels = ['V14', 'V21']
data_all.drop(drop_labels, axis=1, inplace=True)

# data = data_all.iloc[:, :-2]
# stand = StandardScaler()
# data2 = stand.fit_transform(data)
# cols = data_all.columns
# data_all_std = pd.DataFrame(data2, columns=cols[:-2])
# print(data_all_std.shape)
# data_all_index = np.arange(4813)
# data_all_std = pd.merge(data_all_std, data_all.iloc[:, :-2], right_index=True, left_index=True)
# print(data_all_std.columns)

data_all_std = pd.DataFrame(data_all)
ridgeCV = RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50])
cond = data_all_std['origin'] == 'train'
X_train = data_all_std[cond].iloc[:, :-2]
# 真实值
y_train = data_all_std[cond]['target']
ridgeCV.fit(X_train, y_train)
# 预测值，预测值肯定会和真实值有一定的偏差，偏差特别大，当成异常值
y_ = ridgeCV.predict(X_train)
# print(y_train[:5])
# print(y_[:5])

cond = abs(y_train - y_) > y_train.std()
print(cond.sum())

# 画图
# plt.figure(figsize=(12, 6))
# axis = plt.subplot(1, 3, 1)
# axis.scatter(y_train, y_)
# axis.scatter(y_train[cond], y_[cond], c='r', s=20)
#
# axis = plt.subplot(1, 3, 2)
# axis.scatter(y_train, y_train - y_)
# axis.scatter(y_train[cond], (y_train - y_)[cond], c='r')
#
# axis = plt.subplot(1, 3, 3)
# axis.hist(y_train, bins=50)
# plt.show()

# 将异常值点过滤
print(data_all_std.shape)
drop_index = cond[cond].index
data_all_std.drop(drop_index, axis=0, inplace=True)
print(data_all_std.shape)

def detect_model(estimators, data):
    for key, estimator in estimators.items():
        estimator.fit(data[0],  data[2])
        y_ = estimator.predict(data[1])
        mse = mean_squared_error(data[3], y_)
        print("---------------------mse %s" % key, mse)
        r2 = estimator.score(data[1], data[3])
        print("+++++++++++++++++++++r2_score %s" % key, r2)

# cond = data_all_std['origin'] == 'train'
# X = data_all_std[cond].iloc[:, :-2]
# y = data_all_std[cond]['target']
# data = train_test_split(X, y, test_size=0.2)

# estimators = {}
# estimators['knn'] = KNeighborsRegressor()
# estimators['linear'] = LinearRegression()
# estimators['ridge'] = Ridge()
# estimators['lasso'] = Lasso()
# estimators['elasticNet'] = ElasticNet()
# estimators['forest'] = RandomForestRegressor()
# estimators['gbdt'] = GradientBoostingRegressor()
# estimators['ada'] = AdaBoostRegressor()
# estimators['extree'] = ExtraTreesRegressor()
# estimators['svm_rbf'] = SVR(kernel='rbf')
# estimators['svm_poly'] = SVR(kernel='poly')
# estimators['xgb'] = XGBRegressor()
# estimators['light'] = LGBMRegressor()
# detect_model(estimators, data)

# 对于我们测试数据而言：KNN, Lasso, ElasticNet, svm_poly
estimators = {}
estimators['linear'] = LinearRegression()
estimators['ridge'] = Ridge()
estimators['forest'] = RandomForestRegressor()
estimators['gbdt'] = GradientBoostingRegressor()
estimators['ada'] = AdaBoostRegressor()
estimators['extree'] = ExtraTreesRegressor()
estimators['svm_rbf'] = SVR(kernel='rbf')
estimators['xgb'] = XGBRegressor()
estimators['light'] = LGBMRegressor()
# detect_model(estimators, data)

cond = data_all_std['origin'] == 'train'
X_train = data_all_std[cond].iloc[:, :-2]
y_train = data_all_std[cond]['target']
cond = data_all_std['origin'] == 'test'
X_test = data_all_std[cond].iloc[:, :-2]

# # 一个算法预测结果，将结果合并
# y_pred = []
# for key, model in estimators.items():
#     model.fit(X_train, y_train)
#     y_ = model.predict(X_test)
#     y_pred.append(y_)
# y_ = np.mean(y_pred, axis=0)
# pd.Series(y_).to_csv('./ensemble.txt', index=False)
#
# # 预测的结果作为特征，让我们的算法学习，寻找数据和目标值之间的关系
# # y_预测值，和真实值之间差距，将预测值当成新的特征，让我们算法进行再学习
# for key, model in estimators.items():
#     model.fit(X_train, y_train)
#     y_ = model.predict(X_train)
#     X_train[key] = y_
#     y_ = model.predict(X_test)
#     X_test[key] = y_
# print(X_train.head())
# # 一个算法预测结果，将结果合并
# y_pred = []
# for key, model in estimators.items():
#     model.fit(X_train, y_train)
#     y_ = model.predict(X_test)
#     y_pred.append(y_)
# y_ = np.mean(y_pred, axis=0)
# pd.Series(y_).to_csv('./ensemble2.txt', index=False)
#
# # print(y_.shape) # (1907,)
# y_ += np.random.randn(1907)*0.1
# pd.Series(y_).to_csv('./ensemble3.txt', index=False)

# 对数据进行归一化
# 4个测试和训练特征分布不均匀，2个相关性系数小的特征
data = data_all.iloc[:, :-2]
minmaxScaler = MinMaxScaler()
data3 = minmaxScaler.fit_transform(data)
data_all_norm = pd.DataFrame(data3, columns=data_all.columns[:-2])
print(data_all_norm.shape)
data_all_norm = pd.merge(data_all_norm, data_all.iloc[:, -2:], left_index=True, right_index=True)
print(data_all_norm.shape)

def minmax_scale(data):
    return (data - data.min()) / (data.max() - data.min())

fcols = 6
frows = len(data_all_norm.columns[:10])
plt.figure(figsize=(4*fcols, 4*frows))
i = 0
# 绘制前10个
for col in data_all_norm.columns[:10]:
    dat = data_all_norm[[col, 'target']].dropna()

    # 第一个图 这条线就是数据分布dist: distribution(分布)
    i += 1
    plt.subplot(frows, fcols, i)
    sns.distplot(dat[col], fit=stats.norm)
    plt.title(col+'Original')
    plt.xlabel('')
    # 第二个图 skew统计分析中一个属性: skewness 偏斜系数，对正太分布的度量
    i += 1
    plt.subplot(frows, fcols, i)
    _ = stats.probplot(dat[col], plot=plt)
    plt.title('skew='+'{:.4f}'.format(stats.skew(dat[col])))
    plt.xlabel('')
    plt.ylabel('')
    # 第三个图 散点图
    i += 1
    plt.subplot(frows, fcols, i)
    # plt.plot(dat[var], dat['target'], '.', alpha=0.5)
    plt.scatter(dat[col], dat['target'], alpha=0.5)
    plt.title('corr=' + '{:.2f}'.format(np.corrcoef(dat[col], dat['target'])[0][1]))

    # !!! 对数据又进行了处理 !!!
    trans_var, lambda_var = stats.boxcox(dat[col].dropna() + 1)
    trans_var = minmax_scale(trans_var)
    # 第四个图 数据分布图
    i += 1
    plt.subplot(frows, fcols, i)
    sns.distplot(trans_var, fit=stats.norm)
    plt.title(col + 'Tramsformed')
    plt.xlabel('')
    # 第五个图 偏斜度
    i += 1
    plt.subplot(frows, fcols, i)
    _ = stats.probplot(trans_var, plot=plt)
    plt.title('skew=' + '{:.4f}'.format(stats.skew(trans_var)))
    plt.xlabel('')
    plt.ylabel('')
    # # 第六个图 散点图
    i += 1
    plt.subplot(frows, fcols, i)
    plt.plot(trans_var, dat['target'], '.', alpha=0.5)
    # plt.scatter(trans_var, dat['target'], alpha=0.5)
    plt.title('corr=' + '{:.2f}'.format(np.corrcoef(trans_var, dat['target'])[0][1]))
plt.show()