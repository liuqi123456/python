#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/16 10:48
#@Author: Kevin.Liu
#@File  : xgboost.py

from xgboost import XGBClassifier, XGBRegressor
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
import xgboost as xbg
import pandas as pd

train = pd.read_csv('./zhengqi_train.txt', sep='\t')
X = train.iloc[:, 0:-1]
y = train['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 过拟合：训练数据很好，测试数据不行
lr = LinearRegression()
lr.fit(X_train, y_train)
# print("r2_core : ", lr.score(X_test, y_test))
y_ = lr.predict(X_test)
pd.Series(y_).to_csv('./myXgboost2Linear.txt', index=False)
print("均方误差 : ", mean_squared_error(y_test, y_))

ada = AdaBoostRegressor()
ada.fit(X_train, y_train)
y_ = ada.predict(X_test)
pd.Series(y_).to_csv('./myXgboost2Ada.txt', index=False)
print("均方误差 : ", mean_squared_error(y_test, y_))

xgb = XGBRegressor()
xgb.fit(X_train, y_train)
y_ = xgb.predict(X_test)
pd.Series(y_).to_csv('./myXgboost2XGB.txt', index=False)
print("均方误差 : ", mean_squared_error(y_test, y_))

gbdt = GradientBoostingRegressor()
gbdt.fit(X_train, y_train)
y_ = gbdt.predict(X_test)
pd.Series(y_).to_csv('./myXgboost2Gbdt.txt', index=False)
print("均方误差 : ", mean_squared_error(y_test, y_))

forest = RandomForestRegressor()
forest.fit(X_train, y_train)
y_ = forest.predict(X_test)
pd.Series(y_).to_csv('./myXgboost2Forest.txt', index=False)
print("均方误差 : ", mean_squared_error(y_test, y_))
