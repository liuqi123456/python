#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/16 14:53
#@Author: Kevin.Liu
#@File  : myLightGbm.py

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('./zhengqi_train.txt', sep='\t')
test = pd.read_csv('./zhengqi_test.txt', sep='\t')
X_train = train.iloc[:, :-1]
y_train = train['target']
# print(X_train.shape, y_train.shape)

# light = LGBMRegressor()
# light.fit(X_train, y_train)
# y_ = light.predict(test)
# pd.Series(y_).to_csv('./myLightGbmLight.txt', index=False)

# xgb = XGBRegressor(max_depth=3, n_estimators=100)
# xgb.fit(X_train, y_train)
# y_ = xgb.predict(test)
# pd.Series(y_).to_csv('./myLightGbmXgb.txt', index=False)

# 协方差：两个属性之间得关系
# 协方差绝对值越大，两个属性之间的关系越密切
cov = train.cov()
# 目标值和属性之间的关系怎样
# print(cov)

print(X_train.shape)
print(test.shape)
# 不重要的参数
drop_labels = cov.index[cov.loc['target'].abs() < 0.1]
print(drop_labels)
X_train.drop(drop_labels, axis=1, inplace=True)
print(X_train.shape)
test.drop(drop_labels, axis=1, inplace=True)
print(test.shape)

light = LGBMRegressor()
light.fit(X_train, y_train)
y_ = light.predict(test)
pd.Series(y_).to_csv('./myLightGbmLight2.txt', index=False)

xgb = XGBRegressor(max_depth=3, n_estimators=100)
xgb.fit(X_train, y_train)
y_ = xgb.predict(test)
pd.Series(y_).to_csv('./myLightGbmXgb2.txt', index=False)
