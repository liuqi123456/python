#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/16 10:48
#@Author: Kevin.Liu
#@File  : xgboost.py

from xgboost import XGBClassifier, XGBRFRegressor
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xbg
import pandas as pd

# X, y = datasets.load_iris(True)
# X_train, X_test, y_train, y_test = train_test_split(X, y)
#
# xcf = XGBClassifier(n_estimators=100)
# xcf.fit(X_train, y_train)
# print(xcf.score(X_test, y_test))
#
# forest = RandomForestClassifier(max_depth=1, n_estimators=100)
# forest.fit(X_train, y_train)
# print(forest.score(X_test, y_test))
#
# ada = AdaBoostClassifier(n_estimators=100)
# ada.fit(X_train, y_train)
# print(ada.score(X_test, y_test))
#
# gdbt = GradientBoostingClassifier(max_depth=1, n_estimators=100)
# gdbt.fit(X_train, y_train)
# print(gdbt.score(X_test, y_test))