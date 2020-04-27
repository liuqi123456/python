#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/26 20:19
#@Author: Kevin.Liu
#@File  : titanicPy.py

# 泰坦尼克号乘客生存预测

# 数据特征
# 字段	字段说明
# PassengerId	乘客编号
# Survived	存活情况（存活：1 ; 死亡：0）
# Pclass	客舱等级
# Name	乘客姓名
# Sex	性别
# Age	年龄
# SibSp	同乘的兄弟姐妹/配偶数
# Parch	同乘的父母/小孩数
# Ticket	船票编号
# Fare	船票价格
# Cabin	客舱号
# Embarked	登船港口
# PassengerId 是数据唯一序号；Survived 是存活情况，为预测标记特征；剩下的10个是原始特征数据。

from pandas import Series,DataFrame
import pandas as pd
import numpy as np
#线性回归
from sklearn.linear_model import LinearRegression
# 训练集交叉验证，得到平均值
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

# 数据处理
data_train = pd.read_csv("./titanic_train.csv")
data_test = pd.read_csv("./titanic_test.csv")
# print(data_train.head(10))
# print(data_test.head(10))
# print(data_train.info())
# print(data_train.describe())

# 特征选取
# 数据空值处理
# 1.客舱号Cabin列由于存在大量的空值，如果直接对空值进行填空，带来的误差影响会比较大，先不选用Cabin列做特征
# 2.年龄列对于是否能够存活的判断很重要，采用Age均值对空值进行填充
# 3.PassengerId是一个连续的序列，对于是否能够存活的判断无关，不选用PassengerId作为特征
# Age列中的缺失值用Age中位数进行填充
data_train["Age"] = data_train['Age'].fillna(data_train['Age'].median())
print(data_train.describe())

# 选取简单的可用输入特征
predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
# 初始化现行回归算法
alg = LinearRegression()
# 样本平均分成3份，3折交叉验证
kf = KFold(n_splits=3, shuffle=False, random_state=1)
predictions = []
for train, test in kf.split(data_train):
    train_predictors = (data_train[predictors].iloc[train, :])
    train_target = data_train["Survived"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(data_train[predictors].iloc[test, :])
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0
accuracy = sum(predictions == data_train["Survived"]) / len(predictions)
print("准确率为: ", accuracy)

# 初始化逻辑回归算法
logRegAlg = LogisticRegression(random_state=1)
re = logRegAlg.fit(data_train[predictors], data_train["Survived"])
# 使用sklearn库里面的交叉验证函数获取预测准确率分数
scores = model_selection.cross_val_score(logRegAlg, data_train[predictors], data_train["Survived"], cv=3)
# 使用交叉验证分数的平均值作为最终的准确率
print("准确率为: ", scores.mean())

# Sex性别列处理：male用0，female用1
data_train.loc[data_train["Sex"] == "male", "Sex"] = 0
data_train.loc[data_train["Sex"] == "female", "Sex"] = 1
# 缺失值用最多的S进行填充
data_train["Embarked"] = data_train["Embarked"].fillna('S')
# 地点用0,1,2
data_train.loc[data_train["Embarked"] == "S", "Embarked"] = 0
data_train.loc[data_train["Embarked"] == "C", "Embarked"] = 1
data_train.loc[data_train["Embarked"] == "Q", "Embarked"] = 2

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
logRegAlg = LogisticRegression(random_state=1)
re = logRegAlg.fit(data_train[predictors], data_train["Survived"])
scores = model_selection.cross_val_score(logRegAlg, data_train[predictors], data_train["Survived"], cv=3)
print("准确率为: ", scores.mean())

# 新增：对测试集数据进行预处理，并进行结果预测
# Age列中的缺失值用Age均值进行填充
data_test["Age"] = data_test["Age"].fillna(data_test["Age"].median())
# Fare列中的缺失值用Fare最大值进行填充
data_test["Fare"] = data_test["Fare"].fillna(data_test["Fare"].max())
# Sex性别列处理：male用0，female用1
data_test.loc[data_test["Sex"] == "male", "Sex"] = 0
data_test.loc[data_test["Sex"] == "female", "Sex"] = 1
# 缺失值用最多的S进行填充
data_test["Embarked"] = data_test["Embarked"].fillna('S')
# 地点用0,1,2
data_test.loc[data_test["Embarked"] == "S", "Embarked"] = 0
data_test.loc[data_test["Embarked"] == "C", "Embarked"] = 1
data_test.loc[data_test["Embarked"] == "Q", "Embarked"] = 2
test_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# 构造测试集的Survived列，
data_test["Survived"] = -1
test_predictors = data_test[test_features]
data_test["Survived"] = logRegAlg.predict(test_predictors)
print(data_test.head(10))

# 使用随机森林算法
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# 10棵决策树，停止的条件：样本个数为2，叶子节点个数为1
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
kf = model_selection.KFold(n_splits=3, shuffle=False, random_state=1)
scores = model_selection.cross_val_score(alg, data_train[predictors], data_train["Survived"], cv=kf)
print(scores)
print(scores.mean())

# 增加决策树的个数到30棵决策树，交叉验证方法采用10折交叉验证
alg = RandomForestClassifier(random_state=1, n_estimators=30, min_samples_split=2, min_samples_leaf=1)
kf = model_selection.KFold(n_splits=10, shuffle=False, random_state=1)
scores = model_selection.cross_val_score(alg, data_train[predictors], data_train["Survived"], cv=kf)
print(scores)
print(scores.mean())