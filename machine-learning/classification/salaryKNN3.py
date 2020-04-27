#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/10 14:52
#@Author: Kevin.Liu
#@File  : salaryKNN3.py

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

salary = pd.read_csv('./salary.txt')
# print(salary.head())

salary.drop(['fnlwgt', 'education', 'capital-gain', 'capital-loss'],
          axis=1, inplace=True)
# print(salary)

ordinalEncoder = OrdinalEncoder()
data = ordinalEncoder.fit_transform(salary)
# print(data)

salary_ordinal = DataFrame(data, columns=salary.columns)
# print(salary_ordinal.head())

labelEncoder = LabelEncoder()
# salary_label = labelEncoder.fit_transform(salary['salary'])
# print(salary_label)
education_label = labelEncoder.fit_transform(salary['education-num'])
# print(education_label)
# for column in salary.columns:
#     salary[column] = labelEncoder.fit_transform(salary[column])
# print(salary.head())

education = salary[['education-num']]
# print(education)
# print(education.drop_duplicates().count())

oneHotEncoder = OneHotEncoder()

oneHot = oneHotEncoder.fit_transform(education)
# print(oneHot)
nd1 = oneHot.toarray()[:10]
# print(nd1)
print(nd1.argmax(axis=1))