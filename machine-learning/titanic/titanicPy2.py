#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/26 20:45
#@Author: Kevin.Liu
#@File  : titanicPy2.py

# 1. 数据总览
# 2. 缺失值处理
# 3. 数据分析处理
# 4. 变量转换
# 5. 特征工程
# 6. 模型融合及测试
# 7. 学习曲线
# 8. 超参数调试

# from .missing_age import fill_missing_age
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')

def fill_missing_age(missing_age_train, missing_age_test):
    missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
    missing_age_Y_train = missing_age_train['Age']
    missing_age_X_test = missing_age_test.drop(['Age'], axis=1)

    # model 1  gbm
    gbm_reg = GradientBoostingRegressor(random_state=42)
    gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [4], 'learning_rate': [0.01], 'max_features': [3]}
    gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
    gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))
    print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))
    print('GB Train Error for "Age" Feature Regressor:' + str(gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_GB'][:4])

    # model 2 rf
    rf_reg = RandomForestRegressor()
    rf_reg_param_grid = {'n_estimators': [200], 'max_depth': [5], 'random_state': [0]}
    rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
    rf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('Age feature Best RF Params:' + str(rf_reg_grid.best_params_))
    print('Age feature Best RF Score:' + str(rf_reg_grid.best_score_))
    print('RF Train Error for "Age" Feature Regressor' + str(rf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_RF'] = rf_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_RF'][:4])

    # two models merge
    print('shape1', missing_age_test['Age'].shape, missing_age_test[['Age_GB', 'Age_RF']].mode(axis=1).shape)
    # missing_age_test['Age'] = missing_age_test[['Age_GB', 'Age_LR']].mode(axis=1)

    missing_age_test.loc[:, 'Age'] = np.mean([missing_age_test['Age_GB'], missing_age_test['Age_RF']])
    print(missing_age_test['Age'][:4])

    missing_age_test.drop(['Age_GB', 'Age_RF'], axis=1, inplace=True)

    return missing_age_test


def get_top_n_features(titanic_train_data_X, titanic_train_data_Y, top_n_features):
    # randomforest
    rf_est = RandomForestClassifier(random_state=0)
    rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
    rf_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
    print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
    print('Top N Features RF Train Score:' + str(rf_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': rf_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample 10 Feeatures from RF Classifier')
    print(str(features_top_n_rf[:10]))

    # AdaBoost
    ada_est = AdaBoostClassifier(random_state=0)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
    ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
    print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
    print('Top N Features Ada Train Score:' + str(ada_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),
                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 10 Features from Ada Classifier:')
    print(str(features_top_n_ada[:10]))

    # ExtraTree
    et_est = ExtraTreesClassifier(random_state=0)
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
    et_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best ET Params:' + str(et_grid.best_params_))
    print('Top N Features Best DT Score:' + str(et_grid.best_score_))
    print('Top N Features ET Train Score:' + str(et_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': et_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 10 Features from ET Classifier:')
    print(str(features_top_n_et[:10]))

    # GradientBoosting
    gb_est = GradientBoostingClassifier(random_state=0)
    gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
    gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=25, cv=10, verbose=1)
    gb_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best GB Params:' + str(gb_grid.best_params_))
    print('Top N Features Best GB Score:' + str(gb_grid.best_score_))
    print('Top N Features GB Train Score:' + str(gb_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_gb = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': gb_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']
    print('Sample 10 Feature from GB Classifier:')
    print(str(features_top_n_gb[:10]))

    # DecisionTree
    dt_est = DecisionTreeClassifier(random_state=0)
    dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
    dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=25, cv=10, verbose=1)
    dt_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Bset DT Params:' + str(dt_grid.best_params_))
    print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
    print('Top N Features DT Train Score:' + str(dt_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_dt = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': dt_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']
    print('Sample 10 Features from DT Classifier:')
    print(str(features_top_n_dt[:10]))

    # merge the three models
    features_top_n = pd.concat(
        [features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt],
        ignore_index=True).drop_duplicates()
    features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et,
                                     feature_imp_sorted_gb, feature_imp_sorted_dt], ignore_index=True)

    return features_top_n, features_importance

train_data = pd.read_csv('./titanic_train.csv')
test_data = pd.read_csv('./titanic_test.csv')
sns.set_style('whitegrid')
# print(train_data.head())
# print("-" * 40)
# print(train_data.info())
# print("-" * 40)
# print(test_data.info())

# labeldistance，文本的位置离远点有多远，1.1指1.1倍半径的位置
# autopct，圆里面的文本格式，%3.1f%%表示小数有三位，整数有一位的浮点数
# shadow，饼是否有阴影
# startangle，起始角度，0，表示从0开始逆时针转，为第一块。一般选择从90度开始比较好看
# pctdistance，百分比的text离圆心的距离
# patches, l_texts, p_texts，为了得到饼图的返回值，p_texts饼图内部文本的，l_texts饼图外label的文本
# train_data['Survived'].value_counts().plot.pie(labeldistance=1.1, autopct='%1.2f%%', shadow=False, startangle=90, pctdistance=0.6)
# plt.show()

# 对于缺失值，一般有以下几种处理方法:
# 如果数据集很多，但有很少的缺失值，可以删掉带缺失值的行；
# 如果该属性相对学习来说不是很重要，可以对缺失值赋均值或者众数。
# 对于标称属性，可以赋一个代表缺失的值，比如‘U0’。因为缺失本身也可能代表着一些隐含信息。比如船舱号Cabin这一属性，缺失可能代表并没有船舱。
train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values
# replace missing value with U0
train_data['Cabin'] = train_data.Cabin.fillna('U0')

#choose training data to predict age
age_df = train_data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_notnull = age_df.loc[(train_data['Age'].notnull())]
age_df_isnull = age_df.loc[(train_data['Age'].isnull())]
X = age_df_notnull.values[:, 1:]
Y = age_df_notnull.values[:, 0]
# use RandomForestRegression to train data
RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
RFR.fit(X, Y)
predictAges = RFR.predict(age_df_isnull.values[:, 1:])
train_data.loc[train_data['Age'].isnull(), ['Age']] = predictAges
# print(train_data.info())

# 分析数据关系
# 性别与是否生存的关系 Sex
# print(train_data.groupby(['Sex', 'Survived'])['Survived'].count())
# print(train_data[['Sex', 'Survived']].groupby(['Sex']).mean())
# train_data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar()
# plt.show()

# 船舱等级和生存与否的关系 Pclass
# print(train_data.groupby(['Pclass', 'Survived'])['Pclass'].count())
# print(train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean())
# train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean().plot.bar()
# plt.show()
# 不同等级船舱的男女生存率
# print(train_data.groupby(['Sex','Pclass','Survived'])['Survived'].count())
# train_data[['Sex', 'Pclass', 'Survived']].groupby(['Pclass', 'Sex']).mean().plot.bar()
# plt.show()

# 年龄与存活与否的关系 Age
# fig, ax = plt.subplots(1, 2, figsize=(18, 5))
# ax[0].set_yticks(range(0, 110, 10))
# sns.violinplot("Pclass", "Age", hue="Survived", data=train_data, split=True, ax=ax[0])
# ax[0].set_title('Pclass and Age vs Survived')
# ax[1].set_yticks(range(0, 110, 10))
# sns.violinplot("Sex", "Age", hue="Survived", data=train_data, split=True, ax=ax[1])
# ax[1].set_title('Sex and Age vs Survived')
# plt.show()
# # 分析总体的年龄分布
# plt.figure(figsize=(15, 5))
# plt.subplot(121)
# train_data['Age'].hist(bins=100)
# plt.xlabel('Age')
# plt.ylabel('Num')
# plt.subplot(122)
# train_data.boxplot(column='Age', showfliers=False)
# plt.show()
# # 不同年龄下的生存和非生存的分布情况
# facet = sns.FacetGrid(train_data,hue="Survived",aspect=4)
# facet.map(sns.kdeplot, 'Age', shade=True)
# facet.set(xlim=(0, train_data['Age'].max()))
# facet.add_legend()
# plt.show()
# # 不同年龄下的平均生存率
# # average survived passengers by age
# fig, axis1 = plt.subplots(1, 1, figsize=(18, 4))
# train_data['Age_int'] = train_data['Age'].astype(int)
# average_age = train_data[["Age_int", "Survived"]].groupby(['Age_int'], as_index=False).mean()
# sns.barplot(x='Age_int', y='Survived', data=average_age)
# plt.show()
# print(train_data['Age'].describe())
# # 按照年龄，将乘客划分为儿童、少年、成年、老年，分析四个群体的生还情况
# bins = [0, 12, 18, 65, 100]
# train_data['Age_group'] = pd.cut(train_data['Age'], bins)
# by_age = train_data.groupby('Age_group')['Survived'].mean()
# print(by_age)
# by_age.plot(kind='bar')
# plt.show()

# 称呼与存活与否的关系 Name
# train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# pd.crosstab(train_data['Title'], train_data['Sex'])
# train_data[['Title', 'Survived']].groupby(['Title']).mean().plot.bar()
# plt.show()
# # 对于名字，我们还可以观察名字长度和生存率之间存在关系的可能
# fig, axis1 = plt.subplots(1, 1, figsize=(18, 4))
# train_data['Name_length'] = train_data['Name'].apply(len)
# name_length = train_data[['Name_length', 'Survived']].groupby(['Name_length'], as_index=False).mean()
# sns.barplot(x='Name_length', y='Survived', data=name_length)
# plt.show()

# 有无兄弟姐妹和存活与否的关系 SibSp
# 将数据分为有兄弟姐妹和没有兄弟姐妹的两组：
# sibsp_df = train_data[train_data['SibSp'] != 0]
# no_sibsp_df = train_data[train_data['SibSp'] == 0]
# plt.figure(figsize=(11, 5))
# plt.subplot(121)
# sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct='%1.1f%%')
# plt.xlabel('sibsp')
# plt.subplot(122)
# no_sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct='%1.1f%%')
# plt.xlabel('no_sibsp')
# plt.show()
# # 有无父母子女和存活与否的关系 Parch
# parch_df = train_data[train_data['Parch'] != 0]
# no_parch_df = train_data[train_data['Parch'] == 0]
# plt.figure(figsize=(11, 5))
# plt.subplot(121)
# parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct='%1.2f%%')
# plt.xlabel('parch')
# plt.subplot(122)
# no_parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct='%1.2f%%')
# plt.xlabel('no_parch')
# plt.show()
# # 亲友的人数和存活与否的关系 SibSp & Parch
# fig, ax = plt.subplots(1, 2,figsize=(15, 5))
# train_data[['Parch', 'Survived']].groupby(['Parch']).mean().plot.bar(ax=ax[0])
# ax[0].set_title('Parch and Survived')
# train_data[['SibSp', 'Survived']].groupby(['SibSp']).mean().plot.bar(ax=ax[1])
# ax[1].set_title('SibSp and Survived')
# # 若独自一人，那么其存活率比较低；但是如果亲友太多的话，存活率也会很低。
# train_data['Family_Size'] = train_data['Parch'] + train_data['SibSp']+1
# train_data[['Family_Size', 'Survived']].groupby(['Family_Size']).mean().plot.bar()
# plt.show()

# 票价分布和存活与否的关系 Fare
# plt.figure(figsize=(10, 5))
# train_data['Fare'].hist(bins=70)
# train_data.boxplot(column='Fare', by='Pclass', showfliers=False)
# plt.show()
# # print(train_data['Fare'].describe())
# # 绘制生存与否与票价均值和方差的关系：
# fare_not_survived = train_data['Fare'][train_data['Survived'] == 0]
# fare_survived = train_data['Fare'][train_data['Survived'] == 1]
# average_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
# std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])
# average_fare.plot(yerr=std_fare, kind='bar', legend=False)
# plt.show()

# 船舱类型和存活与否的关系 Cabin
# # Replace missing values with "U0"
# train_data.loc[train_data.Cabin.isnull(),'Cabin'] = 'U0'
# train_data['Has_Cabin'] = train_data['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)
# train_data[['Has_Cabin','Survived']].groupby(['Has_Cabin']).mean().plot.bar()
# plt.show()
# # 对不同类型的船舱进行分析
# # create feature for the alphabetical part of the cabin number
# train_data['CabinLetter'] = train_data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
# # convert the distinct cabin letters with incremental integer values
# train_data['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]
# train_data[['CabinLetter','Survived']].groupby(['CabinLetter']).mean().plot.bar()
# plt.show()

# 港口和存活与否的关系 Embarked
# sns.countplot('Embarked', hue='Survived', data=train_data)
# plt.title('Embarked and Survived')
# plt.show()
# sns.factorplot('Embarked','Survived',data = train_data, size=3, aspect=2)
# plt.title('Embarked and Survived rate')
# plt.show()





# 变量转换
# 所有的数据可以分为两类
# 1.定性（Qualitative）变量可以以某种方式，Age就是一个很好的例子。
# 2.定量（Quantitative）变量描述了物体的某一（不能被数学表示的）方面，Embarked就是一个例子。
#               定性（Qualitative）转换
# embark_dummies = pd.get_dummies(train_data['Embarked'])
# train_data = train_data.join(embark_dummies)
# train_data.drop(['Embarked'], axis=1, inplace=True)
# embark_dummies = train_data[['S', 'C', 'Q']]
# print(embark_dummies.head())
# # Replace missing values with "U0"
# train_data['Cabin'][train_data.Cabin.isnull()] = 'U0'
# # create feature for the alphabetical part of the cabin number
# train_data['CabinLetter'] = train_data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
# # convert the distinct cabin letters with incremental integer values
# train_data['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]
# print(train_data[['Cabin', 'CabinLetter']].head())
# #               定量（Quantitative）转换
# assert np.size(train_data['Age']) == 891
# # Scaling可以将一个很大范围的数值映射到一个很小范围（通常是 -1到1，或者是0到1）
# # StandardScaler will subtract the mean from each value then scale to the unit varience
# scaler = preprocessing.StandardScaler()
# train_data['Age_scaled'] = scaler.fit_transform(train_data['Age'].values.reshape(-1, 1))
# print(train_data['Age_scaled'].head())
# # Binning通过观察“邻居”（即周围的值）将连续数据离散化。存储的值被分布到一些“桶”或“箱”中，就像直方图的bin将数据划分成几块一样。
# # Divide all fares into quartiles
# train_data['Fare_bin'] = pd.qcut(train_data['Fare'],5)
# print(train_data['Fare_bin'].head())
# # factorize
# train_data['Fare_bin_id'] = pd.factorize(train_data['Fare_bin'])[0]
# # dummies
# fare_bin_dummies_df = pd.get_dummies(train_data['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))
# train_data = pd.concat([train_data, fare_bin_dummies_df], axis=1)
# print(train_data)

# 特征工程
train_df_org = pd.read_csv('./titanic_train.csv')
test_df_org = pd.read_csv('./titanic_test.csv')
test_df_org['Survived'] = 0
combined_train_test = train_df_org.append(test_df_org)   # 891+418=1309rows, 12columns
PassengerId = test_df_org['PassengerId']
combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().iloc[0], inplace=True)
# 为了后面的特征分析，这里我们将Embarked特征进行factorizing
combined_train_test['Embarked'] = pd.factorize(combined_train_test['Embarked'])[0]
# 使用pd.get_dummies获取one-hot编码
emb_dummies_df = pd.get_dummies(combined_train_test['Embarked'], prefix=combined_train_test[['Embarked']].columns[0])
combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)
# 对Sex也进行one-hot编码，也就是dummy处理
# 为了后面的特征分析，这里我们也将Sex特征进行factorizing
combined_train_test['Sex'] = pd.factorize(combined_train_test['Sex'])[0]
sex_dummies_df = pd.get_dummies(combined_train_test['Sex'], prefix=combined_train_test[['Sex']].columns[0])
combined_train_test = pd.concat([combined_train_test, sex_dummies_df], axis=1)
# 首先从名字中提取各种称呼
# what is each person's title?
combined_train_test['Title'] = combined_train_test['Name'].map(lambda x: re.compile(",(.*?)\.").findall(x)[0])
combined_train_test['Title'] = combined_train_test['Title'].apply(lambda x: x.strip())
title_Dict = {}
title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
title_Dict.update(dict.fromkeys(['Male', 'Miss'], 'Miss'))
title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
combined_train_test['Title'] = combined_train_test['Title'].map(title_Dict)
# 为了后面的特征分析，这里我们也将Title特征进行factorizing
combined_train_test['Title'] = pd.factorize(combined_train_test['Title'])[0]
title_dummies_df = pd.get_dummies(combined_train_test['Title'], prefix=combined_train_test[['Title']].columns[0])
combined_train_test = pd.concat([combined_train_test, title_dummies_df], axis=1)
# 增加名字长度的特征
combined_train_test['Name_length'] = combined_train_test['Name'].apply(len)
# 下面transform将函数np.mean应用到各个group中
combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(combined_train_test.groupby('Pclass').transform(np.mean))
# 我们需要将团体票的票价分配到每个人的头上
combined_train_test['Group_Ticket'] = combined_train_test['Fare'].groupby(by=combined_train_test['Ticket']).transform('count')
combined_train_test['Fare'] = combined_train_test['Fare']/combined_train_test['Group_Ticket']
combined_train_test.drop(['Group_Ticket'], axis=1, inplace=True)
# 使用binning给票价分等级
combined_train_test['Fare_bin'] = pd.qcut(combined_train_test['Fare'], 5)
# 对于5个等级的票价我们可以继续使用dummy为票价等价分列
combined_train_test['Fare_bin'] = pd.qcut(combined_train_test['Fare'], 5)
combined_train_test['Fare_bin_id'] = pd.factorize(combined_train_test['Fare_bin'])[0]
fare_bin_dummies_df = pd.get_dummies(combined_train_test['Fare_bin_id']).rename(columns=lambda x: 'Fare_' + str(x))
combined_train_test = pd.concat([combined_train_test, fare_bin_dummies_df], axis=1)
combined_train_test.drop(['Fare_bin'], axis=1, inplace=True)
# Pclass这一项，其实已经可以不用继续处理了，我们只需将其转换为dummy形式即可
print(combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean())
# 建立Pclass Fare Category
def pclass_fare_category(df, pclass1_mean_fare, pclass2_mean_fare, pclass3_mean_fare):
    if df['Pclass'] == 1:
        if df['Fare'] <= pclass1_mean_fare:
            return 'Pclass1_Low'
        else:
            return 'Pclass1_High'
    elif df['Pclass'] == 2:
        if df['Fare'] <= pclass2_mean_fare:
            return 'Pclass2_Low'
        else:
            return 'Pclass2_High'
    elif df['Pclass'] == 3:
        if df['Fare'] <= pclass3_mean_fare:
            return 'Pclass3_Low'
        else:
            return 'Pclass3_High'
Pclass1_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get(1)
Pclass2_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get(2)
Pclass3_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get(3)
# 建立Pclass_Fare Category
combined_train_test['Pclass_Fare_Category'] = combined_train_test.apply(pclass_fare_category, args=(Pclass1_mean_fare, Pclass2_mean_fare, Pclass3_mean_fare), axis=1)
pclass_level = LabelEncoder()
# 给每一项添加标签
pclass_level.fit(np.array(['Pclass1_Low', 'Pclass1_High', 'Pclass2_Low', 'Pclass2_High', 'Pclass3_Low', 'Pclass3_High']))
# 转换成数值
combined_train_test['Pclass_Fare_Category'] = pclass_level.transform(combined_train_test['Pclass_Fare_Category'])
# dummy 转换
pclass_dummies_df = pd.get_dummies(combined_train_test['Pclass_Fare_Category']).rename(columns=lambda x: 'Pclass_' + str(x))
combined_train_test = pd.concat([combined_train_test, pclass_dummies_df], axis=1)
combined_train_test['Pclass'] = pd.factorize(combined_train_test['Pclass'])[0]
# 亲友的数量没有或者太多会影响到Survived。所以将二者合并为FamliySize这一组合项，同时也保留这两项。
def family_size_category(family_size):
    if family_size <= 1:
        return 'Single'
    elif family_size <= 4:
        return 'Small_Family'
    else:
        return 'Large_Family'
combined_train_test['Family_Size'] = combined_train_test['Parch'] + combined_train_test['SibSp'] + 1
combined_train_test['Family_Size_Category'] = combined_train_test['Family_Size'].map(family_size_category)
le_family = LabelEncoder()
le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
combined_train_test['Family_Size_Category'] = le_family.transform(combined_train_test['Family_Size_Category'])
family_size_dummies_df = pd.get_dummies(combined_train_test['Family_Size_Category'], prefix=combined_train_test[['Family_Size_Category']].columns[0])
combined_train_test = pd.concat([combined_train_test, family_size_dummies_df], axis=1)
# 因为Age项的缺失值较多，所以不能直接填充age的众数或者平均数。以Age为目标值，将Age完整的项作为训练集，将Age缺失的项作为测试集。
missing_age_df = pd.DataFrame(combined_train_test[
    ['Age', 'Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size', 'Family_Size_Category','Fare', 'Fare_bin_id', 'Pclass']])
missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]
print(missing_age_test.head())
combined_train_test.loc[(combined_train_test.Age.isnull()), 'Age'] = fill_missing_age(missing_age_train, missing_age_test)
print(missing_age_test.head())
# 我们将Ticket中的字母分开，为数字的部分则分为一类。
combined_train_test['Ticket_Letter'] = combined_train_test['Ticket'].str.split().str[0]
combined_train_test['Ticket_Letter'] = combined_train_test['Ticket_Letter'].apply(lambda x: 'U0' if x.isnumeric() else x)
# 如果要提取数字信息，则也可以这样做，现在我们对数字票单纯地分为一类。
# combined_train_test['Ticket_Number'] = combined_train_test['Ticket'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
# combined_train_test['Ticket_Number'].fillna(0, inplace=True)
# 将 Ticket_Letter factorize
combined_train_test['Ticket_Letter'] = pd.factorize(combined_train_test['Ticket_Letter'])[0]
# 特征信息的有无也与生存率有一定的关系，所以这里我们暂时保留该特征，并将其分为有和无两类。
combined_train_test.loc[combined_train_test.Cabin.isnull(), 'Cabin'] = 'U0'
combined_train_test['Cabin'] = combined_train_test['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)
# 挑选一些主要的特征，生成特征之间的关联图，查看特征与特征之间的相关性
Correlation = pd.DataFrame(combined_train_test[['Embarked','Sex','Title','Name_length','Family_Size',
                                                'Family_Size_Category','Fare','Fare_bin_id','Pclass',
                                                'Pclass_Fare_Category','Age','Ticket_Letter','Cabin']])
colormap = plt.cm.viridis
plt.figure(figsize=(14, 12))
plt.title('Pearson Correaltion of Feature', y=1.05, size=15)
sns.heatmap(Correlation.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()
# 特征之间的数据分布图
g = sns.pairplot(combined_train_test[[u'Survived', u'Pclass', u'Sex', u'Age', u'Fare', u'Embarked',
                                      u'Family_Size', u'Title', u'Ticket_Letter']], hue='Survived',
                                      palette='seismic', size=1.2, diag_kind='kde', diag_kws=dict(shade=True), plot_kws=dict(s=10))
g.set(xticklabels=[])
plt.show()
# 一些数据的正则化 这里我们将Age和fare进行正则化
scale_age_fare = preprocessing.StandardScaler().fit(combined_train_test[['Age', 'Fare', 'Name_length']])
combined_train_test[['Age','Fare','Name_length']] = scale_age_fare.transform(combined_train_test[['Age', 'Fare', 'Name_length']])
# 弃掉无用特征
combined_data_backup = combined_train_test
combined_train_test.drop(['PassengerId', 'Embarked','Sex', 'Name', 'Fare_bin_id', 'Pclass_Fare_Category',
                          'Parch', 'SibSp', 'Family_Size_Category', 'Ticket'], axis=1, inplace=True)
# 将训练数据和测试数据分开
train_data = combined_train_test[:891]
test_data = combined_train_test[891:]
titanic_train_data_X = train_data.drop(['Survived'], axis=1)
titanic_train_data_Y = train_data['Survived']
titanic_test_data_X = test_data.drop(['Survived'], axis=1)
print(titanic_train_data_X.shape)
print(titanic_train_data_X.info())

# 依据我们筛选出的特征构建训练集和测试集
feature_to_pick = 30
feature_top_n, feature_importance = get_top_n_features(titanic_train_data_X, titanic_train_data_Y, feature_to_pick)
titanic_train_data_X = pd.DataFrame(titanic_train_data_X[feature_top_n])
titanic_test_data_X = pd.DataFrame(titanic_test_data_X[feature_top_n])
# 用视图可视化不同算法筛选的特征排序
rf_feature_imp = feature_importance[:10]
Ada_feature_imp = feature_importance[32:32+10].reset_index(drop=True)
# make importances relative to max importance
rf_feature_importance = 100.0 * (rf_feature_imp['importance'] / rf_feature_imp['importance'].max())
Ada_feature_importance = 100.0 * (Ada_feature_imp['importance'] / Ada_feature_imp['importance'].max())
# Get the indexes of all features over the importance threshold
rf_important_idx = np.where(rf_feature_importance)[0]
Ada_important_idx = np.where(Ada_feature_importance)[0]
# Adapted from http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
pos = np.arange(rf_important_idx.shape[0]) + .5
plt.figure(1, figsize = (18, 8))
plt.subplot(121)
plt.barh(pos, rf_feature_importance[rf_important_idx][::-1])
plt.yticks(pos, rf_feature_imp['feature'][::-1])
plt.xlabel('Relative Importance')
plt.title('RandomForest Feature Importance')
plt.subplot(122)
plt.barh(pos, Ada_feature_importance[Ada_important_idx][::-1])
plt.yticks(pos, Ada_feature_imp['feature'][::-1])
plt.xlabel('Relative Importance')
plt.title('AdaBoost Feature Importance')
plt.show()