#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/26 19:41
#@Author: Kevin.Liu
#@File  : redWine.py

# 红酒数据集分析

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 颜色
color = sns.color_palette()
# 数据print精度
pd.set_option('precision', 3)
# 读取数据
df = pd.read_csv('./winequality-red.csv', sep=';')
# print(df.head(5))
# 查看数据的信息，包括每个字段的名称、非空数量、字段的数据类型
# print(df.info())

# 1.单变量分析
# # 简单的数据统计, 用统计学指标快速描述数据的概要
# # print(df.describe())
# # set plot style
# plt.style.use('ggplot')
# colnm = df.columns.tolist()
# fig = plt.figure(figsize=(10, 6))
# for i in range(12):
#     plt.subplot(2, 6, i+1)
#     sns.boxplot(df[colnm[i]], orient="v", width=0.5, color=color[0])
#     plt.ylabel(colnm[i], fontsize=12)
# # plt.subplots_adjust(left=0.2, wspace=0.8, top=0.9)
# plt.tight_layout()
# plt.show()
# print('\nFigure 1: Univariate Boxplots')
#
# colnm = df.columns.tolist()
# plt.figure(figsize=(10, 8))
# for i in range(12):
#     plt.subplot(4, 3, i+1)
#     df[colnm[i]].hist(bins=100, color=color[0])
#     plt.xlabel(colnm[i], fontsize=12)
#     plt.ylabel('Frequency')
# plt.tight_layout()
# plt.show()
# print('\nFigure 2: Univariate Histograms')
#
# acidityFeat = ['fixed acidity', 'volatile acidity', 'citric acid',
#                'free sulfur dioxide', 'total sulfur dioxide', 'sulphates']
# plt.figure(figsize=(10, 4))
# for i in range(6):
#     ax = plt.subplot(2, 3, i+1)
#     v = np.log10(np.clip(df[acidityFeat[i]].values, a_min=0.001, a_max=None))
#     plt.hist(v, bins=50, color=color[0])
#     plt.xlabel('log(' + acidityFeat[i] + ')', fontsize=12)
#     plt.ylabel('Frequency')
# plt.tight_layout()
# plt.show()
# print('\nFigure 3: Acidity Features in log10 Scale')
#
# plt.figure(figsize=(6, 3))
# bins = 10**(np.linspace(-2, 2))
# plt.hist(df['fixed acidity'], bins=bins, edgecolor='k', label='Fixed Acidity')
# plt.hist(df['volatile acidity'], bins=bins, edgecolor='k', label='Volatile Acidity')
# plt.hist(df['citric acid'], bins=bins, edgecolor='k', alpha=0.8, label='Citric Acid')
# plt.xscale('log')
# plt.xlabel('Acid Concentration (g/dm^3)')
# plt.ylabel('Frequency')
# plt.title('Histogram of Acid Concentration')
# plt.legend()
# plt.tight_layout()
# plt.show()
# print('\nFigure 4')
#
# # 总酸度
# df['total acid'] = df['fixed acidity'] + df['volatile acidity'] + df['citric acid']
# plt.figure(figsize=(8, 3))
# plt.subplot(121)
# plt.hist(df['total acid'], bins=50, color=color[0])
# plt.xlabel('total acid')
# plt.ylabel('Frequency')
# plt.subplot(122)
# plt.hist(np.log(df['total acid']), bins=50 , color=color[0])
# plt.xlabel('log(total acid)')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.show()
# print("\nFigure 5: Total Acid Histogram")
#
# # Residual sugar
# df['sweetness'] = pd.cut(df['residual sugar'], bins=[0, 4, 12, 45], labels=["dry", "medium dry", "semi-sweet"])
# plt.figure(figsize=(5, 3))
# df['sweetness'].value_counts().plot(kind='bar', color=color[0])
# plt.xticks(rotation=0)
# plt.xlabel('sweetness', fontsize=12)
# plt.ylabel('Frequency', fontsize=12)
# plt.tight_layout()
# plt.show()
# print("\nFigure 6: Sweetness")
#
# # 2.双变量分析
# sns.set_style('ticks')
# sns.set_context("notebook", font_scale=1.1)
# colnm = df.columns.tolist()[:11] + ['total acid']
# plt.figure(figsize=(10, 8))
# for i in range(12):
#     plt.subplot(4, 3, i+1)
#     sns.boxplot(x='quality', y=colnm[i], data=df, color=color[1], width=0.6)
#     plt.ylabel(colnm[i], fontsize=12)
# plt.tight_layout()
# plt.show()
# print("\nFigure 7: Physicochemical Properties and Wine Quality by Boxplot")
#
# sns.set_style("dark")
# plt.figure(figsize=(10, 8))
# colnm = df.columns.tolist()[:11] + ['total acid', 'quality']
# mcorr = df[colnm].corr()
# mask = np.zeros_like(mcorr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
# g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
# plt.show()
# print("\nFigure 8: Pairwise Correlation Plot")
#
# # style
# sns.set_style('ticks')
# sns.set_context("notebook", font_scale=1.4)
# plt.figure(figsize=(6,4))
# sns.regplot(x='density', y='alcohol', data=df, scatter_kws={'s': 10}, color=color[1])
# plt.xlim(0.989, 1.005)
# plt.ylim(7, 16)
# plt.show()
# print('\nFigure 9: Density vs Alcohol')
#
# acidity_related = ['fixed acidity', 'volatile acidity', 'total sulfur dioxide',
#                    'sulphates', 'total acid']
# plt.figure(figsize=(10, 6))
# for i in range(5):
#     plt.subplot(2, 3, i+1)
#     sns.regplot(x='pH', y=acidity_related[i], data=df, scatter_kws={'s': 10}, color=color[1])
# plt.tight_layout()
# plt.show()
# print("\nFigure 10: pH vs acid")

# 3.多变量分析
# 与品质相关性最高的三个特征是酒精浓度，挥发性酸度，和柠檬酸。下面图中显示的酒精浓度，挥发性酸和品质的关系。
# 酒精浓度，挥发性酸和品质
# 对于好酒（7，8）以及差酒（3，4），关系很明显。但是对于中等酒（5，6），酒精浓度的挥发性酸度有很大程度的交叉。
plt.style.use('ggplot')
sns.lmplot(x='alcohol', y='volatile acidity', hue='quality', data=df, fit_reg=False, scatter_kws={'s': 10}, size=5)
plt.show()
print("\nFFigure 11-1: Scatter Plots of Alcohol, Volatile Acid and Quality")

sns.lmplot(x='alcohol', y='volatile acidity', col='quality', hue='quality', data=df, fit_reg=False, size=3,  aspect=0.9, col_wrap=3, scatter_kws={'s': 20})
plt.show()
print("\nFigure 11-2: Scatter Plots of Alcohol, Volatile Acid and Quality")

# pH，非挥发性酸，和柠檬酸
# pH和非挥发性的酸以及柠檬酸有相关性。整体趋势也很合理，即浓度越高，pH越低。
# style
sns.set_style('ticks')
sns.set_context("notebook", font_scale=1.4)
plt.figure(figsize=(6, 5))
cm = plt.cm.get_cmap('RdBu')
sc = plt.scatter(df['fixed acidity'], df['citric acid'], c=df['pH'], vmin=2.6, vmax=4, s=15, cmap=cm)
bar = plt.colorbar(sc)
bar.set_label('pH', rotation=0)
plt.xlabel('fixed acidity')
plt.ylabel('citric acid')
plt.xlim(4, 18)
plt.ylim(0, 1)
plt.show()
print('\nFigure 12: pH with Fixed Acidity and Citric Acid')

# 总结
# 整体而言，红酒的品质主要与酒精浓度，挥发性酸，和柠檬酸有关。对于品质优于7，或者劣于4的酒，直观上是线性可分的。但是品质为5，6的酒很难线性区分。