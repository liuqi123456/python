#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/15 15:23
#@Author: Kevin.Liu
#@File  : faceRecognition.py

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

faces = datasets.fetch_olivetti_faces()
X = faces.data
y = faces.target
images = faces.images
# print(X.shape, y.shape, images.shape)
# plt.figure(figsize=(2, 2))
# index = np.random.randint(0, 400, size=1)[0]
# img = images[index]
# plt.imshow(img, cmap=plt.cm.gray)
# plt.show()

# 将X(人脸数据)分成上下图片
X_up = X[:, :2048]
X_down = X[:, 2048:]
index = np.random.randint(0, 400, size=1)[0]

axes = plt.subplot(1, 3, 1)
up_face = X_up[index].reshape(32, 64)
axes.imshow(up_face, cmap='gray')

axes = plt.subplot(1, 3, 2)
down_face = X_down[index].reshape(32, 64)
axes.imshow(down_face, cmap='gray')

axes = plt.subplot(1, 3, 3)
face = X[index].reshape(64, 64)
axes.imshow(face, cmap='gray')
plt.show()
X = X_up.copy()
y = X_down.copy()
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30)
estimators = {}
estimators['linear'] = LinearRegression()
estimators['ridge'] = Ridge(alpha=0.1)
estimators['lasso'] = Lasso(alpha=1)
estimators['knn'] = KNeighborsRegressor(n_neighbors=5)
estimators['tree'] = DecisionTreeRegressor()
result = {}
for key, model in estimators.items():
    model.fit(X_train, y_train)
    y_ = model.predict(X_test) # 预测的下半张人脸
    result[key] = y_
# 结果可视化
for i in range(0, 10):
    # 第一列，上半张人脸
    axes = plt.subplot(10, 7, i*7+1) # 10行7列
    up_face = X_test[i].reshape(32, 64)
    axes.imshow(up_face, cmap=plt.cm.gray)
    axes.axis('off')
    if i == 0:
        axes.set_title('up_face')

    for j ,key in enumerate(result):
        axes = plt.subplot(10, 7, i*7+2+j)
        y_ = result[key]
        predict_down_face = y_[i].reshape(32, 64)
        predict_face = np.concatenate([up_face, predict_down_face])
        axes.imshow(predict_face, cmap=plt.cm.gray)
        axes.axis('off')
        if i == 0:
            axes.set_title(key)

    # # 第七列，整张人脸
    axes = plt.subplot(10, 7, i*7+7)# 10行7列
    down_face = y_test[i].reshape(32, 64)
    true_face = np.concatenate([up_face, down_face])
    axes.imshow(true_face, cmap=plt.cm.gray)
    axes.axis('off')
    if i == 0:
        axes.set_title('true_face')
plt.show()