#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/9 0:35
#@Author: Kevin.Liu
#@File  : number.py

# 数字识别

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 实例1
# digit = cv2.imread('./training_img/0_0.png')
# (32, 32, 3)
# print(digit.shape)
# 将彩色(三维的)图片转化成黑白的(图片灰度化处理)，大大降低了数据量
# digit = cv2.cvtColor(digit, code=cv2.COLOR_BGR2GRAY)
# (32, 32, 3) ---> (32, 32) 数据量大大减少了2/3，只有原来的1/3
# print(digit.shape)
# digit二维的，高度 宽度，像素(只有一个值)
# plt.imshow(digit, cmap=plt.cm.gray)
# plt.show()

# 实例2
# 加载数据，处理(灰度化)
X = []
for i in range(10):
    for j in range(1, 51):
        path = "./training_img/%d_%d.png" % (i, j)
        digit = cv2.imread(path)
        # print(path)
        # print(digit)
        X.append(digit[:, :, 0])
# X 和 y 一一对应
X = np.asarray(X) # numpy对象
y = np.array([i for i in range(10)] * 50)
y.sort()
# print(X.shape)
# print(y.shape)

# index = np.random.randint(0, 500, size=1)[0]
# digit = X[index]
# print("------------------", y[index])
# plt.imshow(digit, cmap=plt.cm.gray)
# plt.show()

# X,y 划分数据 训练和验证
# 训练 0.8  验证 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 算法训练和预测
knn = KNeighborsClassifier(n_neighbors=5)
X_train = X_train.reshape(400, -1) # 三维数据---> 二维数据
knn.fit(X_train, y_train)

# 使用算法进行预测
X_test = X_test.reshape(100, -1)# 三维数据---> 二维数据
y_ = knn.predict(X_test)

# 准确率
# print((y_ == y_test).sum() / 100)
# print((y_ == y_test).mean())

# 打印出图片
# index = np.random.randint(0, 500, size=1)[0]
# digit = X[index]
# print("------------------", y[index])
# plt.imshow(digit, cmap=plt.cm.gray)
# plt.show()

# 实例3
# 将数据二值化操作
# print(X.shape)
for i in range(500):
    for y in range(32):
        for x in range(32):
            if X[i][y, x] < 200:
                X[i][y, x] = 0
            else:
                X[i][y, x] = 255
y = np.array([i for i in range(10)] * 50)
y.sort()

# print(X.shape)
# print(y.shape)

# 打印出图片
# index = np.random.randint(0, 500, size=1)[0]
# digit = X[index]
# print("------------------", y[index])
# plt.imshow(digit, cmap=plt.cm.gray)
# plt.show()

# X,y 划分数据 训练和验证
# 训练 0.8  验证 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# 算法训练和预测
knn = KNeighborsClassifier(n_neighbors=5)
X_train = X_train.reshape(400, -1) # 三维数据---> 二维数据
knn.fit(X_train, y_train)

# 使用算法进行预测
X_test = X_test.reshape(100, -1)# 三维数据---> 二维数据
y_ = knn.predict(X_test)

# 准确率
# print((y_ == y_test).sum() / 100)
print((y_ == y_test).mean())