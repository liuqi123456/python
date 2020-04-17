#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/17 10:32
#@Author: Kevin.Liu
#@File  : main_numpey_matrix.py

import numpy as np

if __name__ == '__main__':
    # 矩阵的创建
    A = np.array([[1, 2], [3, 4]])
    print(A)

    # 矩阵的属性
    print(A.shape)
    print(A.T)

    # 获取矩阵的元素
    print(A[1, 1])
    print(A[0])
    print(A[:, 0])
    print(A[1, :])

    # 矩阵的基本运算
    B = np.array([[5, 6], [7, 8]])
    print(A + B) # 加法
    print(A - B) # 减法
    print(10 * A) # K*矩阵
    print(A * 10) # 矩阵*K
    print(A * B) # 星乘表示矩阵内各对应位置相乘 --> 矩阵*矩阵
    print(A.dot(B)) # 点乘表示求矩阵内积

    p = np.array([10, 100])
    print(A + p)
    print(A + 1)
    print(A.dot(p)) # 点乘

    # 单位矩阵
    I = np.identity(2)
    print(I)
    print(A.dot(I))
    print(I.dot(A))

    # 逆矩阵
    invA = np.linalg.inv(A)
    print(invA)
    print(invA.dot(A))
    print(A.dot(invA))

    # C = np.array([[1, 2, 3], [4, 5, 6]])
    # print(np.linalg.inv(C)) # 只有方正才有可能存在逆矩阵