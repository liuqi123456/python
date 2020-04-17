#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/17 10:31
#@Author: Kevin.Liu
#@File  : mian_numpy_vector.py

import numpy as np

if __name__ == '__main__':
    print(np.__version__)

    # np.array 基础
    list = [1, 2, 3]
    list[0] = 'Linear Algebra'
    print(list)

    vec = np.array([1, 2, 3])
    print(vec)
    # vec[0] = 'Linear Algebra' # ValueError: invalid literal for int() with base 10: 'Linear Algebra'
    vec[0] = 666
    print(vec)

    # np.array的创建
    print(np.zeros(5))
    print(np.ones(5))
    print(np.full(5, 666))

    # np.array的基本属性
    print(vec)
    print("size =", vec.size)
    print('size =', len(vec))
    print(vec[0])
    print(vec[-1])
    print(vec[0:2])
    print(type(vec[0:2]))

    # np.array的基本运算
    vec2 = np.array([4, 5, 6])
    print("{} + {} = {}".format(vec, vec2, vec + vec2)) # 加法
    print("{} - {} = {}".format(vec, vec2, vec - vec2)) # 减法
    print("{} * {} = {}".format(2, vec, 2 * vec)) # k*向量
    print("{} * {} = {}".format(vec, vec2, vec * vec2)) # 星乘 向量*向量
    print("{}.dot({}) = {}".format(vec, vec2, vec.dot(vec2))) # 点乘 向量*向量，再相加
    print(np.linalg.norm(vec)) # 向量的模 --> 默认参数(矩阵整体元素平方和开根号，不保留矩阵二维特性)
    print(vec / np.linalg.norm(vec)) # 向量的规范化 --> 即让向量的长度为1
    print(np.linalg.norm(vec / np.linalg.norm(vec))) # 向量的模

    # zero3 = np.zeros(3)
    # print(zero3 / np.linalg.norm(zero3)) #[nan nan nan]