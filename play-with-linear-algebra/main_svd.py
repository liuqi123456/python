#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/17 10:12
#@Author: Kevin.Liu
#@File  : mian_svd.py

from scipy.linalg import svd
import numpy as np

if __name__ == '__main__':
    A = np.array([[1, 2], [3, 4], [5, 6]])
    U, s, VT = svd(A)
    print(U)
    print(s)
    print(VT)
    print()

    sigma = np.zeros(A.shape)
    for i in range(len(s)):
        sigma[i][i] = s[i]
    print(sigma)
    print(U.dot(sigma).dot(VT))