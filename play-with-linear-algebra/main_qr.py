#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/17 10:18
#@Author: Kevin.Liu
#@File  : main_qr.py

from playLA.Matrix import Matrix
from playLA.GramSchmidtProcess import qr

if __name__ == '__main__':
    A1 = Matrix([[1, 1, 2], [1, 1, 0], [1, 0, 0]])
    Q1, R1 = qr(A1)
    print(Q1)
    print(R1)
    print(Q1.dot(R1))
    print()

    A2 = Matrix([[2, -1, -1], [2, 0, 2], [2, -1, 3]])
    Q2, R2 = qr(A2)
    print(Q2)
    print(R2)
    print(Q2.dot(R2))