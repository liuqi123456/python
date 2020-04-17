#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/3/30 11:25
#@Author: Kevin.Liu
#@File  : Matrix.py

from .Vector import Vector

class Matrix:
    def __init__(self, list2d):
        if isinstance(list2d[0], list):
            self._values = [row[:] for row in list2d]
        elif isinstance(list2d[0], Vector):
            self._values = [row.underlying_list() for row in list2d]

    @classmethod
    def zero(cls, r, c):
        """返回一个r行c列的零矩阵"""
        return cls([[0] * c for _ in range(r)])

    @classmethod
    def identity(cls, n):
        """返回一个n行n列的单位矩阵"""
        m = [[0] * n for _ in range(n)]
        for i in range(n):
            m[i][i] = 1
        return cls(m)

    def T(self):
        """返回矩阵的转置矩阵"""
        return Matrix([[e for e in self.col_vector(i)] for i in range(self.col_num())])

    def __add__(self, other):
        """返回两个矩阵的加法结果"""
        assert self.shape() == other.shape(), "Error in adding. Shape of matrix must be same."
        return Matrix([[a + b for a, b in zip(self.row_vector(i), other.row_vector(i))] for i in range(self.row_num())])

    def __sub__(self, other):
        """返回两个矩阵的减法结果"""
        assert self.shape() == other.shape(), "Error in subtracting. Shape of matrix must be same."
        return Matrix([[a - b for a, b in zip(self.row_vector(i), other.row_vector(i))] for i in range(self.row_num())])

    def dot(self, other):
        """返回矩阵乘法的结果"""
        if isinstance(other, Vector):
            # 矩阵和向量的乘法
            assert self.col_num() == len(other), "Error in Matrix-Vector Multiplication."
            return Vector([self.row_vector(i).dot(other) for i in range(self.row_num())])
        if isinstance(other,Matrix):
            # 矩阵和矩阵的乘法
            assert self.col_num() == other.row_num(), "Error in Matrix-Matrix Multiplication."
            return Matrix([[self.row_vector(i).dot(other.col_vector(j)) for j in range(other.col_num())] for i in range(self.row_num())])

    def __mul__(self, k):
        """返回矩阵的数量乘结果: self * k"""
        return Matrix([[e * k for e in self.row_vector(i)] for i in range(self.row_num())])

    def __rmul__(self, k):
        """返回矩阵的数量乘结果: k * self"""
        return self * k

    def __truediv__(self, k):
        """返回数量除法的结果矩阵：self / k"""
        return (1 / k) * self

    def __pos__(self):
        """返回矩阵取正的结果"""
        return 1 * self

    def __neg__(self):
        """返回矩阵取负的结果"""
        return -1 * self

    def row_vector(self, index):
        """返回矩阵的第index个行向量"""
        return Vector(self._values[index])

    def col_vector(self, index):
        """返回矩阵的第index个列向量"""
        return Vector([row[index] for row in self._values])

    def __getitem__(self, pos):
        """返回矩阵pos位置的元素"""
        r, c = pos
        return self._values[r][c]

    def size(self):
        """返回矩阵的元素个数"""
        r, c = self.shape()
        return r * c

    def row_num(self):
        """返回矩阵的行数"""
        return self.shape()[0]

    __len__ = row_num

    def col_num(self):
        """返回矩阵的列数"""
        return self.shape()[1]

    def shape(self):
        """返回矩阵的形状: (行数， 列数)"""
        return len(self._values), len(self._values[0])

    def __repr__(self):
        return "Matrix({})".format(self._values)

    __str__ = __repr__