#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/3/30 12:54
#@Author: Kevin.Liu
#@File  : _globals.py

EPSILON = 1e-8

def is_zero(x):
    return abs(x) < EPSILON

def is_equal(a, b):
    return abs(a - b) < EPSILON