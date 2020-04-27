#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/11 14:21
#@Author: Kevin.Liu
#@File  : entropy.py

# 熵的计算

import numpy as np

# 账号是否真实 no: 0.3 yes: 0.7
info_D = 0.3*np.log2(1/0.3) + 0.7*np.log2(1/0.7)
# print(info_D)

# 决策树，对目标进行划分
# 三个属性：日志密度，好友密度，是否真实头像

# 使用日志密度进行构建
# s 0.3 ---> no:2, yes:1
# m 0.4 ---> no:1, yes:3
# l 0.3 ---> yes:3
info_L_D = 0.3*(2/3*np.log2(3/2) + 1/3*np.log2(3/1)) + \
           0.4*(1/4*np.log2(4/1) + 3/4*np.log2(4/3)) + \
           0.3*(3/3*np.log2(3/3))
print(info_L_D)
# 信息增益
info_G_L = info_D - info_L_D
print(info_G_L)

# 使用好友密度进行构建
# s 0.3 ---> no:3, yes:1
# m 0.4 ---> yes:4
# l 0.2 ---> yes:2
info_F_D = 0.4*(3/4*np.log2(4/3) + 1/4*np.log2(4/1)) + \
           0.4*0 + \
           0.2*0
print(info_F_D)
# 信息增益
info_G_F = info_D - info_F_D
print(info_G_F)

# 使用是否真实头像进行构建
# no 0.5 ---> no:2, yes:3
# yew 0.5 ---> no:1, yes:4
info_I_D = 0.5*(2/5*np.log2(5/2) + 3/5*np.log2(5/3)) + \
           0.5*(1/5*np.log2(5/1) + 4/5*np.log2(5/4))
print(info_I_D)
# 信息增益
info_G_I = info_D - info_I_D
print(info_G_I)