#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/11 18:24
#@Author: Kevin.Liu
#@File  : gradientTree2.py

# 梯度下降
import matplotlib.pyplot as plt
import numpy as np

# 求解导数令导数=0，求解最小值
# 2*(x - 3)*1 + 2.5 = 0
# 2*x - 3.5 = 0
# x = 1.75

# 梯度下降求解最小值
result = []
# 导数函数
d = lambda x: (x - 3)**2 + 2.5*x -7.5
# 学习率
learning_rate = 0.1
# min_value 瞎蒙的值，方法以最快的速度找到最优解(梯度下降)
min_value = np.random.randint(-3, 5, size=1)[0]
result.append(min_value)
print('---------------------', min_value)
# 记录数据更新了
min_value_last = min_value + 0.1
# 容忍度(误差) tolerance
tol = 0.0001
count = 0
while True:
    # abs 绝对值
    if np.abs(min_value - min_value_last) < tol:
        break
    # 梯度下降
    min_value_last = min_value
    # 更新值：梯度下降
    min_value = min_value - learning_rate*d(min_value)
    result.append(min_value)
    count += 1
    print('+++++++++++++++++++%d' % (count), min_value)
print("====================", min_value)
f = lambda x: (x - 3)**2 + 2.5*x -7.5
# 画图
x = np.linspace(-2, 5, 10)
y = f(x)
plt.plot(x, y)
result = np.asarray(result)
plt.plot(result, f(result), '*')
plt.show()

# 梯度上升求解最大值
result = []
# 导数函数
d2 = lambda x: -(x - 3)**2 + 2.5*x -7.5
# 学习率
learning_rate = 0.1
# max_value 瞎蒙的值，方法以最快的速度找到最优解(梯度上升)
max_value = np.random.randint(2, 10, size=1)[0]
result.append(max_value)
print('---------------------', max_value)
# 记录数据更新了
max_value_last = max_value + 0.1
# 容忍度(误差/精确度) tolerance
tol = 0.0001
count = 0
while True:
    # abs 绝对值
    if np.abs(max_value - max_value_last) < tol:
        break
    # 梯度上升
    max_value_last = max_value
    # 更新值：梯度上升
    max_value = max_value + learning_rate*d2(max_value)
    result.append(max_value)
    count += 1
    print('+++++++++++++++++++%d' % (count), max_value)
print("====================", max_value)

f2 = lambda x: -(x - 3)**2 + 2.5*x -7.5
# 画图
plt.figure(figsize=(18, 12))
x = np.linspace(1, 10, 10)
y = f2(x)
plt.plot(x, y)
result = np.asarray(result)
plt.plot(result, f2(result), '*')
plt.show()