# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:40:02 2019

@author: 非均匀尺度随机均匀采样
"""

from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from time import time
import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler

# In[]:
# 1、生成 0-1 之间的随机数
# 1.1、指定生成随机数矩阵: 产生均匀分布的随机数
a = np.random.rand(100000,1) # 10行2列矩阵： 100个随机数
fig, axe = plt.subplots(1,1,figsize=(15,10))
sns.distplot(a, bins=100, color='green', ax=axe)

# 1.2、随机数列表： 产生均匀分布的随机数
b = np.random.random(100000) # 列表： 50个随机数
fig, axe = plt.subplots(1,1,figsize=(15,10))
sns.distplot(b, bins=100, color='green', ax=axe)

# In[]:
# 2、指定生成范围：
# 2.1、整数: 产生均匀分布的随机数
a = np.random.randint(0,100000,(100000,1)) # 0到10的范围， 4行3列矩阵： 如果只是一个数字，则生成列表
fig, axe = plt.subplots(1,1,figsize=(15,10))
sns.distplot(a, bins=100, color='green', ax=axe)

# 2.2、浮点数: 产生均匀分布的随机数
b = np.random.uniform(0,100000,(100000,1)) # 0到10的范围， 4行3列矩阵： 如果只是一个数字，则生成列表
fig, axe = plt.subplots(1,1,figsize=(15,10))
sns.distplot(b, bins=100, color='green', ax=axe)

# In[]:
# 3、产生标准正态分布随机数
a = np.random.randn(100000,1)
fig, axe = plt.subplots(1,1,figsize=(15,10))
sns.distplot(a, bins=100, color='green', ax=axe)

b = np.random.standard_normal([100000])
fig, axe = plt.subplots(1,1,figsize=(15,10))
sns.distplot(b, bins=100, color='green', ax=axe)



# In[]:
# 非均匀尺度随机均匀采样
# 数据间隔太大，直方图不能很好表示，只能分桶
# 90%的采样点分布在[0.1, 1]之间，只有10%分布在[0.0001, 0.1]之间
a = 0.0001
b = 1
num = 100000
bins = [0.0001, 0.001, 0.01, 0.1, 1]

result = np.random.uniform(a, b, num)

score_cut = pd.cut(result, bins)
print(score_cut.value_counts())

# In[]:
# 10**np.log10(0.0001) = 0.0001, 10**np.log10(1) = 1
result = np.logspace(np.log10(a), np.log10(b), num, base=10) # 默认10为底

score_cut = pd.cut(result, bins)
print(score_cut.value_counts())

# In[]:
m = np.log10(a)
n = np.log10(b)
r = np.random.rand(num,1) # 生成 0-1 之间的随机数： 产生均匀分布的随机数
r = m + (n-m)*r # 2维
result = np.power(10,r).ravel()

score_cut = pd.cut(result, bins)
print(score_cut.value_counts())

# In[]:
m = np.log10(a)
n = np.log10(b)
r = np.random.uniform(m, n, num) # 浮点数: 产生均匀分布的随机数
result = np.power(10,r)

score_cut = pd.cut(result, bins)
print(score_cut.value_counts())






