# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 23:20:48 2019

@author: dell
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import FeatureTools as ft

# In[]:
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# In[]:
X_train_pd = pd.DataFrame(X_train)
X_test_pd = pd.DataFrame(X_test)
# In[]:
for i in X_train_pd.columns:
    f, axes = plt.subplots(1, 2, figsize=(23, 8))
    ft.con_data_distribution(X_train_pd, i, axes)

# In[]:
# 非线性转换
# 1、分位数变换（百分位秩）：映射到均匀分布
quantile_transformer = QuantileTransformer(random_state=0)
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)

# print(np.sort(X_train[:, 0]))
'''
np.percentile()是对数据进行分位数处理，即X_train[:, 0]先选择了X_train的第一列所有数据，然后np.percentile()选择了
排序后的0%、25%、50%、75%和100%的数据元素，总的来说，这就是一种抽查数据的手段
'''
print(np.percentile(X_train[:, 0], [0, 25, 50, 75, 100]))
print(np.percentile(X_train_trans[:, 0], [0, 25, 50, 75, 100]))

print(np.percentile(X_test[:, 0], [0, 25, 50, 75, 100]))
print(np.percentile(X_test_trans[:, 0], [0, 25, 50, 75, 100]))

# In[]:
print("============================================================================================================")

# In[]:
# 2、幂变换（Tukey正态分布打分）：映射到高斯分布
# 2.1、QuantileTransformer(output_distribution='normal'
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)

print(X_train_trans)
print(quantile_transformer.quantiles_)
# In[]:
X_train_trans_pd = pd.DataFrame(X_train_trans)
# In[]:
for i in X_train_trans_pd.columns:
    f, axes = plt.subplots(1, 2, figsize=(23, 8))
    ft.con_data_distribution(X_train_trans_pd, i, axes)

# In[]:
# 2.2、Box-Cox变换：
# 上述例子设置了参数standardize的选项为 False 。 但是，默认情况下，类PowerTransformer将会应用zero-mean,unit-variance normalization到变换出的输出上。
pt = PowerTransformer(method='box-cox', standardize=True)
X_train_box = pt.fit_transform(X_train)
X_test_box = pt.transform(X_test)
# In[]:
X_train_box_pd = pd.DataFrame(X_train_box)
# In[]:
for i in X_train_box_pd.columns:
    f, axes = plt.subplots(1, 2, figsize=(23, 8))
    ft.con_data_distribution(X_train_box_pd, i, axes)

# In[]:
# 2.3、Yeo-Johnson变换：（不是很清楚）
# 上述例子设置了参数standardize的选项为 False 。 但是，默认情况下，类PowerTransformer将会应用zero-mean,unit-variance normalization到变换出的输出上。
pt = PowerTransformer(method='yeo-johnson', standardize=True)
X_train_yeo = pt.fit_transform(X_train)
X_test_yeo = pt.transform(X_test)
# In[]:
X_train_yeo_pd = pd.DataFrame(X_train_yeo)
# In[]:
for i in X_train_yeo_pd.columns:
    f, axes = plt.subplots(1, 2, figsize=(23, 8))
    ft.con_data_distribution(X_train_yeo_pd, i, axes)

# In[]:
# 2.4、Box-Cox变换：（基于scipy库的Box-Cox：scipy.special.boxcox1p）
from scipy.special import boxcox1p

x = np.array([-0.25, -1e-20, 0, 1e-20, 0.25, 1, 3])

# In[]:
# lambda = 0  =>  y = log(1+x)
y = boxcox1p(x, 0)
y, np.log1p(x)

# In[]:
# lambda = 1  =>  y = x
y = boxcox1p(x, 1)
z = ((x + 1) ** 1 - 1) / 1
y, x, z

# In[]:
# lambda = 2  =>  y = 0.5*((1+x)**2 - 1) = 0.5*x*(2 + x)
y = boxcox1p(x, 2)
z = ((x + 1) ** 2 - 1) / 2
y, 0.5 * x * (2 + x), z

# In[]:
# 取lambda = 0  =>  y = log(1+x) = np.log1p(x) 还比较接近 正太分布
X_train_box1p = boxcox1p(X_train_pd, 0)
X_test_box1p = boxcox1p(X_test_pd, 0)
# In[]:
for i in X_train_box1p.columns:
    f, axes = plt.subplots(1, 2, figsize=(23, 8))
    ft.con_data_distribution(X_train_box1p, i, axes)