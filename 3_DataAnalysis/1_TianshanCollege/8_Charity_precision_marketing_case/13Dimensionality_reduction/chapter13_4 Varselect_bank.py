# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 19:53:24 2019

@author: dell
"""

# In[66]:
import os
os.chdir(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\1_TianshanCollege\8_Charity_precision_marketing_case\13Dimensionality_reduction")

import pandas as pd

model_data = pd.read_csv("profile_bank.csv") # (100000, 4)
data = model_data.loc[ :,'CNT_TBM':'CNT_CSC'] # DataFrame
k=3
alphaMax = 5
alphastep=0.2

# In[67]:
from sklearn import preprocessing
import numpy as np
from sklearn.decomposition import SparsePCA
from functools import reduce

# 做主成分之前，进行中心标准化
data = preprocessing.scale(data) # ndarray
n_components = k
pca_n = list()

# In[68]:
# step3:进行SparsePCA计算，选择合适的惩罚项alpha，当恰巧每个原始变量只在一个主成分上有权重时，停止循环
pca_model = SparsePCA(n_components=3, alpha=5)
pca_model.fit(data)

#%%
pca = pd.DataFrame(pca_model.components_).T
print(pca)
print(sum(np.array(pca != 0)))
print(sum(sum(np.array(pca != 0))))

#%%
'''
这里的意思是： 共4个特征，现在留下3个主成分，那么在做 稀疏主成分分析 时的 最优超参数是当：
1、其中2个主成分中 各只有一个特有值，其余3个特征为0；
2、而剩下的1个主成分中，就必须要有 2个特征有值，其余2个特征为0。
3、3个主成分中： 主成分1中有2个特征非0； 主成分2中只有1个特征非0； 主成分3中只有1个特征非0，加起来正好4个特征。
'''
n = data.shape[1] - sum(sum(np.array(pca != 0)))

# In[69]:
# step4:根据上一步得到的惩罚项的取值，估计SparsePCA，并得到稀疏主成分得分
best_alpha = 5

pca_model = SparsePCA(n_components=n_components, alpha=best_alpha)
pca_model.fit(data)

pca = pd.DataFrame(pca_model.components_).T
print(pca)

data = pd.DataFrame(data)

# fit_transform 得到降维后的数据
score = pd.DataFrame(pca_model.fit_transform(data))
print(score.iloc[0:10])

# In[70]:
# step6:计算 原始变量（特征） 与 主成分得分 之间的1-R方值
r = []
R_square = []
for xk in range(data.shape[1]):  # xk输入变量个数
    for paj in range(n_components):  # paj主成分个数
        r.append(abs(np.corrcoef(data.iloc[:, xk], score.iloc[:, paj])[0, 1]))
        r_max1 = max(r)
        r.remove(r_max1)
        r.append(-2)
        r_max2 = max(r)
        R_square.append((1 - r_max1 ** 2) / (1 - r_max2 ** 2))

R = abs(pd.DataFrame(np.array(r).reshape((data.shape[1], n_components))))
R_square = abs(pd.DataFrame(np.array(R_square).reshape((data.shape[1], n_components))))
var_list = []
print(R_square)

# In[71]:
# step7:每个主成分中，选出原始变量的1-R方值最小的。
for i in range(n_components):
    vmin = R_square[i].min()
    print(R_square[i])
    print(vmin)
    print(R_square[R_square[i] == min][i])
    var_list.append(R_square[R_square[i] == vmin][i].index)
    
news_ids =[]
for id in var_list:
    if id not in news_ids:
        news_ids.append(id)
print(news_ids)
orgdata = model_data.loc[:, 'CNT_TBM':'CNT_CSC']
data_vc = orgdata.iloc[:, np.array(news_ids).reshape(len(news_ids))]






