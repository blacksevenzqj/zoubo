# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:31:39 2019

@author: dell
"""

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os
os.chdir(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\3_Feature_engineering")

# In[]:
data = pd.read_csv(r".\digit recognizor.csv")
X = data.iloc[:,1:]
y = data.iloc[:,0]
X.shape

# In[]:
# 画 累计方差贡献率 曲线，找最佳降维后维度的范围
'''
大幅度转折点 发生在 降维后特征维度0至100之间。
'''
pca_line = PCA().fit(X)
# explained_variance_： 解释方差
print(len(pca_line.explained_variance_)) # 建议保留2个主成分
# explained_variance_ratio_： 解释方差占比（累计解释方差占比 自己手动加）
print(len(pca_line.explained_variance_ratio_)) #建议保留2个主成分

plt.figure(figsize=[20,5])
plt.plot(np.cumsum(pca_line.explained_variance_ratio_))
plt.xlabel("number of components after dimension reduction")
plt.ylabel("cumulative explained variance ratio")
plt.show()

# In[]:
# 降维后维度的学习曲线，继续缩小最佳维度的范围
'''
在 降维后特征维度0至100之间 分别进行交叉验证，准确率得分最高点在 降维后特征维度10至24之间
'''
#======【TIME WARNING：2mins 30s】======#
score = []
for i in range(1,101,10):
    X_dr = PCA(i).fit_transform(X)
    once = cross_val_score(RFC(n_estimators=10,random_state=0)
                           ,X_dr,y,cv=5).mean()
    score.append(once)
plt.figure(figsize=[20,5])
plt.plot(range(1,101,10),score)
plt.show()

# In[]:
# 细化学习曲线，找出降维后的最佳维度
'''
在 降维后特征维度10至24之间 分别进行交叉验证，准确率得分最高点在 降维后特征维度20至22之间
'''
score = []
for i in range(10,25):
    X_dr = PCA(i).fit_transform(X)
    once = cross_val_score(RFC(n_estimators=10,random_state=0),X_dr,y,cv=5).mean()
    score.append(once)
plt.figure(figsize=[20,5])
plt.plot(range(10,25),score)
plt.show()

# In[]:
# 细化学习曲线，找出降维后的最佳维度
'''
在 降维后特征维度20至22之间 分别进行交叉验证，准确率得分最高点在 降维后特征维度21
'''
score = []
for i in range(20,23):
    X_dr = PCA(i).fit_transform(X)
    once = cross_val_score(RFC(n_estimators=10,random_state=0),X_dr,y,cv=5).mean()
    score.append(once)
plt.figure(figsize=[20,5])
plt.plot(range(20,23),score)
plt.show()

# In[]:
# 细化学习曲线，找出降维后的最佳维度
'''
在 降维后特征维度20至22之间 并加大随机森林数量100（进一步减小泛化误差，进一步提升模型准确率）分别进行交叉验证，准确率得分最高点在 降维后特征维度22    
'''
score = [] # [0.9430475839429189, 0.9426907585778522, 0.946714832275006]
for i in range(20,23):
    X_dr = PCA(i).fit_transform(X)
    once = cross_val_score(RFC(n_estimators=100,random_state=0),X_dr,y,cv=5).mean()
    score.append(once)
plt.figure(figsize=[20,5])
plt.plot(range(20,23),score)
plt.show()

# In[]:
# 最终使用 降维后特征维度22，随机森林数量100
X_dr = PCA(22).fit_transform(X)
cross_val_score(RFC(n_estimators=100,random_state=0),X_dr,y,cv=5).mean() # 0.9453573095013784


# In[]:
# 使用KNN模型： 因特征维度已经降到了22维，使用KNN
from sklearn.neighbors import KNeighborsClassifier as KNN
cross_val_score(KNN(),X_dr,y,cv=5).mean() # KNN()的值不填写默认=5  0.9687374580540895

# In[]:
# 画KNN中k值的学习曲线： 最高准确率分数在k=3时
score = []
for i in range(10):
    X_dr = PCA(22).fit_transform(X)
    once = cross_val_score(KNN(i+1),X_dr,y,cv=5).mean()
    score.append(once)
plt.figure(figsize=[20,5])
plt.plot(range(10),score)
plt.show()

# In[]:
# 最终参数组合： 降维后特征维度22； KNN的k=3
X_dr = PCA(22).fit_transform(X)
cross_val_score(KNN(3),X_dr,y,cv=5).mean() # 0.9687611768884323



