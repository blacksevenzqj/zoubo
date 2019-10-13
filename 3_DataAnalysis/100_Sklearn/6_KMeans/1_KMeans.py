# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:48:36 2019

@author: dell
"""

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 自定义聚类数据集
X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)
print(type(X), type(y))

fig, ax1 = plt.subplots(2,1,figsize=(10,10))
ax1[0].scatter(X[:, 0], X[:,1], marker='o', s=8)

color = ["red","pink","orange","gray"]
for i in range(4):
    ax1[1].scatter(X[y==i, 0], X[y==i, 1] # y==i 对应 行索引
            ,marker='o' #点的形状
            ,s=8 #点的大小
            ,c=color[i]
           )
    
plt.show()

# In[]:
n_clusters = 3
cluster = KMeans(n_clusters=n_clusters,random_state=0).fit(X)
# 重要属性Labels_，查看聚好的类别，每个样本所对应的类
y_pred = cluster.labels_

# In[]:
# KMeans因为并不需要建立模型或者预测结果，因此我们只需要fit就能够得到聚类结果了
# KMeans也有接口 predict（直接预测） 和 fit_predict（表示学习数据X并对X的类进行预测）
# predict所得到的结果 和 直接fit之后调用属性labels一模一样。
pre = cluster.predict(X)
print(np.sum(y_pred == pre), len(y))

# In[]
# 我们什么时候需要predict呢？当数据量太大的时候！
# 其实我们不必使用所有的数据来寻找质心，少量的数据就可以帮助我们确定质心了，
# 当我们数据量非常大的时候，我们可以使用部分数据来帮助我们确认质心，剩下的数据的聚类结果，使用predict来调用。
cluster_smallsub = KMeans(n_clusters=n_clusters, random_state=0).fit(X[:200])

y_pred_ = cluster_smallsub.predict(X)

# 数据量非常大的时候，效果会好。 但从运行得出这样的结果，肯定与直接fit全部数据会不一致。
# 有时候，当我们不要求那么精确，或者我们的数据量实在太大，那我们可以使用这种方法，使用接口predict
# 如果数据量还行，不是特别大，直接使用fit之后调用属性.labels_提出来
print(np.sum(y_pred == y_pred_), len(y))


# In[]:
# 重要属性：
# 1、cLuster_centers_，查看质心
centroid = cluster.cluster_centers_

# In[]:
# 2、inertia_，查看总距离平方和
inertia = cluster.inertia_

# In[]:
color = ["red","pink","orange","gray"]

fig, ax1 = plt.subplots(1)

for i in range(n_clusters):
    ax1.scatter(X[y_pred==i, 0], X[y_pred==i, 1]
            ,marker='o' #点的形状
            ,s=8 #点的大小
            ,c=color[i]
           )
    
ax1.scatter(centroid[:,0],centroid[:,1]
           ,marker="x"
           ,s=15
           ,c="black")
plt.show()


# In[]:
# 评估指标：
# 1、簇内平方和
'''
1、将 簇内平方和 理解为 KMeans的损失函数，因为 KMeans是不需要求解参数的，所以严格意义上来说KMeans没有损失函数。
2、将 损失函数：簇内平方和 的 结果Inertia 理解为 Kmeans的模型评估指标。当k值固定，统计Inertia指标才有意义；
否则，当k值不断增大时，Inertia会越来越小，没有意义。但实际上 Inertia是不能作为 评估指标，只能作为辅助评估度指标，因为 簇内平方和Inertia 没有界的。
'''
n_clusters = 4
cluster_ = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
inertia_ = cluster_.inertia_
print(inertia_)

n_clusters = 5
cluster_ = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
inertia_ = cluster_.inertia_
print(inertia_)

n_clusters = 6
cluster_ = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
inertia_ = cluster_.inertia_
print(inertia_)

# In[]:
# 2、轮廓系数
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from time import time
import datetime

n_clusters = [3,5,6]
score = []

t0 = time()
for i in n_clusters:
    cluster = KMeans(n_clusters=i,random_state=0).fit(X)
    # 重要属性Labels_，查看聚好的类别，每个样本所对应的类
    y_pred = cluster.labels_
    pre = cluster.predict(X)
    print(np.sum(y_pred == pre) == len(y_pred))
    
    temp = silhouette_score(X,y_pred)
    score.append(temp)

cha = time() - t0
print (int(round(cha * 1000))) #毫秒级时间戳
fig, axis = plt.subplots(1,1, figsize=(10,8))
axis.plot(n_clusters, score)

# In[]:
# 3、Calinski-Harabaz Index
from sklearn.metrics import calinski_harabaz_score

n_clusters = [3,5,6]
score = []

t0 = time()
for i in n_clusters:
    cluster = KMeans(n_clusters=i,random_state=0).fit(X)
    # 重要属性Labels_，查看聚好的类别，每个样本所对应的类
    y_pred = cluster.labels_
    pre = cluster.predict(X)
    print(np.sum(y_pred == pre) == len(y_pred))
    
    temp = calinski_harabaz_score(X, y_pred)
    score.append(temp)

cha = time() - t0
print (int(round(cha * 1000))) #毫秒级时间戳
fig, axis = plt.subplots(1,1, figsize=(10,8))
axis.plot(n_clusters, score)







