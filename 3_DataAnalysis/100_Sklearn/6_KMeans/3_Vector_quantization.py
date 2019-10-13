# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 16:42:19 2019

@author: dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
    #对两个序列中的点进行距离匹配的函数
from sklearn.datasets import load_sample_image
    #导入图片数据所用的类
from sklearn.utils import shuffle #洗牌

# 实例化，导入颐和园的图片
china = load_sample_image("china.jpg")

# In[]:
# 查看数据类型
print(china.dtype) # uint8

print(china.shape)
# 长度 x 宽度 x 像素 > 三个数决定的颜色

#包含多少种不同的颜色?
newimage = china.reshape((427 * 640,3))
print(newimage.shape)

newimage = pd.DataFrame(newimage).drop_duplicates() # (96615, 3) 9W多种不同的颜色（像素点）
print(newimage.shape)

plt.figure(figsize=(15,15))
plt.imshow(china) # 必须是3维数组

# In[]:
n_clusters = 64

china = np.array(china, dtype=np.float64) / china.max()
w, h, d = original_shape = tuple(china.shape)



# In[]:
'''
1、质心替换原数据； 2、没有降低数据矩阵维度（区别于PCA），对数据进行了压缩（用质心代表数据特征，可以理解为降维）
'''
# 1、对数据进行K-Means的矢量量化
image_array = np.reshape(china, (w * h, d))

# 首先，先使用1000个数据来找出质心
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image_array_sample)
print(len(kmeans.labels_)) # 重要属性Labels_，查看聚好的类别，每个样本所对应的类
print(len(kmeans.cluster_centers_)) # 质心
print(kmeans.inertia_) # 簇内平方和

# In[]:
# 找出质心之后，按照已存在的质心对所有数据进行聚类
labels = kmeans.predict(image_array) # 相当于 类.labels_
print(labels.shape)

print(len(set(labels)))

# In[]:
# 使用质心来替换所有的样本
image_kmeans = image_array.copy() # (273280, 3) 27W个样本点 
print(pd.DataFrame(image_kmeans).drop_duplicates().shape) # (96615, 3) 9W多种不同的颜色（像素点）

# In[]:
# 第0个簇的质心
#kmeans.cluster_centers_[labels[0]]

# 将 原始数据副本 的每一个样本数据 替换为 其所属质心数据
for i in range(image_kmeans.shape[0]):
    image_kmeans[i] = kmeans.cluster_centers_[labels[i]]

print(image_kmeans.shape) # (273280, 3)
# 经过 质心替换后 颜色只剩下 64组
print(pd.DataFrame(image_kmeans).drop_duplicates().shape) # (64, 3)

# In[]:
# 恢复图片的结构
image_kmeans = image_kmeans.reshape(w,h,d)
print(image_kmeans.shape)



# 2、对数据进行随机的矢量量化
# In[]:
centroid_random = shuffle(image_array, random_state=0)[:n_clusters] # 随机抽64个点作为质心

# In[]:
labels_random = pairwise_distances_argmin(centroid_random,image_array,axis=0)
# 函数pairwise_distances_argmin(x1,x2,axis) x1 和 x2 分别是序列
# 用来计算x2中的每个样本到x1中的每个样本点的距离，并返回和x2相同形状的，x1中对应的最近的样本点的索引
'''
1、X2--image_array中273280个样本 分别到 X1--centroid_random中随机64个样本点质心 的距离； 
2、计算image_array中一个样本点 到 centroid_random中随机64个样本点质心 的距离，得到64个距离，选择最小的一个距离。
3、image_array中273280个样本，每个样本都要与 centroid_random中随机64个样本点质心 计算距离，并返回64个距离中最小的一个。共计算273280x64次距离。
4、最后返回是和image_array形状相同的 273280个 centroid_random中对应的最近的质心的索引 的列表；
列表的索引对应 image_array中273280个样本索引，列表的值对应 centroid_random中质心索引。
'''
print(labels_random.shape)

print(len(set(labels_random)))

# In[]:
# 使用随机质心来替换所有样本
image_random = image_array.copy()

# 将 原始数据副本 的每一个样本数据 替换为 其所属 随机质心数据
for i in range(image_random.shape[0]):
    image_random[i] = centroid_random[labels_random[i]]

# 经过 随机质心替换后 颜色只剩下 64组
print(pd.DataFrame(image_random).drop_duplicates().shape) # (64, 3)

# 恢复图片的结构
image_random = image_random.reshape(w,h,d)
print(image_random.shape)



# In[]:
# 原图：
plt.figure(figsize=(10,10))
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(china)

# K-Means的矢量量化
plt.figure(figsize=(10,10))
plt.axis('off')
plt.title('Quantized image (64 colors, K-Means)')
plt.imshow(image_kmeans)

# 随机的矢量量化
plt.figure(figsize=(10,10))
plt.axis('off')
plt.title('Quantized image (64 colors, Random)')
plt.imshow(image_random)
plt.show()









