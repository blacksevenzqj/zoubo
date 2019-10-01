# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:20:57 2019

@author: dell
"""

from sklearn.datasets import fetch_lfw_people#7个人的1000多张人脸图片组成的一组人脸数据
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

# In[]:
# 一、迷你案例1： 用人脸识别看PCA降维后的信息保存量
'''
将降维后的数据保存，用于还原 原图像（只能也只需要 还原重要特征），之后用于人脸图像识别。 
'''
faces = fetch_lfw_people(min_faces_per_person=60)#实例化   min_faces_per_person=60：每个人取出60张脸图
print(faces.images.shape) # （1277,62,47）  1277是矩阵中图像的个数，62是每个图像的特征矩阵的行，47是每个图像的特征矩阵的列
print(faces.data.shape) # （1277,2914）   行是样本，列是样本相关的所有特征：2914 = 62 * 47

# In[]:
#X = faces.data
#X = preprocessing.scale(faces.data) # ndarray # 标准化
X = MinMaxScaler().fit_transform(faces.data) # 归一化
print(X.shape)

# In[]:
# 数据本身是图像，和数据本身只是数字，使用的可视化方法不同
# 创建画布和子图对象： fig指的是画布； 
fig, axes = plt.subplots(4,5 # 4行5列个图
                        ,figsize=(8,4) # figsize指的是图的尺寸
                        ,subplot_kw = {"xticks":[],"yticks":[]} #不要显示坐标轴
                        )
# axes：
#不难发现，axes中的一个对象对应fig中的一个空格
#我们希望，在每一个子图对象中填充图像（共20张图），因此我们需要写一个在子图对象中遍历的循环
print(axes.shape) # （4,5）
 
#二维结构，可以有两种循环方式，一种是使用索引，循环一次同时生成一列上的四个图
#另一种是把数据拉成一维，循环一次只生成一个图
#在这里，究竟使用哪一种循环方式，是要看我们要画的图的信息，储存在一个怎样的结构里
#我们使用 子图对象.imshow 来将图像填充到空白画布上
#而imshow要求的数据格式必须是一个(m,n)格式的矩阵，即每个数据都是一张单独的图
#因此我们需要遍历的是faces.images，其结构是(1277, 62, 47)
#要从一个数据集中取出24个图，明显是一次性的循环切片[i,:,:]来得便利
#因此我们要把axes的结构拉成一维来循环
axes.flat # 降低一个维度
#print([*axes.flat]) # 1维

#填充图像（通过axes的flat属性进行遍历）
for i, ax in enumerate(axes.flat):
    ax.imshow(faces.images[i,:,:] 
              ,cmap="gray" #选择色彩的模式
            )
 
# cmap参数取值选择各种颜色：https://matplotlib.org/tutorials/colors/colormaps.html
    
# In[]:
# 原本有2914维，我们现在来降到150维
# 加入白化参数： 降维后特征向量之间的线性独立性（正交）： 消除特征向量之间的相关性
pca = PCA(150, whiten = True).fit(X) #这里X = faces.data，不是faces.images.shape ,因为sklearn只接受2维数组降，不接受高维数组降

V = pca.components_ # 新特征空间
print(V.shape) #V(k，n) (150, 2914)

X_dr = pca.transform(X) # PCA降维后的信息保存量
print(X_dr.shape) # (1348, 150)
    
# In[]:
fig, axes = plt.subplots(3,8,figsize=(8,4),subplot_kw = {"xticks":[],"yticks":[]})
# 原来有2914维特征，现在降维为150维，显示每一维的特征图像：
for i, ax in enumerate(axes.flat):
    ax.imshow(V[i,:].reshape(62,47),cmap="gray")

# In[]:
# 重要接口inverse_transform
'''
在逆转的时候，即便维度升高，原数据中已经被舍弃的信息也不可能再回来了。所以，降维不是完全可逆的。
'''
X_inverse = pca.inverse_transform(X_dr)
print(X_inverse.shape) # (1348, 2914)

print(X_dr.shape) # (1348, 150)
print(V.shape) # V(k，n) (150, 2914)
X_inverse_dot = X_dr.dot(V)
print(X_inverse_dot.shape) # (1348, 2914)

print(faces.images.shape) # (1348, 62, 47)

# In[]:
fig, ax = plt.subplots(3,10,figsize=(10,2.5)
                      ,subplot_kw={"xticks":[],"yticks":[]}
                     )
 
#和2.3.3节中的案例一样，我们需要对子图对象进行遍历的循环，来将图像填入子图中
#那在这里，我们使用怎样的循环？
#现在我们的ax中是2行10列，第一行是原数据，第二行是inverse_transform后返回的数据
#所以我们需要同时循环两份数据，即一次循环画一列上的两张图，而不是把ax拉平
for i in range(10):
    ax[0,i].imshow(faces.images[i,:,:],cmap="binary_r")
    ax[1,i].imshow(X_inverse[i].reshape(62,47),cmap="binary_r")
    ax[2,i].imshow(X_inverse_dot[i].reshape(62,47),cmap="binary_r")



# In[]:
# 二、迷你案例2： 用PCA做噪音过滤
'''
降维的目的之一就是希望抛弃掉对模型带来负面影响的特征，而我们相信，
带有效信息的特征的方差应该是远大于噪音的，所以相比噪音，有效的特征所带的信息应该不会在PCA过程中被大量抛弃。
inverse_transform能够在不恢复原始数据的情况下，将降维后的数据返回到原本的高维空间，即是说能够实现 ”保证维度，
但去掉方差很小特征所带的信息“。利用inverse_transform的这个性质，我们能够实现噪音过滤。
'''
from sklearn.datasets import load_digits
    
digits = load_digits()
print(digits.data.shape) # (1797, 64)
print(set(digits.target.tolist())) # 查看target有哪几个数  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
print(digits.images.shape) # (1797, 8, 8)
    
# In[]:
def plot_digits(data):
    #data的结构必须是（m,n），并且n要能够被分成（8,8）这样的结构
    fig, axes = plt.subplots(4,10,figsize=(10,4)
                            ,subplot_kw = {"xticks":[],"yticks":[]}
                            )
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8,8),cmap="binary")
        
# 显示原始图片： （后续不用）
plot_digits(digits.data)
    
# In[]:
# 人为加入噪音： 在指定的数据集中，随机抽取服从正态分布的数据
rng = np.random.RandomState(42)
# 两个参数，分别是指定的数据集，和抽取出来的正太分布的方差
noisy = rng.normal(digits.data, 2) # np.random.normal(digits.data,2)
print(digits.data.shape)
print(noisy.shape)
# 调用自定义画图函数
plot_digits(noisy)

# In[]:
#noisy_X = noisy
#noisy_X = preprocessing.scale(noisy) # 标准化 
noisy_X = MinMaxScaler().fit_transform(noisy) # 归一化
print(noisy_X.shape)

# In[]:
# 按 信息量占比 选超参数（累计解释方差占比/贡献率）   信息量衡量指标 就是样本方差 又称 可解释性方差
# 加入白化参数： 降维后特征向量之间的线性独立性（正交）： 消除特征向量之间的相关性
pca = PCA(0.5,svd_solver='auto',whiten = True).fit(noisy_X) # full
# explained_variance_： 解释方差
print(pca.explained_variance_) 
# explained_variance_ratio_： 解释方差占比/贡献率
print(pca.explained_variance_ratio_.sum())

V = pca.components_
print(V.shape) #V(k，n) (8, 64)

X_dr = pca.transform(noisy_X)
print(X_dr.shape) # (1797, 8)

# In[]:
# 人为加入噪音的原始图像： 
plot_digits(noisy)

# 人为加入噪音 归一化 后的原始图像： 
plot_digits(noisy_X)

# 去掉噪音后 还原图像：
without_noise = pca.inverse_transform(X_dr)
plot_digits(without_noise)


# 去掉噪音后 还原图像： （手动）
print(X_dr.shape) # (1797, 6)
print(V.shape) # V(k，n) (6, 64)
X_inverse_dot = X_dr.dot(V)
print(X_inverse_dot.shape) # (1797, 64)
plot_digits(X_inverse_dot)

# In[]:
'''
图像降维后的数据，只是用于还原 原图像，并不用于直接生成图像。
'''




    
    
    