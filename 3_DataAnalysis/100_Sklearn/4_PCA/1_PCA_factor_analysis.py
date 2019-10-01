# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:05:30 2019

@author: dell
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# In[]:
'''
每条记录都有 4 项特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度，
可以通过这4个特征预测鸢尾花卉属于（iris-setosa, iris-versicolour, iris-virginica）
'''
iris = load_iris()
original_x = iris.data
original_y = iris.target

X = original_x.copy()
y = original_y.copy()
#作为数组，X是几维？
X.shape#(150, 4)0
#作为数据表或特征矩阵，X是几维？
pd.DataFrame(X).head()

# In[]:
from sklearn import preprocessing
X = preprocessing.scale(X) # ndarray
print(X)

# In[]:
# 画 累计方差贡献率 曲线，找最佳降维后维度的范围
'''
1、主成分个数的选取原则：
1.1、单个主成分解释的变异（方差）不因小于1。
1.2、选取主成分累计的解释变异达到80%-90%。
说明：1、第一次的n_components参数应该设的大一点（保留主成分个数）
说明：2、观察explained_variance_ratio_和explained_variance_的取值变化，
建议explained_variance_ratio_累积大于0.85，explained_variance_需要保留的最后一个主成分大于0.8，
'''
#pca=PCA(n_components=3)
pca = PCA().fit(X)
# explained_variance_： 解释方差
print(pca.explained_variance_) # 建议保留2个主成分
# explained_variance_ratio_： 解释方差占比（累计解释方差占比 自己手动加）
print(pca.explained_variance_ratio_) #建议保留2个主成分

plt.plot([1,2,3,4],np.cumsum(pca.explained_variance_ratio_))
plt.xticks([1,2,3,4]) #这是为了限制坐标轴显示为整数
plt.xlabel("number of components after dimension reduction")
plt.ylabel("cumulative explained variance ratio")
plt.show()

# In[]:
# 使用 最大似然估计 自选超参数： （Minka, T.P. 麻省理工学院）
pca_mle = PCA(n_components="mle") # mle缺点计算量大
pca_mle = pca_mle.fit(X)
X_mle = pca_mle.transform(X)

print(X_mle[0:10]) #3 列的数组
# 可以发现，mle为我们自动选择了3个特征
print(pca_mle.components_)
print(pca_mle.explained_variance_ratio_) # [0.72962445 0.22850762 0.03668922]
# 得到了比设定2个特征时更高的信息含量，对于鸢尾花这个很小的数据集来说，3个特征对应这么高的信息含量，并不
# 需要去纠结于只保留2个特征，毕竟三个特征也可以可视化

# In[]:
# 使用 总解释性方差占比 选超参数：
pca_f = PCA(n_components=0.97,svd_solver="full") # svd_solver="full"不能省略
pca_f = pca_f.fit(X)
X_f = pca_f.transform(X)

print(X_f[0:10]) # 3列的数组
print(pca_f.components_)
print(pca_f.explained_variance_ratio_) # array([0.92461872, 0.05306648])

# In[]:
#调用PCA
pca = PCA(n_components=2)           #实例化
pca = pca.fit(X)                    #拟合模型
X_dr = pca.transform(X)             #获取新矩阵
print(type(X_dr))
#X_dr = PCA(2).fit_transform(X)

# 属性explained_variance_，查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
print(pca.explained_variance_) # 查看方差是否从大到小排列，第一个最大，依次减小   array([4.22824171, 0.24267075])
 
# 属性explained_variance_ratio，查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比
# 又叫做可解释方差贡献率
print(pca.explained_variance_ratio_) # array([0.92461872, 0.05306648])
# 大部分信息都被有效地集中在了第一个特征上
 
print(pca.explained_variance_ratio_.sum()) # 0.977685206318795

# In[]:
'''
计算特征向量矩阵P：（因子旋转前 主成分）
通过主成分在每个变量上的权重的绝对值大小，确定每个主成分的代表性
'''
e_matrix = pd.DataFrame(pca.components_).T # 以 列 的方式呈现
print(e_matrix)

# 因子旋转前的 两个主成分作散点图
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

e_x = e_matrix[0] # 主成分1
e_y = e_matrix[1] # 主成分2
label = ['X1','X2','X3','X4']
plt.xlabel('花瓣长度')
plt.ylabel('花萼长度和宽度')
plt.scatter(e_x, e_y)
for a,b,l in zip(e_x,e_y,label):
    plt.text(a, b, '%s.' % l, ha='center', va= 'bottom',fontsize=14)

plt.show()

# In[]:
#要将三种鸢尾花的数据分布显示在二维平面坐标系中，对应的两个坐标（两个特征向量）应该是三种鸢尾花降维后的x1和x2，怎样才能取出三种鸢尾花下不同的x1和x2呢？
X_dr[y == 0, 0] #这里是布尔索引，看出来了么？
 
#要展示三中分类的分布，需要对三种鸢尾花分别绘图
#可以写成三行代码，也可以写成for循环
"""
plt.figure()
plt.scatter(X_dr[y==0, 0], X_dr[y==0, 1], c="red", label=iris.target_names[0])
plt.scatter(X_dr[y==1, 0], X_dr[y==1, 1], c="black", label=iris.target_names[1])
plt.scatter(X_dr[y==2, 0], X_dr[y==2, 1], c="orange", label=iris.target_names[2])
plt.legend()
plt.title('PCA of IRIS dataset')
plt.show()
"""
 
colors = ['red', 'black', 'orange']
iris.target_names
 
plt.figure()
for i in [0, 1, 2]:
    plt.scatter(X_dr[y == i, 0]
                ,X_dr[y == i, 1]
                ,alpha=.7#指画出的图像的透明度
                ,c=colors[i]
                ,label=iris.target_names[i]
               )
plt.legend()#图例
plt.title('PCA of IRIS dataset')
plt.show()

# In[]:
from fa_kit import FactorAnalysis
from fa_kit import plotting as fa_plotting

# 数据导入和转换
fa = FactorAnalysis.load_data_samples(
        X,
        preproc_demean=True,
        preproc_scale=True
        )

# 抽取主成分 
fa.extract_components()

# In[]:
fa.find_comps_to_retain(method='top_n',num_keep=2)

# In[]:
# varimax： 使用 最大方差法 进行 因子旋转
fa.rotate_components(method='varimax')

# 因子旋转后的 因子权重（因子载荷矩阵A）
temp = pd.DataFrame(fa.comps["rot"]) # rot： 使用因子旋转法
print(temp)

fa_plotting.graph_summary(fa)

# In[]:
# 因子旋转后的 因子权重（因子载荷矩阵A）
fas = pd.DataFrame(fa.comps["rot"])  # rot： 使用因子旋转法
print(fas)

# 因子旋转后的 因子权重（因子载荷矩阵A） 两个因子权重作散点图
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

import matplotlib.pyplot as plt

x_f = fas[0] # 因子1
y_f = fas[1] # 因子2
label = ['X1','X2','X3','X4']
plt.xlabel('花萼长度、花瓣长度、花瓣宽度')
plt.ylabel('花萼宽度')
plt.scatter(x_f, y_f)
for a,b,l in zip(x_f,y_f,label):
    plt.text(a, b, '%s.' % l, ha='center', va= 'bottom',fontsize=14)

plt.show()

# 到目前还没有与PCA中fit_transform类似的函数，因此只能手工计算因子得分
# 以下是矩阵相乘的方式计算因子得分：因子得分 = 原始数据（n*k） * 权重矩阵(k*num_keep)
fa_score = pd.DataFrame(np.dot(X, fas))
print(fa_score.shape)

# In[]:
a = fa_score.rename(columns={0: "花萼长度、花瓣长度、花瓣宽度", 1: "花萼宽度"}) # '经济总量水平', '人均GDP水平'
original_fa = pd.DataFrame(original_x).join(a)
#print(original_fa)

# In[]:
# 因子得分散点图：
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

import matplotlib.pyplot as plt

x_fa = original_fa['花萼长度、花瓣长度、花瓣宽度']
y_fa = original_fa['花萼宽度']
label = original_fa.index
 
colors = ['red', 'black', 'orange']
iris.target_names
 
plt.figure()
for i in [0, 1, 2]:
    plt.scatter(x_fa[y == i]
                ,y_fa[y == i]
                ,alpha=.7#指画出的图像的透明度
                ,c=colors[i]
                ,label=iris.target_names[i]
               )
    
plt.xlabel('花萼长度、花瓣长度、花瓣宽度')
plt.ylabel('花萼宽度')
plt.legend()#图例
plt.title('PCA_factor_analysis of IRIS dataset')

for a,b,l in zip(x_fa,y_fa,label):
    plt.text(a, b+0.1, '%s.' % l, ha='center', va= 'bottom',fontsize=14)

plt.show()



