from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

iris = load_iris()
# print(iris.data[0:10])
# print(iris.target[0:10])


# 2、数据预处理

# 2.1、无量纲化
# 自己的测试数据
np.random.seed(1)
aaa = np.random.randint(0,10,(3,3))
print(aaa)

# 2.1.1、以列为基准
lmean = np.mean(aaa, axis=0)
lstd = np.std(aaa, axis=0) # np.std是总体标准差 除以n；而pandas是样本标准差 除以n-1。
# print(lmean, lstd)
lnum = aaa.shape[0]
lstdmy = []
lstdmy.append(np.sqrt((aaa[0][0] - lmean[0]) ** 2 / lnum + (aaa[1][0] - lmean[0]) ** 2 / lnum + (aaa[2][0] - lmean[0]) ** 2 / lnum))
lstdmy.append(np.sqrt((aaa[0][1] - lmean[2]) ** 2 / lnum + (aaa[1][1] - lmean[1]) ** 2 / lnum + (aaa[2][1] - lmean[1]) ** 2 / lnum))
lstdmy.append(np.sqrt((aaa[0][2] - lmean[2]) ** 2 / lnum + (aaa[1][2] - lmean[2]) ** 2 / lnum + (aaa[2][2] - lmean[2]) ** 2 / lnum))
# print(lstdmy)

# 2.1.2、以行为基准
hmean = np.mean(aaa, axis=1)
hstd = np.std(aaa, axis=1) # np.std是总体标准差 除以n；而pandas是样本标准差 除以n-1。
print(hmean, hstd)


# 2.1.1.1、标准化：以 特征列 为计算维度
from sklearn.preprocessing import StandardScaler
# 标准化，返回值为标准化后的数据
zdata = StandardScaler().fit_transform(aaa) # iris.data
# print(zdata[0:10])
zdatamy = np.zeros((3,3))
for i in range(3):
    zdatamy[0,i] = (aaa[0][i] - lmean[i]) / lstd[i]
    zdatamy[1,i] = (aaa[1][i] - lmean[i]) / lstd[i]
    zdatamy[2,i] = (aaa[2][i] - lmean[i]) / lstd[i]
# print(zdatamy)


# 2.1.1.2、归一化：区间缩放法：以 特征列 为计算维度
from sklearn.preprocessing import MinMaxScaler
#区间缩放，返回值为缩放到[0, 1]区间的数据
mdata = MinMaxScaler().fit_transform(aaa) # iris.data
# print(mdata[0:10])
mdatamy = np.zeros((3,3))
for i in range(3):
    mdatamy[0,i] = (aaa[0][i] - np.min(aaa[:,i], axis=0)) / (np.max(aaa[:,i], axis=0) - np.min(aaa[:,i], axis=0))
    mdatamy[1,i] = (aaa[1][i] - np.min(aaa[:,i], axis=0)) / (np.max(aaa[:,i], axis=0) - np.min(aaa[:,i], axis=0))
    mdatamy[2,i] = (aaa[2][i] - np.min(aaa[:,i], axis=0)) / (np.max(aaa[:,i], axis=0) - np.min(aaa[:,i], axis=0))
# print(mdatamy)


# 2.1.2、归一化：正则化：以 样本行 为计算维度
'''
归一化(正则化)是依照特征矩阵的行处理数据，其目的在于样本向量在点乘运算或其他核函数计算相似性时，
拥有统一的标准，也就是说都转化为“单位向量”。
'''
from sklearn.preprocessing import Normalizer
#归一化，返回值为归一化后的数据
ndata = Normalizer().fit_transform(aaa) # iris.data
print(ndata[0:10])
ndatamy = np.zeros((3,3))
for i in range(3):
    ndatamy[i, 0] = aaa[i][0] / np.sqrt(np.sum(np.square(aaa[i,:])))
    ndatamy[i, 1] = aaa[i][1] / np.sqrt(np.sum(np.square(aaa[i,:])))
    ndatamy[i, 2] = aaa[i][2] / np.sqrt(np.sum(np.square(aaa[i,:])))
print(ndatamy)