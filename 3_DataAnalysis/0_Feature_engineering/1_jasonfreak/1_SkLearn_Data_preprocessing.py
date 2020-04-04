from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


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
# print(hmean, hstd)

# 对于StandardScaler和MinMaxScaler来说，空值NaN会被当做是缺失值，在fit的时候忽略，在transform的时候保持缺失NaN的状态显示。
# 2.1.1.1、标准化：以 特征列 为计算维度
'''
它是基于原始数据的均值和标准差进行的标准化，其标准化的计算公式为x'=(x-mean)/std,其中mean和std为x所在列的均值和标准差。
注：
1、经过标准化后会使每个特征中平均值变为0、标准差变为1。标准化不改变数据的分布，不会把数据变成正态分布。这个方法被广泛的使用在许多机器学习算法中(例如：支持向量机、逻辑回归和类神经网络)
1.1.1、标准化： 不会改变数据分布，数据散点图、偏度、直方图、正太检验 都没有大的变化。
1.1.2、标准化： 改变了 PCA、t-SNE、SVM 的计算结果，因为它们都需要无量纲化处理。
1.2.1、偏态数据取log： 改变数据分布，数据散点图、偏度、直方图、正太检验 都有变化。
1.2.2、偏态数据取log： 改变了 PCA、t-SNE 的结果，但是正确性不好。
1.3.1、偏态数据取log 再 标准化： 改变数据分布，数据散点图、偏度、直方图、正太检验 都有变化（偏态数据取log的作用）
1.3.2、偏态数据取log 再 标准化： 改变了 PCA、t-SNE、SVM 的计算结果，因为它们都需要无量纲化处理（标准化的作用）

2、标准化 的理论取值范围是(-∞,+∞)，但经验上看大多数取值范围在[-4,4]之间：以训练集为标准。
3、但 标准化 方法改变了原始数据的结构（针对稀疏矩阵而言，并不会改变 预设的高斯分布的结构），因此不适宜用于对稀疏矩阵做数据预处理。
4、当数据集中含有离群点，即异常值时，可以用z-score进行标准化，但是标准化后的数据并不理想，因为异常点的特征往往在标准化之后容易失去离群特征。
'''
from sklearn.preprocessing import StandardScaler
# 标准化，返回值为标准化后的数据
zdata = StandardScaler().fit_transform(aaa) # iris.data
print(zdata[0:10], np.mean(zdata, axis=0), np.std(zdata, axis=0))
from sklearn import preprocessing
zdata1 = preprocessing.scale(aaa)
print(zdata1[0:10], np.mean(zdata1, axis=0), np.std(zdata1, axis=0))
zdatamy = np.zeros((3,3))
for i in range(3):
    zdatamy[0,i] = (aaa[0][i] - lmean[i]) / lstd[i]
    zdatamy[1,i] = (aaa[1][i] - lmean[i]) / lstd[i]
    zdatamy[2,i] = (aaa[2][i] - lmean[i]) / lstd[i]
print(zdatamy, np.mean(zdatamy, axis=0), np.std(zdatamy, axis=0))


# 2.1.1.2、归一化：区间缩放法：以 特征列 为计算维度
'''
该方法是用数据的最大值和最小值对原始数据进行预处理其是一种线性变换。其标准化的计算公式为x'=(x-min)/(max-min),min和max是x所在列的最小值和最大值。
此方法得到的数据会完全落入[0,1]区间内（z-score没有类似区间），而且能使数据归一化落到一定的区间内，同时保留原始数据的结构
'''
from sklearn.preprocessing import MinMaxScaler
#区间缩放，返回值为缩放到[0, 1]区间的数据
# 注意：fit_transform(aaa)函数，入参aaa必须为二维矩阵：（使用Padans：必须是DataFrame，不能是Seriers）
mdata = MinMaxScaler().fit_transform(aaa) # iris.data
print(mdata)
print("-"*30)
min_col = np.zeros(3)
max_col = np.zeros(3)
mdatamy = np.zeros((3,3))
for i in range(3):
    min_col[i] = np.min(aaa[:,i], axis=0)
    max_col[i] = np.max(aaa[:,i], axis=0)
    mdatamy[0,i] = (aaa[0][i] - np.min(aaa[:,i], axis=0)) / (np.max(aaa[:,i], axis=0) - np.min(aaa[:,i], axis=0))
    mdatamy[1,i] = (aaa[1][i] - np.min(aaa[:,i], axis=0)) / (np.max(aaa[:,i], axis=0) - np.min(aaa[:,i], axis=0))
    mdatamy[2,i] = (aaa[2][i] - np.min(aaa[:,i], axis=0)) / (np.max(aaa[:,i], axis=0) - np.min(aaa[:,i], axis=0))
print(mdatamy)
print(aaa)
print(min_col)
print(max_col)
# 还原：
re_mdatamy = np.zeros((3,3))
for i in range(3):
    re_mdatamy[0, i] = mdatamy[0, i] * (max_col[i] - min_col[i]) + min_col[i]
    re_mdatamy[1, i] = mdatamy[1, i] * (max_col[i] - min_col[i]) + min_col[i]
    re_mdatamy[2, i] = mdatamy[2, i] * (max_col[i] - min_col[i]) + min_col[i]
print(re_mdatamy)
'''
# 手动：
data['power1'] = ((data['power1'] - np.min(data['power1'])) / (np.max(data['power1']) - np.min(data['power1'])))
# 调MinMaxScaler函数：
data['power2'] = MinMaxScaler().fit_transform(data[['power2']]) # 注意：入参必须是二维矩阵，必须是DataFrame。
'''

# 2.1.1.3、MaxAbscaler归一化
'''
根据最大值得绝对值标准化。其标准化的计算公式为x'=x/|max|，其中max是x所在列的最大值。该方法和Max-Min方法类似，
但该方法的数据区间为[-1,1]，也不会破坏原始数据的结构，因此也可以用于稀疏矩阵、稀疏的CSR或CSC矩阵。
'''
maxab_scaler = preprocessing.MaxAbsScaler()
madata = maxab_scaler.fit_transform(aaa)


# 2.1.1.4、RobustScaler归一化
'''
当数据集中含有离群点，即异常值时，可以用z-score进行标准化，但是标准化后的数据并不理想，
因为异常点的特征往往在标准化之后容易失去离群特征。此时可以用该方法针对离群点做标准化处理。
This Scaler removes the median and scales the data according to the quantile range 移除中位数，并根据四分位距离范围缩放数据，也就是说排除了异常值
'''
robustscaler = preprocessing.RobustScaler()
rdata = robustscaler.fit_transform(aaa)


# 2.1.2、正则化：以 样本行 为计算维度
'''
归一化(正则化)是依照特征矩阵的行处理数据，其目的在于样本向量在点乘运算或其他核函数计算相似性时，
拥有统一的标准，也就是说都转化为“单位向量”。
'''
from sklearn.preprocessing import Normalizer
#归一化，返回值为归一化后的数据
ndata = Normalizer().fit_transform(aaa) # iris.data
# print(ndata[0:10])
ndatamy = np.zeros((3,3))
for i in range(3):
    ndatamy[i, 0] = aaa[i][0] / np.sqrt(np.sum(np.square(aaa[i,:])))
    ndatamy[i, 1] = aaa[i][1] / np.sqrt(np.sum(np.square(aaa[i,:])))
    ndatamy[i, 2] = aaa[i][2] / np.sqrt(np.sum(np.square(aaa[i,:])))
# print(ndatamy)


# -------------------------------------------------------------------------------------------------------------


# 2.2、对定量特征二值化
from sklearn.preprocessing import Binarizer
# 二值化，阈值设置为3，返回值为二值化后的数据（0 或 1）
bdata = Binarizer(threshold=3).fit_transform(aaa) # iris.data
# print(bdata)


# -------------------------------------------------------------------------------------------------------------


# 2.3、对定性特征哑编码
from sklearn.preprocessing import OneHotEncoder

# 哑编码，对IRIS数据集的目标值，返回值为哑编码后的数据
oneArray = np.array(['A','B','C','D','D','D','B','C''A','D'])
# print(type(OneHotEncoder().fit_transform(oneArray.reshape((-1, 1)))))


# -------------------------------------------------------------------------------------------------------------


# 2.4、缺失值计算
from numpy import vstack, array, nan
from sklearn.preprocessing import Imputer

# 缺失值计算，返回值为计算缺失值后的数据
# 参数missing_value为缺失值的表示形式，默认为NaN
# 参数strategy为缺失值填充方式，默认为mean（均值）
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
# 对数据集新增一个样本，4个特征均赋值为NaN，表示数据缺失。vstack从栈顶压入。
idata = imp.fit_transform(vstack((array([nan, nan, nan]), aaa)))
# print(np.mean(aaa, axis=0))
# print(idata)


# -------------------------------------------------------------------------------------------------------------


# 2.5、数据变换
# 2.5.1、多项式转换
from sklearn.preprocessing import PolynomialFeatures
# 参数degree为度，默认值为2
# 它是使用多项式的方法来进行的，如果有a，b两个特征，那么它的2次多项式为(1,a,b, a^2,ab, b^2)；
# 如果有a,b,c三个特征，那么它的2次多项式为(1,a,b,c, a^2,ab,ac, b^2,bc, c^2)。
pdata = PolynomialFeatures().fit_transform(aaa)
# print(pdata)


# 2.5.2、基于函数的数据变换可以使用统一的方式完成，使用preproccessing库的FunctionTransformer对数据进行函数转换的代码如下：
from numpy import log1p, expm1
from sklearn.preprocessing import FunctionTransformer
# 自定义转换函数为对数函数的数据变换
# 第一个参数是单变元函数：可以是对数变换：np.log1p、np.log；可以是指数变换：np.exp 等等。
fdata = FunctionTransformer(log1p).fit_transform(aaa)
# print(fdata)
# print(np.log(aaa + 1))
'''
# 先了解下 log1p 和 expm1：
x = 10**-16
print(x)
# log1p  相当于  ln(x + 1)
print(log1p(3), np.log(3 + 1))
# 但是当x很小的时候，用ln(x + 1)计算得到结果为0，换作log1p(x)计算得到一个很小却不为0的结果。
print(log1p(x), np.log(x + 1))

# expm1  相当于  exp(x) - 1
print(expm1(1.3862943611198906), np.exp(1.3862943611198906) - 1)
# 同样的道理对于expm1，当x特别小，exp(x)-1就会急剧下降出现如上问题，甚至出现错误值。
print(expm1(x), np.exp(x) - 1)
'''