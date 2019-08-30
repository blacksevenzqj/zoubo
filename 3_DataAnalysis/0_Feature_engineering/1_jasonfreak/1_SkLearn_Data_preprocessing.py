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
fdata = FunctionTransformer(np.exp).fit_transform(aaa)
print(fdata)
print(np.log(aaa + 1))
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