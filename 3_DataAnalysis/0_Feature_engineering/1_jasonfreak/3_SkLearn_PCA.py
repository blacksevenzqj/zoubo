from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


iris = load_iris()
print(iris.data[0:5])
# print(iris.target[0:5])
print(set(iris.target))


# 4.1 主成分分析法（PCA）
# 使用decomposition库的PCA类选择特征的代码如下：
from sklearn.decomposition import PCA
# 主成分分析法，返回降维后的数据
# 参数n_components为主成分数目
pdata = PCA(n_components=2).fit_transform(iris.data)
print(pdata[0:5])


# 4.2、线性判别分析法（LDA）包有问题，没试了
# 使用lda库的LDA类选择特征的代码如下：
# from sklearn.lda import LDA
# # 线性判别分析法，返回降维后的数据
# # 参数n_components为降维后的维数
# ldata = LDA(n_components=2).fit_transform(iris.data, iris.target)
# print(ldata[0:5])