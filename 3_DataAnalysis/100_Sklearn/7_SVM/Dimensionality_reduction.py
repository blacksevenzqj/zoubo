# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 00:18:36 2020

@author: dell
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA


def z_score(data):
    return preprocessing.scale(data)


'''
# 此处作主成分分析，主要是进行冗余变量的剔出，因此注意以下两个原则：
# 1、保留的变量个数尽量多，累积的explained_variance_ratio_尽量大，比如阈值设定为0.95
# 2、只剔出单位根非常小的变量，比如阈值设定为0.2
'''


def pca_test(pca_data, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(pca_data)
    # explained_variance_： 解释方差
    variance_ = pca.explained_variance_
    print("解释方差：")
    print(variance_)
    # explained_variance_ratio_： 解释方差占比（累计解释方差占比 自己手动加）
    variance_ratio_ = pca.explained_variance_ratio_
    print("解释方差占比：")
    print(variance_ratio_)

    ratioAccSum = 0.00
    ratioIndex = 0
    ratioValue = 0.00
    for i in pca.explained_variance_ratio_:
        ratioAccSum += i
        ratioIndex += 1
        if ratioAccSum >= 0.95:
            ratioValue = i
            break
    print("累计解释方差占比%f，共个%d个特征，末位特征解释方差占比%f" % (ratioAccSum, ratioIndex, ratioValue))


'''
pca = PCA(n_components=1).fit(data) # 取一个主成分
# pca.fit_transform(data) 就是 PX = Y， data是X、计算结果是Y： 通过 特征矩阵P（特征矩阵P共5维：因X的特征为5维）现只取一个主成分（即只取P的第一行） 将 X 降到 1维。
newdata = pca.fit_transform(data) # newdata 就是 降维结果Y， 意思就是将 5维特征的X 压缩到 1维特征


通过 主成分P 在每个特征上的权重的绝对值大小，确定每个 主成分P 的代表性
print(pca.components_)
print(pd.DataFrame(pca.components_).T) # 以 列 的方式呈现

Dmatrix = pca.components_.dot(data_val) # W · X^T
print(Dmatrix) # 结果和 pca.fit_transform(data) 相同

Dmatrix2 = data.dot(pca.components_.T) # X · W^T
print(Dmatrix2.T) # 结果和 pca.fit_transform(data) 相同
'''


