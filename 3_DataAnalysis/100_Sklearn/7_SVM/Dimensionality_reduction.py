# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 00:18:36 2020

@author: dell
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import FeatureTools as ft


# In[]:


def z_score(data):
    return preprocessing.scale(data)


'''
# 此处作主成分分析，主要是进行冗余变量的剔出，因此注意以下两个原则：
# 1、保留的变量个数尽量多，累积的explained_variance_ratio_尽量大，比如阈值设定为0.95
# 2、只剔出单位根非常小的变量，比如阈值设定为0.2
'''


# 1、主成分分析：
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


# 2、因子分析： 1、data必须经过 标准化（Z分数）；  2、需PCA已经确定 保留主成分个数
# 参考代码： chapter13_2 PCA_FCA_Varselect_city10.py
def factor_analysis(data, num_keep):
    from fa_kit import FactorAnalysis
    from fa_kit import plotting as fa_plotting
    import matplotlib.pyplot as plt

    # 一、计算特征向量矩阵P：（因子旋转前 主成分）； 通过主成分在每个变量上的权重的绝对值大小，确定每个主成分的代表性
    pca = PCA(n_components=num_keep).fit(data)  # PCA确定保留主成分个数
    e_matrix = pd.DataFrame(pca.components_).T  # 以 列 的方式呈现 （默认是以 特征向量 行 的方式呈现）
    print("因子旋转前的 特征向量矩阵P：（因子旋转前 主成分）：（P已降维）")
    print(e_matrix)
    print("-" * 30)

    # 另1： 如果是两个主成分： 可以做 因子旋转前的 两个主成分作散点图

    # 二、因子分析：
    # 1、数据导入和转换
    fa = FactorAnalysis.load_data_samples(
        data,  # 数据必须经过 标准化（Z分数）
        preproc_demean=True,
        preproc_scale=True
    )

    # 2、抽取主成分
    fa.extract_components()

    # 3、使用top_n法保，留2个主成分，上面PCA已算出
    fa.find_comps_to_retain(method='top_n', num_keep=num_keep)  # num_keep 保留主成分个数

    # 4、varimax： 使用 最大方差法 进行 因子旋转
    fa.rotate_components(method='varimax')

    # 5、因子旋转后的 因子权重（因子载荷矩阵A） 相当于 特征矩阵P
    fas = pd.DataFrame(fa.comps["rot"])  # rot： 使用因子旋转法； 默认以 因子权重 列 的方式呈现
    print("因子旋转后的 因子权重（因子载荷矩阵A）：（A已降维）")
    print(fas)
    print("-" * 30)

    # 6、第一图为主成分保留个数； 第二、三图为 因子旋转 前、后 的因子权重 可视化
    fa_plotting.graph_summary(fa)
    plt.show()
    # - 说明：可以通过第三张图观看 因子旋转后 每个因子在每个变量上的权重，权重越高，代表性越强

    # 另2： 如果是两个主成分： 可以做 因子旋转后的 因子权重（因子载荷矩阵A） 两个因子权重作散点图

    # 7、因子得分：
    # 到目前还没有与PCA中fit_transform类似的函数，因此只能手工计算因子得分
    # 以下是矩阵相乘的方式计算因子得分： 因子得分 = 原始数据（n*k） · 权重矩阵(k*num_keep)
    fa_score = pd.DataFrame(np.dot(data, fas))  # 注意data数据需要标准化
    print("因子得分：（结果已降维）")
    print(fa_score)
    print("-" * 30)

    # 另3： 如果是两个主成分： 可以做 因子得分的散点图

    return fa_score


# 3、AGNES层次聚类： 因子分析 → 层次聚类 不常用（只能用于小数据量）
def hierarchical_clustering(data, num_keep, columns, labels):
    if type(columns) != dict:
        raise Exception('columns Type is Error, must dict')
    if type(labels) != list and type(labels) != np.ndarray:
        raise Exception('labels Type is Error, must list or np.ndarray')

    # 一、因子分析：
    fa_score = factor_analysis(data, num_keep)  # 因子得分（DataFrame）
    ft.df_change_colname(fa_score, columns)

    # 二、层次聚类：AGNES
    import scipy.cluster.hierarchy as sch
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    import matplotlib.pyplot as plt

    # 1.2、直接传入 坐标点数据（矩阵）
    Z = sch.linkage(fa_score, metric='euclidean', method='ward')

    # 将层级聚类结果以树状图表示出来并保存为plot_dendrogram.png
    P = sch.dendrogram(Z, labels=labels)  # labels 可以接受 list 或 numpy.ndarray
    plt.show()

    plt.savefig('plot_dendrogram1.png')
    cluster = sch.fcluster(Z, t=1)



