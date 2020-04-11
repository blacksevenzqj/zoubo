# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 00:18:36 2020

@author: dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
# 参考代码： chapter13_1 PCA_Loan_apply.py
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
# 参考代码： chapter13_2 PCA_FCA_Varselect_city10.py、chapter13_3 PCA_FCA_Varselect_bank.py
def factor_analysis(data, num_keep):  # data为只做因子分析的特征列数据
    from fa_kit import FactorAnalysis
    from fa_kit import plotting as fa_plotting

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


# -------------------------因子得分 进 聚类模型-----------------------------------

# 3、AGNES层次聚类： 因子分析 → 层次聚类 不常用（只能用于小数据量）
# 参考代码： chapter14_1 Hclus_FA_city10.py
def hierarchical_clustering(data, num_keep, columns, labels):  # data为只做因子分析的特征列数据
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


'''
一、快速聚类的两种运用场景 K-Means 聚类要点：
1、发现异常情况： 如果不对数据进行任何形式的转换，只是经过中心标准化或级差标准化就进行快速聚类，会根据数据分布特征得到聚类结果。
这种聚类会将极端数据聚为几类。方法一会演示这种情况。这种方法适用于统计分析之前的异常值剔除，对异常行为的挖掘，比如监控银行账户是否有洗钱行为、监控POS机是有从事套现、监控某个终端是否是电话卡养卡客户等等。

2、将个案数据做划分： 出于客户细分目的的聚类分析一般希望聚类结果为大致平均的几大类，因此需要将数据进行转换，
比如使用原始变量的百分位秩、 Turkey正态评分、对数转换等等。在这类分析中数据的具体数值并没有太多的意义，重要的是相对位置。
方法二会演示这种情况。这种方法适用场景包括客户消费行为聚类、客户积分使用行为聚类等等。
'''


# 4、KMeans： 基于 "2" 客户细分目的的聚类分析一般希望聚类结果为大致平均的几大类 的目标。 决策树是作为 验证 KMeans聚类结果 而存在。
# 参考代码： chapter14_2 Kmeans_FA_bank.py
# fa_score = dr.kmeans_roughly_average_categories_part1(data, 3, {0: "ATM_POS", 1: "TBM", 2: "CSC"}, model_data, 3)
def kmeans_roughly_average_categories_part1(data, num_keep, columns, original_data,
                                            init_n_clusters):  # data为只做因子分析的特征列数据
    if type(columns) != dict:
        raise Exception('columns Type is Error, must dict')

    # 一、因子分析：
    fa_score = factor_analysis(data, num_keep)  # 因子得分（DataFrame）
    ft.df_change_colname(fa_score, columns)

    # 1、初始的聚类： （查看 因子得分 的聚类情况，反应了 数据的真实 分布情况）
    kmeans = KMeans(n_clusters=init_n_clusters)  # MiniBatchKMeans()分批处理
    # kmeans = cluster.KMeans(n_clusters=3, init='random', n_init=1)
    result = kmeans.fit(fa_score)  # result 为 因子分析结果 → k-Means的结果
    original_data = original_data.join(pd.DataFrame(result.labels_))
    original_data = original_data.rename(columns={0: "clustor"})
    # 画饼图
    original_data.clustor.value_counts().plot(kind='pie')
    plt.show()

    # 2、检测 因子得分 偏度
    skew, kurt, var_x_ln = ft.skew_distribution_test(fa_score)
    if len(var_x_ln) > 0:
        from sklearn import preprocessing

        # Tukey正态分布打分（聚类模型）（幂变换：隐射到 正太分布）。 而 PowerTransformer 中的  Yeo-Johnson 和 Box-Cox（要求正数）都不行
        quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
        fa_score = quantile_transformer.fit_transform(fa_score)
        fa_score = pd.DataFrame(fa_score)
        fa_score = fa_score.rename(columns=columns)

        skew, kurt, var_x_ln = ft.skew_distribution_test(fa_score)

    return fa_score


# 使用 因子得分 作为特征 进 KMeans 得到聚类结果  →  原始特征（做因子分析的特征列） + 聚类结果作为因变量Y（多分类） 进 决策树 训练， 看叶子节点分类情况
# dr.kmeans_roughly_average_categories_part2(model_data, fa_score, 4, ['CNT_TBM', 'CNT_ATM', 'CNT_POS', 'CNT_CSC'])
def kmeans_roughly_average_categories_part2(original_data, fa_score, n_clusters, columns):  # columns为做 因子分析 的 特征列

    # 2、再聚类： 如果 因子得分 进行了 Tukey正态分布打分（聚类模型）（幂变换：隐射到 正太分布），那么业务就类似于： "2" 客户细分目的的聚类分析一般希望聚类结果为大致平均的几大类 的目标
    kmeans = KMeans(n_clusters=n_clusters)  # MiniBatchKMeans()分批处理
    # kmeans = cluster.KMeans(n_clusters=3, init='random', n_init=1)
    result = kmeans.fit(fa_score)  # result 为 因子分析结果 经过 变量分布正太转换后 k-Means的结果

    original_data = original_data.join(pd.DataFrame(result.labels_))  # model_data_l 只是为了展示 饼图
    original_data = original_data.rename(columns={0: "clustor"})

    # 画饼图
    original_data.clustor.value_counts().plot(kind='pie')
    plt.show()

    # 使用 决策树 对 KMeans结果进行验证（如果 决策树叶子分的不干净，则再重新调整KMeans的k值，再次得到聚类结果 → 决策树验证 ...）
    # 将 KMeans聚类的预测结果 作为 因变量Y（多分类） 和 原始数据特征X（做因子分析的特之列）： 一起进 决策树训练， 看叶子节点的分类结果是否干净（每个叶子节点：某单一类别数量很大，其余类别数量小）
    temp_data = original_data.loc[:, columns]

    from sklearn import tree

    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=2, min_samples_split=100, min_samples_leaf=100,
                                      random_state=12345)
    clf.fit(temp_data, result.labels_)

    import pydotplus
    from IPython.display import display, Image
    import sklearn.tree as tree

    temp_class_names = list()
    for i in range(n_clusters):
        temp_class_names.append(str(i))

    dot_data = tree.export_graphviz(clf,
                                    out_file=None,
                                    feature_names=temp_data.columns,
                                    class_names=temp_class_names,
                                    filled=True)

    graph = pydotplus.graph_from_dot_data(dot_data)
    display(Image(graph.create_png()))