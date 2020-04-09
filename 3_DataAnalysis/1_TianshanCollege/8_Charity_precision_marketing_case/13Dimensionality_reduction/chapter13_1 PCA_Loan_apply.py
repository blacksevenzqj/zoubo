# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 21:26:31 2018

@author: Ben
"""

# coding: utf-8
"""
X1 品格：指客户的名誉；
X2 能力：指客户的偿还能力；
X3 资本：指客户的财务实力和财务状况；
X4 担保：指对申请贷款项担保的覆盖程度；
X5 环境：指外部经济、政策环境对客户的影响
"""

# 一、主成分分析
# - 1、引入数据
# In[1]:
import pandas as pd
import os
os.chdir(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\1_TianshanCollege\8_Charity_precision_marketing_case\13Dimensionality_reduction")
model_data = pd.read_csv("Loan_aply.csv",encoding='gbk')
model_data.head()

# In[2]:
data = model_data.loc[ :,'X1':]
data.head()

# - 2、查看相关系数矩阵，判定做变量降维的必要性（非必须）
# In[3]:
# 直接做 皮尔森相关度检验 和 将数据做了Z分数之和做皮尔森相关度 结果是相同的。
corr_matrix = data.corr(method='pearson')
print(corr_matrix)

# - 3、做主成分之前，进行中心标准化
# In[4]:
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# 等同于 StandardScaler().fit_transform(data)
data = preprocessing.scale(data)
print(data)
data_val = data.T
print(data_val)

# - 4、使用sklearn的主成分分析，用于判断保留主成分的数量
# In[5]:
from sklearn.decomposition import PCA
'''
1、主成分个数的选取原则：
1.1、单个主成分解释的变异（方差）不因小于1。
1.2、选取主成分累计的解释变异达到80%-90%。
说明：1、第一次的n_components参数应该设的大一点（保留主成分个数）
说明：2、观察explained_variance_ratio_和explained_variance_的取值变化，
建议explained_variance_ratio_累积大于0.85，explained_variance_需要保留的最后一个主成分大于0.8，
'''
#pca=PCA(n_components=3)
pca=PCA()
pca.fit(data)
# explained_variance_： 解释方差
print(pca.explained_variance_) # 建议保留1个主成分
'''
解释方差：
[4.67909448 0.42595504 0.33051612 0.0883994  0.03159051]
只有 第一个主成分的方差 > 1
'''

# 有5个特征，做了ZScore标准化： 方差和大概为5，那么以第一个主成分为例： 解释方差4.67909448，解释方差占比 = 4.67909448/大概为5 = 0.93
# 和实际 解释方差占比 0.84223701 有差距， 但可以这样理解。

# explained_variance_ratio_： 解释方差占比（累计解释方差占比 自己手动加）
print(pca.explained_variance_ratio_) #建议保留1个主成分
'''
解释方差占比：
[0.84223701 0.07667191 0.0594929  0.01591189 0.00568629]
累计解释方差占比：
0.84223701 已经达到标准，而 0.84223701 + 0.07667191 = 0.91890892虽然更好，但根据上面 解释方差 还是只选第一个主成分。 
'''

#%%
pca = PCA(n_components=1).fit(data) # 取一个主成分
'''
pca.fit_transform(data) 就是 PX = Y， data是X、计算结果是Y： 通过 特征矩阵P（特征矩阵P共5维：因X的特征为5维）现只取一个主成分（即只取P的第一行） 将 X 降到 1维。
'''
newdata = pca.fit_transform(data) # newdata 就是 降维结果Y， 意思就是将 5维特征的X 压缩到 1维特征
citi10_pca = model_data.join(pd.DataFrame(newdata))
print(citi10_pca)

# In[6]:
'''
通过 主成分P 在每个特征上的权重的绝对值大小，确定每个 主成分P 的代表性
pca.components_ 即为上面PCA模型中训练好的 特征向量矩阵P
因上面只取了一个主成分，所以 pca.components_ 为 1行5列 （按照 特征矩阵 的格式看）
[[0.41348998 0.47289329 0.46559941 0.45465337 0.42650378]] 点乘 X^T 就得到 降维后的数据的转置。
'''
print(pca.components_)
print(pd.DataFrame(pca.components_).T) # 以 列 的方式呈现


Dmatrix = pca.components_.dot(data_val) # W · X^T
print(Dmatrix) # 结果和 pca.fit_transform(data) 相同


Dmatrix2 = data.dot(pca.components_.T) # X · W^T
print(Dmatrix2.T) # 结果和 pca.fit_transform(data) 相同


