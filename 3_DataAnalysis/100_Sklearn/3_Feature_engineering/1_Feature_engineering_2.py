# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 16:24:03 2019

@author: dell
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import os

os.chdir(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\3_Feature_engineering")

# In[]:
# 特征选择：
data = pd.read_csv(r".\digit recognizor.csv")
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
X.shape

# In[]:
# 1、方差选择： （不能有np.nan）
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold()  # 实例化，不填参数默认方差为0
X_var0 = selector.fit_transform(X)  # 获取删除不合格特征之后的新特征矩阵
X_var0.shape  # (42000, 708)

# In[]:
# 按 方差中位数 选取
# X.var() 每一列的方差
X_fsvar = VarianceThreshold(np.median(X.var().values)).fit_transform(X)
X_fsvar.shape  # (42000, 392)
# In[]:
# 按 方差90%分位数 选取
X_fsvar = VarianceThreshold(np.percentile(X.var().values, 90)).fit_transform(X)
X_fsvar.shape  # (42000, 78)

# In[]:
# 若特征是伯努利随机变量，假设p=0.8，即二分类特征中某种类别占到80%以上的时候删除该特征（有80%是同一类别）
X_bvar = VarianceThreshold(.8 * (1 - .8)).fit_transform(X)
X_bvar.shape

# In[]:
# 1.1、方差过滤对模型的影响：
# KNN vs 随机森林在不同方差过滤效果下的对比
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNN

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

X_fsvar = VarianceThreshold(np.median(X.var().values)).fit_transform(X)

# In[]:
'''
# 使用KNN测试 方差选择：
#======【TIME WARNING：20 mins+】======#
cross_val_score(KNN(),X_fsvar,y,cv=5).mean()

#======【TIME WARNING：2 hours】======#
cross_val_score(KNN(),X,y,cv=5).mean()
'''

# 使用随机森林测试 方差选择：
print(cross_val_score(RFC(n_estimators=10, random_state=0), X_fsvar, y, cv=5).mean())
print(cross_val_score(RFC(n_estimators=10, random_state=0), X, y, cv=5).mean())

# In[]:
# 2、卡方检验
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# 假设在这里我一直我需要300个特征
X_fschi = SelectKBest(chi2, k=300).fit_transform(X_fsvar, y)
X_fschi.shape

# In[]:
cross_val_score(RFC(n_estimators=10, random_state=0), X_fschi, y, cv=5).mean()

# In[]:
import matplotlib.pyplot as plt

score = []
for i in range(390, 200, -10):
    X_fschi = SelectKBest(chi2, k=i).fit_transform(X_fsvar, y)
    once = cross_val_score(RFC(n_estimators=10, random_state=0), X_fschi, y, cv=5).mean()
    score.append(once)
plt.plot(range(390, 200, -10), score)
plt.show()

# In[]:
chivalue, pvalues_chi = chi2(X_fsvar, y)
# k取多少？我们想要消除所有p值大于设定值，比如0.05或0.01的特征：
k = chivalue.shape[0] - (pvalues_chi > 0.05).sum()  # 392

# 所有392个特征都显著，再进行交叉检验：
X_fschi = SelectKBest(chi2, k=392).fit_transform(X_fsvar, y)
cross_val_score(RFC(n_estimators=10, random_state=0), X_fschi, y, cv=5).mean()

# In[]:
# F检验：
from sklearn.feature_selection import f_classif

F, pvalues_f = f_classif(X_fsvar, y)
k = F.shape[0] - (pvalues_f > 0.05).sum()

X_fsF = SelectKBest(f_classif, k=392).fit_transform(X_fsvar, y)
cross_val_score(RFC(n_estimators=10, random_state=0), X_fsF, y, cv=5).mean()

# In[]:
# 互信息：
# '''
# 消耗大
from sklearn.feature_selection import mutual_info_classif as MIC

result = MIC(X_fsvar, y)
k = result.shape[0] - sum(result <= 0)  # 392
# In[]:
X_fsmic = SelectKBest(MIC, k=392).fit_transform(X_fsvar, y)
cross_val_score(RFC(n_estimators=10, random_state=0), X_fsmic, y, cv=5).mean()
# '''


# In[]:
# Embedded嵌入法：
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np
import matplotlib.pyplot as plt

# 随机森林实例化
RFC_ = RFC(n_estimators=10, random_state=0)
# 筛选特征（数据）： 针对 树模型的 feature_importances_ 属性删选
X_embedded = SelectFromModel(RFC_, threshold=0.005).fit_transform(X, y)

sfmf = SelectFromModel(RFC_, threshold=0.005).fit(X, y)
X_embedded_index = sfmf.get_support(indices=True)  # 特征选择后 特征的 原列位置索引
X_embedded = sfmf.transform(X)
print(X.columns[X_embedded_index])  # 特征选择后 特征的 原列名称索引
# 在这里我只想取出来有限的特征。0.005这个阈值对于有780个特征的数据来说，是非常高的阈值，因为平均每个特征
# 只能够分到大约0.001 = 1/780 的feature_importances_
# 模型的维度明显被降低了

# In[]:
# 同样的，我们也可以画 threshold 的学习曲线来找最佳阈值（针对feature_importances_）
# ======【TIME WARNING：10 mins】======#
RFC_ = RFC(n_estimators=10, random_state=0)
print(len(RFC_.fit(X, y).feature_importances_))  # 784个，将所有特征打分
# 自定义阈值： 等差数列，从0到分数最大值，20个数 作为阈值。
threshold = np.linspace(0, (RFC_.fit(X, y).feature_importances_).max(), 20)

score = []
for i in threshold:
    X_embedded = SelectFromModel(RFC_, threshold=i).fit_transform(X, y)  # 筛选特征（数据）
    once = cross_val_score(RFC_, X_embedded, y, cv=5).mean()
    score.append(once)
plt.plot(threshold, score)
plt.show()

# In[]:
# 和其他调参一样，我们可以在第一条学习曲线后选定一个范围（0至0.00134），使用细化的学习曲线来找到最佳值：
# ======【TIME WARNING：10 mins】======#
score2 = []
for i in np.linspace(0, 0.00134, 20):  # 只是调整了结束值
    X_embedded = SelectFromModel(RFC_, threshold=i).fit_transform(X, y)
    once = cross_val_score(RFC_, X_embedded, y, cv=5).mean()
    score2.append(once)
plt.figure(figsize=[20, 5])
plt.plot(np.linspace(0, 0.00134, 20), score2)
plt.xticks(np.linspace(0, 0.00134, 20))
plt.show()
# In[]:
# 使用最高点0.000564阈值再行测试：
X_embedded = SelectFromModel(RFC_, threshold=0.000564).fit_transform(X, y)  # 340
cross_val_score(RFC_, X_embedded, y, cv=5).mean()  # 0.9408335415056387 迄今为止最高值
# In[]:
# 我们可能已经找到了现有模型下的最佳结果，如果我们调整一下随机森林的参数呢？n_estimators=100
# =====【TIME WARNING：2 min】=====#
cross_val_score(RFC(n_estimators=100, random_state=0), X_embedded, y, cv=5).mean()  # 0.9639525817795566

# In[]:
# 运算量太大
estimators_list = [30, 50, 100, 150, 200]
for i in estimators_list:
    RFC_ = RFC(n_estimators=i, random_state=0)
    temp_feature_imp = RFC_.fit(X, y).feature_importances_
    print(len(temp_feature_imp))  # 784个，将所有特征打分
    # 自定义阈值： 等差数列，从0到分数最大值，20个数 作为阈值。
    threshold = np.linspace(0, (temp_feature_imp).max(), 20)

    score = []
    for i in threshold:
        X_embedded = SelectFromModel(RFC_, threshold=i).fit_transform(X, y)  # 筛选特征（数据）
        once = cross_val_score(RFC_, X_embedded, y, cv=5).mean()
        score.append(once)
    plt.plot(threshold, score)
    plt.show()

# In[]:
# Wrapper包装法：
from sklearn.feature_selection import RFE

RFC_ = RFC(n_estimators=10, random_state=0)

# 迭代法： 每次迭代删除50个特征
selector = RFE(RFC_, n_features_to_select=340, step=50).fit(X, y)
selector.support_.sum()  # 返回所有的特征的是否最后被选中的布尔矩阵。 340
selector.ranking_  # 返回特征的按数次迭代中综合重要性的排名

X_wrapper = selector.transform(X)
cross_val_score(RFC_, X_wrapper, y, cv=5).mean()

# In[]:
# 学习曲线：
# ======【TIME WARNING: 15 mins】====== #
score = []
for i in range(1, 751, 50):
    X_wrapper = RFE(RFC_, n_features_to_select=i, step=50).fit_transform(X, y)
    once = cross_val_score(RFC_, X_wrapper, y, cv=5).mean()
    score.append(once)
plt.figure(figsize=[20, 5])
plt.plot(range(1, 751, 50), score)
plt.xticks(range(1, 751, 50))
plt.show()






