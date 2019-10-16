from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from time import time
import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler

# In[]:
data_cancer = load_breast_cancer()
X = data_cancer.data.copy()
y = data_cancer.target.copy()

print(X.shape)
print(np.unique(y), set(y))
# In[]:
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


# In[]:
# PCA_SVD联合降维
from sklearn.decomposition import PCA

pca = PCA(2, whiten=True, svd_solver='auto').fit(X)
V = pca.components_  # 新特征空间
print(V.shape)  # V(k，n)

X_dr = pca.transform(X)  # PCA降维后的信息保存量
print(X_dr.shape)

plt.scatter(X_dr[:, 0], X_dr[:, 1], c=y)
plt.show()


# In[]:
# t-SNE降维
from sklearn import manifold

# init='pca'：初始化，默认为random。取值为random为随机初始化，取值为pca为利用PCA进行初始化（常用）
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)
print(X_tsne.shape)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.show()


# In[]:
# 数据分布检测：
# 虽然我们不能说SVM是完全的距离类模型，但是它严重受到数据量纲的影响。让我们来探索一下乳腺癌数据集的量纲
data = pd.DataFrame(X.copy())
temp_desc_svm = data.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T  # 描述性统计
# temp_desc_svm.to_csv("C:\\Users\\dell\\Desktop\\123123\\temp_desc_svm.csv")
'''
从mean列和std列可以看出严重的量纲不统一
从1%的数据和最小值相对比，90%的数据和最大值相对比，查看是否是正态分布或偏态分布，如果差的太多就是偏态分布，谁大方向就偏向谁
可以发现数据大的特征存在偏态问题
这个时候就需要对数据进行标准化
'''


# In[]:
# 正太分布检测：
'''
原假设：样本来自一个正态分布的总体。
备选假设：样本不来自一个正态分布的总体。
w和p同向： w值越小； p-值越小、接近于0； 拒绝原假设。
'''
var = data.columns
shapiro_var = {}
for i in var:
    shapiro_var[i] = stats.shapiro(data[i])  # 返回 w值 和 p值

shapiro = pd.DataFrame(shapiro_var).T.sort_values(by=1, ascending=False)

fig, axe = plt.subplots(1, 1, figsize=(15, 10))
axe.bar(shapiro.index, shapiro[0], width=.4)  # 自动按X轴---skew.index索引0-30的顺序排列
# 在柱状图上添加数字标签
for a, b in zip(shapiro.index, shapiro[0]):
    # a是X轴的柱状体的索引， b是Y轴柱状体高度， '%.4f' % b 是显示值
    plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=12)
plt.show()


# In[]:
# 数据偏度检测：
var = data.columns
skew_var = {}
for i in var:
    skew_var[i] = abs(data[i].skew())

skew = pd.Series(skew_var).sort_values(ascending=False)

fig, axe = plt.subplots(1, 1, figsize=(15, 10))
axe.bar(skew.index, skew, width=.4)  # 自动按X轴---skew.index索引0-30的顺序排列

# 在柱状图上添加数字标签
for a, b in zip(skew.index, skew):
    # a是X轴的柱状体的索引， b是Y轴柱状体高度， '%.4f' % b 是显示值
    plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=12)
plt.show()


# In[]:
var_x_ln = skew.index[skew > 0.9]  # skew的索引 --- data的列名
print(var_x_ln, len(var_x_ln))
# In[]:
fig, axe = plt.subplots(len(var_x_ln), 1, figsize=(20, 28 * 6))

for i, var in enumerate(var_x_ln):
    sns.distplot(data[var], bins=100, color='green', ax=axe[i])
    axe[i].set_title('feature: ' + str(var))
    axe[i].set_xlabel('')
plt.show()


# In[]:
# 将偏度大于1的连续变量 取对数
for i in var_x_ln:
    if min(data[i]) <= 0:
        data[i] = np.log(data[i] + abs(min(data[i])) + 0.01)  # 负数取对数的技巧
    else:
        data[i] = np.log(data[i])

X = data.values