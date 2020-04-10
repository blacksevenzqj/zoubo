
# coding: utf-8
"""
X1	GDP
X2	人均GDP
X3	工业增加值
X4	第三产业增加值
X5	固定资产投资
X6	基本建设投资
X7	社会消费品零售总额
X8	海关出口总额
X9	地方财政收入
"""
# 第十四讲 聚类
# 1、层次聚类
# 第一步：手动测试主成分数量
# 1.1、引入数据
import os
import Dimensionality_reduction as dr
os.chdir(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\1_TianshanCollege\9_Bank_customer_channel\case\14Clustor")

# In[1]:
import pandas as pd

model_data = pd.read_csv("cities_10.csv",encoding='gbk')
model_data.head()

# In[2]:
data = model_data.loc[ :,'X1':]
data.head()

# 2、查看相关系数矩阵，判定做变量降维的必要性（非必须）
# In[3]:
corr_matrix = data.corr(method='pearson')
#corr_matrix = corr_matrix.abs()
corr_matrix

# 3、做主成分之前，进行中心标准化
# In[4]:
from sklearn import preprocessing

data = preprocessing.scale(data)

# 4、使用sklearn的主成分分析，用于判断保留主成分的数量
# In[5]:
from sklearn.decomposition import PCA
'''说明：1、第一次的n_components参数应该设的大一点
   说明：2、观察explained_variance_ratio_和explained_variance_的取值变化，建议explained_variance_ratio_累积大于0.85，explained_variance_需要保留的最后一个主成分大于0.8，
'''
pca=PCA(n_components=3)
newData=pca.fit(data)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)



# In[6]:
# 第二步：根据主成分分析确定需要保留的主成分数量，进行因子分析
# 1、导入包，并对输入的数据进行主成分提取。为保险起见，data需要进行中心标准化
# In[7]:
from fa_kit import FactorAnalysis
from fa_kit import plotting as fa_plotting

fa = FactorAnalysis.load_data_samples(
        data,
        preproc_demean=True,
        preproc_scale=True
        )
fa.extract_components()

# 2、设定提取主成分的方式。默认为“broken_stick”方法，建议使用“top_n”法
# In[8]:
fa.find_comps_to_retain(method='top_n',num_keep=2)

# 3、通过最大方差法进行因子旋转
# In[9]:
fa.rotate_components(method='varimax')

# 因子旋转后的 因子权重（因子载荷矩阵A）
temp = pd.DataFrame(fa.comps["rot"]) # rot： 使用因子旋转法
print(temp)

fa_plotting.graph_summary(fa)
# 说明：可以通过第三张图观看每个因子在每个变量上的权重，权重越高，代表性越强

# 4、获取因子得分（是因子得分，不是因子，相当于PCA降维后的数据结果）
# In[19]:
import numpy as np

fas = pd.DataFrame(fa.comps["rot"])
score = pd.DataFrame(np.dot(data, fas))
# In[19]:
# 自己封装的：
fa_score = dr.factor_analysis(data, 2)



# 第三步：根据因子得分进行数据分析
# In[25]:
a = score.rename(columns={0: "Gross", 1: "Avg"})
citi10_fa = model_data.join(a)

# In[49]:
# 如遇中文显示问题可加入以下代码
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

import matplotlib.pyplot as plt

x = citi10_fa['Gross']
y = citi10_fa['Avg']
label = citi10_fa['AREA']
plt.scatter(x, y)
for a,b,l in zip(x,y,label):
    plt.text(a, b+0.1, '%s.' % l, ha='center', va= 'bottom',fontsize=14)

plt.show()
# 从图中可以看出，如果只有2个主成分，作二维散点图已经很明显的能区分数据趋势

#%%
import scipy.cluster.hierarchy as sch

# 1、层次聚类：AGNES

# 1.1、生成点与点之间的距离矩阵,这里用的欧氏距离:（如：[4,2]的坐标点矩阵 计算后 得到是一维向量，不是距离矩阵形式）
#disMat = sch.distance.pdist(citi10_fa[['Gross','Avg']],'euclidean')
# 进行层次聚类:
#Z = sch.linkage(disMat,method='ward')

# 1.2、直接传入 坐标点数据（矩阵）
Z = sch.linkage(citi10_fa[['Gross','Avg']], metric='euclidean', method='ward')

# 将层级聚类结果以树状图表示出来并保存为plot_dendrogram.png
P = sch.dendrogram(Z,labels=['辽宁','山东','河北','天津','江苏','上海','浙江','福建','广东','广西'])
plt.savefig('plot_dendrogram1.png')

cluster = sch.fcluster(Z, t=1)

# In[49]:
dr.hierarchical_clustering(data, 2, {0: "Gross", 1: "Avg"}, model_data['AREA'].values)
# In[49]:
type(model_data['AREA'].values)
#type(list())




