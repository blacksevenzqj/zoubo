
# coding: utf-8

# 主要功能的帮助文档：
# [matplotlib](http://matplotlib.org/1.4.3/contents.html)
# [seaborn](http://web.stanford.edu/~mwaskom/software/seaborn/tutorial.html)
# [pandas](http://pandas.pydata.org/pandas-docs/version/0.16.0/)
# [scikit-learn](http://scikit-learn.org/stable/)

#subscriberID="个人客户的ID"
#churn="是否流失：1=流失";
#Age="年龄"
#incomeCode="用户居住区域平均收入的代码"
#duration="在网时长"
#peakMinAv="统计期间内最高单月通话时长"
#peakMinDiff="统计期间结束月份与开始月份相比通话时长增加数量"
#posTrend="该用户通话时长是否呈现出上升态势：是=1"
#negTrend="该用户通话时长是否呈现出下降态势：是=1"
#nrProm="电话公司营销的数量"
#prom="最近一个月是否被营销过：是=1"
#curPlan="统计时间开始时套餐类型：1=最高通过200分钟；2=300分钟；3=350分钟；4=500分钟"
#avPlan="统计期间内平均套餐类型"
#planChange="统计结束时和开始时套餐的变化：正值代表套餐档次提升，负值代表下降，0代表不变"
#posPlanChange="统计期间是否提高套餐：1=是"
#negPlanChange="统计期间是否降低套餐：1=是"
#call_10086="拨打10086的次数"

# In[1]:
import pandas as pd
import numpy as np

import os
os.chdir(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\TianshanCollege\6_DecisionTrees_and_NeuralNetworks\DT")

# In[2]:
churn = pd.read_csv('telecom_churn.csv')  # 读取已经整理好的数据
churn.head()
print(churn.shape)

# In[4]:
import matplotlib.pyplot as plt
import seaborn as sns

# In[5]:
sns.barplot(x='edu_class', y='churn',data=churn)
plt.show()

# In[8]:
sns.boxplot(x='churn', y='peakMinDiff', hue=None, data=churn)
plt.show()

# In[10]:
sns.boxplot(x='churn', y='duration', hue='edu_class', data=churn)
plt.show()



# 一、筛选变量： 相关性分析
# * 筛选变量时可以应用专业知识，选取与目标字段相关性较高的字段用于建模，也可通过分析现有数据，用统计量辅助选择
# * 为了增强模型稳定性，自变量之间最好相互独立，可运用统计方法选择要排除的变量或进行变量聚类
# In[11]:
# 连续 - 连续：分类特征转换为数字也可分析，自变量之间检验
# 1.1、spearman 相关性分析 
corrmatrix = churn.corr(method='spearman')  # spearman相关系数矩阵，可选pearson相关系数，目前仅支持这两种,函数自动排除 category 类型（我猜只是排除字符串类型，数字作为分类标识并不排除）
corrmatrix_new = corrmatrix[np.abs(corrmatrix) > 0.5]  # 选取相关系数绝对值大于0.5的变量，仅为了方便查看
#  为了增强模型稳定，根据上述相关性矩阵，可排除'posTrend','planChange','nrProm','curPlan'几个变量
# 1.2、pearson 相关性分析 
corrmatrix1 = churn.corr(method='pearson') 
corrmatrix_new1 = corrmatrix[np.abs(corrmatrix) > 0.5]


# In[12]:
# 2、卡方检验：分类 - 分类：工程上将连续特征 分桶后 转换为分类特征 进行分析，自变量与因变量之间检验
# 连续型变量往往是模型不稳定的原因;
# 如果所有的连续变量都分箱了,可以统一使用卡方检验进行变量重要性检验
churn['duration_bins'] = pd.qcut(churn.duration,5)  #  将duration字段切分为数量（大致）相等的5段
churn['churn'].astype('int64').groupby(churn['duration_bins']).agg(['count', 'mean'])
#print(type(churn['duration_bins']), churn['duration_bins'].dtype) # Series, category

# In[14]:
bins = [0, 4, 8, 12, 22, 73] # 指定分割区间
churn['duration_bins'] = pd.cut(churn['duration'], bins, labels=False) # 不要标签，类型变为int
churn['churn'].astype('int64').groupby(churn['duration_bins']).agg(['mean', 'count'])
#print(type(churn['duration_bins']), churn['duration_bins'].dtype) # Series, int64

# In[15]:
# 2.1、使用 sklearn 包：
# 根据卡方值选择与目标关联较大的分类变量
# 计算卡方值需要应用到sklearn模块，但该模块当前版本不支持pandas的category类型变量，会出现警告信息，可忽略该警告或将变量转换为int类型
import sklearn.feature_selection as feature_selection

# type(churn['gender']) 为 Series， churn['gender'].dtype 为 int32
churn['gender'] = churn['gender'].astype('int') # 将Series中的元素转换为int类型，原本就是了，只是教你怎么转换。
churn['edu_class'] = churn['edu_class'].astype('int')
churn['feton'] = churn['feton'].astype('int')
# 每个特征 分别与 churn['churn'] 进行卡方检验（所以 自由度v = (2-1) * (2-1) = 1）
feature_selection.chi2(churn[['gender', 'edu_class', 'feton', 'prom', 
                              'posPlanChange','duration_bins', 'curPlan', 'call_10086']], churn['churn'])#选取部分字段进行卡方检验

# 使用 SelectKBest 方法直接选出 x 个特征：
#X_new = feature_selection.SelectKBest(feature_selection.chi2, k=2).fit_transform(churn[['gender', 'edu_class', 'feton', 'prom', 
#                              'posPlanChange','duration_bins', 'curPlan', 'call_10086']], churn['churn'])
#print(X_new.shape)
#根据结果显示，'prom'、'posPlanChange'、'curPlan'字段可以考虑排除

#%%
# 2.2、自定义卡方检验：
from scipy import stats

sklearn_chi2 = feature_selection.chi2(churn[['prom']], churn['churn'])
print(sklearn_chi2) # (array([1.11235701]), array([0.29157016])) 卡方值，显著度α

stats_crosstab = pd.crosstab(churn['prom'],churn['churn'], margins=True)
print(stats_crosstab)
# chisq 卡方值
# p-value 卡方值对应的 显著度α，用p-value表示。 显著以否 的衡量标准，和 两样本T检验 的P值是一样的意思。
# expected_freq 卡方检验的 期望频率 = (行合计 * 列合计) / 总和
print('''chisq = %6.4f
p-value = %6.4f
dof = %i 
expected_freq = %s'''  %stats.chi2_contingency(stats_crosstab.iloc[:2, :2]))
# 具体的分析看笔记： “1.1、卡方检验_例子分析”



# 建模
# * 根据数据分析结果选取建模所需字段，同时抽取一定数量的记录作为建模数据
# * 将建模数据划分为训练集和测试集
# * 选择模型进行建模
# In[16]:
# 根据模型不同，对自变量类型的要求也不同，为了示例，本模型仅引入'AGE'这一个连续型变量
#model_data = churn[['subscriberID','churn','gender','edu_class','feton','duration_bins']]
model_data = churn[['subscriberID','churn','gender','edu_class','feton','duration_bins','call_10086','AGE']]#第二可选方案
model_data.head()

# In[17]:
from sklearn.model_selection import train_test_split

target = model_data['churn']  # 选取目标变量（因变量）
data=model_data.ix[:, 'gender':]  # 选取自变量

train_data, test_data, train_target, test_target = train_test_split(data,target,test_size=0.4,train_size=0.6,random_state=12345) # 划分训练集和测试集

# In[18]:
# 选择决策树进行建模
import sklearn.tree as tree

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=5) # 当前支持计算信息增益和GINI
clf.fit(train_data, train_target) # 使用训练数据建模

# 查看模型预测结果
train_est = clf.predict(train_data) # 用模型预测训练集的结果
# 用模型预测训练集的概率，预测为1的概率（注意他只取类别1的预测概率用于后面直方图）
train_est_p = clf.predict_proba(train_data)[:,1] # numpy.ndarray

test_est = clf.predict(test_data) # 用模型预测测试集的结果
# 用模型预测测试集的概率，预测为1的概率（注意他只取类别1的预测概率用于后面直方图）
test_est_p = clf.predict_proba(test_data)[:,1]

#pd.DataFrame({'test_target':test_target,'test_est':test_est,'test_est_p':test_est_p}).T # 查看测试集预测结果与真实结果对比

print(test_data[0:10])
print(test_target[0:10])
print(test_est[0:10])
print(test_est_p[0:10])
print(test_target[0:10] == 1)
print(test_est_p[0:10][test_target[0:10] == 1])

# 模型评估
# In[19]:
import sklearn.metrics as metrics

print(metrics.confusion_matrix(test_target, test_est, labels=[0,1])) # 混淆矩阵
print(metrics.classification_report(test_target, test_est)) # 计算评估指标
print(pd.DataFrame(list(zip(data.columns, clf.feature_importances_))))  # 变量重要性指标


# In[20]:
# 察看预测值的分布情况
red, blue = sns.color_palette("Set1", 2)
sns.distplot(test_est_p[0:10][test_target[0:10] == 1], kde=False, bins=15, color=red)
sns.distplot(test_est_p[0:10][test_target[0:10] == 0], kde=False, bins=15, color=blue)

#sns.distplot(test_est_p[test_target == 1], kde=False, bins=15, color=red)
#sns.distplot(test_est_p[test_target == 0], kde=False, bins=15, color=blue)
plt.show()

# In[21]:
fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_est_p)
fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est_p)
plt.figure(figsize=[6,6])
plt.plot(fpr_train, tpr_train, color=red)
plt.plot(fpr_test, tpr_test, color=blue)
plt.title('ROC curve')
plt.show()
#这里表现出了过渡拟合的情况（训练集ROC 和 测试集ROC 拟合度不好）


#参数调优
#%%
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

param_grid = {
    'max_depth':[2,3,4,5,6,7,8],
    'min_samples_split':[4,8,12,16,20,24,28] # 数值越大模型越简单
}
clf = tree.DecisionTreeClassifier(criterion='entropy')
clfcv = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc', cv=4)
clfcv.fit(train_data, train_target)

#%%
# 查看模型预测结果
train_est = clfcv.predict(train_data)  #  用模型预测训练集的结果
train_est_p = clfcv.predict_proba(train_data)[:,1]  #用模型预测训练集的概率
test_est = clfcv.predict(test_data)  #  用模型预测测试集的结果
test_est_p = clfcv.predict_proba(test_data)[:,1]  #  用模型预测测试集的概率

#%%
fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_est_p)
fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est_p)
plt.figure(figsize=[6,6])
plt.plot(fpr_train, tpr_train, color=red)
plt.plot(fpr_test, tpr_test, color=blue)
plt.title('ROC curve')
plt.show()

#%%
clfcv.best_params_

#%%
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=24) # 当前支持计算信息增益和GINI
clf.fit(train_data, train_target) # 使用训练数据建模

#%%
# 可视化
# 使用dot文件进行决策树可视化需要安装一些工具：
# - 第一步是安装[graphviz](http://www.graphviz.org/)。linux可以用apt-get或者yum的方法安装。如果是windows，就在官网下载msi文件安装。无论是linux还是windows，装完后都要设置环境变量，将graphviz的bin目录加到PATH，比如windows，将C:/Program Files (x86)/Graphviz2.38/bin/加入了PATH
# - 第二步是安装python插件graphviz： pip install graphviz
# - 第三步是安装python插件pydotplus: pip install pydotplus

# In[39]:
import pydotplus
from IPython.display import Image
import sklearn.tree as tree

dot_data = tree.export_graphviz(
    clf, 
    out_file=None, 
    feature_names=train_data.columns,
    max_depth=5,
    class_names=['0','1'],
    filled=True
) 
            
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 


# In[36]:
"""
# 模型保存/读取
import pickle as pickle

model_file = open(r'clf.model', 'wb')
pickle.dump(clf, model_file)
model_file.close()

# In[37]:
model_load_file = open(r'clf.model', 'rb')
model_load = pickle.load(model_load_file)
model_load_file.close()

test_est_load = model_load.predict(test_data)
pd.crosstab(test_est_load,test_est)
#%%
"""
