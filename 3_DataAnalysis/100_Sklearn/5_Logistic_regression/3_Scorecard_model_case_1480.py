# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:15:08 2019

@author: 1480
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import warnings
import os

os.chdir(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\5_Logistic_regression")
train_data = pd.read_csv(r"rankingcard.csv",index_col=0)

# In[]:
# 1.1、查看好坏客户分布
#创建子图及间隔设置
f,ax = plt.subplots(1,1, figsize=(10,5))
sns.countplot('SeriousDlqin2yrs',data=train_data)
plt.show()

badnum=train_data['SeriousDlqin2yrs'].sum()
goodnum=train_data['SeriousDlqin2yrs'].count()-train_data['SeriousDlqin2yrs'].sum()
print('训练集数据中，好客户数量为：%i,坏客户数量为：%i,坏客户所占比例为：%.2f%%' %(goodnum,badnum,(badnum/train_data['SeriousDlqin2yrs'].count())*100))
#样本标签及其不平衡，后面需要使用balance参数

# In[]:
# 1.2、可用额度比值特征分布
# 数据分布及其不正常，中位数和四分之三位数都小于1，但是最大值确达到了50708，可用额度比值应该小于1，所以后面将大于1的值当做异常值剔除。
f,[ax1,ax2]=plt.subplots(1,2,figsize=(12,5))
sns.distplot(train_data['RevolvingUtilizationOfUnsecuredLines'],ax=ax1)
sns.boxplot(y='RevolvingUtilizationOfUnsecuredLines',data=train_data,ax=ax2)
plt.show()
print(train_data['RevolvingUtilizationOfUnsecuredLines'].describe())

# In[]:
# 1.3、年龄分布
f,[ax1,ax2]=plt.subplots(1,2,figsize=(12,5))
sns.distplot(train_data['age'],ax=ax1)
sns.boxplot(y='age',data=train_data,ax=ax2)
plt.show()
print(train_data['age'].describe())

print(len(train_data[train_data['age']<18])) # 只有1条，而且年龄为0，后面当做异常值删除
print(len(train_data[train_data['age']>100])) # 较多且连续，可暂时保留

# In[]:
# 1.4、逾期30-59天 | 60-89天 | 90天笔数分布：
f,[[ax1,ax2],[ax3,ax4],[ax5,ax6]] = plt.subplots(3,2,figsize=(15,15))
sns.distplot(train_data['NumberOfTime30-59DaysPastDueNotWorse'],ax=ax1)
sns.boxplot(y='NumberOfTime30-59DaysPastDueNotWorse',data=train_data,ax=ax2)
sns.distplot(train_data['NumberOfTime60-89DaysPastDueNotWorse'],ax=ax3)
sns.boxplot(y='NumberOfTime60-89DaysPastDueNotWorse',data=train_data,ax=ax4)
sns.distplot(train_data['NumberOfTimes90DaysLate'],ax=ax5)
sns.boxplot(y='NumberOfTimes90DaysLate',data=train_data,ax=ax6)
plt.show()

print(train_data[train_data['NumberOfTime30-59DaysPastDueNotWorse']>13]['NumberOfTime30-59DaysPastDueNotWorse'].value_counts())
print('----------------------')
# 这里可以看出逾期30-59天次数大于13次的有269条，大于80次的也是269条，说明这些是异常值，应该删除
print(train_data[train_data['NumberOfTime60-89DaysPastDueNotWorse']>13]['NumberOfTime60-89DaysPastDueNotWorse'].value_counts())
print('----------------------')
# 这里可以看出逾期60-89天次数大于13次的有269条，大于80次的也是269条，说明这些是异常值，应该删除
print(train_data[train_data['NumberOfTimes90DaysLate']>17]['NumberOfTimes90DaysLate'].value_counts())
print('----------------------')
#这里可以看出逾期90天以上次数大于17次的有269条，大于80次的也是269条，说明这些是异常值，应该删除

# In[]:
# 1.5、负债率特征分布
f,[ax1,ax2] = plt.subplots(1,2,figsize=(12,5))
sns.distplot(train_data['DebtRatio'],ax=ax1)
sns.boxplot(y='DebtRatio',data=train_data,ax=ax2)
plt.show()
print(train_data['DebtRatio'].describe())
print('----------------------')
print(train_data[train_data['DebtRatio']>1].count())
# 因为大于1的有三万多笔，所以猜测可能不是异常值

# In[]:
# 1.6、信贷数量特征分布
f,[ax1,ax2] = plt.subplots(1,2,figsize=(12,5))
sns.distplot(train_data['NumberOfOpenCreditLinesAndLoans'],ax=ax1)
sns.boxplot(y='NumberOfOpenCreditLinesAndLoans',data=train_data,ax=ax2)
plt.show()
print(train_data['NumberOfOpenCreditLinesAndLoans'].describe())
# 由于箱型图的上界值挺连续，所以可能不是异常值

# In[]:
# 1.7、固定资产贷款数量
f,[ax1,ax2] = plt.subplots(1,2,figsize=(12,5))
sns.distplot(train_data['NumberRealEstateLoansOrLines'],ax=ax1)
sns.boxplot(y='NumberRealEstateLoansOrLines',data=train_data,ax=ax2)
plt.show()
print(train_data['NumberRealEstateLoansOrLines'].describe())
# 查看箱型图发现最上方有异常值
print('----------------------')
print(train_data[train_data['NumberRealEstateLoansOrLines']>28]['NumberRealEstateLoansOrLines'].value_counts())
# 固定资产贷款数量大于28的有3个，大于32有一个为54，所以决定把>32的当做异常值剔除

# In[]:
# 1.8、家属数量分布
f,[ax1,ax2] = plt.subplots(1,2,figsize=(12,5))
sns.kdeplot(train_data['NumberOfDependents'],ax=ax1)
sns.boxplot(y='NumberOfDependents',data=train_data,ax=ax2)
plt.show()
print(train_data['NumberOfDependents'].describe())
print('----------------------')
print(train_data[train_data['NumberOfDependents']>10]['NumberOfDependents'].value_counts())
# 由箱型图和描述性统计可以看出，20为异常值，可删除

# 查看缺失比例
x=(train_data['SeriousDlqin2yrs'].count()-train_data['NumberOfDependents'].count())/train_data['SeriousDlqin2yrs'].count()
print('家属数量缺失比例为%.2f%%'%(x*100))
# 缺失比例为2.6%，可直接删除

# In[]:
# 1.9、月收入分布
f,[ax1,ax2] = plt.subplots(1,2,figsize=(12,5))
sns.kdeplot(train_data['MonthlyIncome'],ax=ax1)
sns.boxplot(y='MonthlyIncome',data=train_data,ax=ax2)
plt.show()
print(train_data['MonthlyIncome'].describe())
print('----------------------')
print(train_data[train_data['MonthlyIncome']>2000000]['MonthlyIncome'].value_counts())

# 查看缺失比例
x=(train_data['age'].count()-train_data['MonthlyIncome'].count())/train_data['age'].count()
print('月收入缺失数量比例为%.2f%%'%(x*100))
# 由于月收入缺失数量过大，后面采用随机森林的方法填充缺失值


# In[]:
# 2.1、异常值处理
def strange_delete(data):
    data=data[data['RevolvingUtilizationOfUnsecuredLines']<1]
    data=data[data['age']>18]
    data=data[data['NumberOfTime30-59DaysPastDueNotWorse']<80]
    data=data[data['NumberOfTime60-89DaysPastDueNotWorse']<80]
    data=data[data['NumberOfTimes90DaysLate']<80]
    data=data[data['NumberRealEstateLoansOrLines']<50]
    return data
    
train_data=strange_delete(train_data)

# In[]:
#查看经过异常值处理后是否还存在异常值
train_data.loc[(train_data['RevolvingUtilizationOfUnsecuredLines']>1)|(train_data['age']<18)|(train_data['NumberOfTime30-59DaysPastDueNotWorse']>80)|(train_data['NumberOfTime60-89DaysPastDueNotWorse']>80)|(train_data['NumberOfTimes90DaysLate']>80)|(train_data['NumberRealEstateLoansOrLines']>50)]
print(train_data.shape)
print('----------------------')

# In[]:
# 2.2、缺失值处理
# 2.2.1、对家属数量的缺失值进行删除
train_data=train_data[train_data['NumberOfDependents'].notnull()]
print(train_data.shape)

# In[]:
# 2.2.2、对月收入缺失值用随机森林的方法进行填充--训练集
# 创建随机森林函数
def fillmonthlyincome(data):
    known = data[data['MonthlyIncome'].notnull()]
    unknown = data[data['MonthlyIncome'].isnull()]
    x_train = known.iloc[:,[1,2,3,4,6,7,8,9,10]]
    y_train = known.iloc[:,5]
    x_test = unknown.iloc[:,[1,2,3,4,6,7,8,9,10]]
    rfr = RandomForestRegressor(random_state=0,n_estimators=200,max_depth=3,n_jobs=-1)
    pred_y = rfr.fit(x_train,y_train).predict(x_test)
    return pred_y
# 用随机森林填充训练集缺失值
predict_data=fillmonthlyincome(train_data)
train_data.loc[train_data['MonthlyIncome'].isnull(),'MonthlyIncome']=predict_data
print(train_data.info())

# In[]:
# 缺失值和异常值处理完后进行检查
print(train_data.isnull().sum())


# In[]:
# 3、特征工程——特征共线性（还可以做 方差膨胀系数）
'''
1、特征间共线性：两个或多个特征包含了相似的信息，期间存在强烈的相关关系 
2、常用判断标准：两个或两个以上的特征间的相关性系数高于0.8 
3、共线性的影响： 
3.1、降低运算效率 
3.2、降低一些模型的稳定性 
3.3、弱化一些模型的预测能力
'''
# 建立共线性表格（是检测特征共线性的，所以排除Y）
correlation_table = train_data.iloc[:,1:].corr()
# 皮尔森相似度 绝对值 排序
df_all_corr_abs = correlation_table.abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_all_corr_abs.rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: 'Correlation_Coefficient'}, inplace=True)
temp_corr_abs = df_all_corr_abs[(df_all_corr_abs["Feature_1"] != df_all_corr_abs["Feature_2"])][::2]
#temp_corr_abs2 = temp_corr_abs[(temp_corr_abs['Feature_1'] == 'age') | (temp_corr_abs['Feature_2'] == 'age')]
#temp_corr_abs.to_csv(r"C:\Users\dell\Desktop\123123\temp_corr_abs.csv")
print()
# 皮尔森相似度 排序
df_all_corr = correlation_table.unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_all_corr.rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: 'Correlation_Coefficient'}, inplace=True)
temp_corr = df_all_corr[(df_all_corr["Feature_1"] != df_all_corr["Feature_2"])][::2]
#temp_corr2 = temp_corr[(temp_corr['Feature_1'] == 'age') | (temp_corr['Feature_2'] == 'age')]
#temp_corr.to_csv(r"C:\Users\dell\Desktop\123123\temp_corr.csv")

# 热力图
xticks = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9'] # x轴标签
yticks = list(correlation_table.index) # y轴标签
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(1, 1, 1)
sns.heatmap(correlation_table, annot=True, cmap='rainbow', ax=ax1, annot_kws={'size': 12, 'weight': 'bold', 'color': 'black'})
ax1.set_xticklabels(xticks, rotation=0, fontsize=14)
ax1.set_yticklabels(yticks, rotation=0, fontsize=14)
# 可以看到各个变量间的相关性都不大，所以无需剔除变量


# In[]:
# 4、特征选择
# 4.1、分箱 （核心）
'''
变量分箱（binning）是对连续变量离散化（discretization）的一种称呼。信用评分卡开发中一般有常用的等距分段、等深分段、最优分段。
其中等距分段（Equval length intervals）是指分段的区间是一致的，比如年龄以十年作为一个分段；
等深分段（Equal frequency intervals）是先确定分段数量，然后令每个分段中数据数量大致相等；
最优分段（Optimal Binning）又叫监督离散化（supervised discretizaion），使用递归划分（Recursive Partitioning）将连续变量分为分段，背后是一种基于条件推断查找较佳分组的算法。
'''
# 连续性变量---定义自动分箱函数---最优分箱
'''
使用 斯皮尔曼系数 计算 自变量X 与 因变量Y 之间的相似度
'''
def mono_bin(Y, X, n=10):# X为待分箱的变量，Y为target变量,n为分箱数量
    r = 0    #设定斯皮尔曼 初始值
    badnum=Y.sum()    #计算坏样本数
    goodnum=Y.count()-badnum    #计算好样本数
    #下面这段就是分箱的核心 ，就是机器来选择指定最优的分箱节点，代替我们自己来设置
    while np.abs(r) < 1:  
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)})#用pd.qcut实现最优分箱，Bucket：将X分为n段，n由斯皮尔曼系数决定    
        d2 = d1.groupby('Bucket', as_index = True)# 按照分箱结果进行分组聚合        
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)# 以斯皮尔曼系数作为分箱终止条件
        n = n - 1    
    d3 = pd.DataFrame(d2.X.min(), columns = ['min']) 
    d3['min'] = d2.min().X    #箱体的左边界
    d3['max'] = d2.max().X    #箱体的右边界
    d3['bad'] = d2.sum().Y    #每个箱体中坏样本的数量
    d3['total'] = d2.count().Y    #每个箱体的总样本数
    d3['rate'] = d2.mean().Y
    print(d3['rate'])
    print('----------------------')
    d3['woe']=np.log((d3['bad']/badnum)/((d3['total'] - d3['bad'])/goodnum))# 计算每个箱体的woe值
    d3['badattr'] = d3['bad']/badnum  #每个箱体中坏样本所占坏样本总数的比例
    d3['goodattr'] = (d3['total'] - d3['bad'])/goodnum  # 每个箱体中好样本所占好样本总数的比例
    iv = ((d3['badattr']-d3['goodattr'])*d3['woe']).sum()  # 计算变量的iv值
    d4 = (d3.sort_index(by = 'min')).reset_index(drop=True)   # 对箱体从大到小进行排序
    print('分箱结果：')
    print(d4)
    print('IV值为：')
    print(iv)
    woe=list(d4['woe'].round(3))    
    cut=[]    #  cut 存放箱段节点
    cut.append(float('-inf'))    # 在列表前加-inf
    for i in range(1,n+1):        # n是前面的分箱的分割数，所以分成n+1份
        qua=X.quantile(i/(n+1))         #quantile 分为数  得到分箱的节点
        cut.append(round(qua,4))    # 保留4位小数       #返回cut
    cut.append(float('inf'))    # 在列表后加  inf
    return d4,iv,cut,woe
        
x1_d,x1_iv,x1_cut,x1_woe = mono_bin(train_data['SeriousDlqin2yrs'],train_data.RevolvingUtilizationOfUnsecuredLines)
x2_d,x2_iv,x2_cut,x2_woe = mono_bin(train_data['SeriousDlqin2yrs'],train_data.age) 
x4_d,x4_iv,x4_cut,x4_woe = mono_bin(train_data['SeriousDlqin2yrs'],train_data.DebtRatio) 
x5_d,x5_iv,x5_cut,x5_woe = mono_bin(train_data['SeriousDlqin2yrs'],train_data.MonthlyIncome)
# In[]:
'''
Y = train_data['SeriousDlqin2yrs']
X = train_data.age
r = 0
n=10

while np.abs(r) < 1:  
    d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)})#用pd.qcut实现最优分箱，Bucket：将X分为n段，n由斯皮尔曼系数决定    
    d2 = d1.groupby('Bucket', as_index = True)# 按照分箱结果进行分组聚合        
    r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)# 以斯皮尔曼系数作为分箱终止条件
    n = n - 1    
    print(d2.mean())

d3 = pd.DataFrame(d2.X.min(), columns = ['min']) 
d3['min'] = d2.min().X    #箱体的左边界
d3['max'] = d2.max().X    #箱体的右边界
d3['bad'] = d2.sum().Y    #每个箱体中坏样本的数量
d3['total'] = d2.count().Y    #每个箱体的总样本数
d3['rate'] = d2.mean().Y
print(d3['rate'])
print('----------------------')
d3['woe']=np.log((d3['bad']/badnum)/((d3['total'] - d3['bad'])/goodnum))# 计算每个箱体的woe值
d3['badattr'] = d3['bad']/badnum  #每个箱体中坏样本所占坏样本总数的比例
d3['goodattr'] = (d3['total'] - d3['bad'])/goodnum  # 每个箱体中好样本所占好样本总数的比例
iv = ((d3['badattr']-d3['goodattr'])*d3['woe']).sum()  # 计算变量的iv值
d4 = (d3.sort_index(by = 'min')).reset_index(drop=True)
woe=list(d4['woe'].round(3))   
cut=[]    #  cut 存放箱段节点
cut.append(float('-inf'))    # 在列表前加-inf
for i in range(1,n+1):        # n是前面的分箱的分割数，所以分成n+1份
    qua=X.quantile(i/(n+1))         #quantile 分为数  得到分箱的节点
    cut.append(round(qua,4))    # 保留4位小数       #返回cut
cut.append(float('inf')) 
'''
# In[]:
'''
# 斯皮尔曼测试：
Y = train_data['SeriousDlqin2yrs']
X = train_data.age
r = 0
n=10
d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)})#用pd.qcut实现最优分箱，Bucket：将X分为n段，n由斯皮尔曼系数决定    
d2 = d1.groupby('Bucket', as_index = True)# 按照分箱结果进行分组聚合        
r, p = stats.spearmanr(d2.mean().X, d2.mean().Y) # 以斯皮尔曼系数作为分箱终止条件
# -0.9999999999999999, 6.646897422032013e-64

# 手算：
d2mx = d2.mean().X
d2my = d2.mean().Y
d2mx_index = np.argsort(d2mx)
d2my_index = np.argsort(d2my)
d = np.sum(np.square((d2mx_index - d2my_index)), axis=0)
denominator = len(d2mx)*(np.square(len(d2mx)) - 1)
right = 6 * d / denominator
r1 = 1 - right # -1.0
'''

# In[]:
# 离散型变量-手动分箱
def self_bin(Y,X,cut):    
    badnum=Y.sum()    # 坏用户数量
    goodnum=Y.count()-badnum    #好用户数量
    d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.cut(X, cut)})#建立个数据框 X-- 各个特征变量 ， Y--用户好坏标签 ， Bucket--各个分箱    
    d2 = d1.groupby('Bucket', as_index = True)# 按照分箱结果进行分组聚合
    d3 = pd.DataFrame(d2.X.min(), columns = ['min'])    #  添加 min 列 ,不用管里面的 d2.X.min()
    d3['min']=d2.min().X    
    d3['max'] = d2.max().X    
    d3['bad'] = d2.sum().Y    
    d3['total'] = d2.count().Y    
    d3['rate'] = d2.mean().Y
    d3['woe']=np.log((d3['bad']/badnum)/((d3['total'] - d3['bad'])/goodnum))# 计算每个箱体的woe值
    d3['badattr'] = d3['bad']/badnum  #每个箱体中坏样本所占坏样本总数的比例
    d3['goodattr'] = (d3['total'] - d3['bad'])/goodnum  # 每个箱体中好样本所占好样本总数的比例
    iv = ((d3['badattr']-d3['goodattr'])*d3['woe']).sum()  # 计算变量的iv值  # 计算变量的iv值
    d4 = (d3.sort_index(by = 'min')).reset_index(drop=True)   # 对箱体从大到小进行排序
    woe=list(d4['woe'].round(3))
    return d4,iv,woe

# In[]:
ninf = float('-inf')#负无穷大
pinf = float('inf')#正无穷大
cutx3 = [ninf, 0, 1, 3, 5, pinf]
cutx6 = [ninf, 1, 2, 3, 5, pinf]
cutx7 = [ninf, 0, 1, 3, 5, pinf]
cutx8 = [ninf, 0,1,2, 3, pinf]
cutx9 = [ninf, 0, 1, 3, pinf]
cutx10 = [ninf, 0, 1, 2, 3, 5, pinf]
dfx3,ivx3,woex3 = self_bin(train_data.SeriousDlqin2yrs,train_data['NumberOfTime30-59DaysPastDueNotWorse'], cutx3)
dfx6,ivx6 ,woex6= self_bin(train_data.SeriousDlqin2yrs, train_data['NumberOfOpenCreditLinesAndLoans'], cutx6)
dfx7,ivx7,woex7 = self_bin(train_data.SeriousDlqin2yrs, train_data['NumberOfTimes90DaysLate'], cutx7)
dfx8, ivx8,woex8 = self_bin(train_data.SeriousDlqin2yrs, train_data['NumberRealEstateLoansOrLines'], cutx8)
dfx9, ivx9,woex9 = self_bin(train_data.SeriousDlqin2yrs, train_data['NumberOfTime60-89DaysPastDueNotWorse'], cutx9)
dfx10,ivx10,woex10 = self_bin(train_data.SeriousDlqin2yrs, train_data['NumberOfDependents'], cutx10)


# In[]:
# 4.2.1特征选择---相关系数矩阵
# 建立共线性表格
corr = train_data.corr()#计算各变量的相关性系数
# 皮尔森相似度 绝对值 排序
df_all_corr_abs = corr.abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_all_corr_abs.rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: 'Correlation_Coefficient'}, inplace=True)
print(df_all_corr_abs[(df_all_corr_abs["Feature_1"] != 'SeriousDlqin2yrs') & (df_all_corr_abs['Feature_2'] == 'SeriousDlqin2yrs')])
print()
# 皮尔森相似度 排序
df_all_corr = corr.unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_all_corr.rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: 'Correlation_Coefficient'}, inplace=True)
print(df_all_corr[(df_all_corr["Feature_1"] != 'SeriousDlqin2yrs') & (df_all_corr['Feature_2'] == 'SeriousDlqin2yrs')])


xticks = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']#x轴标签
yticks = list(corr.index)#y轴标签
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(1, 1, 1)
sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax1, annot_kws={'size': 12, 'weight': 'bold', 'color': 'black'})#绘制相关性系数热力图
ax1.set_xticklabels(xticks, rotation=0, fontsize=14)
ax1.set_yticklabels(yticks, rotation=0, fontsize=14)
plt.show()
'''
可见变量RevolvingUtilizationOfUnsecuredLines、NumberOfTime30-59DaysPastDueNotWorse、
NumberOfTimes90DaysLate和NumberOfTime60-89DaysPastDueNotWorse四个特征对于我们所要预测的值SeriousDlqin2yrs(因变量)有较强的相关性。
相关性分析只是初步的检查，进一步检查模型的VI（证据权重）作为变量筛选的依据。
'''

# In[]:
# 4.2.2、IV值筛选
# 通过IV值判断变量预测能力的标准是:小于 0.02: unpredictive；0.02 to 0.1: weak；0.1 to 0.3: medium； 0.3 to 0.5: strong
ivlist=[x1_iv,x2_iv,ivx3,x4_iv,x5_iv,ivx6,ivx7,ivx8,ivx9,ivx10]#各变量IV
index=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']#x轴的标签
fig1 = plt.figure(1,figsize=(8,5))
ax1 = fig1.add_subplot(1, 1, 1)
x = np.arange(len(index))+1 
ax1.bar(x,ivlist,width=.4) #  ax1.bar(range(len(index)),ivlist, width=0.4)#生成柱状图  #ax1.bar(x,ivlist,width=.04)
ax1.set_xticks(x)
ax1.set_xticklabels(index, rotation=0, fontsize=15)
ax1.set_ylabel('IV', fontsize=16)   # IV(Information Value),
#在柱状图上添加数字标签
for a, b in zip(x, ivlist):
    plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=12)
plt.show()
'''
可以看出，DebtRatio (x4)、MonthlyIncome(x5)、NumberOfOpenCreditLinesAndLoans(x6)、NumberRealEstateLoansOrLines(x8)和NumberOfDependents(x10)变量的IV值明显较低，所以予以删除。
故选择特征：RevolvingUtilizationOfUnsecuredLines（x1）、age（x2）、NumberOfTime30-59DaysPastDueNotWorse（x3）、NumberOfTimes90DaysLate（x7）、NumberOfTime60-89DaysPastDueNotWorse（x9）作为后续评分模型建立的对象。
'''
