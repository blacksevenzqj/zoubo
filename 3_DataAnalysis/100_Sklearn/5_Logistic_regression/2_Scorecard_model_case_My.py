# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 23:03:47 2019

@author: dell
"""

'''
SeriousDlqin2yrs 出现 90 天或更长时间的逾期行为（即定义好坏客户）
RevolvingUtilizationOfUnsecuredLines 贷款以及信用卡可用额度与总额度比例
age 借款人借款年龄
NumberOfTime30-59DaysPastDueNotWorse 过去两年内出现35-59天逾期但是没有发展得更坏的次数
DebtRatio 每月偿还债务，赡养费，生活费用除以月总收入
MonthlyIncome 月收入
NumberOfOpenCreditLinesAndLoans 开放式贷款和信贷数量
NumberOfTimes90DaysLate 过去两年内出现90天逾期或更坏的次数
NumberRealEstateLoansOrLines 抵押贷款和房地产贷款数量，包括房屋净值信贷额度
NumberOfTime60-89DaysPastDueNotWorse 过去两年内出现60-89天逾期但是没有发展得更坏的次数
NumberOfDependents 家庭中不包括自身的家属人数（配偶，子女等）
'''

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import Binning_tools as bt
import Tools_customize as tc
import FeatureTools as ft

import os

os.chdir(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\5_Logistic_regression")
data = pd.read_csv(r"rankingcard.csv", index_col=0)  # 第一列变为index

# In[]:
# 1.1、特征类型 Column Types
# Number of each type of column
print(data.dtypes.value_counts())

# In[]:
# 1.2、重复值
# 重复项按特征统计
print(data[data.duplicated()].count())
# 去除重复项
nodup = data[-data.duplicated()]
print(len(nodup))
# 去除重复项
print(len(data.drop_duplicates()))
# 重复项
print(len(data) - len(nodup))

data.drop_duplicates(inplace=True)
print(data.info())  # Int64Index: 149391 entries, 1 to 150000

# 重设索引
data = data.reset_index(drop=True)
# data.index = range(data.shape[0])
print(data.info())  # RangeIndex: 149391 entries, 0 to 149390

# In[]:
# 1.3、异常值检测： 描述性统计 （先删除异常值，再填充缺失值）
# data.describe()
temp_desc = data.describe([0.01, 0.1, 0.25, .5, .75, .9, .99]).T
print(temp_desc[['min', 'max']])
# temp_desc.to_csv("C:\\Users\\dell\\Desktop\\123123\\temp_desc.csv")

# In[]:
# 直方图、箱线图：
'''
v_feat = data.ix[:,1:].columns
f, axes = plt.subplots(10,2, figsize=(20, 28*2))
for i, cn in enumerate(data[v_feat]):
    print(i, cn)
    sns.distplot(data[cn], bins=100, color='green', ax=axes[i][0])
#    sns.distplot(data[cn][data["SeriousDlqin2yrs"] == 1], bins=100, color='red', ax=axes[i][0])
#    sns.distplot(data[cn][data["SeriousDlqin2yrs"] == 0], bins=100, color='blue', ax=axes[i][0])
    axes[i][0].set_title('histogram of feature: ' + str(cn))
    axes[i][0].set_xlabel('')

    sns.boxplot(y=cn, data=data, ax=axes[i][1])
    axes[i][1].set_title('box of feature: ' + str(cn))
    axes[i][1].set_ylabel('')
'''
# In[]:
# 1.3.1、可用额度比值特征分布
f, axes = plt.subplots(2, 2, figsize=(16, 13))
sns.distplot(data['RevolvingUtilizationOfUnsecuredLines'], bins=100, color='green', ax=axes[0][0])
axes[0][0].set_title('histogram of feature: ' + str('RevolvingUtilizationOfUnsecuredLines'))
axes[0][0].set_xlabel('')

sns.boxplot(y='RevolvingUtilizationOfUnsecuredLines', data=data, ax=axes[0][1])
axes[0][1].set_title('box of feature: ' + str('RevolvingUtilizationOfUnsecuredLines'))
axes[0][1].set_ylabel('')

sns.distplot(data['RevolvingUtilizationOfUnsecuredLines'][data["SeriousDlqin2yrs"] == 1], bins=100, color='red',
             ax=axes[1][0])
sns.distplot(data['RevolvingUtilizationOfUnsecuredLines'][data["SeriousDlqin2yrs"] == 0], bins=100, color='blue',
             ax=axes[1][0])
axes[1][0].set_title('histogram of feature: ' + str('RevolvingUtilizationOfUnsecuredLines'))
axes[1][0].set_xlabel('')

sns.boxplot(x='SeriousDlqin2yrs', y='RevolvingUtilizationOfUnsecuredLines', data=data, ax=axes[1][1])
axes[1][1].set_title('box of feature: ' + str('RevolvingUtilizationOfUnsecuredLines'))
axes[1][1].set_ylabel('')
# In[]:
# 1.3.2、年龄分布
f, axes = plt.subplots(2, 2, figsize=(16, 13))
sns.distplot(data['age'], bins=100, color='green', ax=axes[0][0])
axes[0][0].set_title('histogram of feature: ' + str('age'))
axes[0][0].set_xlabel('')

sns.boxplot(y='age', data=data, ax=axes[0][1])
axes[0][1].set_title('box of feature: ' + str('age'))
axes[0][1].set_ylabel('')

sns.distplot(data['age'][data["SeriousDlqin2yrs"] == 1], bins=100, color='red', ax=axes[1][0])
sns.distplot(data['age'][data["SeriousDlqin2yrs"] == 0], bins=100, color='blue', ax=axes[1][0])
axes[1][0].set_title('histogram of feature: ' + str('age'))
axes[1][0].set_xlabel('')

sns.boxplot(x='SeriousDlqin2yrs', y='age', data=data, ax=axes[1][1])
axes[1][1].set_title('box of feature: ' + str('age'))
axes[1][1].set_ylabel('')
# In[]:
# 1.3.3、逾期30-59天 | 60-89天 | 90天笔数分布：
f, [[ax1, ax2], [ax3, ax4], [ax5, ax6]] = plt.subplots(3, 2, figsize=(15, 15))
sns.distplot(data['NumberOfTime30-59DaysPastDueNotWorse'], ax=ax1)
sns.boxplot(y='NumberOfTime30-59DaysPastDueNotWorse', data=data, ax=ax2)
sns.distplot(data['NumberOfTime60-89DaysPastDueNotWorse'], ax=ax3)
sns.boxplot(y='NumberOfTime60-89DaysPastDueNotWorse', data=data, ax=ax4)
sns.distplot(data['NumberOfTimes90DaysLate'], ax=ax5)
sns.boxplot(y='NumberOfTimes90DaysLate', data=data, ax=ax6)
plt.show()
# In[]:
# 1.3.4、负债率特征分布
f, axes = plt.subplots(2, 2, figsize=(16, 13))
sns.distplot(data['DebtRatio'], bins=100, color='green', ax=axes[0][0])
axes[0][0].set_title('histogram of feature: ' + str('DebtRatio'))
axes[0][0].set_xlabel('')

sns.boxplot(y='DebtRatio', data=data, ax=axes[0][1])
axes[0][1].set_title('box of feature: ' + str('DebtRatio'))
axes[0][1].set_ylabel('')

sns.distplot(data['DebtRatio'][data["SeriousDlqin2yrs"] == 1], bins=100, color='red', ax=axes[1][0])
sns.distplot(data['DebtRatio'][data["SeriousDlqin2yrs"] == 0], bins=100, color='blue', ax=axes[1][0])
axes[1][0].set_title('histogram of feature: ' + str('DebtRatio'))
axes[1][0].set_xlabel('')

sns.boxplot(x='SeriousDlqin2yrs', y='DebtRatio', data=data, ax=axes[1][1])
axes[1][1].set_title('box of feature: ' + str('DebtRatio'))
axes[1][1].set_ylabel('')
# In[]:
# 1.3.5、信贷数量特征分布
f, axes = plt.subplots(2, 2, figsize=(16, 13))
sns.distplot(data['NumberOfOpenCreditLinesAndLoans'], bins=100, color='green', ax=axes[0][0])
axes[0][0].set_title('histogram of feature: ' + str('NumberOfOpenCreditLinesAndLoans'))
axes[0][0].set_xlabel('')

sns.boxplot(y='NumberOfOpenCreditLinesAndLoans', data=data, ax=axes[0][1])
axes[0][1].set_title('box of feature: ' + str('NumberOfOpenCreditLinesAndLoans'))
axes[0][1].set_ylabel('')

sns.distplot(data['NumberOfOpenCreditLinesAndLoans'][data["SeriousDlqin2yrs"] == 1], bins=100, color='red',
             ax=axes[1][0])
sns.distplot(data['NumberOfOpenCreditLinesAndLoans'][data["SeriousDlqin2yrs"] == 0], bins=100, color='blue',
             ax=axes[1][0])
axes[1][0].set_title('histogram of feature: ' + str('NumberOfOpenCreditLinesAndLoans'))
axes[1][0].set_xlabel('')

sns.boxplot(x='SeriousDlqin2yrs', y='NumberOfOpenCreditLinesAndLoans', data=data, ax=axes[1][1])
axes[1][1].set_title('box of feature: ' + str('NumberOfOpenCreditLinesAndLoans'))
axes[1][1].set_ylabel('')
# In[]:
# 1.3.6、固定资产贷款数量
f, axes = plt.subplots(2, 2, figsize=(16, 13))
sns.distplot(data['NumberRealEstateLoansOrLines'], bins=100, color='green', ax=axes[0][0])
axes[0][0].set_title('histogram of feature: ' + str('NumberRealEstateLoansOrLines'))
axes[0][0].set_xlabel('')

sns.boxplot(y='NumberRealEstateLoansOrLines', data=data, ax=axes[0][1])
axes[0][1].set_title('box of feature: ' + str('NumberRealEstateLoansOrLines'))
axes[0][1].set_ylabel('')

sns.distplot(data['NumberRealEstateLoansOrLines'][data["SeriousDlqin2yrs"] == 1], bins=100, color='red', ax=axes[1][0])
sns.distplot(data['NumberRealEstateLoansOrLines'][data["SeriousDlqin2yrs"] == 0], bins=100, color='blue', ax=axes[1][0])
axes[1][0].set_title('histogram of feature: ' + str('NumberRealEstateLoansOrLines'))
axes[1][0].set_xlabel('')

sns.boxplot(x='SeriousDlqin2yrs', y='NumberRealEstateLoansOrLines', data=data, ax=axes[1][1])
axes[1][1].set_title('box of feature: ' + str('NumberRealEstateLoansOrLines'))
axes[1][1].set_ylabel('')
# In[]:
# 1.3.7、家属数量分布
f, axes = plt.subplots(2, 2, figsize=(16, 13))
sns.distplot(data['NumberOfDependents'], bins=100, color='green', ax=axes[0][0])
axes[0][0].set_title('histogram of feature: ' + str('NumberOfDependents'))
axes[0][0].set_xlabel('')

sns.boxplot(y='NumberOfDependents', data=data, ax=axes[0][1])
axes[0][1].set_title('box of feature: ' + str('NumberOfDependents'))
axes[0][1].set_ylabel('')

sns.distplot(data['NumberOfDependents'][data["SeriousDlqin2yrs"] == 1], bins=100, color='red', ax=axes[1][0])
sns.distplot(data['NumberOfDependents'][data["SeriousDlqin2yrs"] == 0], bins=100, color='blue', ax=axes[1][0])
axes[1][0].set_title('histogram of feature: ' + str('NumberOfDependents'))
axes[1][0].set_xlabel('')

sns.boxplot(x='SeriousDlqin2yrs', y='NumberOfDependents', data=data, ax=axes[1][1])
axes[1][1].set_title('box of feature: ' + str('NumberOfDependents'))
axes[1][1].set_ylabel('')
# In[]:
# 1.3.8、月收入分布
f, axes = plt.subplots(2, 2, figsize=(16, 13))
sns.distplot(data['MonthlyIncome'], bins=100, color='green', ax=axes[0][0])
axes[0][0].set_title('histogram of feature: ' + str('MonthlyIncome'))
axes[0][0].set_xlabel('')

sns.boxplot(y='MonthlyIncome', data=data, ax=axes[0][1])
axes[0][1].set_title('box of feature: ' + str('MonthlyIncome'))
axes[0][1].set_ylabel('')

sns.distplot(data['MonthlyIncome'][data["SeriousDlqin2yrs"] == 1], bins=100, color='red', ax=axes[1][0])
sns.distplot(data['MonthlyIncome'][data["SeriousDlqin2yrs"] == 0], bins=100, color='blue', ax=axes[1][0])
axes[1][0].set_title('histogram of feature: ' + str('MonthlyIncome'))
axes[1][0].set_xlabel('')

sns.boxplot(x='SeriousDlqin2yrs', y='MonthlyIncome', data=data, ax=axes[1][1])
axes[1][1].set_title('box of feature: ' + str('MonthlyIncome'))
axes[1][1].set_ylabel('')

# In[]:
# 异常值也被我们观察到，年龄的最小值居然有0，这不符合银行的业务需求，即便是儿童账户也要至少8岁，我们可以
# 查看一下年龄为0的人有多少
(data["age"] == 0).sum()
# 发现只有一个人年龄为0，可以判断这肯定是录入失误造成的，可以当成是缺失值来处理，直接删除掉这个样本
data = data[data["age"] != 0]

"""
另外，有三个指标看起来很奇怪：
"NumberOfTime30-59DaysPastDueNotWorse"
"NumberOfTime60-89DaysPastDueNotWorse"
"NumberOfTimes90DaysLate"
这三个指标分别是“过去两年内出现35-59天逾期但是没有发展的更坏的次数”，“过去两年内出现60-89天逾期但是没
有发展的更坏的次数”,“过去两年内出现90天逾期的次数”。这三个指标，在99%的分布的时候依然是2，最大值却是
98，看起来非常奇怪。一个人在过去两年内逾期35~59天98次，一年6个60天，两年内逾期98次这是怎么算出来的？
我们可以去咨询业务人员，请教他们这个逾期次数是如何计算的。如果这个指标是正常的，那这些两年内逾期了98次的
客户，应该都是坏客户。在我们无法询问他们情况下，我们查看一下有多少个样本存在这种异常：
"""
print(data['NumberOfTime30-59DaysPastDueNotWorse'].value_counts())
print(data['NumberOfTime60-89DaysPastDueNotWorse'].value_counts())
print(data['NumberOfTimes90DaysLate'].value_counts())

# 有225个样本存在这样的情况，并且这些样本，我们观察一下，标签并不都是1，他们并不都是坏客户。因此，我们基
# 本可以判断，这些样本是某种异常，应该把它们删除。
print(data.loc[data['NumberOfTime30-59DaysPastDueNotWorse'] >= 90, 'SeriousDlqin2yrs'].value_counts())
print(data.loc[data['NumberOfTime60-89DaysPastDueNotWorse'] >= 90, 'SeriousDlqin2yrs'].value_counts())
print(data.loc[data['NumberOfTimes90DaysLate'] >= 90, 'SeriousDlqin2yrs'].value_counts())

# 删除异常值
data = data[data.loc[:, "NumberOfTime30-59DaysPastDueNotWorse"] < 90]
data = data[data.loc[:, "NumberOfTime60-89DaysPastDueNotWorse"] < 90]
data = data[data.loc[:, "NumberOfTimes90DaysLate"] < 90]
# 一定要恢复索引
data.index = range(data.shape[0])
data.info()


# In[]:
# 1.4、缺失值 Examine Missing Values （先删除异常值，再填充缺失值）
# Function to calculate missing values by column# Funct
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * mis_val / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# Missing values statistics
missing_values = missing_values_table(data)
missing_values.head(20)

# In[]:
# 1.4.1、填充家庭人数 NumberOfDependents（不太重要的，就用均值填充）
data['NumberOfDependents'].fillna(data['NumberOfDependents'].mean(), inplace=True)

# from sklearn.preprocessing import Imputer
# imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
# data['NumberOfDependents'] = imp.fit_transform(data[['NumberOfDependents']].values)

# In[]:
# 1.4.2、填充月收入（随机森林）
'''
这里是将 训练集 与 测试集 合并之后一起进行填充； 那么，真实的测试集是没有标签的，怎么填充？？？
'''


def fill_missing_rf(X, y, to_fill):
    """
    使用随机森林填补一个特征的缺失值的函数
    参数：
    X：要填补的特征矩阵
    y：完整的，没有缺失值的标签
    to_fill：字符串，要填补的那一列的名称
    """

    # 构建我们的新特征矩阵和新标签
    df = X.copy()
    fill = df.loc[:, to_fill]
    df = pd.concat([df.loc[:, df.columns != to_fill], pd.DataFrame(y)], axis=1)

    # 找出我们的训练集和测试集
    Ytrain = fill[fill.notnull()]
    Ytest = fill[fill.isnull()]
    Xtrain = df.iloc[Ytrain.index, :]
    Xtest = df.iloc[Ytest.index, :]

    # 用随机森林回归来填补缺失值
    from sklearn.ensemble import RandomForestRegressor as rfr
    rfr = rfr(n_estimators=100)  # random_state=0,n_estimators=200,max_depth=3,n_jobs=-1
    rfr = rfr.fit(Xtrain, Ytrain)
    Ypredict = rfr.predict(Xtest)

    return Ypredict


# In[]:
X = data.iloc[:, 1:]
y = data["SeriousDlqin2yrs"]  # y = data.iloc[:,0]
X.shape  # (149391, 10)
y_pred = fill_missing_rf(X, y, "MonthlyIncome")

# 通过以下代码检验数据是否数量相同
temp_b = (y_pred.shape == data.loc[data.loc[:, "MonthlyIncome"].isnull(), "MonthlyIncome"].shape)
# 确认我们的结果合理之后，我们就可以将数据覆盖了
if temp_b:
    data.loc[data.loc[:, "MonthlyIncome"].isnull(), "MonthlyIncome"] = y_pred

data.info()

# In[]:
# 1.5、为什么不统一量纲，也不标准化数据分布？
'''
一旦我们将数据统一量纲，或者标准化了之后，数据大小和范围都会改变，统计结果是漂亮了，但是对于业务人员
来说，他们完全无法理解，标准化后的年龄在0.00328~0.00467之间为一档是什么含义。并且，新客户填写的信
息，天生就是量纲不统一的，我们的确可以将所有的信息录入之后，统一进行标准化，然后导入算法计算，但是最
终落到业务人员手上去判断的时候，他们会完全不理解为什么录入的信息变成了一串统计上很美但实际上根本看不
懂的数字。由于业务要求，在制作评分卡的时候，我们要尽量保持数据的原貌，年龄就是8~110的数字，收入就是
大于0，最大值可以无限的数字，即便量纲不统一，我们也不对数据进行标准化处理。
'''


# In[]:
# 1.6、样本不均衡：
def Sample_imbalance(data):
    print(data.shape)  # (150000, 11)
    print(data.info())

    print('No Default', round(len(data[data.SeriousDlqin2yrs == 0]) / len(data) * 100, 2), "% of the dataset")
    print('Default', round(len(data[data.SeriousDlqin2yrs == 1]) / len(data) * 100, 2), "% of the dataset")

    # 查看目标列Y的情况
    print(data.groupby('SeriousDlqin2yrs').size())
    print(data['SeriousDlqin2yrs'].value_counts())

    # 目标变量Y分布可视化
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    sns.countplot(x='SeriousDlqin2yrs', data=data, ax=axs[0])  # 柱状图
    axs[0].set_title("Frequency of each TARGET")

    data['SeriousDlqin2yrs'].value_counts().plot(x=None, y=None, kind='pie', ax=axs[1], autopct='%1.2f%%')  # 饼图
    axs[1].set_title("Percentage of each TARGET")
    plt.show()


# In[]:
Sample_imbalance(data)

# In[]:
# 1.6.1、上采样：
import imblearn
from imblearn.over_sampling import SMOTE

X_temp = data.iloc[:, 1:]
y_temp = data["SeriousDlqin2yrs"]  # y = data.iloc[:,0]

sm = SMOTE(random_state=42)  # 实例化
X, y = sm.fit_sample(X_temp, y_temp)

n_sample_ = X.shape[0]  # 278584
pd.Series(y).value_counts()
n_1_sample = pd.Series(y).value_counts()[1]
n_0_sample = pd.Series(y).value_counts()[0]
print('样本个数：{}; 1占{:.2%}; 0占{:.2%}'.format(n_sample_, n_1_sample / n_sample_, n_0_sample / n_sample_))
# 样本个数：278584; 1占50.00%; 0占50.00%

# In[]:
# 上采样之后，切分训练集、测试集； 保存 前期处理 结果
from sklearn.model_selection import train_test_split

X = pd.DataFrame(X)
y = pd.DataFrame(y)

X_train, X_vali, Y_train, Y_vali = train_test_split(X, y, test_size=0.3, random_state=420)
model_data = pd.concat([Y_train, X_train], axis=1)  # 训练数据构建模型
model_data.index = range(model_data.shape[0])
model_data.columns = data.columns

vali_data = pd.concat([Y_vali, X_vali], axis=1)  # 验证集
vali_data.index = range(vali_data.shape[0])
vali_data.columns = data.columns

n_sample_ = len(Y_train)
n_1_sample = Y_train.loc[:, 0].value_counts()[1]
n_0_sample = Y_train.loc[:, 0].value_counts()[0]
print('样本个数：{}; 1占{:.2%}; 0占{:.2%}'.format(n_sample_, n_1_sample / n_sample_, n_0_sample / n_sample_))

n_sample_ = len(Y_vali)
n_1_sample = Y_vali.loc[:, 0].value_counts()[1]
n_0_sample = Y_vali.loc[:, 0].value_counts()[0]
print('样本个数：{}; 1占{:.2%}; 0占{:.2%}'.format(n_sample_, n_1_sample / n_sample_, n_0_sample / n_sample_))

print(model_data.shape)  # (195008, 11)
print(vali_data.shape)  # (83576, 11)
model_data.to_csv(
    r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\5_Logistic_regression\model_data.csv")  # 训练数据
vali_data.to_csv(
    r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\5_Logistic_regression\vali_data.csv")  # 验证数据
# In[]:
model_data = pd.read_csv(
    r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\5_Logistic_regression\model_data.csv")
vali_data = pd.read_csv(
    r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\5_Logistic_regression\vali_data.csv")
print(model_data.shape)  # (195008, 12)
print(vali_data.shape)  # (83576, 12)
model_data.drop('Unnamed: 0', inplace=True, axis=1)
vali_data.drop('Unnamed: 0', inplace=True, axis=1)
print(model_data.shape)  # (195008, 11)
print(vali_data.shape)  # (83576, 11)

Sample_imbalance(model_data)
# Sample_imbalance(vali_data)


# In[]
# 1.7、分箱：
# 1.7.1、按照 等频 对需要分箱的列进行分箱
# model_data["qcut"], updown = pd.qcut(model_data["age"], retbins=True, q=20)#等频分箱
model_data["qcut"], updown = bt.qcut(model_data, "age", 20, True)
"""
pd.qcut，基于分位数的分箱函数，本质是将连续型变量离散化
只能够处理一维数据。返回箱子的上限和下限
参数q： 要分箱的个数
参数retbins=True： 返回箱子上下限数组
"""
# 在这里时让model_data新添加一列叫做“分箱”，这一列其实就是每个样本所对应的箱子
print(model_data["qcut"].value_counts())
# 所有箱子的上限和下限
print(updown)

# In[]:
# 1.7.2、统计每个分箱中0和1的数量（这里使用了数据透视表的功能groupby）
'''
coount_y0 = model_data[model_data["SeriousDlqin2yrs"] == 0].groupby(by="qcut").count()["SeriousDlqin2yrs"]
coount_y1 = model_data[model_data["SeriousDlqin2yrs"] == 1].groupby(by="qcut").count()["SeriousDlqin2yrs"]

# num_bins值分别为每个区间的上界，下界，0出现的次数，1出现的次数
num_bins = [*zip(updown,updown[1:],coount_y0,coount_y1)]
# 注意zip会按照最短列来进行结合
print(num_bins)
'''
num_bins = bt.qcut_per_bin_twoClass_num(model_data, "SeriousDlqin2yrs", "qcut", updown)

# In[]:
num_bins[0:2]

# In[]：
# 1.7.3、确保每个箱中都有0和1 （看不懂）
for i in range(20):  # 20个箱子
    print("第一处i", i)
    # 如果第一个组没有包含正样本或负样本，向后合并
    if 0 in num_bins[0][2:]:
        print("第一处合并", i)
        num_bins[0:2] = [(
            num_bins[0][0],  # 第一行/桶 下界
            num_bins[1][1],  # 第二行/桶 上界
            num_bins[0][2] + num_bins[1][2],
            num_bins[0][3] + num_bins[1][3])]
        continue

    """
    合并了之后，第一行的组是否一定有两种样本了呢？不一定
    如果原本的第一组和第二组都没有包含正样本，或者都没有包含负样本，那即便合并之后，第一行的组也还是没有
    包含两种样本
    所以我们在每次合并完毕之后，还需要再检查，第一组是否已经包含了两种样本
    这里使用continue跳出了本次循环，开始下一次循环，所以回到了最开始的for i in range(20), 让i+1
    这就跳过了下面的代码，又从头开始检查，第一组是否包含了两种样本
    如果第一组中依然没有包含两种样本，则if通过，继续合并，每合并一次就会循环检查一次，最多合并20次
    如果第一组中已经包含两种样本，则if不通过，就开始执行下面的代码
    """
    # 已经确认第一组中肯定包含两种样本了，如果其他组没有包含两种样本，就向前合并
    # 此时的num_bins已经被上面的代码处理过，可能被合并过，也可能没有被合并
    # 但无论如何，我们要在num_bins中遍历，所以写成in range(len(num_bins))
    print("2")
    for i in range(len(num_bins)):
        print("第二处i", i)
        if 0 in num_bins[i][2:]:
            print("第二处合并", i)
            num_bins[i - 1:i + 1] = [(
                num_bins[i - 1][0],
                num_bins[i][1],
                num_bins[i - 1][2] + num_bins[i][2],
                num_bins[i - 1][3] + num_bins[i][3])]
            break  # 跳出当前这里的循环， 不执行下面的 else， 直接跳到开始for i in range(20)处
            """
            第一个break解释：
            这个break，只有在if被满足的条件下才会被触发
            也就是说，只有发生了合并，才会打断for i in range(len(num_bins))这个循环
            为什么要打断这个循环？因为我们是在range(len(num_bins))中遍历
            但合并发生后，len(num_bins)发生了改变，但循环却不会重新开始
            举个例子，本来num_bins是5组，for i in range(len(num_bins))在第一次运行的时候就等于for i in 
            range(5)
            range中输入的变量会被转换为数字，不会跟着num_bins的变化而变化，所以i会永远在[0,1,2,3,4]中遍历
            进行合并后，num_bins变成了4组，已经不存在=4的索引了，但i却依然会取到4，循环就会报错
            因此在这里，一旦if被触发，即一旦合并发生，我们就让循环被破坏，使用break跳出当前循环
            循环就会回到最开始的for i in range(20)处，for i in range(len(num_bins))却会被重新运行
            这样就更新了i的取值，循环就不会报错了
            """
    else:  # 这个 else: 是单独的，没有和 开头的 if 是一组的，真TM坑啊
        print("3")
        # 如果对第一组和对后面所有组的判断中，都没有进入if去合并，则提前结束所有的循环
        # 顺序执行下来 就进这里的 else， break结束循环
        break

# In[]:
# 对上面代码的 测试 （真TM坑屎人啊）
for i in range(20):  # 20个箱子
    print("第一处i", i)
    # 如果第一个组没有包含正样本或负样本，向后合并
    if 0 in num_bins[0][2:]:
        print("第一处合并", i)
        num_bins[0:2] = [(
            num_bins[0][0],  # 第一行/桶 下界
            num_bins[1][1],  # 第二行/桶 上界
            num_bins[0][2] + num_bins[1][2],
            num_bins[0][3] + num_bins[1][3])]
        continue  # 下一个外层循环

    print("2")
    for i in range(len(num_bins)):  # 第一个if执行完，执行这里
        print("第二处i", i)
        #        if 0 in num_bins[i][2:]:
        if 1 == 1:
            print("第二处合并", i)
            #            num_bins[i-1:i+1] = [(
            #                num_bins[i-1][0],
            #                num_bins[i][1],
            #                num_bins[i-1][2]+num_bins[i][2],
            #                num_bins[i-1][3]+num_bins[i][3])]
            break  # 跳出当前这里的循环， 不执行下面的 else， 直接跳到开始for i in range(20)处

    else:  # 这个 else: 是单独的，没有和 开头的 if 是一组的，真TM坑啊
        print("3")
        break

    # In[]:


# 1.7.4、计算WOE和BAD RATE
def get_num_bins(data, col, y, bins):
    df = data[[col, y]].copy()
    df["cut"], bins = pd.cut(df[col], bins, retbins=True)
    coount_y0 = df.loc[df[y] == 0].groupby(by="cut").count()[y]
    coount_y1 = df.loc[df[y] == 1].groupby(by="cut").count()[y]
    num_bins = [*zip(bins, bins[1:], coount_y0, coount_y1)]
    return num_bins


# BAD RATE是一个箱中，坏的样本占一个箱子里边所有样本数的比例 (bad/total)
# 而bad%是一个箱中的坏样本占整个特征中的坏样本的比例
def get_woe(num_bins):
    # 通过 num_bins 数据计算 woe
    columns = ["min", "max", "count_0", "count_1"]
    df = pd.DataFrame(num_bins, columns=columns)

    df["total"] = df.count_0 + df.count_1  # 一个箱子当中所有的样本数： 按列相加
    df["percentage"] = df.total / df.total.sum()  # 一个箱子里的样本数，占所有样本的比例
    df["bad_rate"] = df.count_1 / df.total  # 一个箱子坏样本的数量占一个箱子里边所有样本数的比例
    df["good%"] = df.count_0 / df.count_0.sum()
    df["bad%"] = df.count_1 / df.count_1.sum()
    df["woe"] = np.log(df["good%"] / df["bad%"])
    return df


# 计算IV值
def get_iv(df):
    rate = df["good%"] - df["bad%"]
    iv = np.sum(rate * df.woe)
    return iv


# In[]:
# 1.7.5、卡方检验，合并箱体，画出IV曲线
num_bins_ = num_bins.copy()

import matplotlib.pyplot as plt
import scipy

IV = []
axisx = []
PV = []
pv_state = True
spearmanr_state = True
columns = ["min", "max", "count_0", "count_1"]

while len(num_bins_) > 2:  # 大于设置的最低分箱个数
    pvs = []
    # 获取 num_bins_两两之间的卡方检验的置信度（或卡方值）
    for i in range(len(num_bins_) - 1):
        x1 = num_bins_[i][2:]
        x2 = num_bins_[i + 1][2:]
        # 0 返回 chi2 值，1 返回 p 值。
        pv = scipy.stats.chi2_contingency([x1, x2])[1]  # p值
        # chi2 = scipy.stats.chi2_contingency([x1,x2])[0] # 计算卡方值
        pvs.append(pv)

    # 通过 卡方p值 进行处理。 合并 卡方p值 最大的两组
    '''
     2、独立性检验
     可以看成：一个特征中的多个类别/分桶  与  另一个特征中多个类别/分桶  的一个条件（类别数量）观测值 与 期望值 的计算。
     原假设： X与Y不相关   特征（两个箱子/类别） 与 因变量Y 不相关， 箱子需要合并
     备选假设： X与Y相关   特征（两个箱子/类别） 与 因变量Y 相关， 箱子不需要合并
     理论上应该这样做，但这里不是
    '''
    if max(pvs) < 0.001 and pv_state:
        # pv最大值都 < 0.001， 拒绝原假设，接受备选假设： 特征（两个箱子/类别） 与 因变量Y 相关， 箱子不需要合并
        num_bins_pv = num_bins_.copy()
        pv_state = False
        bins_df_pv = get_woe(num_bins_pv)
        iv_pv = get_iv(bins_df_pv)
        # break

    if spearmanr_state:
        df_temp = pd.DataFrame(num_bins_, columns=columns)
        bin_list = tc.set_union(df_temp["min"], df_temp["max"])
        X = model_data["age"]
        Y = model_data['SeriousDlqin2yrs']
        d1 = pd.DataFrame(
            {"X": X, "Y": Y, "Bucket": pd.cut(X, bin_list)})  # 用pd.qcut实现最优分箱，Bucket：将X分为n段，n由斯皮尔曼系数决定
        d2 = d1.groupby('Bucket', as_index=True)  # 按照分箱结果进行分组聚合
        r, p = scipy.stats.spearmanr(d2.mean().X, d2.mean().Y)  # 以斯皮尔曼系数作为分箱终止条件
        if abs(r) == 1:
            num_bins_spearmanr = num_bins_.copy()
            spearmanr_state = False
            bins_df_spearmanr = get_woe(num_bins_spearmanr)
            iv_spearmanr = get_iv(bins_df_spearmanr)

    i = pvs.index(max(pvs))
    num_bins_[i:i + 2] = [(
        num_bins_[i][0],
        num_bins_[i + 1][1],
        num_bins_[i][2] + num_bins_[i + 1][2],
        num_bins_[i][3] + num_bins_[i + 1][3])]

    bins_df = get_woe(num_bins_)
    axisx.append(len(num_bins_))
    IV.append(get_iv(bins_df))
    PV.append(max(pvs))  # 卡方p值

plt.figure()
plt.plot(axisx, IV)
# plt.plot(axisx,PV)
plt.xticks(axisx)
plt.xlabel("number of box")
plt.ylabel("IV")
plt.show()
# 选择转折点处，也就是下坠最快的折线点，6→5折线点最陡峭，所以这里对于age来说选择箱数为6
# In[]:
import scipy

num_bins_ = num_bins.copy()
x1 = num_bins_[0][2:]  # (0:4243, 1:6052)
x2 = num_bins_[0 + 1][2:]  # (0:3571, 1:5635)
# 0 返回 chi2 值，1 返回 p 值。
pv = scipy.stats.chi2_contingency([x1, x2])[1]  # p值 0.0005943767678537757
chi2 = scipy.stats.chi2_contingency([x1, x2])[0]  # 计算卡方值 11.793506471136046
# In[]:
afterbins, bins_woe, bins_iv, bins_pv, bins_woe_pv, bins_iv_pv, \
bins_spearmanr, bins_woe_spearmanr, bins_iv_spearmanr = \
    bt.chi_test_merge_boxes_IV_curve(num_bins=num_bins, data=model_data, x_name="age", y_name="SeriousDlqin2yrs",
                                     min_bins=6, is_spearmanr=True)


# In[]:
# 1.7.6、合并箱体 函数：
def get_bin(num_bins_, n):
    while len(num_bins_) > n:
        pvs = []
        for i in range(len(num_bins_) - 1):
            x1 = num_bins_[i][2:]
            x2 = num_bins_[i + 1][2:]
            pv = scipy.stats.chi2_contingency([x1, x2])[1]
            # chi2 = scipy.stats.chi2_contingency([x1,x2])[0]
            pvs.append(pv)

        i = pvs.index(max(pvs))
        num_bins_[i:i + 2] = [(
            num_bins_[i][0],
            num_bins_[i + 1][1],
            num_bins_[i][2] + num_bins_[i + 1][2],
            num_bins_[i][3] + num_bins_[i + 1][3])]
    return num_bins_


afterbins = num_bins.copy()
get_bin(afterbins, 6)

# In[]:
# 希望每组的bad_rate相差越大越好；
# woe差异越大越好，应该具有单调性，随着箱的增加，要么由正到负，要么由负到正，只能有一个转折过程；
# 如果woe值大小变化是有两个转折，比如呈现w型，证明分箱过程有问题
# num_bins保留的信息越多越好
bins_df = get_woe(afterbins)
print(bins_df)


# In[]:
# 1.7.7、将选取最佳分箱个数的过程包装为函数 （所有功能函数）
def graphforbestbin(Data, X, Y, n=5, q=20, graph=True):
    '''
    自动最优分箱函数，基于卡方检验的分箱
    参数：
    DF: 需要输入的数据
    X: 需要分箱的列名
    Y: 分箱数据对应的标签 Y 列名
    n: 保留分箱个数
    q: 初始分箱的个数
    graph: 是否要画出IV图像

    区间为前开后闭 (]
    '''

    DF = Data[[X, Y]].copy()

    DF["qcut"], bins = pd.qcut(DF[X], retbins=True, q=q, duplicates="drop")
    coount_y0 = DF.loc[DF[Y] == 0].groupby(by="qcut").count()[Y]
    coount_y1 = DF.loc[DF[Y] == 1].groupby(by="qcut").count()[Y]
    num_bins = [*zip(bins, bins[1:], coount_y0, coount_y1)]

    for i in range(q):
        if 0 in num_bins[0][2:]:
            num_bins[0:2] = [(
                num_bins[0][0],
                num_bins[1][1],
                num_bins[0][2] + num_bins[1][2],
                num_bins[0][3] + num_bins[1][3])]
            continue

        for i in range(len(num_bins)):
            if 0 in num_bins[i][2:]:
                num_bins[i - 1:i + 1] = [(
                    num_bins[i - 1][0],
                    num_bins[i][1],
                    num_bins[i - 1][2] + num_bins[i][2],
                    num_bins[i - 1][3] + num_bins[i][3])]
                break
        else:
            break

    def get_woe(num_bins):
        columns = ["min", "max", "count_0", "count_1"]
        df = pd.DataFrame(num_bins, columns=columns)
        df["total"] = df.count_0 + df.count_1
        df["percentage"] = df.total / df.total.sum()
        df["bad_rate"] = df.count_1 / df.total
        df["good%"] = df.count_0 / df.count_0.sum()
        df["bad%"] = df.count_1 / df.count_1.sum()
        df["woe"] = np.log(df["good%"] / df["bad%"])
        return df

    def get_iv(df):
        rate = df["good%"] - df["bad%"]
        iv = np.sum(rate * df.woe)
        return iv

    IV = []
    axisx = []
    pv_state = True
    while len(num_bins) > n:
        pvs = []
        for i in range(len(num_bins) - 1):
            x1 = num_bins[i][2:]
            x2 = num_bins[i + 1][2:]
            pv = scipy.stats.chi2_contingency([x1, x2])[1]
            pvs.append(pv)

        # 通过 卡方p值 进行处理。 合并 卡方p值 最大的两组
        '''
         2、独立性检验
         可以看成：一个特征中的多个类别/分桶  与  另一个特征中多个类别/分桶  的一个条件（类别数量）观测值 与 期望值 的计算。
         原假设： X与Y不相关   特征（两个箱子/类别） 与 因变量Y 不相关， 箱子需要合并
         备选假设： X与Y相关   特征（两个箱子/类别） 与 因变量Y 相关， 箱子不需要合并
         理论上应该这样做，但这里不是
        '''
        if max(pvs) < 0.001 and pv_state:
            # pv最大值都 < 0.001， 拒绝原假设，接受备选假设： 特征（两个箱子/类别） 与 因变量Y 相关， 箱子不需要合并
            num_bins_pv = num_bins.copy()
            pv_state = False
        #           break

        i = pvs.index(max(pvs))
        num_bins[i:i + 2] = [(
            num_bins[i][0],
            num_bins[i + 1][1],
            num_bins[i][2] + num_bins[i + 1][2],
            num_bins[i][3] + num_bins[i + 1][3])]

        bins_df = pd.DataFrame(get_woe(num_bins))
        axisx.append(len(num_bins))
        iv = get_iv(bins_df)
        IV.append(iv)

    if graph:
        plt.figure()
        plt.plot(axisx, IV)
        plt.xticks(axisx)
        plt.xlabel("number of box")
        plt.ylabel("IV")
        plt.show()

    return bins_df, num_bins_pv, iv


# In[]:
# 测试：
# print(model_data.columns)
# for i in model_data.columns[1:-1]:
#    print(i)
#    graphforbestbin(model_data,i,"SeriousDlqin2yrs",n=2,q=20)

# In[]:
auto_col_bins = {"RevolvingUtilizationOfUnsecuredLines": 6,
                 "age": 6,
                 "DebtRatio": 4,
                 "MonthlyIncome": 3,
                 "NumberOfOpenCreditLinesAndLoans": 5}

# 不能使用自动分箱的变量（稀疏数据）
hand_bins = {"NumberOfTime30-59DaysPastDueNotWorse": [0, 1, 2, 13]
    , "NumberOfTimes90DaysLate": [0, 1, 2, 17]
    , "NumberRealEstateLoansOrLines": [0, 1, 2, 4, 54]
    , "NumberOfTime60-89DaysPastDueNotWorse": [0, 1, 2, 8]
    , "NumberOfDependents": [0, 1, 2, 3]}

# 保证区间覆盖使用 np.inf替换最大值，用-np.inf替换最小值
# 原因：比如一些新的值出现，例如家庭人数为30，以前没出现过，改成范围为极大值之后，这些新值就都能分到箱里边了
# hand_bins = {k:[-np.inf,*v[:-1],np.inf] for k,v in hand_bins.items()} # 1维数组
# hand_bins = {k:[[-np.inf,*v[:-1],np.inf]] for k,v in hand_bins.items()} # 扩为2维数组
# hand_bins = bt.hand_bins_customize(hand_bins) # 换到后面的 综合函数 运行

# In[]:
'''
bins_of_col = {}
# 生成自动分箱的分箱区间和分箱后的 IV 值
for col in auto_col_bins:
    print(col)
#    bins_df_temp, num_bins_pv_temp, iv_temp = graphforbestbin(model_data,col
#                             ,"SeriousDlqin2yrs"
#                             ,n=auto_col_bins[col]
#                             #使用字典的性质来取出每个特征所对应的箱的数量
#                             ,q=20
#                             ,graph=True)
    afterbins, bins_df_temp, iv_temp, bins_pv, bins_woe_pv, bins_iv_pv = bt.graphforbestbin(
                                model_data, col, "SeriousDlqin2yrs", 
                                min_bins = auto_col_bins[col], q_num=20
                                )

    bins_list = sorted(set(bins_df_temp["min"]).union(bins_df_temp["max"]))
    #保证区间覆盖使用 np.inf 替换最大值 -np.inf 替换最小值
    bins_list[0],bins_list[-1] = -np.inf,np.inf
    bins_of_col[col] = [bins_list, iv_temp]
# In[]:
for col in hand_bins:
    print(col)
     # 手动分箱区间已给定，使用cut函数指定分箱后，求WOE及其IV值。
    num_bins_temp = get_num_bins(model_data, col, 'SeriousDlqin2yrs', hand_bins[col][0])
    iv_temp = get_iv(get_woe(num_bins_temp))
    hand_bins[col].append(iv_temp)

# 合并手动分箱数据    
bins_of_col.update(hand_bins)
'''
# In[]:
bins_of_col = bt.automatic_hand_binning_all(model_data, 'SeriousDlqin2yrs', auto_col_bins, hand_bins)

# In[]:
# 1.7.8、探索性分析：   使用 上采样 → 卡方检验分桶 后的数据
# 1.7.8.1、单变量分析：
# 年龄
'''
model_data['cut'] = pd.cut(model_data.age, bins_of_col['age'][0])
age_cut_grouped_good = model_data[model_data["SeriousDlqin2yrs"] == 0].groupby('cut')["SeriousDlqin2yrs"].count()
ft.seriers_change_colname(age_cut_grouped_good, "good")
age_cut_grouped_bad = model_data[model_data["SeriousDlqin2yrs"] == 1].groupby('cut')["SeriousDlqin2yrs"].count()
ft.seriers_change_colname(age_cut_grouped_bad, "bad")
#df1 = pd.merge(pd.DataFrame(age_cut_grouped_good), pd.DataFrame(age_cut_grouped_bad), left_index=True, right_index=True)
df1 = pd.concat([age_cut_grouped_good, age_cut_grouped_bad], axis=1)
df1.insert(2,"badgrade", df1["bad"] / (df1["good"] + df1["bad"]))
ax1 = df1[["good","bad"]].plot.bar()
ax1.set_xticklabels(df1.index,rotation=15)
ax1.set_ylabel("Num")
ax1.set_title("bar of age")
# In[]:
ax11=df1["badgrade"].plot()
ax11.set_xticklabels(df1.index,rotation=50)
ax11.set_ylabel("badgrade")
ax11.set_title("badgrade of age")
'''
# In[]
bt.box_indicator_visualization(model_data, "age", "SeriousDlqin2yrs", bins_of_col)

# In[]:
# 月收入
'''
model_data['cut'] = pd.cut(model_data.MonthlyIncome, bins_of_col['MonthlyIncome'][0])
MonthlyIncome_cut_grouped_good = model_data[model_data["SeriousDlqin2yrs"] == 0].groupby('cut').count()["SeriousDlqin2yrs"]
MonthlyIncome_cut_grouped_bad = model_data[model_data["SeriousDlqin2yrs"] == 1].groupby('cut').count()["SeriousDlqin2yrs"]
df2 = pd.merge(pd.DataFrame(MonthlyIncome_cut_grouped_good), pd.DataFrame(MonthlyIncome_cut_grouped_bad),right_index=True,left_index=True)
df2.rename(columns={"SeriousDlqin2yrs_x":"good","SeriousDlqin2yrs_y":"bad"},inplace=True)
df2.insert(2,"badgrade", df2["bad"] / (df2["good"] + df2["bad"]))
ax2 = df2[["good","bad"]].plot.bar()
ax2.set_xticklabels(df2.index,rotation=15)
ax2.set_ylabel("Num")
ax2.set_title("bar of MonthlyIncome")
# In[]:
ax22=df2["badgrade"].plot()
ax22.set_xticklabels(df2.index,rotation=50)
ax22.set_ylabel("badgrade")
ax22.set_title("badgrade of MonthlyIncome")
'''
# In[]:
bt.box_indicator_visualization(model_data, "MonthlyIncome", "SeriousDlqin2yrs", bins_of_col)

# In[]:
# 家庭成员数量
'''
model_data['cut'] = pd.cut(model_data.NumberOfDependents, bins_of_col['NumberOfDependents'][0])
NumberOfDependents_cut_grouped_good = model_data[model_data["SeriousDlqin2yrs"] == 0].groupby('cut').count()["SeriousDlqin2yrs"]
NumberOfDependents_cut_grouped_bad = model_data[model_data["SeriousDlqin2yrs"] == 1].groupby('cut').count()["SeriousDlqin2yrs"]
df3 = pd.merge(pd.DataFrame(NumberOfDependents_cut_grouped_good), pd.DataFrame(NumberOfDependents_cut_grouped_bad),right_index=True,left_index=True)
df3.rename(columns={"SeriousDlqin2yrs_x":"good","SeriousDlqin2yrs_y":"bad"},inplace=True)
df3.insert(2,"badgrade", df3["bad"] / (df3["good"] + df3["bad"]))
ax3 = df3[["good","bad"]].plot.bar()
ax3.set_xticklabels(df3.index,rotation=15)
ax3.set_ylabel("Num")
ax3.set_title("bar of NumberOfDependents")
# In[]:
ax33=df3["badgrade"].plot()
ax33.set_xticklabels(df3.index,rotation=50)
ax33.set_ylabel("badgrade")
ax33.set_title("badgrade of NumberOfDependents")
'''
# In[]:
bt.box_indicator_visualization(model_data, "NumberOfDependents", "SeriousDlqin2yrs", bins_of_col)

# In[]:
bt.box_indicator_visualization(model_data, "NumberOfTime30-59DaysPastDueNotWorse", "SeriousDlqin2yrs", bins_of_col)


# In[]:
# 1.7.8.2、多变量分析：
# 1.7.8.2.1、皮尔森相似度：
def corrFunction(data_corr):
    corr = data_corr.corr()  # 计算各变量的相关性系数
    # 皮尔森相似度 绝对值 排序
    df_all_corr_abs = corr.abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_all_corr_abs.rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: 'Correlation_Coefficient'},
                           inplace=True)
    print(df_all_corr_abs[(df_all_corr_abs["Feature_1"] != 'SeriousDlqin2yrs') & (
            df_all_corr_abs['Feature_2'] == 'SeriousDlqin2yrs')])
    print()
    # 皮尔森相似度 排序
    df_all_corr = corr.unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_all_corr.rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: 'Correlation_Coefficient'},
                       inplace=True)
    print(df_all_corr[
              (df_all_corr["Feature_1"] != 'SeriousDlqin2yrs') & (df_all_corr['Feature_2'] == 'SeriousDlqin2yrs')])

    xticks = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']  # x轴标签
    yticks = list(corr.index)  # y轴标签
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax1,
                annot_kws={'size': 12, 'weight': 'bold', 'color': 'black'})  # 绘制相关性系数热力图
    ax1.set_xticklabels(xticks, rotation=0, fontsize=14)
    ax1.set_yticklabels(yticks, rotation=0, fontsize=14)
    plt.show()


corrFunction(model_data.iloc[:, :-2])
# NumberOfTime30-59DaysPastDueNotWorse、NumberOfTimes90DaysLate、89DaysPastDueNotWorse、age
# In[]:
# 1、
temp_corr_abs, temp_corr = ft.corrFunction_withY(model_data.iloc[:, :-2], "SeriousDlqin2yrs")
# In[]:
# 2、
temp_corr_abs1, temp_corr1 = ft.corrFunction(model_data.iloc[:, :-2])

# In[]:
# 1.7.8.2.2、斯皮尔曼相关系数[-1,1]： （对之前的 分箱结果 进行 检测验证）
rlist = []
index = []  # x轴的标签
collist = []
for i, col in enumerate(bins_of_col):
    print("x" + str(i + 1), col, bins_of_col[col][0])
    X = model_data[col]
    Y = model_data['SeriousDlqin2yrs']
    d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.cut(X, bins_of_col[col][0])})
    d2 = d1.groupby('Bucket', as_index=True)  # 按照分箱结果进行分组聚合
    r, p = scipy.stats.spearmanr(d2.mean().X, d2.mean().Y)  # 源码中 以斯皮尔曼系数作为分箱终止条件 while np.abs(r) < 1:
    rlist.append(r)
    index.append("x" + str(i + 1))
    collist.append(col)

fig1 = plt.figure(1, figsize=(8, 5))
ax1 = fig1.add_subplot(1, 1, 1)
x = np.arange(len(index)) + 1
ax1.bar(x, rlist, width=.4)
ax1.plot(x, rlist)
ax1.set_xticks(x)
ax1.set_xticklabels(index, rotation=0, fontsize=15)
ax1.set_ylabel('R', fontsize=16)  # IV(Information Value),
# 在柱状图上添加数字标签
for a, b in zip(x, rlist):
    plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=12)
plt.show()
# RevolvingUtilizationOfUnsecuredLines、age、NumberOfTime30-59DaysPastDueNotWorse、NumberOfTimes90DaysLate、NumberRealEstateLoansOrLines、NumberOfTime60-89DaysPastDueNotWorse
# In[]:
df_spearmanr = bt.spearmanr_visualization(model_data, 'SeriousDlqin2yrs', bins_of_col)

# In[]:
# 1.7.8.2.3、卡方值比较： （在样本量很大的情况下，很容易显著，只能做参考而已）
'''
2、独立性检验
可以看成：一个特征中的多个类别/分桶  与  另一个特征中多个类别/分桶  的一个条件（类别数量）观测值 与 期望值 的计算。
原假设：X与Y不相关
备选假设：X与Y相关
'''
pvlist = []
index = []  # x轴的标签
for i, col in enumerate(bins_of_col):
    print("x" + str(i + 1), col, bins_of_col[col][0])
    model_data['cut'] = pd.cut(model_data[col], bins_of_col[col][0])
    #    print(model_data['SeriousDlqin2yrs'].astype('int64').groupby(model_data['cut']).agg(['count', 'mean']))
    # print(model_data_My['cut'].value_counts())
    bins_stats_crosstab = pd.crosstab(model_data['cut'], model_data['SeriousDlqin2yrs'], margins=True)
    print(bins_stats_crosstab.iloc[:-1, :-1])
    pvlist.append(scipy.stats.chi2_contingency(bins_stats_crosstab.iloc[:-1, :-1])[1])  # P值
    index.append("x" + str(i + 1))

fig1 = plt.figure(1, figsize=(8, 5))
ax1 = fig1.add_subplot(1, 1, 1)
x = np.arange(len(index)) + 1
ax1.bar(x, pvlist, width=.4)  # ax1.bar(range(len(index)),ivlist, width=0.4)#生成柱状图  #ax1.bar(x,ivlist,width=.04)
ax1.set_xticks(x)
ax1.set_xticklabels(index, rotation=0, fontsize=15)
ax1.set_ylabel('PV', fontsize=16)  # IV(Information Value),
# 在柱状图上添加数字标签
for a, b in zip(x, pvlist):
    plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=12)
plt.show()

# In[]:
# 1.7.8.2.4、IV值 比较选择： （从大到小排列 选择特征）
ivlist = []  # 各变量IV
index = []  # x轴的标签
collist = []
for i, col in enumerate(bins_of_col):
    print("x" + str(i + 1), col, bins_of_col[col][1])
    ivlist.append(bins_of_col[col][1])
    index.append("x" + str(i + 1))
    collist.append(col)

fig1 = plt.figure(1, figsize=(8, 5))
ax1 = fig1.add_subplot(1, 1, 1)
x = np.arange(len(index)) + 1
ax1.bar(x, ivlist, width=.4)  # ax1.bar(range(len(index)),ivlist, width=0.4)#生成柱状图  #ax1.bar(x,ivlist,width=.04)
ax1.set_xticks(x)
ax1.set_xticklabels(index, rotation=0, fontsize=15)
ax1.set_ylabel('IV', fontsize=16)  # IV(Information Value),
# 在柱状图上添加数字标签
for a, b in zip(x, ivlist):
    plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=12)
plt.show()
# RevolvingUtilizationOfUnsecuredLines、NumberOfTime30-59DaysPastDueNotWorse、NumberOfTimes90DaysLate、NumberOfTime60-89DaysPastDueNotWorse
# In[]:
df_iv = bt.iv_visualization(bins_of_col)

# In[]:
# 1.7.9、计算WOE值
# 测试
# 函数pd.cut，可以根据已知的分箱间隔把数据分箱
# 参数为 pd.cut(数据，以列表表示的分箱间隔)
data = model_data[["age", "SeriousDlqin2yrs"]].copy()
data["cut"] = pd.cut(data["age"], bins_of_col['age'][0])
# In[]:
# 将数据按分箱结果聚合，并取出其中的标签值
data.groupby("cut")["SeriousDlqin2yrs"].value_counts()
# In[]:
# 使用unstack()来将树状结构变成表状结构
bins_df = data.groupby("cut")["SeriousDlqin2yrs"].value_counts().unstack()
woe = bins_df["woe"] = np.log((bins_df[0] / bins_df[0].sum()) / (bins_df[1] / bins_df[1].sum()))
print(woe)


# In[]:
# 单独计算出woe： 因为测试集映射数据时使用的是训练集的WOE值（测试集不能使用Y值的）
def get_woe_only(data, col, y, bins):
    df = data[[col, y]].copy()
    df["cut"] = pd.cut(df[col], bins)
    bins_df = df.groupby("cut")[y].value_counts().unstack()
    woe = bins_df["woe"] = np.log((bins_df[0] / bins_df[0].sum()) / (bins_df[1] / bins_df[1].sum()))
    return woe


# 将所有特征的WOE存储到字典当中
woeall = {}
for col in bins_of_col:
    woeall[col] = get_woe_only(model_data, col, "SeriousDlqin2yrs", bins_of_col[col][0])

woeall

# In[]:
# 训练集 WOE数据 映射： （所有数据都隐射为WOE值）
model_woe = pd.DataFrame(index=model_data.index)

# 将原数据分箱后，按箱的结果把WOE结构用map函数映射到数据中
# model_woe["age"] = pd.cut(model_data["age"],bins_of_col["age"]).map(woeall["age"])

# 对所有特征操作可以写成：
for col in bins_of_col:
    model_woe[col] = pd.cut(model_data[col], bins_of_col[col][0]).map(woeall[col])

# 将标签补充到数据中
model_woe["SeriousDlqin2yrs"] = model_data["SeriousDlqin2yrs"]
# 这就是我们的建模数据了
# model_woe.to_csv(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\5_Logistic_regression\model_woe.csv")
# In[]:
woeall = bt.storage_woe_dict(model_data, "SeriousDlqin2yrs", bins_of_col)
# In[]:
save_path = r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\5_Logistic_regression\model_woe.csv"
model_woe = bt.woe_mapping(model_data, "SeriousDlqin2yrs", bins_of_col, woeall, True, save_path)

# In[]:
# 测试集 WOE数据 映射： （所有数据都隐射为WOE值）
vali_woe = pd.DataFrame(index=vali_data.index)
# 只能用训练集的WOE， 不能重新计算测试集WOE， 因为测试集是没有Y值的（测试集Y值是最后评分用）
for col in bins_of_col:
    vali_woe[col] = pd.cut(vali_data[col], bins_of_col[col][0]).map(woeall[col])

vali_woe["SeriousDlqin2yrs"] = vali_data["SeriousDlqin2yrs"]
# 这就是我们的建模数据了
# vali_woe.to_csv(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\5_Logistic_regression\vali_woe.csv")
# In[]:
save_path = r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\5_Logistic_regression\vali_woe.csv"
vali_woe = bt.woe_mapping(vali_data, "SeriousDlqin2yrs", bins_of_col, woeall, True, save_path)

# In[]:
# 1.8、建模：
train_X = model_woe.iloc[:, :-1]
train_y = model_woe.iloc[:, -1]

text_X = vali_woe.iloc[:, :-1]
text_y = vali_woe.iloc[:, -1]

lr = LR().fit(train_X, train_y)
lr.score(text_X, text_y)  # 0.8674858811141954

# In[]:
c_1 = np.linspace(0.01, 1, 20)
score = []
for i in c_1:  # 先c_1大范围，再c_2小范围
    lr = LR(solver='liblinear', C=i).fit(train_X, train_y)
    score.append(lr.score(text_X, text_y))
plt.figure()
plt.plot(c_1, score)
plt.show()
print(max(score), c_1[score.index(max(score))])
print(lr.n_iter_)  # 6

c_2 = np.linspace(0.01, 0.2, 20)
score = []
for i in c_2:  # 先c_1大范围，再c_2小范围： 最高点在0.06
    lr = LR(solver='liblinear', C=i).fit(train_X, train_y)
    score.append(lr.score(text_X, text_y))
plt.figure()
plt.plot(c_2, score)
plt.show()
print(max(score), c_2[score.index(max(score))])
print(lr.n_iter_)  # 6

# In[]:
score = []
iter_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in iter_list:
    lr = LR(solver='liblinear', C=0.06, max_iter=i).fit(train_X, train_y)
    score.append(lr.score(text_X, text_y))

plt.figure()
plt.plot(iter_list, score)
plt.show()
print(max(score), iter_list[score.index(max(score))])  # 0.8656911074949747 7
print(lr.n_iter_)  # 6 收敛了就不迭代了。

# In[]:
lr = LR(solver='liblinear', C=0.06, max_iter=6).fit(train_X, train_y)
print(lr.score(text_X, text_y))
print(lr.n_iter_)  # 6

lr = LR(solver='liblinear', C=0.06).fit(train_X, train_y)
print(lr.score(text_X, text_y))
print(lr.n_iter_)  # 6

# In[]:
import scikitplot as skplt

vali_proba_df = pd.DataFrame(lr.predict_proba(text_X))
skplt.metrics.plot_roc(text_y, vali_proba_df,
                       plot_micro=False, figsize=(6, 6),
                       plot_macro=False)

# In[]:
import numpy as np

print(10 / np.log(2))
