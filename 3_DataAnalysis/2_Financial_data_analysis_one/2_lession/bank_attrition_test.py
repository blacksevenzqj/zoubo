'''
分解学习 bank_attrition.py 的代码
'''
import pandas as pd
import numbers
import numpy as np
import math
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
import random
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import chisquare

import os
os.chdir(r"E:\code\python_workSpace\idea_space\zoubo\3_DataAnalysis\2_Financial_data_analysis")

# In[1]:
bankChurn = pd.read_csv('bankChurn.csv',header=0)
externalData = pd.read_csv('ExternalData.csv',header = 0)
#merge two dataframes
AllData = pd.merge(bankChurn, externalData,on='CUST_ID')

columns = list(set(list(AllData.columns)))
print(len(columns), type(columns))
columns.remove('CHURN_CUST_IND')

numericCols = []
stringCols = []
for var in columns:
    x = list(set(AllData[var])) # set排重
    x = [i for i in x if i==i]  # we need to eliminate the noise, which is nan type
    if isinstance(x[0],numbers.Real):
        numericCols.append(var)
    elif isinstance(x[0], str):
        stringCols.append(var)
    else:
        print('The type of ',var,' cannot be determined')

#numericCols = []
#stringCols = []
#
#print(AllData['hnd_price'].size)
#print(len(set(AllData['hnd_price'])))
#print(len(list(set(AllData['hnd_price']))))
#
#x = list(set(AllData['hnd_price']))
#x = [i for i in x if i==i] 
#print(x[0])
#if isinstance(x[0],numbers.Real):
#    numericCols.append('hnd_price')
#elif isinstance(x[0], str):
#    stringCols.append('hnd_price')
#else:
#    print('The type of ', 'hnd_price', ' cannot be determined'


# In[2]:
# # 对 NumVarPerf 分解测试：
def NumVarPerf(df, col, target, filepath, truncation=False):
    '''
    :param df: the dataset containing numerical independent variable and dependent variable 包含数字自变量和因变量的数据集
    :param col: independent variable with numerical type 具有数字类型的自变量
    :param target: dependent variable, class of 0-1 因变量，类0-1
    :param filepath: the location where we save the histogram 我们保存直方图的位置
    :param truncation: indication whether we need to do some truncation for outliers 截断：指示是否需要对异常值进行一些截断
    :return: the descriptive statistics 描述性统计
    '''
    #extract target variable and specific indepedent variable
    validDf = df.loc[df[col] == df[col]][[col,target]]
    #the percentage of valid elements
    validRcd = validDf.shape[0]*1.0/df.shape[0]
    #format the percentage in the form of percent
    validRcdFmt = "%.2f%%"%(validRcd*100)
    #the descriptive statistics of each numerical column
    descStats = validDf[col].describe()
    mu = "%.2e" % descStats['mean']
    std = "%.2e" % descStats['std']
    maxVal = "%.2e" % descStats['max']
    minVal = "%.2e" % descStats['min']
    #we show the distribution by churn/not churn state 我们通过流失/不流失状态显示分布
    x = validDf.loc[validDf[target]==1][col]
    y = validDf.loc[validDf[target]==0][col]
    xweights = 100.0 * np.ones_like(x) / x.size
    yweights = 100.0 * np.ones_like(y) / y.size
    #if need truncation, truncate the numbers in 95th quantile 如果需要截断，截断第95个分位数中的数字
    if truncation == True:
        pcnt95 = np.percentile(validDf[col],95) # 95%分位数
        x = x.map(lambda x: min(x,pcnt95)) # 大于95%分位数的，全部取95%分位数
        y = y.map(lambda x: min(x,pcnt95))
    fig, ax = pyplot.subplots()
    ax.hist(x, weights=xweights, alpha=0.5,label='Attrition')
    ax.hist(y, weights=yweights, alpha=0.5,label='Retained')
    titleText = 'Histogram of '+ col +'\n'+'valid pcnt ='+validRcdFmt+', Mean ='+mu + ', Std='+std+'\n max='+maxVal+', min='+minVal
    ax.set(title= titleText, ylabel='% of Dataset in Bin')
    ax.margins(0.05)
    ax.set_ylim(bottom=0)
    pyplot.legend(loc='upper right')
#    figSavePath = filepath+str(col)+'.png' # 机器会炸
#    pyplot.savefig(figSavePath)
#    pyplot.close(1) # 只能关闭图片显示，不能关闭图片保存。

# In[2]:
#for var in numericCols:
#    NumVarPerf(AllData, var, 'CHURN_CUST_IND', "123")

# 分解测试：
validDf = AllData[AllData['hnd_price'] == AllData['hnd_price']][['hnd_price', 'CHURN_CUST_IND']]
validRcd = validDf.shape[0]*1.0 / AllData.shape[0]
validRcdFmt = "%.2f%%"%(validRcd*100)
descStats = validDf['hnd_price'].describe()
mu = "%.2e" % descStats['mean']
std = "%.2e" % descStats['std']
maxVal = "%.2e" % descStats['max']
minVal = "%.2e" % descStats['min']
x = validDf[validDf['CHURN_CUST_IND']==1]['hnd_price']
y = validDf[validDf['CHURN_CUST_IND']==0]['hnd_price']
xweights = 100.0 * np.ones_like(x) / x.size
yweights = 100.0 * np.ones_like(y) / y.size
print(xweights[0:20])

pcnt95 = np.percentile(validDf['hnd_price'],95)
x = x.map(lambda x: min(x,pcnt95)) # Series
y = y.map(lambda x: min(x,pcnt95))

fig, ax = pyplot.subplots()
ax.hist(x, weights=xweights, alpha=0.5,label='Attrition')
#ax.hist(y, weights=yweights, alpha=0.5,label='Retained')
titleText = 'Histogram of '+ 'hnd_price' +'\n'+'valid pcnt ='+validRcdFmt+', Mean ='+mu + ', Std='+std+'\n max='+maxVal+', min='+minVal
ax.set(title= titleText, ylabel='% of Dataset in Bin')
ax.margins(0.05)
ax.set_ylim(bottom=0)
pyplot.legend(loc='upper right')

c = x[x.map(lambda x: 25<x<50)] # x:Series, c:Series
even_squares = [i for i in x.tolist() if 25<i<50] # list


# In[3]:
# 对 CharVarPerf 分解测试：
def CharVarPerf(df,col,target,filepath):
    '''
    :param df: the dataset containing numerical independent variable and dependent variable
    :param col: independent variable with numerical type
    :param target: dependent variable, class of 0-1
    :param filepath: the location where we save the histogram
    :return: the descriptive statistics
    '''
    validDf = df.loc[df[col] == df[col]][[col, target]]
    validRcd = validDf.shape[0]*1.0/df.shape[0]
    recdNum = validDf.shape[0]
    validRcdFmt = "%.2f%%"%(validRcd*100)
    freqDict = {}
    churnRateDict = {}
    # for each category in the categorical variable, we count the percentage and churn rate
    # 对于分类变量中的每个类别，我们计算百分比和流失率
    for v in set(validDf[col]): # set去重
        vDf = validDf.loc[validDf[col] == v] # 在 该特征 有值的数据中，该类别数量
        freqDict[v] = vDf.shape[0]*1.0/recdNum # 在 该特征 有值的数据中，该类别占比
        churnRateDict[v] = sum(vDf[target])*1.0/vDf.shape[0] # 因变量=1 占 该类别 比例
    descStats = pd.DataFrame({'percent':freqDict,'churn rate':churnRateDict})
    fig = pyplot.figure()  # Create matplotlib figure
    ax = fig.add_subplot(111)  # Create matplotlib axes
    ax2 = ax.twinx()  # Create another axes that shares the same x-axis as ax.
    pyplot.title('The percentage and churn rate for '+col+'\n valid pcnt ='+validRcdFmt)
    descStats['churn rate'].plot(kind='line', color='red', ax=ax)
    descStats.percent.plot(kind='bar', color='blue', ax=ax2, width=0.2,position = 1)
    ax.set_ylabel('churn rate')
    ax2.set_ylabel('percentage')
#    figSavePath = filepath+str(col)+'.png'
#    pyplot.savefig(figSavePath)
#    pyplot.close(1) # 只能关闭图片显示，不能关闭图片保存。

# In[3]:
#stringCols = ['marital', 'GENDER_CD']
#for val in stringCols:
#    print(val)
#    CharVarPerf(AllData, val,'CHURN_CUST_IND',"123")

## 分解测试：
validDf = AllData[AllData['marital'] == AllData['marital']][['marital', 'CHURN_CUST_IND']]
validRcd = validDf.shape[0]*1.0 / AllData.shape[0]
validRcdFmt = "%.2f%%"%(validRcd*100)
recdNum = validDf.shape[0]

freqDict = {}
churnRateDict = {}
for v in set(validDf['marital']):
    vDf = validDf[validDf['marital'] == v] # 在 该特征 有值的数据中，该类别数量
    freqDict[v] = vDf.shape[0]*1.0 / recdNum # 在 该特征 有值的数据中，该类别占比
    churnRateDict[v] = sum(vDf['CHURN_CUST_IND'])*1.0 / vDf.shape[0] # 因变量=1 占 该类别 比例

print(churnRateDict)
descStats = pd.DataFrame({'percent':freqDict,'churn rate':churnRateDict})
fig = pyplot.figure()  # Create matplotlib figure
ax = fig.add_subplot(111)  # Create matplotlib axes
ax2 = ax.twinx()  # Create another axes that shares the same x-axis as ax.
pyplot.title('The percentage and churn rate for '+ 'marital' +'\n valid pcnt ='+validRcdFmt)
descStats['churn rate'].plot(kind='line', color='red', ax=ax)
descStats['percent'].plot(kind='bar', color='blue', ax=ax2, width=0.2, position = 1)
ax.set_ylabel('churn rate')
ax2.set_ylabel('percentage')



# In[4]:
# 方差分析：
# 连续变量在前，分类变量在后： 自变量X(连续)~因变量Y（分类）
anova_results = anova_lm(ols('ASSET_MON_AVG_BAL~CHURN_CUST_IND',AllData).fit())
anova_results2 = anova_lm(ols('ASSET_MON_AVG_BAL~C(CHURN_CUST_IND)',AllData).fit())



# In[5]:
# chisquare test 卡方检验：
chisqDf = AllData[['GENDER_CD','CHURN_CUST_IND']]
grouped = chisqDf['CHURN_CUST_IND'].groupby(chisqDf['GENDER_CD']) # SeriesGroupBy
count = list(grouped.count())
churn = list(grouped.sum())

# 手动计算卡方值：
chisqTable = pd.DataFrame({'total':count,'churn':churn})
chisqTable['expected'] = chisqTable['total'].map(lambda x: round(x*0.101))
chisqValList = chisqTable[['churn','expected']].apply(lambda x: (x[0]-x[1])**2/x[1], axis=1)
# the 2-degree of freedom chisquare under 0.05 is 5.99, which is smaller than chisqVal = 32.66, so GENDER is significant
# 0.05下的2自由度卡方为5.99，小于chisqVal值= 32.66，因此GENDER显着
chisqVal = sum(chisqValList)

# 直接使用scipy的方法计算卡方值： (观测值在前， 期望值在后)
chisqVal_scipy = chisquare(chisqTable['churn'],chisqTable['expected'])


# 作 散点矩阵 图：
# Part 1: Multi factor analysis for independent variables
# 第1部分：自变量的多因素分析
# use short name to replace the raw name, since the raw names are too long to be shown
# 使用短名称替换原始名称，因为原始名称太长而无法显示
col_to_index = {numericCols[i]:'var'+str(i) for i in range(len(numericCols))}
# sample from the list of columns, since too many columns cannot be displayed in the single plot
# 随机采样15个数字列
corrCols = random.sample(numericCols,6) # 15太多，根本看不清
sampleDf = AllData[corrCols]
for col in corrCols:
    sampleDf.rename(columns = {col:col_to_index[col]}, inplace = True)

scatter_matrix(sampleDf, alpha=0.2, figsize=(6, 6), diagonal='kde')








