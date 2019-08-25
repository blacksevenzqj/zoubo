import pandas as pd
import numbers
import numpy as np
import math
from matplotlib import pyplot
# from pandas.tools.plotting import scatter_matrix
from pandas.plotting import scatter_matrix
import random
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import chisquare

import os
os.chdir(r"E:\code\python_workSpace\idea_space\zoubo\3_DataAnalysis\2_Financial_data_analysis")


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
    #we show the distribution by churn/not churn state
    x = validDf.loc[validDf[target]==1][col]
    y = validDf.loc[validDf[target]==0][col]
    xweights = 100.0 * np.ones_like(x) / x.size
    yweights = 100.0 * np.ones_like(y) / y.size
    #if need truncation, truncate the numbers in 95th quantile
    if truncation == True:
        pcnt95 = np.percentile(validDf[col],95)
        x = x.map(lambda x: min(x,pcnt95))
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
    pyplot.close(1) # 只能关闭图片显示，不能关闭图片保存。


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
    #for each category in the categorical variable, we count the percentage and churn rate
    for v in set(validDf[col]):
        vDf = validDf.loc[validDf[col] == v]
        freqDict[v] = vDf.shape[0]*1.0/recdNum
        churnRateDict[v] = sum(vDf[target])*1.0/vDf.shape[0]
    descStats = pd.DataFrame({'percent':freqDict,'churn rate':churnRateDict})
    fig = pyplot.figure()  # Create matplotlib figure
    ax = fig.add_subplot(111)  # Create matplotlib axes
    ax2 = ax.twinx()  # Create another axes that shares the same x-axis as ax.
    pyplot.title('The percentage and churn rate for '+col+'\n valid pcnt ='+validRcdFmt)
    descStats['churn rate'].plot(kind='line', color='red', ax=ax)
    descStats.percent.plot(kind='bar', color='blue', ax=ax2, width=0.2,position = 1)
    ax.set_ylabel('churn rate')
    ax2.set_ylabel('percentage')
#    figSavePath = filepath+str(col)+'.png' # 机器会炸
#    pyplot.savefig(figSavePath)
    pyplot.close(1) # 只能关闭图片显示，不能关闭图片保存。




#read the data: customer details and dictionary of fields in customer details dataset


#read the internal data and external data
#'path' is the path for students' folder which contains the lectures of xiaoxiang
bankChurn = pd.read_csv('bankChurn.csv',header=0)
externalData = pd.read_csv('ExternalData.csv',header = 0)
#merge two dataframes
AllData = pd.merge(bankChurn,externalData,on='CUST_ID')

#step 1: check the type for each column and descripe the basic profile
columns = set(list(AllData.columns))
columns.remove('CHURN_CUST_IND')  #the target variable is not our object
#we differentiate the numerical type and catigorical type of the columns
numericCols = []
stringCols = []
for var in columns:
    x = list(set(AllData[var]))
    x = [i for i in x if i==i]   #we need to eliminate the noise, which is nan type
    if isinstance(x[0],numbers.Real):
        numericCols.append(var)
    elif isinstance(x[0], str):
        stringCols.append(var)
    else:
        print('The type of ',var,' cannot be determined')

'''
#Part 1: Single factor analysis for independent variables
# we check the distribution of each numerical variable, separated by churn/not churn
filepath = path+'/Notes/02 银行业客户流失预警模型的介绍/Pictures1/'
for var in numericCols:
    NumVarPerf(AllData,var, 'CHURN_CUST_IND',filepath)

#need to do some truncation for outliers
filepath = path+'/Notes/02 银行业客户流失预警模型的介绍/Pictures2/'
for val in numericCols:
    NumVarPerf(AllData,val, 'CHURN_CUST_IND',filepath,True)
'''

#anova test
# 连续变量在前，分类变量在后： 自变量X(连续)~因变量Y（分类）
anova_results = anova_lm(ols('ASSET_MON_AVG_BAL~CHURN_CUST_IND',AllData).fit())

'''
#single factor analysis for categorical analysis
filepath = path+'/Notes/02 银行业客户流失预警模型的介绍/Pictures3/'
for val in stringCols:
    print(val)
    CharVarPerf(AllData,val,'CHURN_CUST_IND',filepath)
'''

#chisquare test
chisqDf = AllData[['GENDER_CD','CHURN_CUST_IND']]
grouped = chisqDf['CHURN_CUST_IND'].groupby(chisqDf['GENDER_CD'])
count = list(grouped.count())
churn = list(grouped.sum())
chisqTable = pd.DataFrame({'total':count,'churn':churn})
chisqTable['expected'] = chisqTable['total'].map(lambda x: round(x*0.101))
chisqValList = chisqTable[['churn','expected']].apply(lambda x: (x[0]-x[1])**2/x[1], axis=1)
chisqVal = sum(chisqValList)
#the 2-degree of freedom chisquare under 0.05 is 5.99, which is smaller than chisqVal = 32.66, so GENDER is significant
#Alternatively, we can use function directly
chisquare(chisqTable['churn'],chisqTable['expected'])

#Part 1: Multi factor analysis for independent variables
#use short name to replace the raw name, since the raw names are too long to be shown
col_to_index = {numericCols[i]:'var'+str(i) for i in range(len(numericCols))}
#sample from the list of columns, since too many columns cannot be displayed in the single plot
corrCols = random.sample(numericCols,6) # 15太多，根本看不清
sampleDf = AllData[corrCols]
for col in corrCols:
    sampleDf.rename(columns = {col:col_to_index[col]}, inplace = True)

scatter_matrix(sampleDf, alpha=0.2, figsize=(6, 6), diagonal='kde')


