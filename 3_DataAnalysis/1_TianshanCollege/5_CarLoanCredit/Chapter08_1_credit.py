
# coding: utf-8

# # 逻辑回归
#信用风险建模案例
##数据说明：本数据是一份汽车贷款违约数据
##名称---中文含义
##application_id---申请者ID
##account_number---帐户号
##bad_ind---是否违约
##vehicle_year---汽车购买时间
##vehicle_make---汽车制造商
##bankruptcy_ind---曾经破产标识
##tot_derog---五年内信用不良事件数量(比如手机欠费消号)
##tot_tr---全部帐户数量
##age_oldest_tr---最久账号存续时间(月)
##tot_open_tr---在使用帐户数量
##tot_rev_tr---在使用可循环贷款帐户数量(比如信用卡)
##tot_rev_debt---在使用可循环贷款帐户余额(比如信用卡欠款)
##tot_rev_line---可循环贷款帐户限额(信用卡授权额度)
##rev_util---可循环贷款帐户使用比例(余额/限额)
##fico_score---FICO打分
##purch_price---汽车购买金额(元)
##msrp---建议售价
##down_pyt---分期付款的首次交款
##loan_term---贷款期限(月)
##loan_amt---贷款金额
##ltv---贷款金额/建议售价*100
##tot_income---月均收入(元)
##veh_mileage---行使历程(Mile)
##used_ind---是否二手车
##weight---样本权重


# In[1]:

import os
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

#pd.set_option('display.max_columns', None)
os.chdir(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\TianshanCollege\5_CarLoanCredit")

# 导入数据和数据清洗
# In[5]:
accepts = pd.read_csv('accepts.csv').dropna()

##衍生变量:
def divMy(x,y):
    if x==np.nan or y==np.nan:
        return np.nan
    elif float(y) == 0 :
        return -1
    else:
        return float(x) / float(y)
divMy('1','0.0')

#%%
accepts[["tot_rev_line","tot_income"]].head()
# 历史负债收入比:tot_rev_line/tot_income
accepts["dti_hist"]=accepts[["tot_rev_line","tot_income"]].apply(lambda x:divMy(x[0],x[1]),axis = 1)
# 本次新增负债收入比:loan_amt/tot_income
accepts["dti_mew"]=accepts[["loan_amt","tot_income"]].apply(lambda x:divMy(x[0],x[1]),axis = 1)
# 本次贷款首付比例:down_pyt/loan_amt
accepts["fta"]=accepts[["down_pyt","loan_amt"]].apply(lambda x:divMy(x[0],x[1]),axis = 1)
# 新增债务比:loan_amt/tot_rev_debt
accepts["nth"]=accepts[["loan_amt","tot_rev_debt"]].apply(lambda x:divMy(x[0],x[1]),axis = 1)
# 新增债务额度比:loan_amt/tot_rev_line
accepts["nta"]=accepts[["loan_amt","tot_rev_line"]].apply(lambda x:divMy(x[0],x[1]),axis = 1)

accepts.head()


# 一、分类变量的相关关系
# 交叉表
# In[6]:
cross_table = pd.crosstab(accepts.used_ind,accepts.bad_ind, margins=True)
#cross_table = pd.crosstab(accepts.bankruptcy_ind,accepts.bad_ind, margins=True)

cross_table

# 列联表检验
# 原假设：不相关； 备选假设：相关。
# In[7]:
#W[0,0] / 行1总计
#W[0,1] / 行1总计
#W[1,0] / 行2总计
#W[1,1] / 行2总计
def percConvert(ser):
    return ser/float(ser[-1])

cross_table.apply(percConvert, axis=1) # cross_table值不变

# In[8]:
# 卡方检验
# chisq 卡方值
# p-value 卡方值对应的 显著度α，用p-value表示。 显著以否 的衡量标准，和 两样本T检验 的P值是一样的意思。
# expected_freq 卡方检验的 期望频率 = (行合计 * 列合计) / 总和
print(cross_table.iloc[:2, :2])
print('''chisq = %6.4f 
p-value = %6.4f
dof = %i 
expected_freq = %s'''  %stats.chi2_contingency(cross_table.iloc[:2, :2]))



# 逻辑回归
# In[9]:
# age_oldest_tr:最久账号存续时间(月)
accepts.plot(x='age_oldest_tr', y='bad_ind', kind='scatter')

# 随机抽样，建立训练集与测试集
# In[10]:
# copy() 浅拷贝
train = accepts.sample(frac=0.7, random_state=1234).copy() # 70%的样本训练
test = accepts[~ accepts.index.isin(train.index)].copy() # ~表示非，那就是剩下30%的样本
print(' 训练集样本量: %i \n 测试集样本量: %i' %(len(train), len(test)))

# In[11]:
# statsmodels包中的逻辑回归
lg = smf.glm('bad_ind ~ age_oldest_tr', data=train, 
             family=sm.families.Binomial(sm.families.links.logit)).fit()
lg.summary() # 逻辑回归不用看指标，都没有用

# 预测
# In[19]:
# 得到的是概率，而非最终的0/1结果，区别于sklearn中的decision_function的𝜃x+b结果
train['proba'] = lg.predict(train)
test['proba'] = lg.predict(test)

test['proba'].head(10)

# In[12]:
# 模型评估
# 设定阈值
# In[20]:
test['prediction'] = (test['proba'] > 0.3).astype('int')

# 混淆矩阵
# In[22]:
pd.crosstab(test.bad_ind, test.prediction, margins=True)

# - 计算准确率
# In[23]:
acc = sum(test['prediction'] == test['bad_ind']) /np.float(len(test))
print('The accurancy is %.2f' %acc)

# In[25]:
for i in np.arange(0.02, 0.3, 0.02):
    prediction = (test['proba'] > i).astype('int')
    confusion_matrix = pd.crosstab(prediction,test.bad_ind, margins = True)
    precision = confusion_matrix.ix[0, 0] /confusion_matrix.ix['All', 0]
    recall = confusion_matrix.ix[0, 0] / confusion_matrix.ix[0, 'All']
    Specificity = confusion_matrix.ix[1, 1] /confusion_matrix.ix[1,'All']
    f1_score = 2 * (precision * recall) / (precision + recall)
    print('threshold: %s, precision: %.2f, recall:%.2f ,Specificity:%.2f , f1_score:%.2f'%(i, precision, recall, Specificity,f1_score))


# - 绘制ROC曲线
# In[27]:
import sklearn.metrics as metrics

fpr_test, tpr_test, th_test = metrics.roc_curve(test.bad_ind, test.proba)
fpr_train, tpr_train, th_train = metrics.roc_curve(train.bad_ind, train.proba)

plt.figure(figsize=[3, 3])
plt.plot(fpr_test, tpr_test, 'b--')
plt.plot(fpr_train, tpr_train, 'r-')
plt.title('ROC curve')
plt.show()

# In[28]:
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))


# 包含分类预测变量的逻辑回归
#%%
formula = '''bad_ind ~ C(used_ind)'''
lg_m = smf.glm(formula=formula, data=train, 
             family=sm.families.Binomial(sm.families.links.logit)).fit()
lg_m.summary()

# In[14]:
#- 多元逻辑回归
# 向前法
def forward_select(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    while remaining:
        aic_with_candidates=[]
        for candidate in remaining:
            formula = "{} ~ {}".format(
                response,' + '.join(selected + [candidate]))
            aic = smf.glm(
                formula=formula, data=data, 
                family=sm.families.Binomial(sm.families.links.logit)
            ).fit().aic
            aic_with_candidates.append((aic, candidate))
        aic_with_candidates.sort(reverse=True)
        best_new_score, best_candidate=aic_with_candidates.pop()
        if current_score > best_new_score: 
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print ('aic is {},continuing!'.format(current_score))
        else:        
            print ('forward selection over!')
            break
            
    formula = "{} ~ {} ".format(response,' + '.join(selected))
    print('final formula is {}'.format(formula))
    model = smf.glm(
        formula=formula, data=data, 
        family=sm.families.Binomial(sm.families.links.logit)
    ).fit()
    return(model)


# In[16]:
#只有连续变量可以进行变量筛选，分类变量需要进行WOE转换才可以进行变量筛选
candidates = ['bad_ind','tot_derog','age_oldest_tr','tot_open_tr','rev_util','fico_score','loan_term','ltv',
              'veh_mileage','dti_hist','dti_mew','fta','nth','nta']
data_for_select = train[candidates]

lg_m1 = forward_select(data=data_for_select, response='bad_ind')
lg_m1.summary()


# Seemingly wrong when using 'statsmmodels.stats.outliers_influence.variance_inflation_factor'
# 方差膨胀因子 共线性检验： 特征向量 选择完之后，还要进行 共线性检验
# In[17]:
def vif(df, col_i):
    from statsmodels.formula.api import ols
    
    cols = list(df.columns)
    cols.remove(col_i)
    cols_noti = cols
    formula = col_i + '~' + '+'.join(cols_noti)
    r2 = ols(formula, df).fit().rsquared
    return 1. / (1. - r2)

# In[18]:
candidates = ['bad_ind','fico_score','ltv','age_oldest_tr','tot_derog','nth','tot_open_tr','veh_mileage','rev_util']
exog = train[candidates].drop(['bad_ind'], axis=1)

for i in exog.columns:
    print(i, '\t', vif(df=exog, col_i=i))

#%%
train['proba'] = lg_m1.predict(train)
test['proba'] = lg_m1.predict(test)
import sklearn.metrics as metrics

fpr_test, tpr_test, th_test = metrics.roc_curve(test.bad_ind, test.proba)
fpr_train, tpr_train, th_train = metrics.roc_curve(train.bad_ind, train.proba)

plt.figure(figsize=[3, 3])
plt.plot(fpr_test, tpr_test, 'b--')
plt.plot(fpr_train, tpr_train, 'r-')
plt.title('ROC curve')
plt.show()

# In[28]:
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))

#%%
#目前vehicle_year、vehicle_make、bankruptcy_ind、used_ind这些分类变量无法通过逐步变量筛选法
#解决方案：
#1、逐一根据显著性测试
#2、使用决策树等方法筛选变量，但是多分类变量需要事先进行变量概化
#3、使用WOE转换，多分类变量也需要事先进行概化，使用scorecardpy包中的woe算法可以自动进行概化
# 使用第一种方法
#formula = '''bad_ind ~ fico_score+ltv+age_oldest_tr+tot_derog+nth+tot_open_tr+veh_mileage+rev_util+C(used_ind)+C(vehicle_year)+C(bankruptcy_ind)'''
formula = '''bad_ind ~ fico_score+ltv+age_oldest_tr+tot_derog+nth+tot_open_tr+veh_mileage+rev_util+C(bankruptcy_ind)'''
lg_m = smf.glm(formula=formula, data=train, 
             family=sm.families.Binomial(sm.families.links.logit)).fit()
lg_m.summary()

#%%



