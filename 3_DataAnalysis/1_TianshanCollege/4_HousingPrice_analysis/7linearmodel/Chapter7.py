# coding: utf-8

# # 线性回归模型与诊断
# 数据说明：本数据是一份汽车贷款数据

# |字段名|中文含义|
# |:--:|:--:|
# |id|id|
# |Acc|是否开卡(1=已开通)|
# |avg_exp|月均信用卡支出（元）|
# |avg_exp_ln|月均信用卡支出的自然对数|
# |gender|性别(男=1)|
# |Age|年龄|
# |Income|年收入（万元）|
# |Ownrent|是否自有住房（有=1；无=0)|
# |Selfempl|是否自谋职业(1=yes, 0=no)|
# |dist_home_val|所住小区房屋均价(万元)|
# |dist_avg_income|当地人均收入|
# |high_avg|高出当地平均收入|
# |edu_class|教育等级：小学及以下开通=0，中学=1，本科=2，研究生=3|

# In[1]:
# get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
# import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression as LR, Ridge, Lasso
import FeatureTools as ft

os.chdir(
    r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\1_TianshanCollege\4_HousingPrice_analysis\7linearmodel")
# pd.set_option('display.max_columns', 8)

# 导入数据和数据清洗
# In[2]:
raw = pd.read_csv(r'creditcard_exp.csv', skipinitialspace=True)
raw.head()

# In[3]:
exp = raw[raw['avg_exp'].notnull()].copy().iloc[:, 2:].drop('age2', axis=1)
exp_new = raw[raw['avg_exp'].isnull()].copy().iloc[:, 2:].drop('age2', axis=1)
exp.describe(include='all')

# 相关性分析
# 散点图
# In[4]:
exp.plot('Income', 'avg_exp', kind='scatter')
plt.show()

# In[5]:
exp[['Income', 'avg_exp', 'Age', 'dist_home_val']].corr(method='pearson')  # 皮尔森相似度矩阵

# 线性回归算法
# 简单线性回归
# In[6]:
# ols类计算 线性回归模型 并得到 预测值 和 残差
'''
P>|t| 解释：
原假设是：β1 = 0，备选假设是:β1≠0。样本量<100：α=10%
P>|t|值越接近于0，说明p值越 < α显著性水平（0.1），越在拒绝域内，启用备选假设β1≠0，该特征对模型有用。
P>|t|值越大于0，说明p值越 ＞ α显著性水平（0.1），越不在拒绝域内，启用原假设β1=0，该特征对模型没有用。
如：Income的P>|t|=0.000，说明概率p在拒绝域内，启用备选假设β1≠0，特征Income对模型有用。
如：Age的P>|t|=0.934，说明概率p不在拒绝域内，启用原假设β1=0，特征Age对模型没有用。
'''
lm_s = ols('avg_exp ~ Income + Age + dist_home_val', data=exp).fit()
lm_s.summary()

# 在原始数据集上得到预测值和残差
# In[8]:
predict_resid = pd.DataFrame([lm_s.predict(exp), lm_s.resid], index=['predict', 'resid']).T

# 在待预测数据集上得到预测值
# In[9]:
lm_s.predict(exp_new)[:5]

# 多元线性回归的变量筛选（自定义函数：向前法）
# In[11]:
'''forward select'''


def forward_select(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    while remaining:
        aic_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {}".format(
                response, ' + '.join(selected + [candidate]))
            aic = ols(formula=formula, data=data).fit().aic
            aic_with_candidates.append((aic, candidate))
        aic_with_candidates.sort(reverse=True)
        best_new_score, best_candidate = aic_with_candidates.pop()
        if current_score > best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print('aic is {},continuing!'.format(current_score))
        else:
            print('forward selection over!')
            break

    formula = "{} ~ {} ".format(response, ' + '.join(selected))
    print('final formula is {}'.format(formula))
    model = ols(formula=formula, data=data).fit()
    return (model)


# In[12]:
data_for_select = exp[['avg_exp', 'Income', 'Age', 'dist_home_val', 'dist_avg_income']]
lm_m = forward_select(data=data_for_select, response='avg_exp')
print(lm_m.rsquared)

# 7.3 线性回归的诊断
# 7.3.1 残差分析
# In[13]:
# ols类计算 线性回归模型 并得到 预测值 和 残差
ana1 = ols('avg_exp ~ Age + Income + dist_home_val', data=exp).fit()
exp['Pred'] = ana1.predict(exp)
exp['resid'] = ana1.resid  # 残差随着x的增大呈现 喇叭口形状，出现异方差
exp.plot('Pred', 'resid', kind='scatter')  # Pred = β*Income，随着预测值的增大，残差resid呈现 喇叭口形状
ana1.summary()
# In[]:
Xtrain = exp[['Age', 'Income', 'dist_home_val']]
Ytrain = exp[['avg_exp']]

reg = LR().fit(Xtrain, Ytrain)
yhat = reg.predict(Xtrain)  # 预测我们的yhat
print(reg.score(Xtrain, Ytrain))

predict = pd.DataFrame(yhat, columns=['Pred'])
print(Ytrain.dtypes, predict.dtypes)

y = Ytrain.copy()
ft.recovery_index([y])
# resid = pd.DataFrame((y['avg_exp'] - predict["Pred"]), columns=['resid'])
resid = pd.DataFrame(y['avg_exp'].sub(predict["Pred"]), columns=['resid'])

resid_1 = pd.concat([predict, resid], axis=1)
resid_1.plot('Pred', 'resid', kind='scatter')

print(ft.r2_score_customize(Ytrain, yhat, 1))
print(ft.r2_score_customize(Ytrain, yhat, 2))
print(ft.adj_r2_customize(Ytrain, yhat, Xtrain.shape[1], 2))

# In[15]:
# ols类计算 线性回归模型 并得到 预测值 和 残差
# 遇到异方差情况,教科书上会介绍使用加权最小二乘法，但是实际上最常用的是对 被解释变量y 取对数
# R-squared为0.454
ana1 = ols('avg_exp ~ Income', exp).fit()
exp['Pred'] = ana1.predict(exp)
exp['resid'] = ana1.resid  # 残差
exp.plot('Income', 'resid', kind='scatter')  # 随着Income的增大，残差resid呈现 喇叭口形状
ana1.summary()

# In[15]:
# 对y取对数得： avg_exp_ln
# y取对数后 减小了 异方差，但 R-squared为0.403，比之前减小了。
ana2 = ols('avg_exp_ln ~ Income', exp).fit()  # avg_exp_ln原始数据中就有了
exp['Pred'] = ana2.predict(exp)
exp['resid'] = ana2.resid  # 残差
exp.plot('Income', 'resid', kind='scatter')  # 随着Income的增大，残差resid稍微形状好了些
ana2.summary()

# In[16]:
# 取对数会使模型更有解释意义
# y和x同时取对数，减小异方差，同时增大 R-squared为0.480
exp['Income_ln'] = np.log(exp['Income'])  # 取对数之后 是按百分比 看指标
ana3 = ols('avg_exp_ln ~ Income_ln', exp).fit()
exp['Pred'] = ana3.predict(exp)
exp['resid'] = ana3.resid
exp.plot('Income_ln', 'resid', kind='scatter')  # 随着Income的增大，残差resid稍微形状好了些
ana3.summary()
# In[16]:
# 调用封装好的函数：残差分析
Ytrain = exp['avg_exp']
Ytrain.name = 'Y'
r_sq = ft.heteroscedastic(exp, Ytrain, ['Age', 'Income', 'dist_home_val'])

# 7.3.2 强影响点分析：学生化残差
# In[18]:
# 就是 R-squared 指标，可以看出 ln(exp)~ln(Income) 模型的 R-squared为0.480最高。
r_sq = {'exp~Income': ana1.rsquared, 'ln(exp)~Income': ana2.rsquared, 'ln(exp)~ln(Income)': ana3.rsquared}
print(r_sq)

# In[19]:
# 7.3.2.1、手动计算 学生化残差
# 使用的就是 ln(exp)~ln(Income) 模型的 残差
# 求 学生化残差  =  (残差 - 残差均值) / 残差的标准差
exp['resid_t'] = (exp['resid'] - exp['resid'].mean()) / exp['resid'].std()

# In[20]:
# 样本量为几百个时：|SR| > 2 为强影响点
# 样本量为上千个时：|SR| > 3 为强影响点
exp[abs(exp['resid_t']) > 2]

# In[21]:
# 剔除强影响点后，重新建立模型。 R-squared为0.494，为截止最高值。
# 异常值 剔除 只做一次。
exp2 = exp[abs(exp['resid_t']) <= 2]
ana4 = ols('avg_exp_ln ~ Income_ln', exp2).fit()
exp2['Pred'] = ana4.predict(exp2)
exp2['resid'] = ana4.resid
exp2.plot('Income_ln', 'resid', kind='scatter')
ana4.summary()
# In[22]:
ana4.rsquared  # 就是 R-squared 指标
# In[]:
# 封装 学生化残差 函数
temp_index = ft.studentized_residual(exp['Income_ln'], exp['avg_exp_ln'], ['Income_ln'], 'avg_exp_ln', num=2)
print(temp_index)
exp.loc[temp_index]
# In[]
# 离群特征检测
ft.outlier_detection(exp, 'Income_ln', exp[['avg_exp_ln']], 'avg_exp_ln')

# In[23]:
# 7.3.2.2、statemodels包提供了更多强影响点判断指标
from statsmodels.stats.outliers_influence import OLSInfluence

# 使用的就是 ln(exp)~ln(Income) 模型，计算 强影响点
OLSInfluence(ana3).summary_frame().head()

# 7.3.3多重共线性分析：方差膨胀因子
# 增加变量
# 经过单变量线性回归的处理，我们基本对模型的性质有了一定的了解，接下来可以放入更多的连续型解释变量。在加入变量之前，要注意变量的函数形式转变。比如当地房屋均价、当地平均收入，其性质和个人收入一样，都需要取对数
# In[24]:
# exp2 是已经剔除了 两个 强影响点 后的数据集
exp2['dist_home_val_ln'] = np.log(exp2['dist_home_val'])  # 所住小区房屋均价(万元)|
exp2['dist_avg_income_ln'] = np.log(exp2['dist_avg_income'])  # 当地人均收入

# ols类计算 线性回归模型
# 第一次： Income_ln 和 dist_avg_income_ln 是强相关性，必须剔除一个（根据方差膨胀因子） R-squared=0.553
ana5 = ols('''avg_exp_ln ~ Income_ln + dist_home_val_ln + dist_avg_income_ln''', exp2).fit()
# 第二次
# ana5 = ols('''avg_exp_ln ~ dist_home_val_ln + dist_avg_income_ln''', exp2).fit() # R-squared=0.552

ana5.summary()

# In[25]:
# Step regression is not always work.
ana5.bse  # The standard errors of the parameter estimates


# The function "statsmodels.stats.outliers_influence.variance_inflation_factor" uses "OLS" to fit data, and it will generates a wrong rsquared. So define it ourselves!
# 函数“statsmodels.stats.outliers_influence.variance_inflation_factor”使用“OLS”来拟合数据，它将生成错误的rsquared。 所以自己定义：方差膨胀因子公式
# 一个自变量 和 多个自变量 计算 方差膨胀因子， > 10 表示 该变量多重共线性严重
# In[27]:
# 方差膨胀因子 公式：
def vif(df, col_i):
    cols = list(df.columns)
    cols.remove(col_i)
    cols_noti = cols
    formula = col_i + '~' + '+'.join(cols_noti)
    r2 = ols(formula, df).fit().rsquared
    return 1. / (1. - r2)


# In[28]:
'''
# Income_ln的R-squared：
r2 = ols('Income_ln ~ dist_home_val_ln + dist_avg_income_ln', exog).fit().rsquared
# Income_ln的方差膨胀因子：
vl = 1. / (1. - r2)

# dist_home_val_ln的R-squared
r2 = ols('dist_home_val_ln ~ Income_ln + dist_avg_income_ln', exog).fit().rsquared
# dist_home_val_ln的方差膨胀因子：
vl = 1. / (1. - r2)

# dist_avg_income_ln的R-squared
r2 = ols('dist_avg_income_ln ~ Income_ln + dist_home_val_ln', exog).fit().rsquared
# dist_avg_income_ln的方差膨胀因子：
vl = 1. / (1. - r2)
'''
# ols类计算 线性回归模型
# 第一次： Income_ln 和 dist_avg_income_ln 是强相关性，必须剔除一个（根据方差膨胀因子）
exog = exp2[['Income_ln', 'dist_home_val_ln', 'dist_avg_income_ln']]
# 第二次
# exog = exp2[['dist_home_val_ln','dist_avg_income_ln']]

# i 相当于 Y因变量 进行预测
for i in exog.columns:
    print(i, '\t', ft.vif(df=exog, col_i=i))  # 封装了

# In[]:
# i 相当于 Y因变量 进行预测
for i in exog.columns:
    print(i, '\t', ft.vif_sklearn(df=exog, col_i=i))  # 封装了

# In[]:
# 先求没有取对数的方差膨胀因子，再求取了对数的方差膨胀因子 方便对比
temp_dict, temp_dict_ln = ft.variance_expansion_coefficient(exp2, ['Income', 'dist_home_val', 'dist_avg_income'], 2)

# In[29]:
# Income_ln与dist_avg_income_ln具有共线性，使用“高出平均收入的比率”代替其中一个
# high_avg：高出当地平均收入； dist_avg_income：当地人均收入
exp2['high_avg_ratio'] = exp2['high_avg'] / exp2['dist_avg_income']

# In[30]:
# 计算 high_avg_ratio 特征加入后的 方差膨胀因子
exog1 = exp2[['high_avg_ratio', 'dist_home_val_ln', 'dist_avg_income_ln']]

for i in exog1.columns:
    print(i, '\t', vif(df=exog1, col_i=i))

# In[31]:
# 使用 自定义函数：向前法 对 选出的特征 再进行筛选
var_select = exp2[['avg_exp_ln', 'high_avg_ratio',
                   'dist_home_val_ln', 'dist_avg_income_ln']]
ana7 = forward_select(data=var_select, response='avg_exp_ln')
print(ana7.rsquared)

# In[32]:
# ols类计算 线性回归模型 包括了 两个样本T检验（还是 方差分析？因为带了 多分类特征）：
formula8 = '''
avg_exp_ln ~ dist_avg_income_ln + dist_home_val_ln + 
C(gender) + C(Ownrent) + C(Selfempl) + C(edu_class)
'''
ana8 = ols(formula8, exp2).fit()
ana8.summary()

# In[33]:
# ols类计算 线性回归模型 包括了 两个样本T检验（还是 方差分析？因为带了 多分类特征），有交互项：
formula9 = '''
avg_exp_ln ~ dist_avg_income_ln + dist_home_val_ln + 
C(Selfempl) + C(gender):C(edu_class)
'''
ana9 = ols(formula9, exp2).fit()
ana9.summary()

# 正则算法（statsmodels包的ols类的正则化算法 和 sklearn的正则化算法） 没看，有需要再看吧
# In[34]:
# 岭回归
lmr = ols('avg_exp ~ Income + dist_home_val + dist_avg_income',
          data=exp).fit_regularized(alpha=1, L1_wt=0)

lmr.summary()
# L1_wt参数为0则使用岭回归，为1使用lasso

# In[35]:
# LASSO算法
lmr1 = ols('avg_exp ~ Age + Income + dist_home_val + dist_avg_income',
           data=exp).fit_regularized(alpha=1, L1_wt=1)
lmr1.summary()

# 使用scikit-learn进行正则化参数调优
# In[36]:
from sklearn.preprocessing import StandardScaler

continuous_xcols = ['Age', 'Income', 'dist_home_val',
                    'dist_avg_income']  # 抽取连续变量
scaler = StandardScaler()  # 标准化
X = scaler.fit_transform(exp[continuous_xcols])
y = exp['avg_exp_ln']

# In[37]:
from sklearn.linear_model import RidgeCV

alphas = np.logspace(-2, 3, 100, base=10)

# Search the min MSE by CV
rcv = RidgeCV(alphas=alphas, store_cv_values=True)
rcv.fit(X, y)

# In[38]:
print('The best alpha is {}'.format(rcv.alpha_))
print('The r-square is {}'.format(rcv.score(X, y)))
# Default score is rsquared

# In[39]:
X_new = scaler.transform(exp_new[continuous_xcols])
np.exp(rcv.predict(X_new)[:5])

# In[40]:
cv_values = rcv.cv_values_
n_fold, n_alphas = cv_values.shape

cv_mean = cv_values.mean(axis=0)
cv_std = cv_values.std(axis=0)
ub = cv_mean + cv_std / np.sqrt(n_fold)
lb = cv_mean - cv_std / np.sqrt(n_fold)

plt.semilogx(alphas, cv_mean, label='mean_score')
plt.fill_between(alphas, lb, ub, alpha=0.2)
plt.xlabel("$\\alpha$")
plt.ylabel("mean squared errors")
plt.legend(loc="best")
plt.show()

# In[41]:
rcv.coef_
# 手动选择正则化系数——根据业务判断

# 岭迹图
# In[42]:
from sklearn.linear_model import Ridge

ridge = Ridge()

coefs = []
for alpha in alphas:
    ridge.set_params(alpha=alpha)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

# In[43]:

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

# In[44]:

ridge.set_params(alpha=40)
ridge.fit(X, y)
ridge.coef_

# In[45]:

ridge.score(X, y)

# 预测

# In[46]:

np.exp(ridge.predict(X_new)[:5])

# lasso

# In[54]:

from sklearn.linear_model import LassoCV

lasso_alphas = np.logspace(-3, 0, 100, base=10)
lcv = LassoCV(alphas=lasso_alphas, cv=10)  # Search the min MSE by CV
lcv.fit(X, y)

print('The best alpha is {}'.format(lcv.alpha_))
print('The r-square is {}'.format(lcv.score(X, y)))
# Default score is rsquared


# In[49]:

from sklearn.linear_model import Lasso

lasso = Lasso()
lasso_coefs = []
for alpha in lasso_alphas:
    lasso.set_params(alpha=alpha)
    lasso.fit(X, y)
    lasso_coefs.append(lasso.coef_)

# In[50]:

ax = plt.gca()

ax.plot(lasso_alphas, lasso_coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Lasso coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

# In[51]:

lcv.coef_

# 弹性网络

# In[52]:

from sklearn.linear_model import ElasticNetCV

l1_ratio = [0, .1, .5, .7, .9, .95, .99, 1]

encv = ElasticNetCV(l1_ratio=l1_ratio)
encv.fit(X, y)

# In[53]:

print('The best l1_ratio is {}'.format(encv.l1_ratio_))
print('The best alpha is {}'.format(encv.alpha_))


