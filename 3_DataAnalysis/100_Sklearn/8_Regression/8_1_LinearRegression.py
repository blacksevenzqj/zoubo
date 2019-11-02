# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:42:26 2019

@author: dell
"""

from sklearn.linear_model import LinearRegression as LR, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing as fch #加利福尼亚房屋价值数据集
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import FeatureTools as ft

# In[]:
housevalue = fch() #会需要下载，大家可以提前运行试试看

X = pd.DataFrame(housevalue.data) #放入DataFrame中便于查看
y = housevalue.target

# In[]:
print(y.min(), y.max()) # 0.14999 5.00001

'''
MedInc ：该街区住户的收入中位数
HouseAge ：该街区房屋使用年代的中位数
AveRooms ：该街区平均的房间数目
AveBedrms ：该街区平均的卧室数目
Population ：街区人口
AveOccup ：平均入住率
Latitude ：街区的纬度
Longitude ：街区的经度
'''
housevalue.feature_names #特征名字

X.columns = housevalue.feature_names

# In[]:
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)

# 索引重排
# 因为y 是array格式的，索引自动重排了。
ft.recovery_index([Xtrain, Xtest])

# In[]:
# 数据分布
# 直方图
for fe in Xtrain.columns:
    f, axes = plt.subplots(1,2, figsize=(23, 8))
    ft.con_data_distribution(Xtrain, fe, axes)
# In[]:
f, axes = plt.subplots(1,2, figsize=(23, 8))
ft.con_data_distribution(pd.DataFrame(Ytrain), 0, axes)

# In[]:
# 正太、偏度检测
ft.normal_comprehensive(Xtrain)


# In[]:
# 散点图
for i in Xtrain.columns:
    f, axes = plt.subplots(1,1, figsize=(10, 6))
    ft.con_data_scatter(Xtrain, i, Ytrain, "House rating", axes)

# In[]:
# 特征的 皮尔森相关度
ft.corrFunction(Xtrain)



# In[]:
'''
测试 SKlearn 和 statsmodels，在 无超参数情况下，是相同的。
'''
reg = LR().fit(Xtrain, Ytrain)
yhat = reg.predict(Xtrain) #预测我们的yhat
print(reg.score(Xtrain, Ytrain))

predict = pd.DataFrame(yhat, columns=['Pred'])
resid = pd.DataFrame((Ytrain - yhat), columns=['resid'])
resid_1 = pd.concat([predict,resid], axis=1)
resid_1.plot('Pred', 'resid',kind='scatter')

print(ft.r2_score_customize(Ytrain, yhat, 2))
print(ft.adj_r2_customize(Ytrain, yhat, Xtrain.shape[1], 2))

# In[]:
from statsmodels.formula.api import ols

temp_Y = pd.DataFrame(Ytrain, columns=['Y'])
temp_X = pd.concat([Xtrain, temp_Y], axis=1)

cols = list(temp_X.columns)
cols.remove("Y")
cols_noti = cols
formula = "Y" + '~' + '+'.join(cols_noti)

temp_X2 = temp_X.copy()
lm_s = ols(formula, data=temp_X).fit()
print(lm_s.rsquared, lm_s.aic)

temp_X2['Pred'] = lm_s.predict(temp_X)
temp_X2['resid'] = lm_s.resid # 残差随着x的增大呈现 喇叭口形状，出现异方差
temp_X2.plot('Pred', 'resid',kind='scatter') # Pred = β*Income，随着预测值的增大，残差resid呈现 喇叭口形状
lm_s.summary()



# In[]:
# 一、使用统计学的先验思路 解决 多重共线性
# 线性回归特征分析：基于统计学方法，自定义封装的函数
# 1、扰动项ε 独立同分布 （异方差检验、DW检验）
'''
MedInc ：该街区住户的收入中位数
HouseAge ：该街区房屋使用年代的中位数
AveRooms ：该街区平均的房间数目
AveBedrms ：该街区平均的卧室数目
Population ：街区人口
AveOccup ：平均入住率
Latitude ：街区的纬度
Longitude ：街区的经度
'''
col_list = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
r_sq = ft.heteroscedastic(Xtrain, Ytrain, col_list)
# In[]:
col = "HouseAge"
r_sq = ft.heteroscedastic_singe(Xtrain, Ytrain, col)

# In[]:
# 2、扰动项ε 服从正太分布 （QQ检验）
ft.disturbance_term_normal(Xtrain, Ytrain, col_list)

# In[]:
# 3、学生化残差
ft.studentized_residual(Xtrain, Ytrain)
# In[]:
#ft.strong_influence_point(Xtrain, Ytrain) # 太耗时，最好不要用了

# In[]:
# 4、方差膨胀因子
temp_variance, temp_variance_ln = ft.variance_expansion_coefficient(Xtrain, col_list)
# In[]:
temp_variance, temp_variance_ln = ft.variance_expansion_coefficient(Xtrain, ["Latitude", "Longitude"])



# In[]:
# 标准化：
# 标准化，返回值为标准化后的数据
ss = StandardScaler().fit(Xtrain)
Xtrain_ = ss.transform(Xtrain)
Xtest_ = ss.transform(Xtest)

Xtrain_ = pd.DataFrame(Xtrain_, columns=Xtrain.columns)
Xtest_ = pd.DataFrame(Xtest_, columns=Xtest.columns)

# In[]:
reg = LR().fit(Xtrain_, Ytrain)
yhat = reg.predict(Xtest_) #预测我们的yhat

print(yhat.min(), yhat.max(), yhat.mean())
print(reg.coef_)
print([*zip(Xtrain.columns,reg.coef_)])
print(reg.intercept_)

# In[]:
# 均方误差
from sklearn.metrics import mean_squared_error as MSE

print(Ytest.min(), Ytest.max(), Ytest.mean()) # 0.14999 5.00001 2.0819292877906976
print(MSE(Ytest, yhat)) # 0.5309012639324573

print(MSE(Ytest, yhat) / Ytest.mean()) # 每个样本误差？
k = np.sum(abs(yhat - Ytest)) / len(Ytest)
print(k / Ytest.mean()) # 每个样本误差？

# In[]:
# 负均方误差
import sklearn
sklearn.metrics.SCORERS.keys()
cross_val_score(reg, Xtest_, Ytest, cv=10, scoring="neg_mean_squared_error") # 负均方误差

# In[]:
# 可解释方差（线性回归模型 基本不使用该指标）
'''
解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量的方差变化，值越小则说明效果越差。
'''
cross_val_score(reg, Xtest_, Ytest, cv=10, scoring="explained_variance")

# In[]:
# R^2
from sklearn.metrics import r2_score # R square

# 第一种方式：
print(r2_score(Ytest, yhat))
print(ft.r2_score_customize(Ytest, yhat, 2))
print(ft.adj_r2_customize(Ytest, yhat, Xtest_.shape[1], 2))

# 第二种方式：
print(reg.score(Xtest_,Ytest))
# 第三种方式：
print(cross_val_score(reg,Xtest_,Ytest,cv=10,scoring="r2").mean())

# In[]:
# 线性回归 数据拟合
ft.fitting_comparison(Ytest, yhat)

# In[]:
# 负的 r2： （数学验证r2可以取到负值）
rng = np.random.RandomState(42)
a = rng.randn(100, 80)
b = rng.randn(100)
cross_val_score(LR(), a, b, cv=5, scoring='r2')



# In[]:
# 二、改进线性回归模型 解决 多重共线性（岭回归、Lasso）
# 这两个模型不是为了提升模型表现，而是为了修复漏洞而设计的（实际上，我们使用岭回归或者Lasso，模型的效果往往会下降一些，因为我们删除了一小部分信息），因此在结果为上的机器学习领域颇有些被冷落的意味。

# 使用岭回归来进行建模： 岭回归模型 泛化能力（R^2均值与R^2方差）
'''
在统计学中，我们会通过VIF或者各种检验来判断数据是否存在共线性，然而在机器学习中，我们可以使用模型来判断——
如果一个数据集在岭回归中使用各种正则化参数取值下模型表现没有明显上升（比如出现持平或者下降），则说明数据没有多重共线性，顶多是特征之间有一些相关性。
反之，如果一个数据集在岭回归的各种正则化参数取值下表现出明显的上升趋势，则说明数据存在多重共线性。
'''

#交叉验证下，与线性回归相比，岭回归的 R^2结果 如何变化？
ft.linear_model_comparison(X, y, cv_customize=5, score_type=1, start=1, end=1001, step=100)
'''
可以看出，加利佛尼亚数据集上，岭回归的结果轻微上升，随后骤降。可以说，加利佛尼亚房屋价值数据集带有很轻
微的一部分共线性，这种共线性被正则化参数α消除后，模型的效果提升了一点点，但是对于整个模型而言是杯水车
薪。在过了控制多重共线性的点后，模型的效果飞速下降，显然是正则化的程度太重，挤占了参数w本来的估计空
间（α如果太大，也会导致w的估计出现较大的偏移，无法正确拟合数据的真实面貌）
从这个结果可以看出，加利佛尼亚数据集的核心问题不在于多重共线性，岭回归不能够提升模型表现。
'''
# In[]:
# 方差 如何变化：
ft.linear_model_comparison(X, y, cv_customize=5, score_type=2, start=1, end=1001, step=100)
# In[]:
# 针对R^2上升，方差也上升的这一段区间进行细化： 细化 R^2 学习曲线； 细化 方差 学习曲线； 并做对比：
ft.linear_model_comparison_all(X, y, cv_customize=5, start=1, end=1001, step=100)
'''
可以发现，模型R^2方差上升快速，R^2方差的上升部分变化 是 R^2的上升部分变化的0.3974倍，
因此只要噪声的状况维持恒定，模型的泛化误差可能还是一定程度上降低了的。
虽然岭回归和Lasso不是设计来提升模型表现，而是专注于解决多重共线性问题的，但当α在一定范围内变动的时候，消除多重共线性也许能够一定程度上提高模型的泛化能力。
'''

# In[]:
# 换波斯顿房价数据集测试：
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score

X = load_boston().data
y = load_boston().target

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
# In[]:
ft.linear_model_comparison(X, y, cv_customize=5, score_type=2, start=1, end=1001, step=100)
# In[]:
ft.linear_model_comparison(X, y, cv_customize=5, score_type=1, start=1, end=1001, step=100)
# In[]:
ft.linear_model_comparison(X, y, cv_customize=5, score_type=1, start=100, end=300, step=10, linear_show=False)
'''
可以发现，比起加利佛尼亚房屋价值数据集，波士顿房价数据集的方差降低明显，偏差也降低明显，可见使用岭回归
还是起到了一定的作用，模型的泛化能力是有可能会上升的。
遗憾的是，没有人会希望自己获取的数据中存在多重共线性，因此发布到scikit-learn或者kaggle上的数据基本都经过一定的多重共线性的处理的，
要找出绝对具有多重共线性的数据非常困难，也就无法给大家展示岭回归在实际数据中大显身手的模样。
我们也许可以找出具有一些相关性的数据，但是大家如果去尝试就会发现，基本上如果我们使用岭回归或者Lasso，
那模型的效果都是会降低的，很难升高，这恐怕也是岭回归和Lasso一定程度上被机器学习领域冷遇的原因。
'''


# In[]:
# 使用 岭回归 中 RidgeCV交叉验证类 （类似 网格搜索GridSearchCV）
from sklearn.linear_model import RidgeCV

X = pd.DataFrame(housevalue.data)
y = housevalue.target
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
ft.recovery_index([Xtrain, Xtest])

# In[]:
# RidgeCV交叉验证 默认计算的是 R^2
Ridge_ = RidgeCV(alphas=np.arange(1,1001,100)
                 #,scoring="neg_mean_squared_error" # 默认R^2
                 ,store_cv_values=True
                 #,cv=5 # 默认 留一验证： 论文证明岭回归最佳交叉验证方式
                ).fit(Xtrain, Ytrain)

# In[]:
# 无关交叉验证的岭回归结果： 根据交叉验证得出的模型，用于预测
Ridge_.score(Xtest, Ytest) # 这个接口只会计算 R^2

# In[]:
# 调用 RidgeCV模型训练 的所有 交叉验证的结果
# 留一交叉验证： 
# 矩阵为14448行： 与 折数相同 与 样本量相同 
# 10列： 与 正则化超参数alphas数量相同 
Ridge_.cv_values_.shape # (14448, 10) 求的是 R^2 折数的均值， 所以要 按行求均值

# In[]:
# 进行平均后可以查看每个正则化系数取值下的交叉验证结果
print(Ridge_.cv_values_.mean(axis=0)) # 默认R^2， 注意： 留一交叉验证， 按行求均值

# 查看被选择出来的最佳正则化系数
print(Ridge_.alpha_) # 根据 默认R^2 值选择出来的 α = 101

# In[]:
# 换成 负均方误差
Ridge_ = RidgeCV(alphas=np.arange(1,1001,100)
                 ,scoring="neg_mean_squared_error" # 默认R^2
                 ,store_cv_values=True
                 #,cv=5 # 默认 留一验证
                ).fit(Xtrain, Ytrain)
# In[]:
# 进行平均后可以查看每个正则化系数取值下的交叉验证结果
print(Ridge_.cv_values_.mean(axis=0)) # 负均方误差， 注意： 留一交叉验证， 按行求均值

# 查看被选择出来的最佳正则化系数
print(Ridge_.alpha_) # 根据 负均方误差 值选择出来的 α = 101
# 可以看出 根据 R^2 和 neg_mean_squared_error 计算出来的 最优正则化系数α 是一样的 = 101



# In[]:
# 三、使用Lasso来进行建模： 
X = pd.DataFrame(housevalue.data)
y = housevalue.target
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
ft.recovery_index([Xtrain, Xtest])

# In[]:
# 线性回归进行拟合
reg = LR().fit(Xtrain,Ytrain)
(reg.coef_*100).tolist()
print(reg.score(Xtest,Ytest)) # 默认R^2

# 岭回归进行拟合
Ridge_ = Ridge(alpha=0).fit(Xtrain,Ytrain)
(Ridge_.coef_*100).tolist()
print(Ridge_.score(Xtest,Ytest)) # 默认R^2

# Lasso进行拟合
lasso_ = Lasso(alpha=0).fit(Xtrain,Ytrain)
(lasso_.coef_*100).tolist()
print(lasso_.score(Xtest,Ytest)) # 默认R^2

# In[]:
# 有了坐标下降，就有迭代和收敛的问题，因此sklearn不推荐我们使用0这样的正则化系数。如果我们的确希望取到0，
# 那我们可以使用一个比较很小的数，比如0.01，或者 10 * e^-3 这样的值：
# 岭回归进行拟合
Ridge_ = Ridge(alpha=0.01).fit(Xtrain,Ytrain)
(Ridge_.coef_*100).tolist()

# Lasso进行拟合
lasso_ = Lasso(alpha=0.01).fit(Xtrain,Ytrain)
(lasso_.coef_*100).tolist()

# In[]:
# 加大正则项系数，观察模型的系数发生了什么变化
Ridge_ = Ridge(alpha=10**4).fit(Xtrain,Ytrain)
(Ridge_.coef_*100).tolist()
lasso_ = Lasso(alpha=1).fit(Xtrain,Ytrain)
(lasso_.coef_*100).tolist()
#将系数进行绘图
plt.plot(range(1,9),(reg.coef_*100).tolist(),color="red",label="LR")
plt.plot(range(1,9),(Ridge_.coef_*100).tolist(),color="orange",label="Ridge")
plt.plot(range(1,9),(lasso_.coef_*100).tolist(),color="k",label="Lasso")
plt.plot(range(1,9),[0]*8,color="grey",linestyle="--")
plt.xlabel('w') #横坐标是每一个特征所对应的系数
plt.legend()
plt.show()
'''
可见，比起岭回归，Lasso所带的L1正则项对于系数的惩罚要重得多，并且它会将系数压缩至0，因此可以被用来做特征选择。
也因此，我们往往让Lasso的正则化系数 在很小的空间中变动，以此来寻找最佳的正则化系数。
'''

# In[]:
from sklearn.linear_model import LassoCV
# LassoCV交叉验证 默认计算的是 均方误差

## 自己建立Lasso进行alpha选择的范围： 其实是形成10为底的指数函数： 10**(-10)到10**(-2)次方
#alpharange = np.logspace(-10, -2, 200,base=10)
#print(alpharange.shape)
#lasso_ = LassoCV(alphas=alpharange #自行输入的alpha的取值范围
#                ,cv=5 #交叉验证的折数
#                ).fit(Xtrain, Ytrain)

# 使用lassoCV自带的正则化路径长度和路径中的alpha个数来自动建立alpha选择的范围
lasso_ = LassoCV(eps=0.00001 # 默认 0.001
              ,n_alphas=300 # 默认 100
              ,cv=5 # 默认就是5折
                ).fit(Xtrain, Ytrain)


# 查看被选择出来的最佳正则化系数
print(lasso_.alpha_)
# 使用正则化路径的长度和路径中α的个数来自动生成的，用来进行交叉验证的正则化参数
print(lasso_.alphas_.shape)

# 调用LassoCV模型训练 的所有 交叉验证的结果： 返回每个alpha下的五折交叉验证结果
# 普通交叉验证： 
# 矩阵为200行： 与 正则化超参数alphas_数量相同 
# 5列： 折数
print(lasso_.mse_path_) # 求的是 均方误差 折数的均值， 所以要 按列求均值

print(lasso_.mse_path_.mean(axis=1).shape) # 按列求均值，(200,)
'''
1、在岭回归当中，我们是留一验证，因此我们的交叉验证结果返回的是，每一个样本在每个alpha下的交叉验证结果
因此我们要求每个alpha下的交叉验证均值，就是axis=0，按行求均值。
2、而在这里，我们返回的是，每一个alpha取值下，每一折交叉验证的结果
因此我们要求每个alpha下的交叉验证均值，就是axis=1，按列求均值。
'''

# 最佳正则化系数下获得的模型的系数结果
print(lasso_.coef_)
print(lasso_.score(Xtest,Ytest)) # R^2 0.60389154238192 

# 与线性回归相比如何？
reg = LR().fit(Xtrain,Ytrain)
print(reg.score(Xtest,Ytest)) # R^2 0.6043668160178817








