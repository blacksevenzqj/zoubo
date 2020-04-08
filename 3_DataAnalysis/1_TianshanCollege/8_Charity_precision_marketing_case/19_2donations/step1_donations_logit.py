# coding: utf-8

# 本代码案例为 **两阶段精准营销模型** 中的 **构造营销响应模型** 部分
#
# # 数据挖掘方法论──SEMMA模型训练使用流程
#
# - Sample──数据取样
#
# - Explore──数据特征探索、分析和予处理
#
# - Modify──问题明确化、数据调整和技术选择
#
# - Model──模型的研发、知识的发现
#
# - Assess──模型和知识的综合解释和评价
#
# # 数据获取与导入的S（抽样）阶段。
#
# ## 规整数据集

# In[1]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from woe import WoE  # 从本地导入
import os
import FeatureTools as ft
import Tools_customize as tc
import Binning_tools as bt
import Dimensionality_reduction as dr

os.chdir(
    r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\1_TianshanCollege\8_Charity_precision_marketing_case\19_2donations")

# In[2]:
# 创建一个列表，用来保存所有的建模数据清洗的相关信息
DATA_CLEAN = []

# In[3]:
model_data = pd.read_csv("donations.csv").drop(["ID", "TARGET_D"], 1)
model_data.head()
print(model_data.shape)

# In[4]:
model_data.dtypes

# In[5]:
# 变量筛选：
# 因变量Y（分类）
y = 'TARGET_B'

# 自变量X（连续）
# var_c = ["GiftCnt36","GiftCntAll","GiftCntCard36",
#         "GiftCntCardAll","GiftTimeLast","GiftTimeFirst",
#         "PromCnt12","PromCnt36","PromCntAll",
#         "PromCntCard12","PromCntCard36","PromCntCardAll",
#         "StatusCatStarAll","DemAge","DemMedHomeValue",
#         "DemPctVeterans","DemMedIncome","GiftAvgLast",
#         "GiftAvg36","GiftAvgAll","GiftAvgCard36"]

var_c = ["GiftCnt36", "GiftCntAll", "GiftCntCard36",
         "GiftCntCardAll", "GiftTimeLast", "GiftTimeFirst",
         "PromCnt12", "PromCnt36", "PromCntAll",
         "PromCntCard12", "PromCntCard36", "PromCntCardAll",

         "StatusCatStarAll", "DemAge",
         #         "DemMedHomeValue", # 数据有问题的特征，教程中显示也是被淘汰的变量

         "DemPctVeterans",
         #         "DemMedIncome",    # 数据有问题的特征，教程中显示也是被淘汰的变量
         "GiftAvgLast",

         "GiftAvg36", "GiftAvgAll", "GiftAvgCard36"]

# 自变量X（分类）
var_d = ['StatusCat96NK', 'DemHomeOwner', 'DemGender', 'DemCluster']

# In[6]:
# 1、数据取样S（变量初筛）阶段
X = model_data[var_c + var_d].copy()
Y = model_data[y].copy()

# 筛选预测能力强的变量

# 1、**WoE类参数说明**:
# + **qnt_num**:int,等频分箱个数,默认16
# + **min_block_size**:int,最小观测数目，默认16
# + **spec_values**:dict,若为分类自变量，指派替换值
# + **v_type**:str,自变量类型,分类:‘d’,连续变量:‘c’，默认'c'
# + **bins**:list,预定义的连续变量分箱区间
# + **t_type**:str,目标变量类型,二分类:‘b’,连续变量:‘c’，默认'b'

# 2、**WoE类重要方法**:
# + **plot**:绘制WOE图
# + **transform**:转换数据为WOE数据
# + **fit_transform**:转换数据为WOE数据
# + **optimize**:连续变量使用最优分箱

# 3、**WoE类重要属性**:
# + **bins**:分箱结果汇总
# + **iv**:变量的信息价值

# In[7]:
# 1.1、根据IV值筛选变量 - 自变量X（分类） 与 因变量Y（分类）
iv_d = {}
for i in var_d:  # 自变量X（分类）
    iv_d[i] = WoE(v_type='d').fit(X[i], Y).iv

pd.Series(iv_d).sort_values(ascending=False)
# In[8]:
# 保留iv值较高的分类变量，> 0.02 作为阈值。
var_d_s = ['StatusCat96NK', 'DemCluster']  # 剔除了 'DemHomeOwner', 'DemGender'
# In[8]:
# 自己封装的
iv_d, var_d_s = ft.get_all_category_feature_ivs(model_data, var_d, 'TARGET_B', False)

# In[9]:
# 1.2、自变量X（连续） 与 因变量Y（分类）
iv_c = {}
for i in var_c:  # 自变量X（连续）
    iv_c[i] = WoE(v_type='c', t_type='b', qnt_num=3).fit(X[i], Y).iv

sort_iv_c = pd.Series(iv_c).sort_values(ascending=False)
print(sort_iv_c)
# In[10]:
# 以 2% 作为选取变量的阈值
'''
剔除了：
PromCntCard12       0.017793
PromCnt12           0.014559
PromCnt36           0.013919
DemAge              0.008953
DemPctVeterans      0.004013
GiftCntCard36       0.000000
StatusCatStarAll    0.000000
'''
var_c_s = list(sort_iv_c[sort_iv_c > 0.02].index)
print(var_c_s)
# In[10]:
# 自己封装的
iv_c, var_c_s = ft.get_all_con_feature_ivs(model_data, var_c, 'TARGET_B')

# In[11]:
X = model_data[var_c_s + var_d_s].copy()  # 第一次变量筛选之后 X第一次赋值（IV值筛选）
Y = model_data[y].copy()

# 2、针对每个变量的E（探索）阶段
# 2.1、对连续变量的统计探索
# In[12]:
X[var_c_s].describe().T

# In[13]:
# 利用 众数 减去 中位数 的差值  除以  四分位距来 查找是否有可能存在异常值
abs((X[var_c_s].mode().iloc[0,] - X[var_c_s].median()) /
    (X[var_c_s].quantile(0.75) - X[var_c_s].quantile(0.25)))
# PromCntAll 0.972222 值很大，需要进一步用直方图观测。
# In[]:
# 自己封装的：
temp_outlier = ft.all_con_mode_median_iqr_outlier(X, var_c_s)

# In[14]:
# 对嫌疑最大的几个变量进行可视化分析
plt.hist(X["PromCntAll"], bins=20);
# 从直方图中可以看出： 数据有2个最大峰值，属于正常数据，不用清洗。


# 2.2、对分类变量的统计探索
# 查看是否 分类的 类别过多
# In[15]:
X["StatusCat96NK"].value_counts()
# 一共6个类别 属于正常，但 L类别 的频次只有34，太少。先记下来。

# In[16]:
len(X["DemCluster"].value_counts())  # 名族共54个。

# In[17]:
# 3、针对有问题的变量进行修改的M（修改）阶段
# 3.1、将连续变量的错误值改为缺失值，本模型中筛选后的变量,没有发现无错误值，将连续变量的缺失值用中位数填补
# In[18]:
# 查看缺失比例
1 - (X.describe().T["count"]) / len(X)  # GiftAvgCard36 有 0.18377 比例的数据缺失。

# In[19]:
'''
1、缺失值填充：不用“多重差补”。用均值或中位数填充，用模型开发时的均值填充，而不是运行时数据的均值填充。
2、要保证模型数据的稳定性，最担心的就是变量飘逸：模型运行时X的均值发生变化，随之预测Y的均值也发生改变，模型就不起效了。
'''
fill_GiftAvgCard36 = X.GiftAvgCard36.median()  # 取GiftAvgCard36的中位数作为填充值。
X.GiftAvgCard36.fillna(value=fill_GiftAvgCard36, inplace=True)  # 填充缺失值。

# In[20]:
# 不管是 test数据集，还是运行时数据集，都按照这个流程走一遍
# 将填补修改信息保存至数据清洗信息当中.1
DATA_CLEAN.append({"fill_GiftAvgCard36": fill_GiftAvgCard36})
print(DATA_CLEAN)

# 3.2、分类变量概化：对分类水平过多的变量进行合并（概化）： 每个箱子中的 样本数量 接近
# In[24]:
print(var_d_s)

# 统计每个水平的对应目标变量的均值，和每个水平数量
'''
注意：因为TARGET_B特征是二分类，取值区间[0,1]，所以求均值 就是 求DemCluster名族的每个类别中 TARGET_B=1的 频次（概率）  
'''
DemCluster_grp = model_data[['DemCluster', 'TARGET_B']].groupby('DemCluster', as_index=False)
# Index索引 变成了 DemCluster
DemC_C = DemCluster_grp['TARGET_B'].agg({'mean': 'mean', 'count': 'count'}).sort_values("mean")

# In[33]:
# 自己做的分箱，和 目标意思 不一致，就是练习一下
# A、按名族类别的count值的区间 对 名族代号 进行分箱（不能均分样本）
# model_data['DemCluster'].value_counts(ascending=True)
## pd.qcut得到Series： 索引是 value_counts 的类别（DemCluster）； 值是 value_counts 的统计值。
# qcats1 = pd.qcut(model_data['DemCluster'].value_counts(ascending=True), q=10, labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) # Series
# print(qcats1)
# print(qcats1.value_counts()) # 每个桶中分别有几个 名族
# X_New = X.copy()
## X["DemCluster"]的值  对齐  qcats1的索引 （2个都是Series）
# X_New["MyDemCluster"] = X["DemCluster"].map(qcats1)
# print(X_New["MyDemCluster"].value_counts()) # 每个桶中分别有几个 样本


# B、按名族类别的样本数量分桶（能均分样本）
# 错的，索引对不齐
# qcats2 = pd.DataFrame(pd.qcut(model_data['DemCluster'],q=10), index=model_data['DemCluster'])
# qcats2 = pd.qcut(model_data['DemCluster'],q=10).reindex(model_data['DemCluster'])
# print(qcats2['DemCluster'].value_counts())

qcats3 = pd.qcut(model_data['DemCluster'], q=10, labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
print(qcats3.value_counts())
X_New2 = X.copy()
X_New2['myIndex'] = X.index
# X["myIndex"]的值  对齐  qcats3的索引 （2个都是Series）
X_New2["MyDemCluster"] = X_New2['myIndex'].map(qcats3)
X_New2.drop('myIndex', axis=1, inplace=True)
print(X_New2["MyDemCluster"].value_counts())

# In[25]:
# '''
# 分类变量的 概化操作
# 将这些类别尽量以 人数大致均等的方式、 以 响应率为序 归结为10个大类。
DemC_C["count_cumsum"] = DemC_C["count"].cumsum()
# 按 值的累加和 分箱
DemC_C["new_DemCluster"] = DemC_C["count_cumsum"].apply(lambda x: x // (len(model_data) / 10))  # float64
DemC_C["new_DemCluster"] = DemC_C["new_DemCluster"].astype(int)

# In[26]:
# 查看 分箱特征 中 每个类别 的 count样本数 是否接近。
DemC_C.groupby("new_DemCluster")["count"].sum()

# In[27]:
# 将重编码信息保存至数据清洗信息当中.2
DemCluster_new_class = DemC_C[["DemCluster", "new_DemCluster"]].set_index("DemCluster")  # DataFrame
print(type(DemCluster_new_class))
DATA_CLEAN.append(DemCluster_new_class.to_dict())
print(DATA_CLEAN)

# 根据重编码替换原数据
# In[28]:
# X中的名族字段 替换为 分箱后的名族字段
X["DemCluster"] = X["DemCluster"].map(DATA_CLEAN[1]['new_DemCluster'])
# X["DemCluster"] = X["DemCluster"].map(DemCluster_new_class['new_DemCluster'])

# In[29]:
X.head()

# In[30]:
# 查看 分箱特征 中 每个类别 的 count样本数 是否接近。
X.DemCluster.value_counts()

# 对分类变量进行woe转换
# In[31]:
X_rep = X.copy()  # 为了降维做准备（所有分类变量转换为连续变量），X第二次赋值。 X中的DemCluster名族字段已经分箱并赋值了。
# In[31]:
# 自己封装的：
model_data_1 = model_data.copy()
bt.category_feature_generalization(model_data_1, 'DemCluster', 'TARGET_B', bin_num=10)
model_data_1 = model_data_1[['DemCluster', 'StatusCat96NK', 'TARGET_B']]
# In[31]:
bins_df, iv = ft.category_feature_iv(model_data_1, 'DemCluster', 'TARGET_B')

# In[32]:
# 3.3、将 分类概化 之后的 分类变量 进行 Woe转换 为 连续变量
# 目前有一个Scorecardpy的包可以实现自动化分箱，可以用来试用，但是还没有经过全面的测试，不可直接使用。
for i in var_d_s:  # ['StatusCat96NK', 'DemCluster']
    X_rep[i + "_woe"] = WoE(v_type='d').fit_transform(X_rep[i], Y)
# In[31]:
# 自己封装的：
ft.all_feature_woe_mapping(model_data_1, var_d_s, 'TARGET_B', use_woe_library=True)

# In[33]:
# 将woe转换的过程保存.3、4
# 因为 WOE转换后，X_rep数据中 StatusCat96NK 和 DemCluster分类变量 的每个相同的 类别 的WOE结果值是相同的，所以要删除重复项： 为了 给DATA_CLEAN 赋值
StatusCat96NK_woe = X_rep[["StatusCat96NK", "StatusCat96NK_woe"]].drop_duplicates().set_index("StatusCat96NK").to_dict()
DemCluster_woe = X_rep[["DemCluster", "DemCluster_woe"]].drop_duplicates().set_index("DemCluster").to_dict()

DATA_CLEAN.append(StatusCat96NK_woe)
DATA_CLEAN.append(DemCluster_woe)
print(DATA_CLEAN)

# In[34]:
# 删除原始的 2个分类 特征
del X_rep["StatusCat96NK"]
del X_rep["DemCluster"]

# In[35]:
X_rep.rename(columns={"StatusCat96NK_woe": "StatusCat96NK", "DemCluster_woe": "DemCluster"}, inplace=True)

# 3.4、通过 随机森林 对 所有连续变量 的重要性进行筛选（现在所有变量都为 连续变量，用随机森林再做一次变量重要性筛选）
# In[36]:
import sklearn.ensemble as ensemble

'''
#n_estimators：最大的弱学习器的个数
#max_features：RF划分时考虑的最大特征数（如果是浮点数，代表考虑特征百分比，即考虑（百分比 x N）取整后的特征数； 默认是"auto",意味着划分时最多考虑√N个特征）
#min_samples_split：内部节点再划分所需最小样本数
'''
rfc = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=3, max_features=0.5, min_samples_split=5)
rfc_model = rfc.fit(X_rep, Y)
rfc_model.feature_importances_
rfc_fi = pd.DataFrame()
rfc_fi["features"] = list(X.columns)
rfc_fi["importance"] = list(rfc_model.feature_importances_)
rfc_fi = rfc_fi.set_index("features", drop=True).sort_values(by='importance', ascending=False)  # drop=True 删除掉原始索引
rfc_fi.plot(kind="bar");

# In[37]:
# 以 2% 作为选取变量的阈值
var_x = list(rfc_fi.importance[rfc_fi.importance > 0.02].index)  # 第二次变量筛选，还有13个特征
# StatusCat96NK 的重要性 < 0.02，剔除。
print(var_x)
# print(X_rep.columns)
del X_rep["StatusCat96NK"]
print(X_rep.columns)

# 3.5、查看 连续变量 的分布情况（连续变量 分布转换）
# In[38]:
for i in var_x:
    print(i)
    plt.hist(X_rep[i], bins=20)
    plt.show()

# In[39]:
# 计算连续变量 的 偏度
skew_var_x = {}
for i in var_x:
    skew_var_x[i] = abs(X_rep[i].skew())

skew = pd.Series(skew_var_x).sort_values(ascending=False)
print(skew)

# In[40]:
# 求偏度大于1的连续变量
# ['GiftAvg36', 'GiftAvgLast', 'GiftCnt36', 'GiftCntCardAll'] 偏度>1，取对数
var_x_ln = skew.index[skew > 1]
print(var_x_ln)

# In[41]:
# 加入数据清洗.5
DATA_CLEAN.append({"val_x_ln": var_x_ln})
print(DATA_CLEAN)

# In[42]:
# 将偏度大于1的连续变量 取对数
for i in var_x_ln:
    if min(X_rep[i]) <= 0:
        X_rep[i] = np.log(X_rep[i] + abs(min(X_rep[i])) + 0.01)  # 负数取对数的技巧
    else:
        X_rep[i] = np.log(X_rep[i])

# In[43]:
# 再次 计算特征：偏度，可以看到 特征偏度 有了明显下降
skew_var_x = {}
for i in var_x:
    skew_var_x[i] = abs(X_rep[i].skew())

skew = pd.Series(skew_var_x).sort_values(ascending=False)
print(skew)

# 3.6、变量压缩
# In[44]:
# 3.6.1、做主成分之前，进行中心标准化（Z分数）
from sklearn import preprocessing

pcadata = preprocessing.scale(X_rep)  # 当然是不改变 X_rep

# 3.6.2、使用sklearn的主成分分析，用于判断保留主成分的数量
# In[5]:
from sklearn.decomposition import PCA

'''
#此处作主成分分析，主要是进行冗余变量的剔出，因此注意以下两个原则：
# 1、保留的变量个数尽量多，累积的explained_variance_ratio_尽量大，比如阈值设定为0.95
# 2、只剔出单位根非常小的变量，比如阈值设定为0.2
'''
pca = PCA(n_components=13)  # 经过 随机森林 变量筛选后，还剩下13个特征，全部特征 都做PCA 观测重要性。
pca.fit(pcadata)
# explained_variance_： 解释方差
print(pca.explained_variance_)  # 建议保留9个主成分
# explained_variance_ratio_： 解释方差占比（累计解释方差占比 自己手动加）
print(pca.explained_variance_ratio_)  # 建议保留8个主成分

# %%
ratioAccSum = 0.00
ratioIndex = 0
ratioValue = 0.00
for i in pca.explained_variance_ratio_:
    ratioAccSum += i
    ratioIndex += 1
    if ratioAccSum >= 0.95:  # 保留8个主成分
        ratioValue = i
        break
    # In[]:
# 自己封装的：
dr.pca_test(pcadata)

# In[45]:
# 3.6.3、SparsePCA稀疏主成分分析 + 变量压缩 选择 原始特征
from VarSelec import Var_Select

'''
#非常非常耗资源
#k-预期最大需要保留的最大变量个数，实际保留数量不能多于这个数值。 k的个数 就是 主成分个数。
'''
# X_rep 的数据 已经做过 1、偏度大于1取对数； 2、分类变量 WOE转换
# Var_Select(orgdata, k,alphaMin=10, alphaMax=20, alphastep=0.2)
X_rep_reduc = Var_Select(X_rep, k=8, alphaMin=0.1, alphaMax=200, alphastep=0.5)
print(X_rep_reduc.head())

# 如果报best_alpha没有定义的错误，请扩大alphaMax的取值
# In[46]:
X_rep_reduc_corr = X_rep_reduc.corr()

# In[47]:
# 最后选择的变量为
list(X_rep_reduc.columns)  # SparsePCA 和 变量压缩 选择出的 8个 原始变量（特征）

# In[48]:
# 添加清洗.6
DATA_CLEAN.append({"final_var": list(X_rep_reduc.columns)})
print(DATA_CLEAN)

# In[49]:
assert len(DATA_CLEAN) == 6, "确保没有重复添加清洗需要的数据"

# In[50]:
X_rep_reduc.head()

# 4、建立逻辑回归模型M（建模）阶段
# 分成训练集和测试集，比例为6:4
# In[51]:
import sklearn.model_selection as model_selection

ml_data = model_selection.train_test_split(X_rep_reduc, Y, test_size=0.3, random_state=0)
train_data, test_data, train_target, test_target = ml_data

# 模型训练
# 使用全部变量进行logistic回归
# In[52]:
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
train_data = min_max_scaler.fit_transform(train_data)
test_data = min_max_scaler.fit_transform(test_data)

# In[53]:
import sklearn.linear_model as linear_model

logistic_model = linear_model.LogisticRegression(class_weight=None,
                                                 dual=False,
                                                 fit_intercept=True,
                                                 intercept_scaling=1,
                                                 penalty='l1',
                                                 random_state=None,
                                                 tol=0.001)

# In[54]:
from sklearn.model_selection import ParameterGrid, GridSearchCV

C = np.logspace(-3, 0, 20, base=10)

param_grid = {'C': C}

clf_cv = GridSearchCV(estimator=logistic_model,
                      param_grid=param_grid,
                      cv=5,
                      scoring='roc_auc')

clf_cv.fit(train_data, train_target)

# In[55]:
logistic_model = linear_model.LogisticRegression(C=clf_cv.best_params_["C"],
                                                 class_weight=None,
                                                 dual=False,
                                                 fit_intercept=True,
                                                 intercept_scaling=1,
                                                 penalty='l1',
                                                 random_state=None,
                                                 tol=0.001)
logistic_model.fit(train_data, train_target)

# In[59]:
print(logistic_model.coef_)  # 表明第一个变量被剔除
X_rep_reduc = X_rep_reduc.drop(["PromCntCardAll"], 1)  # 不影响 train_test_split 的切分结果。

# In[60]:
import statsmodels.api as sm
import statsmodels.formula.api as smf

model = X_rep_reduc.join(train_target)

formula = "TARGET_B ~ " + "+".join(X_rep_reduc)
lg_m = smf.glm(formula=formula, data=model,
               family=sm.families.Binomial(sm.families.links.logit)).fit()
lg_m.summary()

# 模型验证A（验证）阶段
# 对逻辑回归模型进行评估
# In[58]:
train_est = logistic_model.predict(train_data)
test_est = logistic_model.predict(test_data)
print(test_est)

# In[59]:
train_est_p = logistic_model.predict_proba(train_data)[:, 1]
test_est_p = logistic_model.predict_proba(test_data)[:, 1]
print(test_est_p)

# - 目标样本和非目标样本的分数分布
# In[64]:
import seaborn as sns
from sklearn import metrics

red, blue = sns.color_palette("Set1", 2)

# In[65]:
sns.kdeplot(test_est_p[test_target == 1], shade=True, color=red)
sns.kdeplot(test_est_p[test_target == 0], shade=True, color=blue)

# ROC曲线
# In[66]:
plt.figure(figsize=[6, 6])
# fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_est_p)
fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est_p)
# plt.plot(fpr_train, tpr_train, color=red)
plt.plot(fpr_test, tpr_test, color=blue)
plt.title('ROC curve')
print('AUC = %6.4f' % metrics.auc(fpr_test, tpr_test))

# In[67]:
# 它这个 KS 是没看懂呀，看我自己实现的吧！
plt.figure(figsize=[6, 6])
# train_x_axis = np.arange(len(fpr_train))/float(len(fpr_train))
test_x_axis = np.arange(len(fpr_test)) / float(len(fpr_test))
# plt.plot(fpr_train, train_x_axis, color=red)
# plt.plot(tpr_train, train_x_axis, color=red)
plt.plot(fpr_test, test_x_axis, color=blue)
plt.plot(tpr_test, test_x_axis, color=blue)
plt.title('KS curve')
# '''


# 构建神经网络并评估
# In[68]:
'''

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(train_data)

scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)


# In[69]:


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(10,), 
                    activation='logistic', alpha=0.1, max_iter=1000)


# In[70]:


from sklearn.model_selection import GridSearchCV
from sklearn import metrics

param_grid = {
    'hidden_layer_sizes':[(10, ), (15, ), (20, ), (5, 5)],
    'activation':['logistic', 'tanh', 'relu'], 
    'alpha':[0.001, 0.01, 0.1, 0.2, 0.4, 1, 10]
}
mlp = MLPClassifier(max_iter=1000)
gcv = GridSearchCV(estimator=mlp, param_grid=param_grid, 
                   scoring='roc_auc', cv=4, n_jobs=-1)
gcv.fit(scaled_train_data, train_target)


# In[71]:


gcv.best_params_


# In[72]:


mlp = MLPClassifier(hidden_layer_sizes=gcv.best_params_["hidden_layer_sizes"], 
                    activation=gcv.best_params_["activation"], alpha=gcv.best_params_["alpha"], max_iter=1000)

mlp.fit(scaled_train_data, train_target)


# In[73]:


train_predict = mlp.predict(scaled_train_data)
test_predict = mlp.predict(scaled_test_data)


# In[74]:


train_proba = mlp.predict_proba(scaled_train_data)[:, 1]  
test_proba = mlp.predict_proba(scaled_test_data)[:, 1]


# In[75]:


from sklearn import metrics

print(metrics.confusion_matrix(test_target, test_predict, labels=[0, 1]))
print(metrics.classification_report(test_target, test_predict))


# In[76]:


fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_proba)
fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_proba)

plt.figure(figsize=[4, 4])
plt.plot(fpr_test, tpr_test, 'b-')
plt.plot(fpr_train, tpr_train, 'r-')
plt.title('ROC curve')
plt.show()

print('AUC = %6.4f' %metrics.auc(fpr_test, tpr_test))

#发现神经网络没有提升效果，因此仍然保留逻辑回归模型
# ## 模型永久化

# In[77]:


import pickle as pickle


# In[78]:


# 使用with语句确保文件关闭
with open(r'logitic.model', 'wb') as f:
    pickle.dump(logistic_model, f)


# In[79]:


with open(r'logitic.model', 'rb') as f:
    model_load = pickle.load(f)


# In[80]:


test_est_load = model_load.predict(test_data)


# In[81]:


pd.crosstab(test_est_load,test_est)


# 把清洗过程中使用的数据也保存到文件当中

# In[82]:


with open(r'logitic.dataclean', 'wb') as f:
    pickle.dump(DATA_CLEAN, f)
#%%
'''