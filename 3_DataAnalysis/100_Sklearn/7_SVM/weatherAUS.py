# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:59:57 2019

@author: dell
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer  # 0.20, conda, pip
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import re
import os

os.chdir(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\7_SVM")

# In[]:
weather = pd.read_csv(r"weatherAUS5000.csv", index_col=0)  # 第一列变为index

X = weather.iloc[:, :-1]
Y = weather.iloc[:, -1]

# In[]:
import FeatureTools as ft

print(X.isnull().mean())  # 缺失值所占总值的比例 isnull().sum(全部的True)/X.shape[0]
print(ft.missing_values_table(X))

print(set(Y), Y.isnull().sum())

# In[]:
# Ori_Xtrain, Ori_Xtest, Ori_Ytrain, Ori_Ytest = train_test_split(X,Y,test_size=0.3,random_state=420) #随机抽样
Ori_Xtrain, Ori_Xtest, Ori_Ytrain, Ori_Ytest = ft.data_segmentation_skf(X, Y, test_size=0.3)

Xtrain = Ori_Xtrain.copy()
Xtest = Ori_Xtest.copy()
Ytrain = Ori_Ytrain.copy()
Ytest = Ori_Ytest.copy()

# 恢复索引
for i in [Xtrain, Xtest, Ytrain, Ytest]:
    i.index = range(i.shape[0])

# In[]:
# 类别比重
ft.sample_category(Ytest, Ytrain)

# In[]:
# 一、特征工程：
# 将标签编码
from sklearn.preprocessing import LabelEncoder

encorder = LabelEncoder().fit(Ytrain)  # 允许一维数据的输入

# 使用训练集进行训练，然后在训练集和测试集上分别进行transform
Ytrain = pd.DataFrame(encorder.transform(Ytrain))
Ytest = pd.DataFrame(encorder.transform(Ytest))
# 如果我们的测试集中，出现了训练集中没有出现过的标签类别
# 比如说，测试集中有YES, NO, UNKNOWN
# 而我们的训练集中只有YES和NO， 那么只能 重新建模。

# 备份数据，好习惯
# Ytrain.to_csv(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\7_SVM\数据备份\Ytrain.csv")
# Ytest.to_csv(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\7_SVM\数据备份\Ytest.csv")


# In[]:
# 描述性统计
Xtrain.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T

data_temp = pd.concat([Xtrain, Ytrain], axis=1)  # 新对象

f, axes = plt.subplots(2, 2, figsize=(16, 13))
ft.class_data_distribution(data_temp, "MinTemp", 0, axes)

# In[]:
linearv = ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustDir",
           "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
           "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"]

for i in linearv[0:3]:
    f, axes = plt.subplots(2, 2, figsize=(16, 13))
    ft.class_data_distribution(data_temp, i, 0, axes)

# In[]:
# 特征类别
# 日期
temp_group = Xtrain.groupby(["Date"])["Location"].count()
temp_group_ = temp_group[temp_group > 1]

Xtrainc = Xtrain.copy()
Xtrainc.sort_values(by="Location")

Xtrainc[Xtrainc["Date"] == '2008-10-30']

# In[]:
print(Xtrain["Rainfall"].isnull().sum())
tRainfall_mean = np.mean(Xtrain["Rainfall"])
Xtrain["Rainfall"].fillna(tRainfall_mean, inplace=True)

Xtrain.loc[Xtrain["Rainfall"] >= 1, "RainToday"] = 1
Xtrain.loc[Xtrain["Rainfall"] < 1, "RainToday"] = 0
# Xtrain[Xtrain["Rainfall"].isnull()]["RainToday"] = np.nan
print(Xtrain["RainToday"].isnull().sum())

# In[]:
print(Xtest["Rainfall"].isnull().sum())
Xtest["Rainfall"].fillna(tRainfall_mean, inplace=True)
Xtest.loc[Xtest["Rainfall"] >= 1, "RainToday"] = 1
Xtest.loc[Xtest["Rainfall"] < 1, "RainToday"] = 0
# Xtest.loc[Xtest["Rainfall"].isnull()]["RainToday"] = np.nan
print(Xtest["RainToday"].isnull().sum())

# In[]:
Xtrain["Month"] = Xtrain["Date"].apply(lambda x: int(x.split("-")[1]))
Xtest["Month"] = Xtest["Date"].apply(lambda x: int(x.split("-")[1]))

# In[]:
# 训练集中 气象站 名称
Xtrain.loc[:, "Location"].value_counts().count()

# In[]:
# 爬虫 得到的 澳大利亚城市 和 澳大利亚城市对应气候
# 澳大利亚城市 经纬度
cityll = pd.read_csv(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\7_SVM\cityll.csv",
                     index_col=0)
# 澳大利亚城市对应气候
city_climate = pd.read_csv(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\7_SVM\Cityclimate.csv")

# In[]:
cityll.head()
# In[]:
city_climate.head()
print(set(city_climate['Climate']), len(set(city_climate['Climate'])))

# In[]:
float(cityll.loc[0, 'Latitude'][:-1])

# 纬度
cityll["Latitudenum"] = cityll['Latitude'].apply(lambda x: float(x[:-1]))
# 经度
cityll["Longitudenum"] = cityll['Longitude'].apply(lambda x: float(x[:-1]))

# 观察一下所有的经纬度方向都是一致的，全部是南纬，东经，因为澳大利亚在南半球，东半球
print(cityll['Latitudedir'].value_counts())
print(cityll['Longitudedir'].value_counts())

citylld = cityll[['City', 'Latitudenum', 'Longitudenum']]

# In[]:
# 为 citylld 添加 气候
city_climate_index = city_climate.set_index('City')
citylld['Climate'] = citylld['City'].map(city_climate_index['Climate'])

# 地图中有8种气候，这里只有7种，是因为 第8种是极地气候，这些城市都不在其范围内
citylld['Climate'].value_counts()

# In[]:
# 训练集中 气象站 经纬度
samplecity = pd.read_csv(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\7_SVM\samplecity.csv",
                         index_col=0)
samplecity.head()

# 纬度
samplecity["Latitudenum"] = samplecity['Latitude'].apply(lambda x: float(x[:-1]))
# 经度
samplecity["Longitudenum"] = samplecity['Longitude'].apply(lambda x: float(x[:-1]))

samplecityd = samplecity[['City', 'Latitudenum', 'Longitudenum']]

samplecityd.head()

# In[]:
# 气象站 和 城市 的距离
# 首先使用radians将角度转换成弧度
from math import radians, sin, cos, acos

# radians：角度（纬度、经度） 转 弧度
citylld.loc[:, "slat"] = citylld.iloc[:, 1].apply(lambda x: radians(x))
citylld.loc[:, "slon"] = citylld.iloc[:, 2].apply(lambda x: radians(x))
samplecityd.loc[:, "elat"] = samplecityd.iloc[:, 1].apply(lambda x: radians(x))
samplecityd.loc[:, "elon"] = samplecityd.iloc[:, 2].apply(lambda x: radians(x))

# In[]:
for i in range(samplecityd.shape[0]):
    slat = citylld.loc[:, "slat"]  # 所有城市的 纬度
    slon = citylld.loc[:, "slon"]  # 所有城市的 经度
    elat = samplecityd.loc[i, "elat"]  # 第i个气象站 纬度
    elon = samplecityd.loc[i, "elon"]  # 第i个气象站 经度
    # 第i个气象站 到 所有城市 的距离
    dist = 6371.01 * np.arccos(np.sin(slat) * np.sin(elat) +
                               np.cos(slat) * np.cos(elat) * np.cos(slon.values - elon))
    # 与 第i个气象站 距离最近的 城市的索引
    city_index = np.argsort(dist)[0]

    # 每次计算后，取距离最近的城市，然后将最近的 城市 和 城市对应的气候 都匹配到samplecityd中
    samplecityd.loc[i, "closest_city"] = citylld.loc[city_index, "City"]
    samplecityd.loc[i, "Climate"] = citylld.loc[city_index, "Climate"]

# In[]:
# 查看 气象站 气候的分布 （samplecityd的City字段 是 气象站名称）
samplecityd["Climate"].value_counts()

# 确认无误后，取出样本城市所对应的气候，并保存
locafinal = samplecityd.loc[:, ['City', 'Climate']]
locafinal.columns = ['Location', 'Climate']

# 为.map做准备
locafinal = locafinal.set_index('Location')
# locafinal.to_csv(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\7_SVM\samplelocation.csv")

# In[]:
# 1、把location替换成气候的是我们的map的映射
Xtrain['Climate'] = Xtrain['Location'].map(locafinal['Climate'])
# In[]:
# 将location中的内容替换，并且确保匹配进入的气候字符串中不含有逗号，气候两边不含有空格
# 我们使用re这个模块来消除逗号
# re.sub(希望替换的值，希望被替换成的值，要操作的字符串) #去掉逗号
# x.strip()是去掉空格的函数
Xtrain["Climate"] = Xtrain["Climate"].apply(lambda x: re.sub(",", "", x.strip()))

# 合并运行
Xtest["Climate"] = Xtest["Location"].map(locafinal['Climate']).apply(lambda x: re.sub(",", "", x.strip()))

# In[]:
# 查看分布：
# 地图中有8种气候，这里只有7种，是因为 第8种是极地气候，这些城市都不在其范围内
print(set(locafinal['Climate']), len(set(locafinal['Climate'])))

print(set(Xtrain["Location"]), len(set(Xtrain["Location"])))
print(set(Xtest["Location"]), len(set(Xtest["Location"])))
temp_Location_list = ft.set_diff(set(Xtrain['Location']), set(Xtest['Location']))

print(set(Xtrain["Climate"]), len(set(Xtrain["Climate"])))
print(set(Xtest["Climate"]), len(set(Xtest["Climate"])))
temp_Climate_list = ft.set_diff(set(Xtrain['Climate']), set(Xtest['Climate']))

# In[]:
Xtrain.groupby(['Location', 'Climate'])['Date'].count()
Xtest.groupby(['Location', 'Climate'])['Date'].count()

# In[]:
# 查看缺失值的缺失情况
print(Xtrain.isnull().mean())
print(Xtest.isnull().mean())
# In[]:
print(Xtrain.info())
print(Xtrain.info())

# In[]:
Xtrain_ = Xtrain.copy()
Xtest_ = Xtest.copy()
Ytrain_ = Ytrain.copy()
Ytest_ = Ytest.copy()

# In[]:
# 分类（离散）特征
Xtrain_.drop(['Date', 'Location'], inplace=True, axis=1)
Xtest_.drop(['Date', 'Location'], inplace=True, axis=1)

cate = Xtrain_.columns[Xtrain_.dtypes == "object"].tolist()

# 除了特征类型为"object"的特征们，还有虽然用数字表示，但是本质为分类型特征的云层遮蔽程度
cloud = ["Cloud9am", "Cloud3pm", 'RainToday', 'Month']
cate = cate + cloud
# WindGustDir、WindDir9am、WindDir3pm、Climate、Cloud9am、Cloud3pm

print(Xtrain_[cate].isnull().mean())
print(Xtest_[cate].isnull().mean())

# In[]:
# 对于分类型特征，我们使用众数来进行填补
si = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
# 注意，我们使用训练集数据来训练我们的填补器，本质是在生成训练集中的众数
si.fit(Xtrain_.loc[:, cate])
# In[]:
# 然后我们用训练集中的众数来同时填补训练集和测试集
Xtrain_.loc[:, cate] = si.transform(Xtrain_.loc[:, cate])
Xtest_.loc[:, cate] = si.transform(Xtest_.loc[:, cate])
# In[]
print(Xtrain_[cate].isnull().mean())
print(Xtest_[cate].isnull().mean())

# In[]:
# 将所有的分类型变量编码为数字，一个类别是一个数字
from sklearn.preprocessing import OrdinalEncoder  # 只允许二维以上的数据进行输入

oe = OrdinalEncoder()

# In[]:
# 先行测试： （LabelEncoder 是单特征转换）
# encorder_t = LabelEncoder().fit(Xtrain_['Location'])
# Xtrain_['Location'] = encorder_t.transform(Xtrain_['Location'])
# Xtest_['Location'] = encorder_t.transform(Xtest_['Location'])
#
# print(len(Xtest[Xtest['Location'] == 'Adelaide']))
# print(len(Xtrain[Xtrain['Location'] == 'Adelaide']))
'''
Xtest中有Xtrain没有的气象站，所以转换失败。
'''

# In[]:
# 利用训练集进行fit
'''
oe = oe.fit(Xtrain_[cate])

# 用训练集的编码结果来编码训练和测试特征矩阵
# 在这里如果测试特征矩阵报错，就说明测试集中出现了训练集中从未见过的类别
Xtrain_.loc[:,cate] = oe.transform(Xtrain_.loc[:,cate])
Xtest_.loc[:,cate] = oe.transform(Xtest_.loc[:,cate])
'''
Xtrain_.loc[:, cate] = oe.fit_transform(Xtrain_.loc[:, cate])
Xtest_.loc[:, cate] = oe.fit_transform(Xtest_.loc[:, cate])

# In[]:
# 连续特征 缺失值处理：
col = Xtrain_.columns.tolist()
for i in cate:
    col.remove(i)

# temp_null = Xtrain_[col].isnull().sum()
# col = temp_null[temp_null>0].index.tolist()

# In[]:
# 实例化模型，填补策略为"mean"表示均值
impmean = SimpleImputer(missing_values=np.nan, strategy="mean")
# 用训练集来fit模型
impmean = impmean.fit(Xtrain_.loc[:, col])
# 分别在训练集和测试集上进行均值填补
Xtrain_.loc[:, col] = impmean.transform(Xtrain_.loc[:, col])
Xtest_.loc[:, col] = impmean.transform(Xtest_.loc[:, col])

print(Xtrain_[col].isnull().sum())
print(Xtest_[col].isnull().sum())

# In[]:
# 标准化
from sklearn.preprocessing import StandardScaler  # 数据转换为均值为0，方差为1的数据

# 标准化不改变数据的分布，不会把数据变成正态分布的
ss = StandardScaler()
ss = ss.fit(Xtrain_.loc[:, col])
Xtrain_.loc[:, col] = ss.transform(Xtrain_.loc[:, col])
Xtest_.loc[:, col] = ss.transform(Xtest_.loc[:, col])

# In[]:
# 二、建模
from time import time  # 随时监控我们的模型的运行时间
import datetime
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import RocLib as rlb

# In[]:
Ytrain_ = Ytrain_.iloc[:, 0].ravel()
Ytest_ = Ytest_.iloc[:, 0].ravel()

# In[]:
# 建模选择自然是我们的支持向量机SVC，首先用核函数的学习曲线来选择核函数
# 我们希望同时观察，精确性，recall以及AUC分数
times = time()  # 因为SVM是计算量很大的模型，所以我们需要时刻监控我们的模型运行时间

# kernel_list = ["linear","poly","rbf","sigmoid"]
kernel_list = ["linear"]
for kernel in kernel_list:
    clf = SVC(kernel=kernel
              , gamma="auto"
              , degree=1
              #              ,cache_size = 5000
              ).fit(Xtrain_, Ytrain_)
    result = clf.predict(Xtest_)
    score = clf.score(Xtest_, Ytest_)
    recall = recall_score(Ytest_, result)
    clf_decision_scores = clf.decision_function(Xtest_)
    auc = roc_auc_score(Ytest_, clf_decision_scores)
    print("%s 's testing accuracy %f, recall is %f', auc is %f" % (kernel, score, recall, auc))
    #    print(datetime.datetime.fromtimestamp(time()-times).strftime("%M:%S:%f"))
    fig, axe = plt.subplots(2, 2, figsize=(30, 20))
    rlb.ComprehensiveIndicatorFigure(Ytest_, clf_decision_scores, axe[0], 1)
    rlb.ComprehensiveIndicatorSkLibFigure(Ytest_, clf_decision_scores, axe[1])

# In[]:
'''
其实首先调的是 默认阈值=0 时模型的指标，模型确定（暂时）之后 再通过 decision_function 阈值进行指标调整。
'''
# 1、追求 少数类别Y=1 尽量高的recall召回率：（善用 class_weight：调整样本 类别权重）
# 1.1、使用“balanced”模式，这个模式使用y的值自动调整与输入数据中的类频率成反比的权重为n_samples/(n_classes * np.bincount(y))
'''
注意： class_weight = balanced 其实找的是模型的平衡点（AUC面积最大化 相当于 KS最大值化）： 默认阈值 与 KS阈值 近似完全重合，达到模型默认阈值的平衡点。
'''
# kernel_list = ["linear","poly","rbf","sigmoid"]
kernel_list = ["linear"]
times = time()
for kernel in kernel_list:
    clf = SVC(kernel=kernel
              , gamma="auto"
              , degree=1
              , cache_size=5000
              , class_weight="balanced"  # 使用balanced
              ).fit(Xtrain_, Ytrain_)
    result = clf.predict(Xtest_)
    score = clf.score(Xtest_, Ytest_)  # 0.803239
    precision = precision_score(Ytest_, result)  # 0.551793
    recall = recall_score(Ytest_, result)  # 0.728947
    clf_decision_scores = clf.decision_function(Xtest_)
    auc = roc_auc_score(Ytest_, clf_decision_scores)  # 0.857590
    print("testing accuracy is %f, precision is %f, recall is %f', auc is %f" % (score, precision, recall, auc))
    #    print(datetime.datetime.fromtimestamp(time()-times).strftime("%M:%S:%f"))
    fig, axe = plt.subplots(2, 2, figsize=(30, 20))
    rlb.ComprehensiveIndicatorFigure(Ytest_, clf_decision_scores, axe[0], 1)
    rlb.ComprehensiveIndicatorSkLibFigure(Ytest_, clf_decision_scores, axe[1])
# In[]:
# 1.2、自定义设置 样本 类别权重
'''
1、通过超参数class_weight和C结合起效：超参数C模型自动设置
class_weight设置为：{"标签的值1"：权重1，"标签的值2"：权重2}的字典，则超参数C将会自动被设为：
标签的值1的C：权重1 * C，标签的值2的C：权重2 * C
2、class_weight = {1:15} 表示 类别1：15，隐藏了类别0：1这个比例 （不显示设置的类别，权重默认为1）
3、只通过调整模型 样本权重超参数 就能提高recall召回率，可以不用选择 decision_function阈值 进行自定义预测。 
'''
times = time()
clf = SVC(kernel="linear"
          , gamma="auto"
          , cache_size=5000
          , class_weight={1: 15}  # 注意，这里写的其实是，类别1：15，隐藏了类别0：1这个比例
          ).fit(Xtrain_, Ytrain_)
result = clf.predict(Xtest_)
score = clf.score(Xtest_, Ytest_)
recall = recall_score(Ytest_, result)
clf_decision_scores_test = clf.decision_function(Xtest_)
auc = roc_auc_score(Ytest_, clf_decision_scores_test)
print("testing accuracy %f, recall is %f', auc is %f" % (score, recall, auc))
# print(datetime.datetime.fromtimestamp(time()-times).strftime("%M:%S:%f"))

fig, axe = plt.subplots(2, 2, figsize=(30, 20))
rlb.ComprehensiveIndicatorFigure(Ytest_, clf_decision_scores_test, axe[0], 1)
rlb.ComprehensiveIndicatorSkLibFigure(Ytest_, clf_decision_scores_test, axe[1])

# Train 与 Test ROC曲线比较
fig, axe = plt.subplots(1, 1, figsize=(13, 10))
clf_decision_scores_train = clf.decision_function(Xtrain_)
rlb.comparedRoc(Ytrain_, clf_decision_scores_train, Ytest_, clf_decision_scores_test, axe)

# In[]:
# 2、追求  少数类别Y=1  尽量高的precision精准度：
# 2.1、以 模型默认 超参数检测：
clf = SVC(kernel="linear"
          , gamma="auto"
          , cache_size=5000
          ).fit(Xtrain_, Ytrain_)

result = clf.predict(Xtest_)
clf_decision_scores = clf.decision_function(Xtest_)

fig, axe = plt.subplots(2, 2, figsize=(30, 20))
rlb.ComprehensiveIndicatorFigure(Ytest_, clf_decision_scores, axe[0], 1)
rlb.ComprehensiveIndicatorSkLibFigure(Ytest_, clf_decision_scores, axe[1])

# In[]:
# 查看模型 特异度： 1-FPR
# 少数类别Y=1 的特异度  等于  多数类别Y=0 的召回率
print(rlb.SPE(Ytest_, result))  # 1-FPR： print(rlb.TPR(Ytest_, result, 0))
print(rlb.FPR(Ytest_, result))  # 1-SPE

print(rlb.confusion_matrix_customize(Ytest_, result))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(Ytest_, result, labels=[0, 1]))

# In[]:
# 2.2、以 小范围 样本权重超参数 检测：
# 第一次测试 class_weight 范围：（class_weight在很小值范围内）
# irange = np.linspace(0.01,0.05,10)
# precision 峰值 0.745614 ： under ratio 1:1.014444 testing accuracy is 0.839232, precision is 0.745614, recall is 0.447368', auc is 0.855629

# 第二次测试 class_weight 范围： 根据第一步的precision峰值对应的class_weight值（上一个值与下一个值为边界）， 进一步缩小class_weight值范围
irange = np.linspace(0.010000, 0.018889, 10)
# under ratio 1:1.013951 testing accuracy is 0.839232, precision is 0.745614, recall is 0.447368', auc is 0.855670
'''
没有什么质的改变，在很小的样本权重范围内 模型的学习曲线 少数类别Y=1 的precision精准度 与 原始模型相当，
如果继续增大 少数类别Y=1 的样本权重，则会变成 追求最高recall召回率，精准度进一步降低(FP增长速度 远大于 TP增长速度)
也就是说 原始不调整样本权重的SVM模型 已经是 precision精准度相对高的。
'''
# In[]:
score_list = []
precision_list = []
recall_list = []
auc_list = []

for i in irange:
    times = time()
    clf = SVC(kernel="linear"
              , gamma="auto"
              , cache_size=5000
              , class_weight={1: 1 + i}  # 注意，这里写的其实是，类别1：1+i，隐藏了类别0：1这个比例
              ).fit(Xtrain_, Ytrain_)
    result = clf.predict(Xtest_)
    score = clf.score(Xtest_, Ytest_)
    precision = precision_score(Ytest_, result)
    recall = recall_score(Ytest_, result)
    auc = roc_auc_score(Ytest_, clf.decision_function(Xtest_))
    score_list.append(score)
    precision_list.append(precision)
    recall_list.append(recall)
    auc_list.append(auc)
    print("under ratio 1:%f testing accuracy is %f, precision is %f, recall is %f', auc is %f" % (
    1 + i, score, precision, recall, auc))
#    print(datetime.datetime.fromtimestamp(time()-times).strftime("%M:%S:%f"))

fig, axe = plt.subplots(1, 1, figsize=(10, 10))
# axe.plot(irange, score_list)
axe.plot(irange, precision_list)
axe.plot(irange, recall_list)
# axe.plot(irange, auc_list)
# In[]:
# 2.2.1、细化权重阈值（只通过调整模型 样本权重超参数 已无法再提高precision精准度）
times = time()
clf = SVC(kernel="linear"
          , gamma="auto"
          , cache_size=5000
          , class_weight={1: 1.013951}
          ).fit(Xtrain_, Ytrain_)
result = clf.predict(Xtest_)
score = clf.score(Xtest_, Ytest_)
precision = precision_score(Ytest_, result)
recall = recall_score(Ytest_, result)
clf_decision_scores = clf.decision_function(Xtest_)
auc = roc_auc_score(Ytest_, clf_decision_scores)
print("testing accuracy is %f, precision is %f, recall is %f', auc is %f" % (score, precision, recall, auc))
# print(datetime.datetime.fromtimestamp(time()-times).strftime("%M:%S:%f"))

fig, axe = plt.subplots(2, 2, figsize=(30, 20))
rlb.ComprehensiveIndicatorFigure(Ytest_, clf_decision_scores, axe[0], 1)
rlb.ComprehensiveIndicatorSkLibFigure(Ytest_, clf_decision_scores, axe[1])
# In[]:
# 2.2.2、选择 decision_function阈值 进行自定义预测
'''
因无法 只通过调整模型 样本权重超参数 就能提高precision精准度，根据这个模型，选择 decision_function阈值 进行自定义预测：
以 precision精准度 > 90% 为目标选择 decision_function阈值：（根据2.2.1模型指标图，precision对应的阈值上下限为：0到2）
'''
threshold = rlb.thresholdSelection(Ytest_, clf_decision_scores, 0, 2, 0.9, 1, 1)
my_predict = np.array(clf_decision_scores >= threshold, dtype='int')

score = rlb.precision_scoreAll_customize(Ytest_, my_predict)
precision = rlb.precision_score_customize(Ytest_, my_predict)
recall = rlb.recall_score_customize(Ytest_, my_predict)
f1 = rlb.f1_score_customize(Ytest_, my_predict)
print("testing accuracy is %f, precision is %f, recall is %f', f1 is %f" % (score, precision, recall, f1))

# In[]:
# 换模型测试：
from sklearn.linear_model import LogisticRegression as LR

logclf = LR(solver="liblinear").fit(Xtrain_, Ytrain_)
result = logclf.predict(Xtest_)
score = logclf.score(Xtest_, Ytest_)

precision = precision_score(Ytest_, result)
recall = recall_score(Ytest_, result)
clf_decision_scores = logclf.decision_function(Xtest_)
print("testing accuracy is %f, precision is %f, recall is %f" % (score, precision, recall))
fig, axe = plt.subplots(2, 2, figsize=(30, 20))
rlb.ComprehensiveIndicatorFigure(Ytest_, clf_decision_scores, axe[0], 1)
rlb.ComprehensiveIndicatorSkLibFigure(Ytest_, clf_decision_scores, axe[1])

# In[]
C_range = np.linspace(5, 10, 10)

for C in C_range:
    logclf = LR(solver="liblinear", C=C).fit(Xtrain_, Ytrain_)
    result = logclf.predict(Xtest_)
    score = logclf.score(Xtest_, Ytest_)
    precision = precision_score(Ytest_, result)
    recall = recall_score(Ytest_, result)
    print("C is %f testing accuracy is %f, precision is %f, recall is %f" % (C, score, precision, recall))

# 加正则项C后，精准度微降低，召回率微升高（和 SVM 情况类似），也不是很理想，上 集成模型 吧。


# In[]:
# 3、追求 平衡： （AUC面积最大化 相当于 KS值越大越好）

'''
# 要运行10分钟！！！
C_range = np.linspace(0.01,20,20)

recallall = []
aucall = []
scoreall = []
for C in C_range:
    times = time()
    clf = SVC(kernel = "linear",C=C,cache_size = 5000
    ,class_weight = "balanced"
    ).fit(Xtrain, Ytrain)
    result = clf.predict(Xtest)
    score = clf.score(Xtest,Ytest)
    recall = recall_score(Ytest, result)
    auc = roc_auc_score(Ytest,clf.decision_function(Xtest))
    recallall.append(recall)
    aucall.append(auc)
    scoreall.append(score)
    print("under C %f, testing accuracy is %f,recall is %f', auc is %f" %
    (C,score,recall,auc))
    print(datetime.datetime.fromtimestamp(time()-times).strftime("%M:%S:%f"))

print(max(aucall),C_range[aucall.index(max(aucall))])
plt.figure()
plt.plot(C_range,recallall,c="red",label="recall")
plt.plot(C_range,aucall,c="black",label="auc")
plt.plot(C_range,scoreall,c="orange",label="accuracy")
plt.legend()
plt.show()
'''
'''
首先，我们注意到，随着C值逐渐增大，模型的运行速度变得越来越慢。对于SVM这个本来运行就不快的模型来说，
巨大的C值会是一个比较危险的消耗。所以正常来说，我们应该设定一个较小的C值范围来进行调整。
其次，C很小的时候，模型的各项指标都很低，但当C到1以上之后，模型的表现开始逐渐稳定，在C逐渐变大之后，
模型的效果并没有显著地提高。可以认为我们设定的C值范围太大了，然而再继续增大或者缩小C值的范围，
AUC面积也只能够在0.86上下进行变化了，调节C值不能够让模型的任何指标实现质变。
'''
# In[]:
'''
注意： class_weight = balanced 其实找的是模型的平衡点（AUC面积最大化 相当于 KS最大值化）： 默认阈值 与 KS阈值 近似完全重合，达到模型默认阈值的平衡点。
本模型设置了C超参数，使近似重合的 默认阈值 与 KS阈值 之间有一段距离，那么平衡点应以 KS阈值 为准。
'''
times = time()
clf = SVC(kernel="linear", C=3.1663157894736838, cache_size=5000
          , class_weight="balanced"
          ).fit(Xtrain_, Ytrain_)

result = clf.predict(Xtest_)
score = clf.score(Xtest_, Ytest_)  # 0.801440
precision = precision_score(Ytest_, result)  # 0.548323
recall = recall_score(Ytest_, result)  # 0.731579
clf_decision_scores = clf.decision_function(Xtest_)
auc = roc_auc_score(Ytest_, clf_decision_scores)  # 0.857490
print("testing accuracy is %f, precision is %f, recall is %f', auc is %f" % (score, precision, recall, auc))
# print(datetime.datetime.fromtimestamp(time()-times).strftime("%M:%S:%f"))

fig, axe = plt.subplots(2, 2, figsize=(30, 20))
rlb.ComprehensiveIndicatorFigure(Ytest_, clf_decision_scores, axe[0], 1)
rlb.ComprehensiveIndicatorSkLibFigure(Ytest_, clf_decision_scores, axe[1])
# In[]:
# 以 KS最大值 对应的 decision_function阈值 进行自定义预测
my_predict = np.array(clf_decision_scores >= -0.4176, dtype='int')
score = rlb.precision_scoreAll_customize(Ytest_, my_predict)  # 0.752849
precision = rlb.precision_score_customize(Ytest_, my_predict)  # 0.475758
recall = rlb.recall_score_customize(Ytest_, my_predict)  # 0.826316
f1 = rlb.f1_score_customize(Ytest_, my_predict)  # 0.603846
print("testing accuracy is %f, precision is %f, recall is %f', f1 is %f" % (score, precision, recall, f1))
'''
加入 超参数C  和  class_weight = balanced 权重超参数 的本模型， 默认阈值0 时模型显得更 平衡些（和 只设置balanced 的模型相似）。
本模型正因为 加入了 C超参数，相比 只设置 balanced 的模型（默认阈值 与 KS阈值 近似完全重合，达到模型默认阈值的平衡点）, 
KS阈值向左移动了，KS最大值时的指标 貌似没那么 平衡了。
通过对比 本模型 和 只设置balanced 的模型，如果 追求模型平衡的话，还是选择 只设置balanced 的模型。
'''

