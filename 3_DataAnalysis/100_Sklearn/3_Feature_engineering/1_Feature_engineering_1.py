# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 20:11:10 2019

@author: dell
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import os
os.chdir(r"F:\新笔记\机器学习\教程\4、sklearn\菜菜的机器学习sklearn课堂\课件\03数据预处理和特征工程")

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

# In[]:
'''
对于StandardScaler和MinMaxScaler来说，空值NaN会被当做是缺失值，在fit的时候忽略，在transform的时候
保持缺失NaN的状态显示。并且，尽管去量纲化过程不是具体的算法，但在fit接口中，依然只允许导入至少二维数
组，一维数组导入会报错。通常来说，我们输入的X会是我们的特征矩阵，现实案例中特征矩阵不太可能是一维所
以不会存在这个问题
'''
# In[]:
# 1、实现归一化
scaler = MinMaxScaler()                             #实例化
scaler.fit(data)                           #fit，在这里本质是生成min(x)和max(x)
result = scaler.transform(data)                     #通过接口导出结果

# In[]:
print(scaler.fit_transform(data))                #训练和导出结果一步达成
print(scaler.inverse_transform(result))                    #将归一化后的结果逆转
 
# In[]:
# 使用MinMaxScaler的参数feature_range实现将数据归一化到[0,1]以外的范围中
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler(feature_range=[5,10])         #依然实例化
result = scaler.fit_transform(data)                 #fit_transform一步导出结果

# In[]:
# 当X中的特征数量非常多的时候，fit会报错并表示，数据量太大了我计算不了
# 此时使用partial_fit作为训练接口
scaler = MinMaxScaler()
scaler.partial_fit(data)
result = scaler.transform(data)

# In[]:
X = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
#归一化
X_nor = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#逆转归一化
X_returned = X_nor * (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0)


# In[]:
# 2、标准化：
from sklearn.preprocessing import StandardScaler
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
 
scaler = StandardScaler()                           #实例化
scaler.fit(data)                                    #fit，本质是生成均值和方差
scaler.mean_                                        #查看均值的属性mean_
scaler.var_                                         #查看方差的属性var_
 
# In[]:
x_std = scaler.transform(data)                      #通过接口导出结果
x_std.mean()                                        #导出的结果是一个数组，用mean()查看均值
x_std.std()                                         #用std()查看方差

# In[]: 
print(scaler.fit_transform(data))                       #使用fit_transform(data)一步达成结果
print(scaler.inverse_transform(x_std))



# In[]:
# 3、缺失值处理
data = pd.read_csv(r"Narrativedata.csv"
                   ,index_col=0
                  )#index_col=0将第0列作为索引，不写则认为第0列为特征
 
data.info()

#填补年龄
Age = data.loc[:,"Age"].values.reshape(-1,1)            #sklearn当中特征矩阵必须是二维
 
imp_mean = SimpleImputer()                              #实例化，默认均值填补
imp_median = SimpleImputer(strategy="median")           #用中位数填补
imp_0 = SimpleImputer(strategy="constant",fill_value=0) #用0填补
 
imp_mean = imp_mean.fit_transform(Age)
imp_median = imp_median.fit_transform(Age)
imp_0 = imp_0.fit_transform(Age)
#在这里我们使用中位数填补Age
data.loc[:,"Age"] = imp_median
 
#使用众数填补Embarked
Embarked = data.loc[:,"Embarked"].values.reshape(-1,1)
imp_mode = SimpleImputer(strategy = "most_frequent")
data.loc[:,"Embarked"] = imp_mode.fit_transform(Embarked)

# In[]:
data_ = pd.read_csv(r"Narrativedata.csv"
                   ,index_col=0
                  )#index_col=0将第0列作为索引，不写则认为第0列为特征
data_.loc[:,"Age"] = data_.loc[:,"Age"].fillna(data_.loc[:,"Age"].median())
 
data_.dropna(axis=0,inplace=True)
#.dropna(axis=0)删除所有有缺失值的行，.dropna(axis=1)删除所有有缺失值的列



# In[]:
# 4、分类变量转换：
from sklearn.preprocessing import LabelEncoder

y = data_.iloc[:,-1]                         #要输入的是标签，不是特征矩阵，所以允许一维
 
le = LabelEncoder()                         #实例化
le.fit(y)                                   #导入数据
label = le.transform(y)                     #transform接口调取结果
 
print(le.classes_)                                 #属性.classes_查看标签中究竟有多少类别
print(label)                                       #查看获取的结果label
 
print(le.fit_transform(y))                         #也可以直接fit_transform一步到位
print(le.inverse_transform(label))                 #使用inverse_transform可以逆转

# In[]:
from sklearn.preprocessing import OrdinalEncoder

y = data_.iloc[:,-1].reshape(-1,1)
## 接口categories_对应LabelEncoder的接口classes_，一模一样的功能
enc = OrdinalEncoder()
enc.fit(y)
print(enc.categories_)
data_.iloc[:,-1] = enc.transform(y)

data_.iloc[:,1:-1] = enc.fit_transform(data_.iloc[:,1:-1]) # 一步到位

# In[]:
from sklearn.preprocessing import OneHotEncoder

X = data_.iloc[:,1:-1]
 
enc = OneHotEncoder(categories='auto').fit(X)
result = enc.transform(X).toarray()
 
#依然可以直接一步到位，但为了给大家展示模型属性，所以还是写成了三步
OneHotEncoder(categories='auto').fit_transform(X).toarray()
 
#依然可以还原
pd.DataFrame(enc.inverse_transform(result))
 
print(enc.get_feature_names()) # 返回每一个经过哑变量后生成稀疏矩阵列的名字
 
# axis=1,表将两表左右相连，如果是axis=0，就是将量表上下相连
newdata = pd.concat([data,pd.DataFrame(result)],axis=1)
newdata.drop(["Sex","Embarked"],axis=1,inplace=True)
newdata.columns = ["Age","Survived","Female","Male","Embarked_C","Embarked_Q","Embarked_S"]



# In[]:
# 5、连续变量转换：
# 将年龄二值化
data_2 = data.copy()
 
from sklearn.preprocessing import Binarizer
X = data_2.iloc[:,0].values.reshape(-1,1)               #类为特征专用，所以不能使用一维数组
transformer = Binarizer(threshold=30).fit_transform(X)
 
data_2.iloc[:,0] = transformer
data_2.head()

# In[]:
# 分箱 编码 一同完成：
from sklearn.preprocessing import KBinsDiscretizer
 
X = data.iloc[:,0].values.reshape(-1,1) 
# 普通转换、等宽
est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform') # 等宽分箱
res = est.fit_transform(X)
# 查看转换后分的箱：变成了一列中的三箱
print(set(res.ravel()))
unique_label, counts_label = np.unique(res, return_counts=True)
print(counts_label/ len(res)) 

# 普通转换、等位/等深
est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile') # 等位/深分箱
res = est.fit_transform(X)
# 查看转换后分的箱：变成了一列中的三箱
print(set(res.ravel()))
unique_label, counts_label = np.unique(res, return_counts=True)
print(counts_label/ len(res)) 

# one-hot转换 默认
est = KBinsDiscretizer(n_bins=3, encode='onehot', strategy='uniform') # 等宽分箱
#查看转换后分的箱：变成了哑变量
res2 = est.fit_transform(X).toarray()
print(set(res2.ravel()))

# In[]:
# 分箱：
# 等宽
data.iloc[:,0].describe()
bins = [0, 28, 35, 80]
score_cut = pd.cut(data.iloc[:,0], bins) # labels=False
print(score_cut.value_counts())
print(score_cut.value_counts() / len(score_cut))

# 等深
qcats3 = pd.qcut(data.iloc[:,0],q=3, labels=['0', '1', '2'])
print(qcats3.value_counts())
print(qcats3.value_counts() / len(qcats3))







