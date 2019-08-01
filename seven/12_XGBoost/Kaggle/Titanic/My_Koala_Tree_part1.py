# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 17:03:12 2019

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')
import re


#pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

train_data = pd.read_csv('C:\\Users\\dell\\Desktop\\titanic\\train.csv')
test_data = pd.read_csv('C:\\Users\\dell\\Desktop\\titanic\\test.csv')

# print(train_data.head())
# print(train_data.info())
# train_data['Survived'].value_counts().plot.pie(autopct = '%1.2f%%') # 342/891



# 2. 缺失值处理的方法
# 2.1、Embarked---在哪上船 属性：只有2个缺失值，使用众数填充。
# train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values # 889 -> 891
# train_data['Embarked'] = train_data.Embarked.fillna(train_data.Embarked.dropna().mode().values[0])
train_data['Embarked'].fillna(train_data['Embarked'].dropna().mode().iloc[0], inplace=True)

# 2.2、Cabin---船舱 属性：缺失值太多。缺失本身也可能代表着一些隐含信息，可能代表并没有船舱。
train_data['Cabin'] = train_data.Cabin.fillna('U0')
# train_data.Cabin[train_data.Cabin.isnull()] = 'U0'
# train_data.loc[train_data.Cabin.isnull(), 'Cabin'] = 'U0'

# 2.3、使用 回归随机森林 等模型来预测缺失属性的值（实际的应用中需要将非数值特征转换为数值特征）
age_df = train_data[['Age','Survived','Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_notnull = age_df.loc[(train_data['Age'].notnull())]
age_df_isnull = age_df.loc[(train_data['Age'].isnull())]
X = age_df_notnull.values[:,1:]
Y = age_df_notnull.values[:,0]
RFR = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=-1)
RFR.fit(X,Y)
predictAges = RFR.predict(age_df_isnull.values[:,1:])
train_data.loc[train_data['Age'].isnull(), ['Age']] = predictAges
# print(train_data.info())



# 3. 分析数据关系
# 3.1、性别与是否生存的关系 Sex
print(train_data.groupby(['Sex', 'Survived'])['Survived'].count())
print(train_data[['Sex', 'Survived']].groupby(['Sex']).count() / train_data[['Sex']].count().values)
print(train_data[['Sex', 'Survived']].groupby(['Sex']).mean())

# 3.2、船舱等级和生存与否的关系 Pclass
print(train_data.groupby(['Pclass', 'Survived'])['Survived'].count())
print(train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean())

print(train_data.groupby(['Sex', 'Pclass', 'Survived'])['Survived'].count())
print(train_data[['Sex', 'Pclass', 'Survived']].groupby(['Sex', 'Pclass']).mean())

# 3.3、年龄与存活与否的关系 Age
print(train_data['Age'].describe())
bins = [0, 12, 18, 65, 100]
train_data['Age_group'] = pd.cut(train_data['Age'], bins)
by_age = train_data.groupby(['Age_group'])['Survived'].mean()
print(by_age)

# 3.4、称呼与存活与否的关系 Name
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print(pd.crosstab(train_data['Title'], train_data['Sex']))
print(train_data[['Title','Survived']].groupby(['Title']).mean())
train_data['Name_length'] = train_data['Name'].apply(len)
print(train_data[['Name_length','Survived']].groupby(['Name_length'],as_index=False).mean())

# 3.5、有无兄弟姐妹和存活与否的关系 SibSp
sibsp_df = train_data[train_data['SibSp'] != 0]
no_sibsp_df = train_data[train_data['SibSp'] == 0]
print(sibsp_df[['SibSp', 'Survived']].groupby(['Survived']).count() / sibsp_df.Survived.count())
# print(sibsp_df['Survived'].value_counts() / sibsp_df.Survived.count())
print(no_sibsp_df[['SibSp', 'Survived']].groupby(['Survived']).count() / no_sibsp_df.Survived.count())

# 3.6、有无父母子女和存活与否的关系 Parch
parch_df = train_data[train_data['Parch'] != 0]
no_parch_df = train_data[train_data['Parch'] == 0]
print(parch_df['Survived'].value_counts() / parch_df.Survived.count())
print(no_parch_df['Survived'].value_counts() / no_parch_df.Survived.count())

# 3.7、亲友的人数和存活与否的关系 SibSp & Parch
print(train_data[['SibSp', 'Survived']].groupby(['SibSp']).mean())
print(train_data[['Parch', 'Survived']].groupby(['Parch']).mean())
train_data['Family_Size'] = train_data['SibSp'] + train_data['Parch'] + 1 # +1是算上自己
print(train_data[['Family_Size', 'Survived']].groupby(['Family_Size']).mean())

# 3.8、票价分布和存活与否的关系 Fare
print(train_data[['Pclass', 'Fare', 'Survived']].groupby(['Pclass', 'Fare']).mean())
print(train_data['Fare'].describe())
# 生存与否 与 票价均值和标准差的关系
fare_not_survived = train_data['Fare'][train_data['Survived'] == 0]
fare_survived = train_data.Fare[train_data['Survived'] == 1]
average_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
print(average_fare.iloc[:,:])
std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])
print(std_fare.iloc[:])

# 3.9、船舱类型和存活与否的关系 Cabin --- 定性(Qualitative)转换：Factorizing
# 简单地将数据分为是否有Cabin记录作为特征，与生存与否进行分析：
# print(train_data.Cabin[train_data['Cabin'].notnull()])
train_data['Has_Cabin'] = train_data['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)
print(train_data[['Has_Cabin', 'Survived']].groupby(['Has_Cabin']).mean())
# 只保留 第一个船舱
train_data['CabinLetter'] = train_data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
train_data['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]
print(train_data[['CabinLetter', 'Survived']].groupby(['CabinLetter']).mean())
# 可见，不同的船舱生存率也有不同，但是差别不大。所以在处理中，我们可以直接将特征删除。

# 3.10、港口和存活与否的关系 Embarked
print(train_data[['Pclass', 'Embarked', 'Survived']].groupby(['Embarked', 'Survived']).count())
print(train_data[['Embarked', 'Survived']].groupby(['Embarked']).mean())



# 4、变量转换
# 4.1、定性(Qualitative)转换：
# 4.1.1、Dummy Variables
embark_dummies  = pd.get_dummies(train_data['Embarked'])
train_data = train_data.join(embark_dummies)
# train_data.drop(['Embarked'], axis=1,inplace=True)
# print(train_data[['S', 'C', 'Q']])

# 4.1.2、 Factorizing


# 4.2、定量(Quantitative)转换：
# 4.2.1、Scaling Z分数
from sklearn import preprocessing

assert np.size(train_data['Age']) == 891
# Z分数
scaler = preprocessing.StandardScaler()
train_data['Age_scaled'] = scaler.fit_transform(train_data['Age'].values.reshape(-1, 1))
# print(train_data['Age_scaled'][0:10])

# 4.2.2、Binning 分为5个桶
# 4.2.2.1、票价Fare
train_data['Fare_bin'] = pd.qcut(train_data['Fare'], 5)
print(train_data['Fare_bin'].value_counts())
# 定量 之后 定性1：factorize
train_data['Fare_bin_id'] = pd.factorize(train_data['Fare_bin'])[0]
# 定量 之后 定性2：dummies
fare_bin_dummies_df = pd.get_dummies(train_data['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))
train_data = pd.concat([train_data, fare_bin_dummies_df], axis=1)

# 4.2.2.2、年龄Age
train_data['Age_group']
# 定量 之后 定性1：factorize
train_data['Age_group_id'] = pd.factorize(train_data['Age_group'])[0]
# 定量 之后 定性2：dummies
age_bin_dummies_df = pd.get_dummies(train_data['Age_group']).rename(columns=lambda x: 'Age_' + str(x))
train_data = pd.concat([train_data, age_bin_dummies_df], axis=1)


train_data.to_csv('C:\\Users\\dell\\Desktop\\titanic\\out\\train_data.csv')

