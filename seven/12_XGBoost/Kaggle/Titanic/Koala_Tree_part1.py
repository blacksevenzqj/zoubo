# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 17:03:12 2019

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
import re


#pd.set_option('display.max_columns', None)

train_data = pd.read_csv('C:\\Users\\dell\\Desktop\\titanic\\train.csv')
test_data = pd.read_csv('C:\\Users\\dell\\Desktop\\titanic\\test.csv')

sns.set_style('whitegrid')
#print(train_data.head())


#print(train_data.info())
#print("-" * 40)
#print(test_data.info())
#print("-" * 40)
#print(train_data.describe())


#train_data['Survived'].value_counts().plot.pie(autopct = '%1.2f%%')

# 众数填充缺失值：Embarked字段 上船地点
train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values

train_data['Cabin'] = train_data.Cabin.fillna('U0')


age_df = train_data[['Age','Survived','Fare', 'Parch', 'SibSp', 'Pclass']]
#print(age_df.head())
age_df_notnull = age_df.loc[(train_data['Age'].notnull())]
age_df_isnull = age_df.loc[(train_data['Age'].isnull())]


from sklearn.ensemble import RandomForestRegressor

#choose training data to predict age
age_df = train_data[['Age','Survived','Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_notnull = age_df.loc[(train_data['Age'].notnull())]
age_df_isnull = age_df.loc[(train_data['Age'].isnull())]
X = age_df_notnull.values[:,1:]
Y = age_df_notnull.values[:,0]
#print(X)
#print(Y)

#print("-" * 40)

# use RandomForestRegression to train data   随机森林填充缺失值，不懂
RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
RFR.fit(X,Y)
predictAges = RFR.predict(age_df_isnull.values[:,1:])
train_data.loc[train_data['Age'].isnull(), ['Age']]= predictAges
#print(train_data.info())

print("-" * 40)

# 1、性别与是否生存的关系 Sex
print(train_data.groupby(['Sex','Survived'])['Survived'].count())
print("-" * 40)
train_data.groupby(['Sex','Survived'])['Survived'].count().plot.bar()
print(train_data[['Sex','Survived']].groupby(['Sex']).mean())
print("-" * 40)
train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()


# 2、船舱等级与是否生存的关系 Pclass
print(train_data.groupby(['Pclass','Survived'])['Survived'].count())
print("-" * 40)
train_data.groupby(['Pclass','Survived'])['Survived'].count().plot.bar()
print(train_data[['Pclass','Survived']].groupby(['Pclass']).mean())
print("-" * 40)
train_data[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()

# 不同船舱男女生存比例
print(train_data.groupby(['Sex','Pclass','Survived'])['Survived'].count())
print("-" * 40)
print(train_data.groupby(['Sex','Pclass','Survived'])['Survived'].count().plot.bar())

print(train_data[['Sex','Pclass','Survived']].groupby(['Sex','Pclass']).mean())
print("-" * 40)
train_data[['Sex','Pclass','Survived']].groupby(['Sex','Pclass']).mean().plot.bar()


# 3、不同年龄生存比例关系 Age
fig, ax = plt.subplots(1, 2, figsize = (18, 8))

sns.violinplot("Pclass", "Age", hue="Survived", data=train_data, split=True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0, 110, 10))

sns.violinplot("Sex", "Age", hue="Survived", data=train_data, split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0, 110, 10))

plt.show()


plt.figure(figsize=(12,5))
plt.subplot(121)
#train_data['Age'].hist(bins=70) #
plt.hist(train_data['Age'], bins=70) # bins为分段数
plt.xlabel('Age')
plt.ylabel('Num')

plt.subplot(122)
train_data.boxplot(column='Age', showfliers=False)
plt.show()

print(train_data['Age'].describe())
# 可以看到 均值是29.65，而标准差是13.73，已经是偏态分布。从直方图中也能看到。

print("-" * 40)

# 不同年龄下的生存和非生存的分布情况：不懂这个API
#facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
#facet.map(sns.kdeplot,'Age',shade= True)
#facet.set(xlim=(0, train_data['Age'].max()))
#facet.add_legend()

# 不同年龄下的平均生存率：
# average survived passengers by age
#fig, axis1 = plt.subplots(1,1,figsize=(18,4))
#train_data["Age_int"] = train_data["Age"].astype(int)
#average_age = train_data[["Age_int", "Survived"]].groupby(['Age_int'],as_index=False).mean()
#sns.barplot(x='Age_int', y='Survived', data=average_age)


bins = [0, 12, 18, 65, 100]
train_data['Age_group'] = pd.cut(train_data['Age'], bins)
by_age = train_data.groupby(['Age_group'])['Survived'].mean()
print(by_age)
#by_age.plot(kind = 'bar')
by_age.plot.bar()

print("-" * 40)

# 名字对生存率的影响：
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print(pd.crosstab(train_data['Title'], train_data['Sex']))

print("-" * 40)

print(train_data[['Title','Survived']].groupby(['Title']).mean())
train_data[['Title','Survived']].groupby(['Title']).mean().plot.bar()

# 同时，对于名字，我们还可以观察名字长度和生存率之间存在关系的可能：
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
train_data['Name_length'] = train_data['Name'].apply(len)
name_length = train_data[['Name_length','Survived']].groupby(['Name_length'],as_index=False).mean()
sns.barplot(x='Name_length', y='Survived', data=name_length)


# (5) 有无兄弟姐妹和存活与否的关系 SibSp
# 将数据分为有兄弟姐妹的和没有兄弟姐妹的两组：
sibsp_df = train_data[train_data['SibSp'] != 0]
no_sibsp_df = train_data[train_data['SibSp'] == 0]
plt.figure(figsize=(10,5))
plt.subplot(121)
sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('sibsp')

plt.subplot(122)
no_sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('no_sibsp')

plt.show()


# (6) 有无父母子女和存活与否的关系 Parch
parch_df = train_data[train_data['Parch'] != 0]
no_parch_df = train_data[train_data['Parch'] == 0]

plt.figure(figsize=(10,5))
plt.subplot(121)
parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('parch')

plt.subplot(122)
no_parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('no_parch')

plt.show()

# (7) 亲友的人数和存活与否的关系 SibSp & Parch
fig,ax=plt.subplots(1,2,figsize=(18,8))
train_data[['Parch','Survived']].groupby(['Parch']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Parch and Survived')
train_data[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar(ax=ax[1])
ax[1].set_title('SibSp and Survived')

# 从图表中可以看出，若独自一人，那么其存活率比较低；但是如果亲友太多的话，存活率也会很低。
train_data['fimary_size'] = train_data['Parch'] + train_data['SibSp'] + 1
train_data[['fimary_size', 'Survived']].groupby('fimary_size').mean().plot.bar()


# (8) 票价分布和存活与否的关系 Fare
# 绘制票价的分布情况
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.hist(train_data['Fare'], bins=70)
plt.xlabel('Fare')
plt.ylabel('Num')

train_data.boxplot(column='Fare', by='Pclass', showfliers=False)
plt.show()

print(train_data['Fare'].describe())
# 绘制生存与否与票价均值和方差的关系：
fare_not_survived = train_data['Fare'][train_data['Survived'] == 0]
fare_survived = train_data['Fare'][train_data['Survived'] == 1]

average_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])

#average_fare.plot.bar()
#std_fare.plot.bar()
average_fare.plot(yerr=std_fare, kind='bar', legend=False)

plt.show()


# (9) 船舱类型和存活与否的关系 Cabin
# 有船舱和没有船舱的存活对比
train_data.loc[train_data.Cabin.isnull(), 'Cabin'] = 'U0'
train_data['Has_Cabin'] = train_data['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)
train_data[['Has_Cabin','Survived']].groupby(['Has_Cabin']).mean().plot.bar()

# 对不同类型的船舱进行分析：
train_data['CabinLetter'] = train_data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
train_data['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]
train_data[['CabinLetter','Survived']].groupby(['CabinLetter']).mean().plot.bar()


# (10) 港口和存活与否的关系 Embarked
sns.countplot('Embarked', hue='Survived', data=train_data)
plt.title('Embarked and Survived')

sns.factorplot('Embarked', 'Survived', data=train_data, size=3, aspect=2)
plt.title('Embarked and Survived rate')
plt.show()




from sklearn import preprocessing

assert np.size(train_data['Age']) == 891
# 定量：Z分数
scaler = preprocessing.StandardScaler()
train_data['Age_scaled'] = scaler.fit_transform(train_data['Age'].values.reshape(-1, 1))
#print(train_data['Age_scaled'][0:10])


# 定量：分为5个桶
train_data['Fare_bin'] = pd.qcut(train_data['Fare'], 5)
print(train_data['Fare_bin'].value_counts())

# 定性：factorize
train_data['Fare_bin_id'] = pd.factorize(train_data['Fare_bin'])[0]
# dummies
fare_bin_dummies_df = pd.get_dummies(train_data['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))
train_data = pd.concat([train_data, fare_bin_dummies_df], axis=1)
#train_data.to_csv('C:\\Users\\dell\\Desktop\\titanic\\out\\train_data.csv')



