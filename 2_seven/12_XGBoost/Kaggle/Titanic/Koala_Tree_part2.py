# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:11:47 2019

@author: Administrator

---Feature engineering---
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
import re

# pd.set_option('display.max_columns', None)


train_df_org = pd.read_csv('C:\\Users\\dell\\Desktop\\titanic\\train.csv')
test_df_org = pd.read_csv('C:\\Users\\dell\\Desktop\\titanic\\test.csv')

sns.set_style('whitegrid')
# print(train_data.head())

test_df_org['Survived'] = 0
# 行向相加（从上到下）
combined_train_test = train_df_org.append(test_df_org)
PassengerId = test_df_org['PassengerId']

# (1) Embarked 众数填充
# combined_train_test.Embarked[combined_train_test.Embarked.isnull()] = combined_train_test.Embarked.dropna().mode().values
combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().iloc[0], inplace=True)
# 定性转换：factorize
combined_train_test['Embarked'] = pd.factorize(combined_train_test['Embarked'])[0]
# 定性转换：get_dummies
emb_dummies_df = pd.get_dummies(combined_train_test['Embarked'], prefix=combined_train_test[['Embarked']].columns[0])
combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)

# (2) 对Sex也进行one-hot编码，也就是dummy处理：
combined_train_test['Sex'] = pd.factorize(combined_train_test['Sex'])[0]
sex_dummies_df = pd.get_dummies(combined_train_test['Sex'], prefix=combined_train_test[['Sex']].columns[0])
combined_train_test = pd.concat([combined_train_test, sex_dummies_df], axis=1)

# (3) Name
# combined_train_test['Title'] = combined_train_test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# print(pd.crosstab(combined_train_test['Title'], combined_train_test['Sex']))
combined_train_test['Title'] = combined_train_test['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
# 将各式称呼进行统一化处理：
title_Dict = {}
title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
print(title_Dict)
combined_train_test['Title'] = combined_train_test['Title'].map(title_Dict)
# 定性转换：factorize
combined_train_test['Title'] = pd.factorize(combined_train_test['Title'])[0]
# 定性转换：get_dummies
title_dummies_df = pd.get_dummies(combined_train_test['Title'], prefix=combined_train_test[['Title']].columns[0])
combined_train_test = pd.concat([combined_train_test, title_dummies_df], axis=1)
combined_train_test['Name_length'] = combined_train_test['Name'].apply(len)

# combined_train_test.to_csv('C:\\Users\\dell\\Desktop\\titanic\\out\\ddd.csv')


# (4) Fare
# ccc = combined_train_test.groupby('Pclass').transform(np.mean) # 所有列的数据均参与平均值计算，并返回所有列，接收时也必须是所有列接收。
# combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(combined_train_test.groupby('Pclass').transform(np.mean))
# 上面这个写法太难理解
# 1.1、combined_train_test.groupby('Pclass')['Fare'].transform(np.mean) 将Pclass分组后计算每组的Fare均值。
# 1.2、combined_train_test['Fare'].fillna(xxx) 填充Fare字段的缺失值，注意：会按照缺失值的Pclass所属组 查找1.1中计算得到的相对应的Pclass组均值进行填充。
# ccc = combined_train_test.groupby('Pclass')['Fare'].transform(np.mean) # 指明参与平均值计算的列，返回单列，
combined_train_test['Fare'] = combined_train_test['Fare'].fillna(
    combined_train_test.groupby('Pclass')['Fare'].transform(np.mean))
# combined_train_test.to_csv('C:\\Users\\dell\\Desktop\\titanic\\out\\eee.csv')


# 团体票 除以 团体人数 = 单票价 重新赋值给 相应字段
# 先取数据集，再分组。而数据集中没有分组所需字段数据，则使用 by=combined_train_test['Ticket'] 特殊指定。
# ccc = combined_train_test['Fare'].groupby(by=combined_train_test['Ticket']).transform('count')
# ccc.to_csv('C:\\Users\\dell\\Desktop\\titanic\\out\\fff222.csv')
# ccc = combined_train_test.groupby('Ticket')['Fare'].transform('count') #等价于上面的代码
# ccc.to_csv('C:\\Users\\dell\\Desktop\\titanic\\out\\fff333.csv')

combined_train_test['Group_Ticket'] = combined_train_test['Fare'].groupby(by=combined_train_test['Ticket']).transform(
    'count')
combined_train_test['Fare'] = combined_train_test['Fare'] / combined_train_test['Group_Ticket']
# combined_train_test.drop(['Group_Ticket'], axis=1, inplace=True)
# combined_train_test.to_csv('C:\\Users\\dell\\Desktop\\titanic\\out\\ggg.csv')


# 使用binning给票价分等级：
combined_train_test['Fare_bin'] = pd.qcut(combined_train_test['Fare'], 5)
combined_train_test['Fare_bin_id'] = pd.factorize(combined_train_test['Fare_bin'])[0]

# 如果用 Fare_bin，则lambda取到的x是Fare_(-0.001, 7.229]，使用Fare_bin_id，lambda取到的x是1、2、3...
fare_bin_dummies_df = pd.get_dummies(combined_train_test['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))
combined_train_test = pd.concat([combined_train_test, fare_bin_dummies_df], axis=1)
# combined_train_test.drop(['Fare_bin'], axis=1, inplace=True)
# combined_train_test.to_csv('C:\\Users\\dell\\Desktop\\titanic\\out\\hhh.csv')


# (5) Pclass
from sklearn.preprocessing import LabelEncoder

# 建立PClass Fare Category
def pclass_fare_category(df, pclass1_mean_fare, pclass2_mean_fare, pclass3_mean_fare):
    if df['Pclass'] == 1:
        if df['Fare'] <= pclass1_mean_fare:
            return 'Pclass1_Low'
        else:
            return 'Pclass1_High'
    elif df['Pclass'] == 2:
        if df['Fare'] <= pclass2_mean_fare:
            return 'Pclass2_Low'
        else:
            return 'Pclass2_High'
    elif df['Pclass'] == 3:
        if df['Fare'] <= pclass3_mean_fare:
            return 'Pclass3_Low'
        else:
            return 'Pclass3_High'


# Pclass1_mean_fare = combined_train_test.groupby('Pclass')['Fare'].mean().get(1) # 等价于下面3行代码
Pclass1_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([1]).values[0]
Pclass2_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([2]).values[0]
Pclass3_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([3]).values[0]

# 建立Pclass_Fare Category
combined_train_test['Pclass_Fare_Category'] = combined_train_test.apply(pclass_fare_category, args=(
Pclass1_mean_fare, Pclass2_mean_fare, Pclass3_mean_fare), axis=1)

pclass_level = LabelEncoder()
# 给每一项添加标签
pclass_level.fit(
    np.array(['Pclass1_Low', 'Pclass1_High', 'Pclass2_Low', 'Pclass2_High', 'Pclass3_Low', 'Pclass3_High']))

# 转换成数值
# combined_train_test['Pclass_Fare_Category'] = pclass_level.transform(combined_train_test['Pclass_Fare_Category'])
combined_train_test['Pclass_Fare_Category2'] = pclass_level.transform(combined_train_test['Pclass_Fare_Category'])
# combined_train_test['Pclass_Fare_Category3'] = pd.factorize(combined_train_test['Pclass_Fare_Category'])[0] # 效果和使用 LabelEncoder 一样的

# dummy 转换
pclass_dummies_df = pd.get_dummies(combined_train_test['Pclass_Fare_Category']).rename(
    columns=lambda x: 'Pclass_' + str(x))
combined_train_test = pd.concat([combined_train_test, pclass_dummies_df], axis=1)

# combined_train_test['Pclass2'] = pd.factorize(combined_train_test['Pclass'])[0] # 没有意义的 定性转换
# combined_train_test.to_csv('C:\\Users\\dell\\Desktop\\titanic\\out\\iii.csv')


# (6) Parch and SibSp
def family_size_category(family_size):
    if family_size <= 1:
        return 'Single'
    elif family_size <= 4:
        return 'Small_Family'
    else:
        return 'Large_Family'

combined_train_test['Family_Size'] = combined_train_test['Parch'] + combined_train_test['SibSp'] + 1
combined_train_test['Family_Size_Category'] = combined_train_test['Family_Size'].map(family_size_category)

le_family = LabelEncoder()
le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
combined_train_test['Family_Size_Category2'] = le_family.transform(combined_train_test['Family_Size_Category'])

family_size_dummies_df = pd.get_dummies(combined_train_test['Family_Size_Category'], prefix=combined_train_test[['Family_Size_Category']].columns[0])
combined_train_test = pd.concat([combined_train_test, family_size_dummies_df], axis=1)

#combined_train_test.to_csv('C:\\Users\\dell\\Desktop\\titanic\\out\\jjj.csv')



# Age
missing_age_df = pd.DataFrame(combined_train_test[
    ['Age', 'Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size', 'Family_Size_Category2', 'Fare', 'Fare_bin_id', 'Pclass']])

notmissing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]
print(missing_age_test.head())










