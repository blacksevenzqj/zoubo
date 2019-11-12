# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:11:18 2019

@author: dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.impute import SimpleImputer  # 填补缺失值的类
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# In[]:
dataset = load_boston()
dataset  # 是一个字典
dataset.target  # 查看数据标签
dataset.data  # 数据的特征矩阵
dataset.data.shape  # 数据的结构
# 总共506*13=6578个数据

X_full, y_full = dataset.data, dataset.target
n_samples = X_full.shape[0]  # 506
n_features = X_full.shape[1]  # 13

# In[]:
# 首先确定我们希望放入的缺失数据的比例，在这里我们假设是50%，那总共就要有3289个数据缺失
rng = np.random.RandomState(0)  # 设置一个随机种子，方便观察
missing_rate = 0.5
n_missing_samples = int(np.floor(n_samples * n_features * missing_rate))  # 3289
# np.floor向下取整，返回.0格式的浮点数

# 所有数据要随机遍布在数据集的各行各列当中，而一个缺失的数据会需要一个行索引和一个列索引
# 如果能够创造一个数组，包含3289个分布在0~506中间的行索引，和3289个分布在0~13之间的列索引，那我们就可以利用索引来为数据中的任意3289个位置赋空值
# 然后我们用0，均值和随机森林来填写这些缺失值，然后查看回归的结果如何
missing_features = rng.randint(0, n_features, n_missing_samples)  # randint（下限，上限，n）指在下限和上限之间取出n个整数
len(missing_features)  # 3289
missing_samples = rng.randint(0, n_samples, n_missing_samples)
len(missing_samples)  # 3289
# missing_samples = rng.choice(n_samples,n_missing_samples,replace=False)
# 我们现在采样了3289个数据，远远超过我们的样本量506，所以我们使用随机抽取的函数randint。
# 但如果我们需要的数据量小于我们的样本量506，那我们可以采用np.random.choice来抽样，choice会随机抽取不重复的随机数，
# 因此可以帮助我们让数据更加分散，确保数据不会集中在一些行中!
# 这里我们不采用np.random.choice,因为我们现在采样了3289个数据，远远超过我们的样本量506，使用np.random.choice会报错

X_missing = X_full.copy()
y_missing = y_full.copy()
X_missing[missing_samples, missing_features] = np.nan
X_missing = pd.DataFrame(X_missing)
# 转换成DataFrame是为了后续方便各种操作，numpy对矩阵的运算速度快到拯救人生，但是在索引等功能上却不如pandas来得好用
X_missing.head()
# 并没有对y_missing进行缺失值填补，原因是有监督学习，不能缺标签啊

# In[]:
# 使用均值进行填补
from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')  # 实例化
X_missing_mean = imp_mean.fit_transform(X_missing)  # 特殊的接口fit_transform = 训练fit + 导出predict
# pd.DataFrame(X_missing_mean).isnull()#但是数据量大的时候还是看不全
# 布尔值False = 0， True = 1
# pd.DataFrame(X_missing_mean).isnull().sum()#如果求和为0可以彻底确认是否有NaN

# 使用0进行填补
imp_0 = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)  # constant指的是常数
X_missing_0 = imp_0.fit_transform(X_missing)

# In[]:
# 使用随机森林填补缺失值
'''
使用随机森林回归填补缺失值
任何回归都是从特征矩阵中学习，然后求解连续型标签y的过程，之所以能够实现这个过程，是因为回归算法认为，特征
矩阵和标签之前存在着某种联系。实际上，标签和特征是可以相互转换的，比如说，在一个“用地区，环境，附近学校数
量”预测“房价”的问题中，我们既可以用“地区”，“环境”，“附近学校数量”的数据来预测“房价”，也可以反过来，
用“环境”，“附近学校数量”和“房价”来预测“地区”。而回归填补缺失值，正是利用了这种思想。
对于一个有n个特征的数据来说，其中特征T有缺失值，我们就把特征T当作标签，其他的n-1个特征和原本的标签组成新
的特征矩阵。那对于T来说，它没有缺失的部分，就是我们的Y_train，这部分数据既有标签也有特征；
而它缺失的部分，只有特征没有标签，就是我们需要预测的部分Y_test。
特征T不缺失的值对应的其他n-1个特征 + 本来的标签：X_train
特征T不缺失的值：Y_train
特征T缺失的值对应的其他n-1个特征 + 本来的标签：X_test
特征T缺失的值：未知，我们需要预测的Y_test
这种做法，对于某一个特征大量缺失，其他特征却很完整的情况，非常适用。
那如果数据中除了特征T之外，其他特征也有缺失值怎么办？
答案是遍历所有的特征，从缺失最少的开始进行填补（因为填补缺失最少的特征所需要的准确信息最少）。
填补一个特征时，先将其他特征的缺失值用0代替，每完成一次回归预测，就将预测值放到原本的特征矩阵中，再继续填
补下一个特征。每一次填补完毕，有缺失值的特征会减少一个，所以每次循环后，需要用0来填补的特征就越来越少。当
进行到最后一个特征时（这个特征应该是所有特征中缺失值最多的），已经没有任何的其他特征需要用0来进行填补了，
而我们已经使用回归为其他特征填补了大量有效信息，可以用来填补缺失最多的特征。
遍历所有的特征后，数据就完整，不再有缺失值了。
'''
X_missing_reg = X_missing.copy()
# 找出数据集中，缺失值从小到大排列的特征们的顺序，并且有了这些的索引
sortindex = np.argsort(X_missing_reg.isnull().sum(axis=0)).values  # np.argsort()返回的是从小到大排序的顺序所对应的索引
# 从 缺失值 最少的特征 开始填充：
for i in sortindex:
    # 构建我们的新特征矩阵（没有被选中去填充的特征 + 原始的标签）和新标签（被选中去填充的特征）
    df = X_missing_reg.copy()
    fillc = df.iloc[:, i]  # 新标签(有缺失值的 特征)
    df = pd.concat([df.iloc[:, df.columns != i], pd.DataFrame(y_missing)], axis=1)  # 新特征矩阵： 除了现特征缺失列（剩余所有列） + 原标签Y

    # 在新特征矩阵中，对含有缺失值的列，进行0的填补： 除了现特征缺失列（剩余所有特征列中有缺失值） 都填充0
    df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)

    # 找出我们的训练集和测试集
    Ytrain = fillc[fillc.notnull()]  # Ytrain是被选中要填充的特征中（现在是我们的标签），存在的那些值：非空值
    Ytest = fillc[fillc.isnull()]  # Ytest 是被选中要填充的特征中（现在是我们的标签），不存在的那些值：空值。注意我们需要的不是Ytest的值，需要的是Ytest所带的索引
    Xtrain = df_0[Ytrain.index, :]  # 在新特征矩阵上，被选出来的要填充的特征的非空值所对应的记录
    Xtest = df_0[Ytest.index, :]  # 在新特征矩阵上，被选出来的要填充的特征的空值所对应的记录

    # 用随机森林回归来填补缺失值
    rfc = RandomForestRegressor(n_estimators=100)  # 实例化
    rfc.fit(Xtrain, Ytrain)  # 导入训练集进行训练
    Ypredict = rfc.predict(Xtest)  # 用predict接口将Xtest导入，得到我们的预测结果（回归结果），就是我们要用来填补空值的这些值

    # 将填补好的特征返回到我们的原始的特征矩阵中
    X_missing_reg.loc[X_missing_reg.iloc[:, i].isnull(), i] = Ypredict

# 检验是否有空值
X_missing_reg.isnull().sum()

# In[]:
# 对所有数据进行建模，取得MSE结果
x_labels = ['Full data',
            'Zero Imputation',
            'Mean Imputation',
            'Regressor Imputation']
X = [X_full, X_missing_mean, X_missing_0, X_missing_reg]
mse = []

for x in X:
    estimator = RandomForestRegressor(random_state=0, n_estimators=100)  # 实例化
    scores = cross_val_score(estimator, x, y_missing, scoring='neg_mean_squared_error', cv=5).mean()
    mse.append(scores * -1)

# In[]:
result_X = pd.DataFrame(np.array(mse).reshape(-1, 1),
                        index=x_labels, columns=['squared']).sort_values(by='squared')

colors = ['r', 'g', 'b', 'orange']

plt.figure(figsize=(12, 6))  # 画出画布
ax = plt.subplot(111)  # 添加子图
for i, v in enumerate(result_X.index):
    ax.barh(i, result_X.loc[v], color=colors[i], alpha=0.6, align='center')  # bar为条形图，barh为横向条形图，alpha表示条的粗度
ax.set_title('Imputation Techniques with Boston Data')
ax.set_xlim(left=result_X['squared'].min() * 0.9, right=result_X['squared'].max() * 1.1)  # 设置x轴取值范围
ax.set_yticks(np.arange(len(mse)))
ax.set_xlabel('MSE')
ax.set_yticklabels(result_X.index)
plt.show()


