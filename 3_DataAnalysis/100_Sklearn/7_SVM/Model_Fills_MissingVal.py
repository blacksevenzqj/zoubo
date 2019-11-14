# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:57:25 2019

@author: dell
"""
import numpy as np
import pandas as pd

# In[]:
'''
以下填充缺失值方法，都是使用因变量Y 反向 预测特征缺失值。

将自己碰到的问题（特征工程导致的过拟合）分享出来，求高人解惑：
这个问题主要是数据泄露的问题。在训练集中作了一个新特征（每平米单价），本地CV, 结果RMSE非常好（0.05左右)。PB的成绩就非常差（0.3左右）。
主要原因是简单的利用了销售价格来产生新特征，通常这种做法，即降低了泛化成绩，也是不可取的。

即，训练集 有 因变量Y 所以能这样做，但是怎么延伸到 真实测试集呢？
如果 不 或者 不能 将该填充缺失值方式延伸至 真实测试集， 那么使用 填充了缺失值的特征 训练出的模型， 直接预测 有缺失值特征的 真实测试集，效果如何？？？
'''

# In[]:
# 1.4.2、填充月收入（随机森林）
'''
这里是将 训练集 与 测试集 合并之后一起进行填充； 那么，真实的测试集是没有标签的，怎么填充？？？ 代码：2_Scorecard_model_case_My.py
'''


def fill_missing_rf(X, y, to_fill):
    """
    使用随机森林填补一个特征的缺失值的函数
    参数：
    X：要填补的特征矩阵
    y：完整的，没有缺失值的标签
    to_fill：字符串，要填补的那一列的名称
    """

    # 构建我们的新特征矩阵和新标签
    df = X.copy()
    fill = df.loc[:, to_fill]
    df = pd.concat([df.loc[:, df.columns != to_fill], pd.DataFrame(y)], axis=1)

    # 找出我们的训练集和测试集
    Ytrain = fill[fill.notnull()]
    Ytest = fill[fill.isnull()]
    Xtrain = df.iloc[Ytrain.index, :]
    Xtest = df.iloc[Ytest.index, :]

    # 用随机森林回归来填补缺失值
    from sklearn.ensemble import RandomForestRegressor as rfr
    rfr = rfr(n_estimators=100)  # random_state=0,n_estimators=200,max_depth=3,n_jobs=-1
    rfr = rfr.fit(Xtrain, Ytrain)
    Ypredict = rfr.predict(Xtest)

    return Ypredict


# In[]:
# X = data.iloc[:,1:]
# y = data["SeriousDlqin2yrs"] # y = data.iloc[:,0]
# X.shape # (149391, 10)
# y_pred = fill_missing_rf(X,y,"MonthlyIncome")
#
## 通过以下代码检验数据是否数量相同
# temp_b = (y_pred.shape == data.loc[data.loc[:,"MonthlyIncome"].isnull(),"MonthlyIncome"].shape)
## 确认我们的结果合理之后，我们就可以将数据覆盖了
# if temp_b :
#    data.loc[data.loc[:,"MonthlyIncome"].isnull(),"MonthlyIncome"] = y_pred
#
# data.info()


# In[]:
# The Kmean idea was referenced from The 11th place solution
'''
可以作为一种思路参考，但是不能直接使用：
1.1、直接按照 房屋总面积分组： 应该按 房屋总面积区间分组。
1.2、直接按房屋总面积分组 → 取价格均值 → 再按价格均值聚类，得到聚类标签 → 再直接按房屋总面积分组 → 取聚类标签中位数。这种做法是否多此一举？
再直接按房屋总面积分组 → 取聚类标签中位数： 因是按房屋总面积分组，所以 房屋总面积相同的 聚类标签一定相同，还取聚类标签中位数？ 

2.1、房屋的地段、装修等等情况没有考虑进去，小面积也有高价格情况。
2.2、如果按代码中的逻辑，那么扩展到测试集，当测试集中 小面积好地段、好专修的房屋也会被降等级，造成错误分类。
'''


def add_kmean_col(pre_combined, ntrain, Y_train, target_col="SalePrice", kmean_col="GrLivArea", n_cluster=10, SEED=42):
    from xgboost import XGBRegressor, XGBClassifier
    # from lightgbm import LGBMRegressor,LGBMClassifier
    from sklearn.cluster import KMeans

    # pre_combined 为特征矩阵train 和 text 行向合并的DataFrame； ntrain为ntrain = train.shape[0]； target_col为因变量列名； kmean_col为特征列名。 代码在：House_Prices_wangyong.py
    col = kmean_col
    X_train = pre_combined[:ntrain]
    X_test = pre_combined[ntrain:]

    class cluster_target_encoder:

        def make_encoding(self, df):
            self.encoding = df.groupby('X')['y'].mean()

        def fit(self, X, y):
            df = pd.DataFrame(columns=['X', 'y'], index=X.index)
            df['X'] = X
            df['y'] = y
            self.make_encoding(df)

            clust = KMeans(n_cluster, random_state=SEED)

            labels = clust.fit_predict(self.encoding[df['X'].values].values.reshape(-1, 1))
            df['labels'] = labels
            self.clust_encoding = df.groupby('X')['labels'].median()

        def transform(self, X):
            res = X.map(self.clust_encoding).astype(float)
            return res

        def fit_transform(self, X, y):
            self.fit(X, y)
            return self.transform(X)

    enc1 = cluster_target_encoder()

    # fit & transform
    labels_train = enc1.fit_transform(X_train[col], Y_train)
    labels_test = enc1.transform(X_test[col])

    # fill na of label_test 延伸到 真实测试集： 是否合理，可以思考
    est = XGBClassifier()
    est.fit(X_train.select_dtypes(include=[np.number]), labels_train)
    labels_test[np.isnan(labels_test)] = est.predict(X_test.select_dtypes(include=[np.number]))[np.isnan(labels_test)]

    labels_train.name = "Glv_K"
    pre_combined = pd.concat([pre_combined, labels_train], axis=1)
    pre_combined.Glv_K = pre_combined.Glv_K.fillna(labels_test).astype("str")
    return pre_combined