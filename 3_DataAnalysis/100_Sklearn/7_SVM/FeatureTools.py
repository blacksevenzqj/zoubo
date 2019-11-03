# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:48:40 2019

@author: dell
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit, cross_val_score as CVS
import matplotlib.pyplot as plt
import seaborn as sns


# In[]:
# 缺失值
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * mis_val / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# 特征返回非缺失值部分
def get_notMissing_values(data_temp, feature):
    return data_temp[data_temp[feature] == data_temp[feature]]  # 返回全部Data


# 重复值处理
def duplicate_value(data):
    # 重复项按特征统计
    print(data[data.duplicated()].count())
    # 去除重复项 后 长度
    nodup = data[-data.duplicated()]
    print("去除重复项后长度：%d" % len(nodup))
    # 去除重复项 后 长度
    print("去除重复项后长度：%d" % len(data.drop_duplicates()))
    # 重复项 长度
    print("重复项长度：%d" % len(data) - len(nodup))

    # 在原数据集上 删除重复项
    data.drop_duplicates(inplace=True)
    print(data.info())

    # 重设索引
    data = data.reset_index(drop=True)
    # data.index = range(data.shape[0])

    print(data.info())


# In[]:
# 分类模型 数据类别 样本不均衡
def sample_category(ytrain, ytest):
    train_unique_label, train_counts_label = np.unique(ytrain, return_counts=True)
    test_unique_label, test_counts_label = np.unique(ytest, return_counts=True)
    print('-' * 60)
    print('Label Distributions: \n')
    print("训练集类别%s，数量%s，占比%s" % (train_unique_label, train_counts_label, (train_counts_label / len(ytrain))))
    print("测试集类别%s，数量%s，占比%s" % (test_unique_label, test_counts_label, (test_counts_label / len(ytest))))


# 分类模型 数据类别 样本不均衡
def Sample_imbalance(data, y_name):
    print(data.shape)
    print(data.info())

    print('Y = 0', round(len(data[data.y_name == 0]) / len(data) * 100, 2), "% of the dataset")
    print('Y = 1', round(len(data[data.y_name == 1]) / len(data) * 100, 2), "% of the dataset")

    # 查看目标列Y的情况
    print(data.groupby(y_name).size())
    print(data[y_name].value_counts())

    # 目标变量Y分布可视化
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    sns.countplot(x=y_name, data=data, ax=axs[0])  # 柱状图
    axs[0].set_title("Frequency of each TARGET")  # 每个目标的出现频率

    data[y_name].value_counts().plot(x=None, y=None, kind='pie', ax=axs[1], autopct='%1.2f%%')  # 饼图
    axs[1].set_title("Percentage of each TARGET")  # 每个目标的百分比
    plt.show()


def set_diff(set_one, set_two):
    temp_list = []
    temp_list.append(list(set(set_one) & set(set_two)))  # 交
    temp_list.append(list(set(set_one) - (set(set_two))))  # 差
    temp_list.append(list(set(set_one) ^ set(set_two)))  # 补
    return temp_list


# In[]:
# 数据切分
def data_segmentation_skf(X, y, test_size=0.3):
    n_splits_temp = int(1 / test_size)
    # StratifiedKFold用法类似Kfold，但是他是分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同。
    # StratifiedKFold 其实是 5折 交叉验证 的 分层采样： 这里用于 将原始数据集 分为 训练集 和 测试集（共5次循环，其实一次就够了）
    sss = StratifiedKFold(n_splits=n_splits_temp, random_state=None, shuffle=False)

    for train_index, test_index in sss.split(X, y):  # 每一次循环赋值 都 分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同
        print("Train:", train_index, "Train_len:", len(train_index), "Test:", test_index, "Test_len:", len(test_index))
        print(len(train_index) / len(y), len(test_index) / len(y))
        original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
        original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
        break

    return original_Xtrain, original_Xtest, original_ytrain, original_ytest


# In[]:
# ================================数据分布==============================
# In[]:
# 分类模型 连续特征 数据分布： （不能有缺失值）
def class_data_distribution(data, feature, label, axes):
    data = get_notMissing_values(data, feature)
    sns.distplot(data[feature], bins=100, color='green', ax=axes[0][0])
    axes[0][0].set_title('feature: ' + str(feature))
    axes[0][0].set_xlabel('')

    sns.boxplot(y=feature, data=data, ax=axes[0][1])
    axes[0][1].set_title('feature: ' + str(feature))
    axes[0][1].set_ylabel('')

    sns.distplot(data[feature][data[label] == 1], bins=100, color='red', ax=axes[1][0])
    sns.distplot(data[feature][data[label] == 0], bins=100, color='blue', ax=axes[1][0])
    axes[1][0].set_title('feature: ' + str(feature))
    axes[1][0].set_xlabel('')

    sns.boxplot(x=label, y=feature, data=data, ax=axes[1][1])
    axes[1][1].set_title('feature: ' + str(feature))
    axes[1][1].set_xlabel('')
    axes[1][1].set_ylabel('')


# 分类模型 连续特征 2个特征之间 散点分布：
def class_data_scatter(x_data, one_f, two_f, y, axes):
    axes.scatter(x_data[:, one_f], x_data[:, two_f], c=y, s=10, cmap="rainbow")  # 蓝色
    axes.set_xlabel(one_f)  # x轴标签
    axes.set_ylabel(two_f)  # y轴标签


# 连续/分类模型 连续特征 直方图分布： （不能有缺失值）
def con_data_distribution(data, feature, axes):
    data = get_notMissing_values(data, feature)
    sns.distplot(data[feature], bins=100, color='green', ax=axes[0])
    axes[0].set_title('feature: ' + str(feature))
    axes[0].set_xlabel('')

    sns.boxplot(y=feature, data=data, ax=axes[1])
    axes[1].set_title('feature: ' + str(feature))
    axes[1].set_ylabel('')


# 连续模型 连续特征 与 Y 散点分布：
def con_data_scatter(x_data, i, y, j, axes):
    axes.scatter(x_data[i], y, c='#0000FF', s=10, cmap="rainbow")  # 蓝色
    axes.set_xlabel(i)  # x轴标签
    axes.set_ylabel(j)  # y轴标签


# In[]:
# -----------------------------正太、偏度检测-------------------------------
# In[]:
# 正太分布检测：
'''
原假设：样本来自一个正态分布的总体。
备选假设：样本不来自一个正态分布的总体。
w和p同向： w值越小； p-值越小、接近于0； 拒绝原假设。
'''


def normal_distribution_test(data):
    from scipy import stats

    var = data.columns
    shapiro_var = {}
    for i in var:
        shapiro_var[i] = stats.shapiro(data[i])  # 返回 w值 和 p值

    shapiro = pd.DataFrame(shapiro_var).T.sort_values(by=1, ascending=False)

    fig, axe = plt.subplots(1, 1, figsize=(15, 10))
    axe.bar(shapiro.index, shapiro[0], width=.4)  # 自动按X轴---skew.index索引0-30的顺序排列
    axe.set_title("Normal distribution for shapiro")

    # 在柱状图上添加数字标签
    for a, b in zip(shapiro.index, shapiro[0]):
        # a是X轴的柱状体的索引， b是Y轴柱状体高度， '%.4f' % b 是显示值
        plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=12)
    plt.show()


'''
偏度>1 为偏斜数据，需要取log
'''


def skew_distribution_test(data):
    var = data.columns
    skew_var = {}
    for i in var:
        skew_var[i] = abs(data[i].skew())

    skew = pd.Series(skew_var).sort_values(ascending=False)

    fig, axe = plt.subplots(1, 1, figsize=(15, 10))
    axe.bar(skew.index, skew, width=.4)  # 自动按X轴---skew.index索引0-30的顺序排列
    axe.set_title("Normal distribution for skew")

    # 在柱状图上添加数字标签
    for a, b in zip(skew.index, skew):
        # a是X轴的柱状体的索引， b是Y轴柱状体高度， '%.4f' % b 是显示值
        plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=12)
    plt.show()

    return skew


def normal_comprehensive(data):
    temp_data = data.copy()

    normal_distribution_test(temp_data)
    skew = skew_distribution_test(temp_data)

    var_x_ln = skew.index[skew > 1]  # skew的索引 --- data的列名
    print(var_x_ln, len(var_x_ln))

    for i, var in enumerate(var_x_ln):
        f, axes = plt.subplots(1, 2, figsize=(23, 8))
        con_data_distribution(temp_data, var, axes)

    # 将偏度大于1的连续变量 取对数
    logarithm(temp_data, var_x_ln, 2)
    normal_distribution_test(temp_data)
    skew = skew_distribution_test(temp_data)
    var_x_ln = skew.index[skew > 1]  # skew的索引 --- data的列名
    for i, var in enumerate(var_x_ln):
        f, axes = plt.subplots(1, 2, figsize=(23, 8))
        con_data_distribution(temp_data, var, axes)


# Q-Q图检测
def normal_QQ_test(data, feature):
    import statsmodels.api as sm

    temp_data = data.copy()
    # 包实现
    fig = sm.qqplot(temp_data[feature], fit=True, line='45')
    fig.show()

    # ========================================================================

    # 手动
    mean = temp_data[feature].mean()
    std = temp_data[feature].std()
    temp_data.sort_values(by=feature, inplace=True)

    # 注意： Data.列名 是传统调用方式，但 Data.index 是特殊的，是DataFrame的索引，不是列名。
    s_r = temp_data.reset_index(drop=False)
    s_r['p'] = (s_r.index - 0.5) / len(s_r)  # 计算百分位数 p(i)
    # print(s_r['p'])
    s_r['q'] = (s_r[feature] - mean) / std  # 计算q值：Z分数

    st = temp_data[feature].describe()
    x1, y1 = 0.25, st['25%']
    x2, y2 = 0.75, st['75%']

    fig = plt.figure(figsize=(10, 9))
    ax1 = fig.add_subplot(3, 1, 1)  # 创建子图1
    # 绘制数据分布图
    ax1.scatter(temp_data.index, temp_data[feature])
    plt.grid()

    ax2 = fig.add_subplot(3, 1, 2)  # 创建子图2
    # 绘制直方图
    temp_data[feature].hist(bins=30, alpha=0.5, normed=True, ax=ax2)
    temp_data[feature].plot(kind='kde', secondary_y=True, ax=ax2)
    plt.grid()

    ax3 = fig.add_subplot(3, 1, 3)  # 创建子图3
    ax3.plot(s_r['p'], s_r[feature], 'k.', alpha=0.1)
    ax3.plot([x1, x2], [y1, y2], '-r')  # 绘制QQ图，直线为 四分之一位数、四分之三位数的连线，基本符合正态分布
    plt.grid()


# In[]:
# -----------------------------正太、偏度检测-------------------------------
# In[]:
# ================================数据分布==============================


# In[]:
# 恢复索引
def recovery_index(data_list):
    for i in data_list:
        i.index = range(i.shape[0])


# 取对数
def logarithm(X_rep, var_x_ln, f_type=1):
    if f_type == 1:
        for i in var_x_ln:
            if min(X_rep[i]) <= 0:
                X_rep[i + "_ln"] = np.log(X_rep[i] + abs(min(X_rep[i])) + 0.01)  # 负数取对数的技巧
            else:
                X_rep[i + "_ln"] = np.log(X_rep[i])
    else:
        for i in var_x_ln:
            if min(X_rep[i]) <= 0:
                X_rep[i] = np.log(X_rep[i] + abs(min(X_rep[i])) + 0.01)  # 负数取对数的技巧
            else:
                X_rep[i] = np.log(X_rep[i])


# In[]:
# 相似度计算
# 特征选择：特征共线性（还可以做 方差膨胀系数）
def corrFunction(data_corr):
    '''
    1、特征间共线性：两个或多个特征包含了相似的信息，期间存在强烈的相关关系
    2、常用判断标准：两个或两个以上的特征间的相关性系数高于0.8
    3、共线性的影响：
    3.1、降低运算效率
    3.2、降低一些模型的稳定性
    3.3、弱化一些模型的预测能力
    '''
    # 建立共线性表格（是检测特征共线性的，所以排除Y）
    correlation_table = data_corr.corr()
    # 皮尔森相似度 绝对值 排序
    df_all_corr_abs = correlation_table.abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_all_corr_abs.rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: 'Correlation_Coefficient'},
                           inplace=True)
    temp_corr_abs = df_all_corr_abs[(df_all_corr_abs["Feature_1"] != df_all_corr_abs["Feature_2"])][::2]
    #    temp_corr_abs.to_csv(r"C:\Users\dell\Desktop\123123\temp_corr_abs.csv")
    print()
    # 皮尔森相似度 排序
    df_all_corr = correlation_table.unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_all_corr.rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: 'Correlation_Coefficient'},
                       inplace=True)
    temp_corr = df_all_corr[(df_all_corr["Feature_1"] != df_all_corr["Feature_2"])][::2]
    #    temp_corr.to_csv(r"C:\Users\dell\Desktop\123123\temp_corr.csv")

    # 热力图
    temp_x = []
    for i, fe in enumerate(data_corr.columns):
        temp_x.append("x" + str(i))
    xticks = temp_x  # x轴标签
    yticks = list(correlation_table.index)  # y轴标签
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    sns.heatmap(correlation_table, annot=True, cmap='rainbow', ax=ax1,
                annot_kws={'size': 12, 'weight': 'bold', 'color': 'black'})  #
    ax1.set_xticklabels(xticks, rotation=0, fontsize=14)
    ax1.set_yticklabels(yticks, rotation=0, fontsize=14)


# 特征选择：（带 因变量Y）
def corrFunction_withY(data_corr, label):  # label： 因变量Y名称
    corr = data_corr.corr()  # 计算各变量的相关性系数
    # 皮尔森相似度 绝对值 排序
    df_all_corr_abs = corr.abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_all_corr_abs.rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: 'Correlation_Coefficient'},
                           inplace=True)
    print(df_all_corr_abs[(df_all_corr_abs["Feature_1"] != label) & (df_all_corr_abs['Feature_2'] == label)])
    print()
    # 皮尔森相似度 排序
    df_all_corr = corr.unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_all_corr.rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: 'Correlation_Coefficient'},
                       inplace=True)
    print(df_all_corr[(df_all_corr["Feature_1"] != label) & (df_all_corr['Feature_2'] == label)])

    xticks = data_corr.columns  # x轴标签
    yticks = list(corr.index)  # y轴标签
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax1,
                annot_kws={'size': 12, 'weight': 'bold', 'color': 'black'})  # 绘制相关性系数热力图
    ax1.set_xticklabels(xticks, rotation=0, fontsize=14)
    ax1.set_yticklabels(yticks, rotation=0, fontsize=14)
    plt.show()


# In[]:
# ================================线性回归特征分析==============================
# In[]:
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square

# 拟合优度
# R^2
'''
从 真实值的内部差异（方差）为出发点思考 R^2 公式：
1、mse 代表 回归值与真实值之间的（均方误差MSE）  
mse = np.sum(np.square(y_predict - y_true))  /  len(y_ture)

2、variance 表示 真实值 的 内部差异（方差）
variance = np.sum(np.square(y_true - np.mean(y_true)))  /  len(y_ture)

3、R^2
R^2  =  1 - (mse/variance)  =  1 - (RSS残差平方和 / TSS总离差平方和)
注意： mse/variance 相除之后，各自分母的 len(y_ture) 被消去，得到上述公式：分子上得到RSS残差平方和，分母上这时不能说是方差（len(y_ture)被消除了），而是TSS总离差平方和。

4、所以，消去分母 len(y_ture) 后，公式变换得到 最终 R^2 公式：
rss = np.sum(np.square(y_predict - y_true))
tss = np.sum(np.square(y_true - np.mean(y_true)))
R^2 = 1 - (rss/tss)

因此R-squared既考量了回归值与真实值的差异，也兼顾了真实值的离散程度。【模型对样本数据的拟合度】R-squared 取值范围 (-∞,1]，
值越大表示模型越拟合训练数据，最优解是1；当模型预测为随机值的时候，R^2有可能为负；若预测值恒为样本期望，R^2为0。

其中y是我们的真实标签，y^^ 是我们的预测结果， y^- 是我们的均值， 分母 ∑(y - y^-)^2  如果除以样本量m就是我们的方差。
方差的本质是任意一个y值和样本均值的差异，差异越大，这些值所带的信息越多。在R^2中，分子（RSS离差平方和）是模型没有捕获到的信息总量，
分母是真实标签所带的信息量（可以理解为方差：m被消除了），所以其衡量的是：
1 - 模型没有捕获到的信息量 占 真实标签中所带的信息量的比例  =  模型捕获到的信息量 占 真实标签中所带的信息量的比例，
所以，R^2 越接近1越好。

上述公式 R^2 = 1 - RSS/TSS 是正确公式；而 R^2 = ESS/TSS 是不完整、有缺陷的公式（不能使用）
'''


def r2_score_customize(y_true, y_predict, customize_type=1):
    if customize_type == 1:
        return r2_score(y_true, y_predict)
    else:
        rss = np.sum(np.square(y_predict - y_true))
        tss = np.sum(np.square(y_true - np.mean(y_true)))
        return 1 - (rss / tss)


# 如果解释不同解释变量的模型之间的模型优劣（样本一样）
# Adj.R^2
'''
i=1 当有截距项时，反之等于0
n=用于拟合该模型的观察值数量
p=模型中参数的个数
Adj.R^2  =  1  -  (n-i)(1-R^2) / (n-p) 
各处查询后用下面的公式：
Adj.R^2  =  1  -  (n-i)(1-R^2) / (n-p-1) 
'''


def adj_r2_customize(y_true, y_predict, coef_num, customize_type=1):
    r2_score = r2_score_customize(y_true, y_predict, customize_type)
    adj_r2 = 1 - ((len(y_true) - 1) * (1 - r2_score)) / (len(y_true) - coef_num - 1)
    return adj_r2


# 多元线性回归选择自变量指标（AIC/BIC、R^2/Adj.R^2）
# AIC： （有问题，公式都不能确定。。。）
'''
AIC指标： 多元线性回归选择自变量指标（AIC/BIC、R^2/Adj.R^2）
k = 解释变量个数
RSS = 残差的离差平方和
AIC = 2k + n(log(RSS/n)) 
'''
# def aic_customize(y_true, y_predict, coef_num):
#    rss = np.sum(np.square(y_predict - np.mean(y_true)))
#    aic = 2*coef_num + len(y_true)*(np.log(rss/len(y_true)))
#    return aic


# 残差分析：
'''
残差分析： 
残差中是否有离群值？
残差散点图是否和某个解释变量X有曲线关系？
残差的离散程度是否和某个解释变量有关？
残差 = (Ytrue - Yhat)
'''


# 1.1、扰动项ε 独立同分布 （异方差检验、DW检验）
# 1.1.1、异方差： 随着x的增大，残差呈扇面型分布，残差的方差呈放大趋势。出现在“横截面”数据中（样本是同一时间采集到的）
def heteroscedastic(X, Y, col_list):
    from statsmodels.formula.api import ols

    temp_X = X[col_list].copy()
    temp_Y = pd.DataFrame(Y, columns=['Y'])
    temp_data = pd.concat([temp_X, temp_Y], axis=1)

    formula = "Y" + '~' + '+'.join(col_list)
    print(formula)
    lm_s1 = ols(formula, data=temp_data).fit()
    print(lm_s1.rsquared, lm_s1.aic)
    temp_data['Pred'] = lm_s1.predict(temp_data)
    temp_data['resid'] = lm_s1.resid  # 残差随着x的增大呈现 喇叭口形状，出现异方差
    temp_data.plot('Pred', 'resid', kind='scatter')  # Pred = β*Income，随着预测值的增大，残差resid呈现 喇叭口形状
    print(lm_s1.summary())

    print("-" * 30)

    logarithm(temp_data, "Y")
    formula = "Y_ln" + '~' + '+'.join(col_list)
    print(formula)
    lm_s2 = ols(formula, data=temp_data).fit()
    print(lm_s2.rsquared, lm_s2.aic)
    temp_data['Pred'] = lm_s2.predict(temp_data)
    temp_data['resid'] = lm_s2.resid  # 残差随着x的增大呈现 喇叭口形状，出现异方差
    temp_data.plot('Pred', 'resid', kind='scatter')  # Pred = β*Income，随着预测值的增大，残差resid呈现 喇叭口形状
    print(lm_s2.summary())

    print("-" * 30)

    col_list_ln = [i + "_ln" for i in col_list]
    logarithm(temp_data, col_list)
    formula = "Y_ln" + '~' + '+'.join(col_list_ln)
    print(formula)
    lm_s3 = ols(formula, data=temp_data).fit()
    print(lm_s3.rsquared, lm_s3.aic)
    temp_data['Pred'] = lm_s3.predict(temp_data)
    temp_data['resid'] = lm_s3.resid  # 残差随着x的增大呈现 喇叭口形状，出现异方差
    temp_data.plot('Pred', 'resid', kind='scatter')  # Pred = β*Income，随着预测值的增大，残差resid呈现 喇叭口形状
    print(lm_s3.summary())

    r_sq = {'Y~Features': lm_s1.rsquared, 'ln(Y)~Features': lm_s2.rsquared, 'ln(Y)~ln(Features)': lm_s3.rsquared}
    return r_sq


def heteroscedastic_singe(X, Y, col):
    from statsmodels.formula.api import ols

    temp_X = X[col].copy()
    temp_Y = pd.DataFrame(Y, columns=['Y'])
    temp_data = pd.concat([temp_X, temp_Y], axis=1)

    formula = "Y" + '~' + '+' + col
    print(formula)
    lm_s1 = ols(formula, data=temp_data).fit()
    print(lm_s1.rsquared, lm_s1.aic)
    temp_data['Pred'] = lm_s1.predict(temp_data)
    temp_data['resid'] = lm_s1.resid  # 残差随着x的增大呈现 喇叭口形状，出现异方差
    temp_data.plot(col, 'resid', kind='scatter')  # Pred = β*Income，随着预测值的增大，残差resid呈现 喇叭口形状
    print(lm_s1.summary())

    print("-" * 30)

    temp_data["Y_ln"] = np.log(temp_data["Y"])
    formula = "Y_ln" + '~' + '+' + col
    print(formula)
    lm_s2 = ols(formula, data=temp_data).fit()
    print(lm_s2.rsquared, lm_s2.aic)
    temp_data['Pred'] = lm_s2.predict(temp_data)
    temp_data['resid'] = lm_s2.resid  # 残差随着x的增大呈现 喇叭口形状，出现异方差
    temp_data.plot(col, 'resid', kind='scatter')  # Pred = β*Income，随着预测值的增大，残差resid呈现 喇叭口形状
    print(lm_s2.summary())

    print("-" * 30)

    temp_data[col + "_ln"] = np.log(temp_data[col])
    formula = "Y_ln" + '~' + '+' + col + "_ln"
    print(formula)
    lm_s3 = ols(formula, data=temp_data).fit()
    print(lm_s3.rsquared, lm_s3.aic)
    temp_data['Pred'] = lm_s3.predict(temp_data)
    temp_data['resid'] = lm_s3.resid  # 残差随着x的增大呈现 喇叭口形状，出现异方差
    temp_data.plot(col + "_ln", 'resid', kind='scatter')  # Pred = β*Income，随着预测值的增大，残差resid呈现 喇叭口形状
    print(lm_s3.summary())

    r_sq = {'Y~' + col: lm_s1.rsquared, 'ln(Y)~' + col: lm_s2.rsquared, 'ln(Y)~ln(' + col + ')': lm_s3.rsquared}
    return r_sq


# 1.2、扰动项ε 服从正太分布 （QQ检验）
def disturbance_term_normal(X, Y, col_list):
    from statsmodels.formula.api import ols

    temp_X = X[col_list].copy()
    temp_Y = pd.DataFrame(Y, columns=['Y'])
    temp_data = pd.concat([temp_X, temp_Y], axis=1)

    formula = "Y" + '~' + '+'.join(col_list)
    print(formula)
    lm_s1 = ols(formula, data=temp_data).fit()
    print(lm_s1.rsquared, lm_s1.aic)
    temp_data['Pred'] = lm_s1.predict(temp_data)
    temp_data['resid'] = lm_s1.resid  # 残差随着x的增大呈现 喇叭口形状，出现异方差
    temp_data.plot('Pred', 'resid', kind='scatter')  # Pred = β*Income，随着预测值的增大，残差resid呈现 喇叭口形状
    print(lm_s1.summary())

    # 扰动项ε QQ检测
    normal_QQ_test(temp_data, "resid")


# 2.1、学生化残差： 强影响点分析
'''
学生化残差  =  (残差 - 残差均值) / 残差的标准差
样本量为几百个时：|SR| > 2 为强影响点
样本量为上千个时：|SR| > 3 为强影响点
'''


def studentized_residual(Xtrain, Ytrain):
    from statsmodels.formula.api import ols

    temp_Y = pd.DataFrame(Ytrain, columns=['Y'])
    temp_data = pd.concat([Xtrain, temp_Y], axis=1)
    cols = list(temp_data.columns)
    cols.remove("Y")
    cols_noti = cols
    formula = "Y" + '~' + '+'.join(cols_noti)

    lm_s = ols(formula, data=temp_data).fit()
    print(lm_s.rsquared, lm_s.aic)
    temp_data['Pred'] = lm_s.predict(temp_data)
    temp_data['resid'] = lm_s.resid  # 残差随着x的增大呈现 喇叭口形状，出现异方差
    temp_data.plot('Pred', 'resid', kind='scatter')  # Pred = β*Income，随着预测值的增大，残差resid呈现 喇叭口形状

    temp_data['resid_t'] = (temp_data['resid'] - temp_data['resid'].mean()) / temp_data['resid'].std()

    temp_data2 = temp_data[abs(temp_data['resid_t']) <= 3].copy()
    lm_s2 = ols(formula, temp_data2).fit()
    print(lm_s2.rsquared, lm_s2.aic)
    temp_data2['Pred'] = lm_s2.predict(temp_data2)
    temp_data2['resid'] = lm_s2.resid
    temp_data2.plot('Pred', 'resid', kind='scatter')
    lm_s2.summary()


# 2.2、强影响点分析 更多指标： statemodels包提供了更多强影响点判断指标 （太耗时，最好不要用了）
def strong_influence_point(Xtrain, Ytrain):
    from statsmodels.formula.api import ols
    from statsmodels.stats.outliers_influence import OLSInfluence

    temp_Y = pd.DataFrame(Ytrain, columns=['Y'])
    temp_data = pd.concat([Xtrain, temp_Y], axis=1)
    cols = list(temp_data.columns)
    cols.remove("Y")
    cols_noti = cols
    formula = "Y" + '~' + '+'.join(cols_noti)

    lm_s = ols(formula, data=temp_data).fit()
    OLSInfluence(lm_s).summary_frame().head()


# 解释变量X 之间不能强线性相关 （膨胀系数）
# 3、方差膨胀因子
'''
3.1、两个特征检验：
['Income_ln', 'dist_avg_income_ln']，得到：
Income_ln           36.439462480880216
dist_avg_income_ln  36.439462480880216
> 10 表示 该变量多重共线性严重 （两个特征检验，一定是都大于或都小于）

3.2、三个特征检验：
['Income_ln', 'dist_home_val_ln','dist_avg_income_ln']，得到：
Income_ln           36.653639058963186
dist_home_val_ln    1.053596313570258
dist_avg_income_ln  36.894876856102

Income_ln ~ dist_home_val_ln + dist_home_val_ln > 10  说明Income_ln 与 这两特征 存在多重共线性
dist_avg_income_ln ~ Income_ln + dist_home_val_ln > 10  说明dist_avg_income_ln 与 这两特征 存在多重共线性
dist_home_val_ln<10 ~ Income_ln + dist_avg_income_ln < 10  说明dist_home_val_ln 与 这两特征 不存在多重共线性
所以 Income_ln 和 dist_avg_income_ln 存在多重共线性严重。

如果 特征更多，则需要 选多个特征筛选之后，再进一步小范围筛选。
'''


def vif(df, col_i):
    from statsmodels.formula.api import ols

    cols = list(df.columns)
    cols.remove(col_i)
    cols_noti = cols
    formula = col_i + '~' + '+'.join(cols_noti)
    print(formula)
    r2 = ols(formula, df).fit().rsquared
    return 1. / (1. - r2), formula


def variance_expansion_coefficient(df, cols):
    temp_df = df[cols].copy()
    temp_dict = {}
    temp_dict_ln = {}

    for i in temp_df.columns:
        temp_v = vif(df=temp_df, col_i=i)
        temp_dict[temp_v[1]] = temp_v[0]
        print(i, '\t', temp_v[0])
        print()

    print("-" * 30)

    logarithm(temp_df, temp_df.columns)
    col_list_ln = [i + "_ln" for i in cols]
    temp_df = temp_df[col_list_ln]
    for i in temp_df.columns:
        temp_v = vif(df=temp_df, col_i=i)
        temp_dict_ln[temp_v[1]] = temp_v[0]
        print(i, '\t', temp_v[0])
        print()

    return temp_dict, temp_dict_ln


# 训练集 与 测试集 拟合：
def fitting_comparison(y_true, y_predict):
    plt.plot(range(len(y_true)), sorted(y_true), c="black", label="Data")
    plt.plot(range(len(y_predict)), sorted(y_predict), c="red", label="Predict")
    plt.legend()
    plt.show()


# 线性回归模型 泛化能力：
def linear_model_comparison(X, y, cv_customize=5, start=1, end=1001, step=100, linear_show=True):
    from sklearn.linear_model import LinearRegression as LR, Ridge, Lasso
    from sklearn.model_selection import cross_val_score

    alpharange = np.arange(start, end, step)
    linear_r2_scores, ridge_r2_scores = [], []
    linear_r2var_scores, ridge_r2var_scores = [], []
    linear_ge, ridge_ge = [], []
    for alpha in alpharange:
        linear = LR()
        ridge = Ridge(alpha=alpha)

        linear_score = cross_val_score(linear, X, y, cv=cv_customize, scoring="r2")
        ridge_score = cross_val_score(ridge, X, y, cv=cv_customize, scoring="r2")

        # 因 R^2=(-∞,1]， R^2拟合优度： 模型捕获到的信息量 占 真实标签中所带的信息量的比例
        # 1 - R^2均值 = 偏差， 所以用 R^2均值 代表偏差（R^2均值越小，偏差越大； R^2均值越大，偏差越小）
        # 偏差：交叉验证 的 R^2均值：不同训练集训练出多个模型 分别预测不同测试集得到多个预测值集合 --- 多个R^2拟合优度， 多个R^2拟合优度 的 均值： 不同模型R^2拟合优度的准确性
        ridge_r2_score = ridge_score.mean()  # R^2均值 代表 偏差
        linear_r2_score = linear_score.mean()
        title_mean = "R2_Mean"
        ridge_r2_scores.append(ridge_r2_score)
        linear_r2_scores.append(linear_r2_score)

        # 方差：交叉验证 的 R^2方差：不同训练集训练出多个模型 分别预测不同测试集得到多个预测值集合 --- 多个R^2拟合优度， 多个R^2拟合优度 的 方差： 不同模型R^2拟合优度的离散程度
        ridge_r2var_score = ridge_score.var()  # R^2 方差
        linear_r2var_score = linear_score.var()
        title_var = "R2_Var"
        ridge_r2var_scores.append(ridge_r2var_score)
        linear_r2var_scores.append(linear_r2var_score)

        # 计算泛化误差的可控部分
        ridge_ge.append((1 - ridge_r2_score) ** 2 + ridge_r2var_score)
        linear_ge.append((1 - linear_r2_score) ** 2 + linear_r2var_score)

    maxR2_Alpha, maxR2 = alpharange[ridge_r2_scores.index(max(ridge_r2_scores))], max(ridge_r2_scores)
    start_Alpha, start_R2 = alpharange[0], ridge_r2_scores[0]
    diff_R2 = maxR2 - start_R2
    print("R^2起始阈值%f:R^2起始值%f，R^2最大值阈值%f:R^2最大值%f，R^2差值%f" % (start_Alpha, start_R2, maxR2_Alpha, maxR2, diff_R2))

    # 当R^2最大值时，求 R^2方差Var的最大值，用R^2的变化差值 与 R^2方差的变化差值 再进行比较
    start_R2VaR = ridge_r2var_scores[0]
    R2VarR_Index = alpharange.tolist().index(maxR2_Alpha)
    R2varR = ridge_r2var_scores[R2VarR_Index]
    diff_R2varR = R2varR - start_R2VaR
    print("R^2方差起始阈值%f:R^2方差起始值%f，R^2方差对应阈值%f:R^2方差对应最大值%f，R^2方差差值%f" % (
    alpharange[0], start_R2VaR, maxR2_Alpha, R2varR, diff_R2varR))
    print("R^2方差差值/R^2差值 = %f" % (diff_R2varR / diff_R2))

    # 1、打印R2最高所对应的参数取值； 2、并打印这个参数下的R2； 3、并打印这个参数下的R2方差
    print(alpharange[ridge_r2_scores.index(max(ridge_r2_scores))], max(ridge_r2_scores),
          ridge_r2var_scores[ridge_r2_scores.index(max(ridge_r2_scores))])
    # 1、打印R2方差最低时对应的参数取值； 2、并打印这个参数下的R2； 3、并打印这个参数下的R2方差
    print(alpharange[ridge_r2var_scores.index(min(ridge_r2var_scores))],
          ridge_r2_scores[ridge_r2var_scores.index(min(ridge_r2var_scores))], min(ridge_r2var_scores))
    # 1、打印泛化误差可控部分的参数取值； 2、并打印这个参数下的R2； 3、并打印这个参数下的R2方差
    print(alpharange[ridge_ge.index(min(ridge_ge))], ridge_r2_scores[ridge_ge.index(min(ridge_ge))],
          ridge_r2var_scores[ridge_ge.index(min(ridge_ge))], min(ridge_ge))

    plt.figure(figsize=(10, 8))
    plt.plot(alpharange, ridge_r2_scores, color="red", label="Ridge")
    if linear_show:
        plt.plot(alpharange, linear_r2_scores, color="orange", label="LR")
    plt.title(title_mean)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(alpharange, ridge_r2var_scores, color="red", label="Ridge")
    if linear_show:
        plt.plot(alpharange, linear_r2var_scores, color="orange", label="LR")
    plt.title(title_var)
    plt.legend()
    plt.show()

    # R2_Var值非常小，从0.0038→0.0045平稳缓慢增加，所以看不出变宽的痕迹。
    plt.figure(figsize=(10, 8))
    plt.plot(alpharange, ridge_r2_scores, c="k", label="R2_Mean")
    plt.plot(alpharange, ridge_r2_scores + np.array(ridge_r2var_scores), c="red", linestyle="--", label="R2_Var")
    plt.plot(alpharange, ridge_r2_scores - np.array(ridge_r2var_scores), c="red", linestyle="--")
    plt.legend()
    plt.title("R2_Mean vs R2_Var")
    plt.show()

    # 绘制 化误差的可控部分
    plt.figure(figsize=(10, 8))
    plt.plot(alpharange, ridge_ge, c="gray", linestyle='-.')
    plt.title("Generalization error")
    plt.show()


# In[]:
# ================================线性回归特征分析==============================


# In[]:
# ====================================学习曲线==================================
# In[]:
# -----------------------------1、基于样本量-------------------------------
# 基于MSE绘制学习曲线（样本量）
def plot_learning_curve_mse_customize(algo, X_train, X_test, y_train, y_test):
    train_score = []
    test_score = []
    for i in range(1, len(X_train) + 1):
        algo.fit(X_train[:i], y_train[:i])

        y_train_predict = algo.predict(X_train[:i])
        train_score.append(mean_squared_error(y_train[:i], y_train_predict))

        y_test_predict = algo.predict(X_test)
        test_score.append(mean_squared_error(y_test, y_test_predict))

    plt.plot([i for i in range(1, len(X_train) + 1)],
             np.sqrt(train_score), label="train")
    plt.plot([i for i in range(1, len(X_train) + 1)],
             np.sqrt(test_score), label="test")
    plt.legend()
    plt.show()


# 基于R^2值绘制学习曲线（样本量）
def plot_learning_curve_r2_customize(algo, X_train, X_test, y_train, y_test):
    train_score = []
    test_score = []
    for i in range(1, len(X_train) + 1):
        algo.fit(X_train[:i], y_train[:i])

        y_train_predict = algo.predict(X_train[:i])
        train_score.append(r2_score(y_train[:i], y_train_predict))

        y_test_predict = algo.predict(X_test)
        test_score.append(r2_score(y_test, y_test_predict))

    plt.plot([i for i in range(1, len(X_train) + 1)],
             train_score, label="train")
    plt.plot([i for i in range(1, len(X_train) + 1)],
             test_score, label="test")
    plt.legend()
    plt.axis([0, len(X_train) + 1, -0.1, 1.1])
    plt.show()


# 基于learning_curve函数：
def plot_learning_curve(estimator, title, X, y, scoring=None,
                        ax=None,  # 选择子图
                        ylim=None,  # 设置纵坐标的取值范围
                        cv=None,  # 交叉验证
                        n_jobs=None  # 设定索要使用的线程
                        ):
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np

    # learning_curve如果不显示设置scoring，则会按照模型默认的指标返回
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y
                                                            , scoring=scoring
                                                            #                                                            ,shuffle=True
                                                            , cv=cv
                                                            , random_state=420
                                                            , n_jobs=n_jobs)
    if ax == None:
        ax = plt.gca()
    else:
        ax = plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid()  # 绘制网格，不是必须
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-'
            , color="r", label="Training score")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-'
            , color="g", label="Test score")
    ax.legend(loc="best")
    return ax


# In[]:
# -----------------------------1、基于样本量-------------------------------

# In[]:
# -----------------------------2、基于超参数-------------------------------
def getModel(i, model_name, hparam_name, prev_hparam_value, random_state):
    from xgboost import XGBRegressor as XGBR

    if model_name == "XGBR":
        if hparam_name == "n_estimators":
            reg = XGBR(n_estimators=i, random_state=random_state)
        elif hparam_name == "subsample" and prev_hparam_value is not None:
            reg = XGBR(n_estimators=prev_hparam_value, subsample=i, random_state=random_state)
        else:
            raise RuntimeError('Hparam Error')
    return reg


def learning_curve_r2_customize(axisx, Xtrain, Ytrain, cv, model_name="XGBR", hparam_name="n_estimators",
                                prev_hparam_value=None, random_state=420):
    rs = []
    var = []
    ge = []
    for i in axisx:
        reg = getModel(i, model_name, hparam_name, prev_hparam_value, random_state)
        cvresult = CVS(reg, Xtrain, Ytrain, cv=cv)
        # 记录1-偏差
        rs.append(cvresult.mean())
        # 记录方差
        var.append(cvresult.var())
        # 计算泛化误差的可控部分
        ge.append((1 - cvresult.mean()) ** 2 + cvresult.var())
    # 1、打印R2最高所对应的参数取值； 2、并打印这个参数下的R2； 3、并打印这个参数下的R2方差
    print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])
    # 1、打印R2方差最低时对应的参数取值； 2、并打印这个参数下的R2； 3、并打印这个参数下的R2方差
    print(axisx[var.index(min(var))], rs[var.index(min(var))], min(var))
    # 1、打印泛化误差可控部分的参数取值； 2、并打印这个参数下的R2； 3、并打印这个参数下的R2方差
    print(axisx[ge.index(min(ge))], rs[ge.index(min(ge))], var[ge.index(min(ge))], min(ge))

    rs = np.array(rs)
    var = np.array(var)
    plt.figure(figsize=(20, 5))
    plt.plot(axisx, rs, c="black", label=model_name)
    # 添加 方差
    plt.plot(axisx, rs + var, c="red", linestyle='-.')
    plt.plot(axisx, rs - var, c="red", linestyle='-.')
    plt.legend()
    plt.show()

    # 绘制 化误差的可控部分
    #    plt.figure(figsize=(20,5))
    #    plt.plot(axisx,ge,c="gray",linestyle='-.')
    #    plt.show()
    '''
    270 0.8628488903325771 0.0010233348954168013
    170 0.8590591457326957 0.0009745514733593459
    270 0.8628488903325771 0.0010233348954168013 0.019833761778422276
    最后选择 n_estimators=270 的超参数
    '''

# In[]:
# -----------------------------2、基于超参数-------------------------------
# In[]:
# ====================================学习曲线==================================
