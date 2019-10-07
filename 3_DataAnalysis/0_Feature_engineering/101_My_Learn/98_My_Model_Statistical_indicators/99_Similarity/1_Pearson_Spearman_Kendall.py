import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

os.chdir(r"E:\code\python_workSpace\idea_space\zoubo\3_DataAnalysis\0_Feature_engineering\101_My_Learn\98_My_Model_Statistical_indicators\100_Inferential_statistics\1_Normal_distribution")


df = pd.read_csv("countries of the world.csv")
print(df.columns)

df = df.dropna()
df = df.iloc[:,2:].apply(lambda x: x.astype(str).str.replace(',','.').astype(float))


# 检验人均GDP和手机使用率的相关性
sns.regplot(x='GDP ($ per capita)', y='Phones (per 1000)', data=df)
plt.show()

'''
看起来有些像是线性关系，但是方差随着变量的值有所变动，看起来并不是同方差。另外，我们得检验一下，两个变量是不是接近正态分布的。
Scipy.stats中有多个方法可以用来检验正态分布，比如normaltest() 、shapiro()、kstest(rvs='norm')等，这里我们选用shapiro()，
分别检验各国人均GDP和手机使用率是否符合正态分布。
原假设：样本来自一个正态分布的总体。
备选假设：样本不来自一个正态分布的总体。
'''
print(stats.shapiro(df['GDP ($ per capita)']))
# (0.8052586317062378, 3.5005310282387736e-14)
print(stats.shapiro(df['Phones (per 1000)']))
# (0.8678628206253052, 2.0484371143769664e-11)
# 返回的结果是一个包含统计量w和p-值的元组。可以看到，p-值非常小，接近于0，于是可以拒绝原假设。
# 我们认为各国人均GDP和手机使用率都不符合正态分布。


# 用Pandas计算相关系数
'''
低度相关：0 <= |r| <= 0.3
中度相关：0.3 <= |r| <= 0.8
高度相关：0.8 <= |r| <= 1
'''
# 因 各国人均GDP和手机使用率都不符合正态分布，所以 不适用皮尔森相似度pearson
print(df['GDP ($ per capita)'].corr(df['Phones (per 1000)'], method='pearson'))
# 0.88352010541116632
print(df['GDP ($ per capita)'].corr(df['Phones (per 1000)'], method='spearman'))
# 0.90412918508969042
print(df['GDP ($ per capita)'].corr(df['Phones (per 1000)'], method='kendall'))
# 0.72385173233005073


# 用Scipy.stats计算相关系数
# 原假设：线性无关
# 备选假设：线性相关
print(stats.pearsonr(df['GDP ($ per capita)'], df['Phones (per 1000)']))
# (0.88352010541116643, 3.3769381277913882e-60) 相关系数，PV值
print(stats.spearmanr(df['GDP ($ per capita)'], df['Phones (per 1000)']))
# SpearmanrResult(correlation=0.90412918508969042, pvalue=2.8375903612871671e-67)
print(stats.kendalltau(df['GDP ($ per capita)'], df['Phones (per 1000)']))
# KendalltauResult(correlation=0.72385173233005073, pvalue=1.3086853817834578e-46)