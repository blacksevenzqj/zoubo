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