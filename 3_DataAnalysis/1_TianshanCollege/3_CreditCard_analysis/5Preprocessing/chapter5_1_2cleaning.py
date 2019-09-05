
# coding: utf-8

# # 第5章 数据整合和数据清洗
# ## 5.4 数据清洗
# 发现数据问题类型
# In[1]:
import pandas as pd
import os 
import numpy as np
import matplotlib.pyplot as plt

os.chdir(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\1_TianshanCollege\3_CreditCard_analysis\5Preprocessing")
camp = pd.read_csv('teleco_camp_orig.csv')
camp.head()

# In[2]:
# 1、脏数据或数据不正确
#%%
plt.hist(camp['AvgIncome'], bins=20, normed=True) #查看分布情况：可以看出存在缺失值
camp['AvgIncome'].describe(include='all')

# In[3]:
plt.hist(camp['AvgHomeValue'], bins=20, normed=True) #查看分布情况：可以看出存在缺失值
camp['AvgHomeValue'].describe(include='all')

# In[4]:
# 这里的0值应该是缺失值
camp['AvgIncome']=camp['AvgIncome'].replace({0: np.NaN})
# 像这种外部获取的数据要比较小心，经常出现意义不清晰 或 错误值。 AvgHomeValue也有这种情况
plt.hist(camp['AvgIncome'], bins=20, normed=True,range=(camp.AvgIncome.min(),camp.AvgIncome.max()))#由于数据中存在缺失值,需要指定绘图的值域
camp['AvgIncome'].describe(include='all')

# In[5]:
camp['AvgHomeValue']=camp['AvgHomeValue'].replace({0: np.NaN})
plt.hist(camp['AvgHomeValue'], bins=20, normed=True,range=(camp.AvgHomeValue.min(),camp.AvgHomeValue.max()))#由于数据中存在缺失值,需要指定绘图的值域
camp['AvgHomeValue'].describe(include='all')



# In[6]:
# 2、数据不一致-
# 这个问题需要详细的结合描述统计进行变量说明核对
# 2.1、数据重复
# In[ ]:
camp['dup'] = camp.duplicated() # 生成重复标识变量，就是哑变量列
camp.dup.head()

# In[ ]:
# 本数据没有重复记录，此处只是示例
camp_dup = camp[camp['dup'] == True] # 把有重复的数据保存出来，以备核查
camp_nodup = camp[camp['dup'] == False] # 注意与camp.drop_duplicates()的区别
camp_nodup.head()

# In[ ]:
camp['dup1'] = camp['ID'].duplicated() # 按照主键进行重复记录标识
# accepts['fico_score'].duplicated() # 没有实际意义
camp.head()


# 2.2、缺失值处理
# In[ ]:
camp.describe()
# 如果count数量少于样本量，说明存在缺失
# 缺失最多的两个变量是Age和AvgIncome,缺失了大概20%。

# In[ ]:
# Age 和 Age_empflag 同时进模型，让模型来选择取舍
vmean = camp['Age'].mean(axis=0, skipna=True)
camp['Age_empflag'] = camp['Age'].isnull()
camp['Age']= camp['Age'].fillna(vmean)
camp['Age'].describe()

# In[ ]:
# AvgHomeValue 和 AvgHomeValue_empflag 同时进模型，让模型来选择取舍
vmean = camp['AvgHomeValue'].mean(axis=0, skipna=True)
camp['AvgHomeValue_empflag'] = camp['AvgHomeValue'].isnull() # 
camp['AvgHomeValue']= camp['AvgHomeValue'].fillna(vmean)
camp['AvgHomeValue'].describe()

# In[ ]:
# AvgIncome 和 AvgIncome_empflag 同时进模型，让模型来选择取舍
vmean = camp['AvgIncome'].mean(axis=0, skipna=True)
camp['AvgIncome_empflag'] = camp['AvgIncome'].isnull()
camp['AvgIncome']= camp['AvgIncome'].fillna(vmean)
camp['AvgIncome'].describe()

# - 其他有缺失变量请自行填补，找到一个有缺失的分类变量，使用众数进行填补
# - 多重插补：sklearn.preprocessing.Imputer仅可用于填补均值、中位数、众数，多重插补可考虑使用Orange、impute、Theano等包
# - 多重插补的处理有两个要点：1、被解释变量有缺失值的观测不能填补，只能删除；2、只对放入模型的解释变量进行插补。


# 2.3、噪声值处理
# 2.3.1、盖帽法
# In[ ]:
def blk(floor, root): # 'blk' will return a function
    def f(x):       
        if x < floor:
            x = floor
        elif x > root:
            x = root
        return x
    return f

q1 = camp['Age'].quantile(0.01) # 计算百分位数
q99 = camp['Age'].quantile(0.99)
blk_tot = blk(floor=q1, root=q99) # 'blk_tot' is a function
camp['Age']= camp['Age'].map(blk_tot)
camp['Age'].describe()


# 2.3.2、分箱（等深，等宽）
# 2.3.2.1、分箱法——等深分箱
# In[ ]:
camp['Age_group1'] = pd.qcut( camp['Age'], 4) # 等深分箱，按相同样本数量分4箱
camp['Age_group1'].value_counts()

# 2.3.2.2、分箱法——等宽分箱
# In[ ]:
camp['Age_group2'] = pd.cut( camp['Age'], 4) # 等宽分箱，按值范围分4箱
camp['Age_group2'].value_counts()

# In[ ]:
#camp.to_csv('tele_camp_ok.csv')





