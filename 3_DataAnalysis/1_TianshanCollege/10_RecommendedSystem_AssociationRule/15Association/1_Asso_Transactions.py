
# coding: utf-8

# # 关联规则介绍
# In[ ]:
import pandas as pd
#from Apriori import *
import Apriori as apri
import matplotlib.pyplot as plt
import os

os.chdir(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\1_TianshanCollege\10_RecommendedSystem_AssociationRule\15Association")

# 数据载入
# 原数据为倒排表数据
# In[ ]:
#Transactions---自行车及周边物品的销售数据
inverted = pd.read_csv("Transactions.csv")
inverted.head()

# 数据转换
# 倒排表数据转换为相应的二维列表数据
# In[ ]:
idataset = apri.dataconvert(inverted,tidvar='OrderNumber',itemvar='Model',data_type='inverted')
idataset[:5]
        
# In[]:
# ## 关联规则
# 参数说明:
# 
# + minSupport:最小 支持度 阈值 （必须 ≥ 该阈值）
# + minConf:最小 置信度 阈值 （必须 ≥ 该阈值）
# + minlen:规则 最小长度 （必须 ≥ 该阈值： 其中minlen = 1或2 时 效果相同）
# + maxlen:规则 最大长度 （必须 ≤ 该阈值）

# 这里 minSupport 或 minConf设定越低，产生的规则越多，计算量也就越大
# 设定参数为：minSupport=0.05,minConf=0.5,minlen=1,maxlen=10
# In[ ]:
res = apri.arules(idataset,minSupport=0.01,minConf=0.1,minlen=2,maxlen=2) # DataFrame
print(type(res))

# ## 产生关联规则
# + 规定 提升度 要大于1,并按照 置信度 进行排序
# In[ ]:
res[res.lift>1].sort_values('support',ascending=False).head(20)

# ## 关联规则结果汇总
# In[ ]:
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.scatter.html
res.plot.scatter(3,4,c=5,figsize=(4,4))
plt.xlabel('support')
plt.ylabel('confidence')

#%%
# 互补品
res.loc[res.lift>1,['lhs','rhs','lift']].sort_values('lift',ascending=False).head(20)

#%%
# 互斥品
res.loc[res.lift<1,['lhs','rhs','lift']].sort_values('lift',ascending=True).head(20)

#%%
# 如果一个新客户刚刚下单了Mountain-200这个产品,如果希望获得最高的营销响应率,那在他的付费成功页面上最应该推荐什么产品?
# 1、如果只是为了提高 营销响应率： 则选择 置信度高的
Mountain_200 = res.loc[res.lhs==frozenset({'Mountain-200'}),:] # 左手规则

res.loc[res.lhs==frozenset({'Mountain-200'}),['lhs','rhs','support','confidence','lift']].sort_values('confidence',ascending=False).head(20)

#%%
# 如果一个新客户刚刚下单了Mountain-200这个产品,如果希望最大化提升总体的产品销售额,那在他的付费成功页面上最应该推荐什么产品?
# 2、但有时 置信度 高的 已经是畅销产品，不用推荐。 这时我们推荐 提升度 高的产品： 相对小众一些的产品（相对于 置信度高 的产品而言） 对整体的销售有利。
Mountain_200 = res.loc[res.lhs==frozenset({'Mountain-200'}),:]

res.loc[res.lhs==frozenset({'Mountain-200'}),['lhs','rhs','support','confidence','lift']].sort_values('lift',ascending=False).head(20)

#%%
# 如果希望推荐Sport-100自行车，应该如何制定营销策略？
'''
置信度 或 提升度 按倒序选择都可以， 因为被推荐的产品已定（右手规则已定），那么：
1、置信度（条件概率）是定值； 
2、提升度也是定值，因为 提示度 公式中的 无条件概率（被推荐商品概率）已定。
3、所以 置信度 和 提升度 的值有相同的趋势。
'''
Sport_100 = res.loc[res.rhs==frozenset({'Sport-100'}),:] # 右手规则

res.loc[res.rhs==frozenset({'Sport-100'}),['lhs','support','confidence','lift']].sort_values('lift',ascending=False).head(20)




#%%


















