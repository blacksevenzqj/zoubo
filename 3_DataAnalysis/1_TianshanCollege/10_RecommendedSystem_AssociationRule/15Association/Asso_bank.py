# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 04:05:17 2018

@author: Ben
"""


# coding: utf-8

# # 关联规则介绍

# In[ ]:


import pandas as pd
#from Apriori import *
import Apriori as apri
import matplotlib.pyplot as plt


# ## 数据载入

# 原数据为倒排表数据

# In[ ]:


inverted=pd.read_csv(r'D:\Python_Training\script_Python\15Association\bank.csv',encoding='gbk')
inverted.head()


# ## 数据转换

# 倒排表数据转换为相应的二维列表数据

# In[ ]:


idataset=apri.dataconvert(inverted,tidvar='CSR_ID',itemvar='PROD',data_type = 'inverted')
idataset[:5]


# ## 关联规则

# 参数说明:
# 
# + minSupport:最小支持度阈值
# + minConf:最小置信度阈值
# + minlen:规则最小长度
# + maxlen:规则最大长度

# 这里，minSupport或minConf设定越低，产生的规则越多，计算量也就越大
# 
# 设定参数为:minSupport=0.05,minConf=0.5,minlen=1,maxlen=10

# In[ ]:


res = apri.arules(idataset,minSupport=0.05,minConf=0.5,minlen=1,maxlen=10)


# ## 产生关联规则

# + 规定提升度要大于1,并按照置信度进行排序

# In[ ]:


res.ix[res.lift>1,:].sort_values('support',ascending=False).head(20)


# ## 关联规则结果汇总

# In[ ]:


res.plot.scatter(3,4,c=5,figsize=(4,4))
plt.xlabel('support')
plt.ylabel('confidence')

#%%
