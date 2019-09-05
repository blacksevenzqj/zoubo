
# coding: utf-8

# # 第5章3 RFM
# - pandas学习参考： [十分钟搞定pandas](http://www.cnblogs.com/chaosimple/p/4153083.html)

# ### 1. 导入数据

# In[5]:

import pandas as pd
import numpy as np
trad_flow = pd.read_csv('F:\script\RFM_TRAD_FLOW.csv', encoding='gbk')
trad_flow.head(10)


# ### 2.计算 RFM

# In[6]:

M=trad_flow.groupby(['cumid','type'])[['amount']].sum()


# In[7]:

M_trans=pd.pivot_table(M,index='cumid',columns='type',values='amount')


# In[8]:

F=trad_flow.groupby(['cumid','type'])[['transID']].count()
F.head()


# In[9]:

F_trans=pd.pivot_table(F,index='cumid',columns='type',values='transID')
F_trans.head()


# In[10]:

R=trad_flow.groupby(['cumid','type'])[['time']].max()
R.head()


# In[11]:

#R_trans=pd.pivot_table(R,index='cumid',columns='type',values='time')
#R_trans.head()


# ### 3.衡量客户对打折商品的偏好

# In[12]:

M_trans['Special_offer']= M_trans['Special_offer'].fillna(0)


# In[13]:

M_trans['spe_ratio']=M_trans['Special_offer']/(M_trans['Special_offer']+M_trans['Normal'])
M_rank=M_trans.sort_values('spe_ratio',ascending=False,na_position='last').head()


# In[16]:

M_rank['spe_ratio_group'] = pd.qcut( M_rank['spe_ratio'], 4) # 这里以age_oldest_tr字段等宽分为4段
M_rank.head()


# In[ ]:




