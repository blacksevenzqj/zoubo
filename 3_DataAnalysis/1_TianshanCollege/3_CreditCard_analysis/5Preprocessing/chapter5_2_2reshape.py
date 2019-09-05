
# coding: utf-8

# 第5章 数据整合和数据清洗
# pandas学习参考： [十分钟搞定pandas](http://www.cnblogs.com/chaosimple/p/4153083.html)

# 5.1　数据整合
# 5.1.1 行列操作 
# 1. 单列
# In[7]:
import pandas as pd
import numpy as np

#%%
# 拆分、堆叠列
# In[3]:
import pandas as pd
table = pd.DataFrame({'cust_id':[10001,10001,10002,10002,10003],
                      'type':['Normal','Special_offer',\
                              'Normal','Special_offer','Special_offer'],
                      'Monetary':[3608,420,1894,3503,4567]})

# In[24]:
table1 = pd.pivot_table(table,index='cust_id',columns='type',values='Monetary')

# In[25]:
table2 = pd.pivot_table(table,index='cust_id',columns='type',values='Monetary',fill_value=0,aggfunc='sum')

# In[27]:
table3 = pd.pivot_table(table,index='cust_id',
                        columns='type',
                        values='Monetary',
                        fill_value=0,
                        aggfunc=np.sum).reset_index()

# In[28]:
table4 = pd.melt(table3,
	 id_vars='cust_id',
    value_vars=['Normal','Special_offer'],
    value_name='Monetary',
    var_name='TYPE')


