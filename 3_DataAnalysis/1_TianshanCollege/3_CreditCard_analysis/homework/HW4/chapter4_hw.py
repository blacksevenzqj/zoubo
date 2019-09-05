
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
os.chdir(r'E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\1_TianshanCollege\3_CreditCard_analysis\homework\HW4')

# # 使用auto_ins作如下分析
# ### 1、首先对loss重新编码为1/0，有数值为1，命名为loss_flag
# In[132]:
auto = pd.read_csv(r'auto_ins.csv',encoding = 'gbk')

# In[134]:
def codeMy(x):
    if x>0:
        return 1
    else:
        return 0
    
auto["loss_flag"]= auto.Loss.map(codeMy)
#%%
#auto["loss_flag1"]= auto.Loss.map(lambda x: 1 if x >0 else 0)


# In[116]:
# ###2、对loss_flag分布情况进行描述分析
auto.loss_flag.value_counts()
# In[117]:
auto.loss_flag.value_counts()/auto.Loss.count()
# In[118]:
#auto.loss_flag.value_counts().plot(kind='bar')
auto.loss_flag.value_counts().plot(kind='pie')


# In[116]:
# ### 3、分析是否出险和年龄、驾龄、性别、婚姻状态等变量之间的关系
# In[119]:
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
#是否出险和年龄
sns.boxplot(x = 'loss_flag',y = 'Age',data = auto, ax = ax1)
#是否出险和驾龄
sns.boxplot(x = 'loss_flag',y = 'exp',data = auto, ax = ax2)

# In[120]:
#是否出险和性别
from stack2dim import *
stack2dim(auto,'Gender','loss_flag')
#stack2dim(auto,'loss_flag','Gender')

# In[121]:
#是否出险和婚姻状态
stack2dim(auto,'Marital','loss_flag')
# In[126]:



