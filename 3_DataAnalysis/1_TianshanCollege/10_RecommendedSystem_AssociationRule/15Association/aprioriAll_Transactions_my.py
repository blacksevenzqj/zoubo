# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 20:22:42 2019

@author: dell
"""

# ### 应用
# In[ ]:
import os, sys
os.chdir(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\TianshanCollege\10_RecommendedSystem_AssociationRule\15Association")

import itertools
import pandas as pd

sys.path.append('./myscripts')

from aprioriAll import aprioriAll

transactions = pd.read_csv("Transactions.csv")


# In[ ]:
'''
Sort phase: The database is sorted, with customer-id as the major key 
and transaction-time as the minor key. This step implicitly 
converts the original transaction database into a database of 
customer sequences.
'''
# In[ ]:
def aggFunc(*args):
    agg = itertools.chain(*args)
    return list(agg)

baskets = transactions['Model'].groupby([transactions['OrderNumber'], transactions['LineNumber']]).apply(aggFunc)
baskets.head()
# 必须要像上面这种分组，下面这个分组不能达到效果
#baskets2 = transactions[['OrderNumber','LineNumber','Model']].groupby(['OrderNumber','LineNumber']).apply(aggFunc)

# In[ ]:
dataSet = list(baskets.groupby(level=0).apply(list))
dataSet[:3]

# In[ ]:
seq = aprioriAll(dataSet, min_support=0.04)

#%%











