# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 07:39:20 2018

@author: Ben
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 23:26:04 2018

@author: Ben
"""
from VarSelec import Var_Select,Var_Select_auto

#Var_Select_auto(orgdata, alphaMax=100, alphastep=0.2,eig_csum_retio=0.95,eigVals_min=0.6)
#Var_Select(orgdata, k,alphaMin=0.1, alphaMax=200, alphastep=0.2)
# In[66]:
import os
os.chdir(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\TianshanCollege\8_Charity_precision_marketing_case\13Dimensionality_reduction")
import pandas as pd
model_data = pd.read_csv("profile_bank.csv")
data = model_data.ix[ :,'CNT_TBM':'CNT_CSC']


# In[67]:
Varseled_data = Var_Select(data,k=3)
print(Varseled_data.head())

#%%
Varseled_Auto_data=Var_Select_auto(data)
#%%

















#%%
