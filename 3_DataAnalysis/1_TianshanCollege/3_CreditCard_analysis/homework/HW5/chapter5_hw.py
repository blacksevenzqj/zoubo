# -*- coding: utf-8 -*-
"""
Created on Sat May 19 08:36:14 2018

@author: ben
"""
import os
import pandas as pd
os.chdir(r'E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\1_TianshanCollege\3_CreditCard_analysis\homework\HW5')

# In[ ]:
import sqlite3 # sqlite3相当于轻量版，更多功能可使用SQLAlchemy
con = sqlite3.connect(':memory:') # 数据库连接

#%%
card=pd.read_csv(r"card.csv",encoding="gbk")
disp=pd.read_csv(r"disp.csv",encoding="gbk")
clients=pd.read_csv(r"clients.csv",encoding="gbk")

card.to_sql('card', con)
disp.to_sql('disp', con)
clients.to_sql('clients', con)

#%%
car_sql = '''
            select a.*,c.sex,c.birth_date,c.district_id
              from card as a
              left join disp as b on a.disp_id=b.disp_id
              left join clients as c on b.client_id=c.client_id
              where b.type="所有者"
          '''

card_t = pd.read_sql(car_sql, con)

#%%
# 设置
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

#%% 
# 1、发卡趋势
# 总体发卡趋势
from  datetime  import  *  
card_t['issued_date']=pd.to_datetime(card_t['issued'])
card_t['issued_year']=card_t['issued_date'].map(lambda x:x.year)

card_t.card_id.groupby(card_t['issued_year']).count().plot(kind="bar")

#%%
# 2、不同卡的分布
card_t.type.value_counts().plot(kind="pie",autopct='%.1f%%')

#%%
# 分类型发卡趋势
#https://blog.csdn.net/roguesir/article/details/78178365
pd.crosstab(card_t.issued_year,card_t.type).plot(kind = 'bar')

#%%
# 分类型发卡占比
t1=pd.crosstab(card_t.issued_year,card_t.type)
t1["sum1"]=t1.sum(1)
t2=t1.div(t1.sum1,axis = 0)
t2.drop("sum1",1).plot(kind = 'bar',stacked= True)

#%%
# 面积图
import matplotlib.pyplot as plt

labels=["青年卡","普通卡","金卡"]
y1=t1.loc[:,"青年卡"].astype('int')
y2=t1.loc[:,"普通卡"].astype('int')
y3=t1.loc[:,"金卡"].astype('int')
x=t1.index.astype('int')
plt.stackplot(x,y1,y2,y3,labels = labels)
plt.title('发卡趋势')
plt.ylabel('发卡量')
plt.legend(loc = 'upper left')
plt.show()

#%%
# 3、不同持卡人的性别对比
sub_sch=pd.crosstab(card_t.type,card_t.sex)
sub_sch.div(sub_sch.sum(1),axis = 0).plot(kind = 'bar',stacked= True)

#%%
# 或者
from stack2dim import *
stack2dim(card_t,'type','sex')

#%%
# 4、不同类型卡的持卡人在办卡时的平均年龄对比
import seaborn as sns
import time
card_t['age']=(pd.to_datetime(card_t['issued'])-pd.to_datetime(card_t['birth_date']))

card_t['age1']=card_t['age'].map(lambda x:x.days/365)
sns.boxplot(x = 'type', y = 'age1', data = card_t)

#%%
# 5、不同类型卡的持卡人在办卡前一年内的平均帐户余额对比
trans=pd.read_csv(r"trans.csv",encoding="gbk")
trans.to_sql('trans', con)

#%%
card_t.to_sql('card_t', con)

#%%
car_sql='''
select a.card_id,a.issued,a.type,c.type as t_type,c.amount,c.balance,c.date as t_date
  from card as a
  left join disp as b on a.disp_id=b.disp_id
  left join trans as c on b.account_id=c.account_id
  where b.type="所有者"
  order by a.card_id,c.date
'''

card_t2 = pd.read_sql(car_sql, con)

#%%
card_t2['issued']=pd.to_datetime(card_t2['issued'])
card_t2['t_date']=pd.to_datetime(card_t2['t_date'])

# 将对账户余额进行清洗
# In[9]:
import datetime
card_t2['balance2'] = card_t2['balance'].map(
    lambda x: int(''.join(x[1:].split(','))))
card_t2['amount2'] = card_t2['amount'].map(
    lambda x: int(''.join(x[1:].split(','))))

card_t2.head()
card_t3 = card_t2[card_t2.issued>card_t2.t_date][
    card_t2.issued<card_t2.t_date+datetime.timedelta(days=365)]

#card_t3["avg_balance"] = card_t3.groupby('card_id')['balance2'].mean()
card_t4=card_t3.groupby(['type','card_id'])['balance2'].agg([('avg_balance','mean')])
card_t4.to_sql('card_t4', con)

#%%
card_t5=card_t4.reset_index()
#card_t5=pd.read_sql('select * from card_t4', con)
sns.boxplot(x = 'type', y = 'avg_balance', data = card_t5)

#%%
# 6、不同类型卡的持卡人在办卡前一年内的平均收入对比
type_dict = {'借':'out','贷':'income'}
card_t3['type1'] = card_t3.t_type.map(type_dict)
card_t6= card_t3.groupby(['type','card_id','type1'])[['amount2']].sum()
card_t6.head()
card_t6.to_sql('card_t6', con)

#%%
card_t7=card_t6.reset_index()
#card_t7=pd.read_sql('select * from card_t6', con)
card_t7.to_sql('card_t7', con)
card_t8=pd.read_sql('select * from card_t7 where type1="income"', con)

# In[13]:
sns.boxplot(x = 'type', y = 'amount2', data = card_t8)

#%%
card_t9=pd.read_sql('select * from card_t7 where type1="out"', con)

# In[13]:
sns.boxplot(x = 'type', y = 'amount2', data = card_t9)