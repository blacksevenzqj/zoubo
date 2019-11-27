# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:09:12 2019

@author: dell
"""

import pandas as pd
import time;  
import datetime
import sys
import numpy as np
import Tools_customize as tc

# In[]:
# 一、time模块
# 1、time.time() → Timestamp
ticks = time.time()
print("当前时间戳为:", ticks, type(ticks))

# In[]:
# 2、Timestamp → struct_time
localtime = time.localtime(time.time())
print("本地时间为 :", localtime, type(localtime))

# In[]:
# 3.1、struct_time → str
localtime = time.asctime(time.localtime(time.time()))
print("本地时间为 :", localtime, type(localtime))
# In[]:
# 3.2、3、struct_time → str 格式化
time1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(time1, type(time1))
time2 = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
print(time2, type(time2))

# In[]:
# 4、Timestamp → str
localtime = time.ctime(time.time())
print("本地时间为 :", localtime, type(localtime))

# In[]:
# 5、str → struct_time → Timestamp
# 时间字符串转成时间格式时： 匹配格式化字符串 需要与 时间字符串 格式相同
b = time.strptime("Sat Mar 28 22:24:24 2016", "%a %b %d %H:%M:%S %Y") # str → struct_time
print(b, type(b))
time3 = time.mktime(b) # struct_time → Timestamp
print(time3, type(time3))

# In[]:
# 6、休眠
time.sleep(3) #表示延迟3秒后程序继续运行

# In[]:
# struct_time
# UTC时间实际就是格林尼治时间，它 比 中国时间 晚八个小时
time1 = time.gmtime()
print(time1, type(time1))
time2 = time.strftime("%Y-%m-%d %H:%M:%S", time1)
print(time2, type(time2))



# In[]:
# 二、datetime模块
# 1、<class 'datetime.datetime'>
# 获取当前时间和日期
print(datetime.datetime.today(), type(datetime.datetime.today()))
print(datetime.datetime.now(), type(datetime.datetime.now()))
# In[]:
# 获取指定日期
dt = datetime.datetime(2015, 4, 19, 12, 20)
print(dt, type(dt))

# In[]:
# 2.1、datetime → timestamp
df = datetime.datetime(2015, 4, 19, 12, 20).timestamp()
print(df, type(df))
# In[]:  
# 2.2、timestamp → datetime  
t = 1429417200.0
print(datetime.datetime.fromtimestamp(t), type(datetime.datetime.fromtimestamp(t)))

# In[]:
# 3.1、str → datetime
cday = datetime.datetime.strptime('2015-6-1 18:19:59', '%Y-%m-%d %H:%M:%S')
print(cday, type(cday))
cday = datetime.datetime.strptime("0001-01-01", '%Y-%m-%d') # datetime允许最小日期
print(cday, type(cday))
# In[]:
# 3.2、datetime → str 
now = datetime.datetime.now()
print(now.strftime('%a, %b %d %H:%M'))
print(now.strftime("%Y-%m-%d %H:%M:%S"))

# In[]
# 4、datetime时间加减
now = datetime.datetime.now()
result = now + datetime.timedelta(hours=8)
print(result, type(result))

result = now + datetime.timedelta(days=2, hours=6)
print(result, type(result))



# In[]:
# 三、Pandas时间模块
# arg : integer, float, string, datetime, list, tuple, 1-d array, Series
# 1.1、str → PD时间格式（直接转换）
# 时间字符串 必须是 日期时间格式（如下格式都可行）； PD时间格式为： 2017-06-10 00:00:00
str1 = "6/10/2017"
#str1 = "2017/06/10"
#str1 = "2017-06-10"

time1 = pd.Timestamp(str1)
print(time1, type(time1))

test2 = pd.to_datetime(str1)
print(test2, type(test2))
# In[]:
# 1.2、str → PD时间格式（使用匹配格式化字符串）
# 时间字符串转成时间格式时： 匹配格式化字符串 需要与 时间字符串 格式相同
str1 = "6/10/2019"
'''
errors：
If ‘raise’, then invalid parsing will raise an exception
If ‘coerce’, then invalid parsing will be set as NaT
If ‘ignore’, then invalid parsing will return the input
'''
test1 = pd.to_datetime(str1, format = "%m/%d/%Y", errors = 'coerce')
print(test1, type(test1))

'''
Passing infer_datetime_format=True can often-times speedup a parsing if its not an ISO8601 format exactly, but in a regular format.
不会处理异常的，infer_datetime_format=True只是加速处理
'''
test1 = pd.to_datetime(str1, infer_datetime_format=True)
print(test1, type(test1))

# In[]:
# 2、PD时间格式 → str
str1 = "6/10/2017"
#str1 = "2017/06/10"
#str1 = "2017-06-10"

test1 = pd.to_datetime(str1)
print(test1, type(test1))

test2 = test1.strftime('%Y-%m-%d')
print(test2)

# In[]:
# 3、timestamp → PD时间格式
'''
unit of the arg (D,s,ms,us,ns) denote the unit, which is an integer or float number
s是秒
ms是毫秒=0.001秒
us是微秒=0.000001秒
ns是纳秒=0.000000001秒
'''
test1 = pd.to_datetime(1490195805, unit='s') # 秒
print(test1, type(test1))

test1 = pd.to_datetime(1490195805433502912, unit='ns') # 纳秒
print(test1, type(test1))
      
# In[]:
# 4、PD时间格式 → PD时间格式（意义？）
time1 = pd.Timestamp('1997-07-01')
print(time1, type(time1))

test1 = pd.to_datetime(time1, unit='D') # unit='D'看不出有什么作用
print(test1, type(test1))

# In[]:
# 5、datetime → PD时间格式
str1 = "6/10/2017"
#str1 = "2017/06/10"
#str1 = "2017-06-10"

cday = datetime.datetime.strptime(str1, '%m/%d/%Y')
print(cday, type(cday), cday.strftime('%Y-%m-%d'), type(cday.strftime('%Y-%m-%d')))

test1 = pd.to_datetime(cday)
print(test1, type(test1))

# In[]:
# 6、DataFrame → Series（PD时间格式）
df = pd.DataFrame({'year': [2015, 2016],
                   'month': [2, 3],
                   'day': [4, 5]
                   })
df1 = pd.to_datetime(df)
print(df1, df1.dtypes)
# In[]:
print(df1.dt.date, df1.dt.date.dtypes, df1.dt.date[0], type(df1.dt.date[0])) # <class 'datetime.date'>
print(df1.dt.year)
print(df1.dt.month)
print(df1.dt.day)
print(df1.dt.minute)
print(df1.dt.second)
print(df1.dt.quarter)
# In[]:
# 方法 1
df2 = df1.map(lambda x : x.strftime('%Y-%m'))
print(df2, df2.dtypes)
# 方法 2
df3 = df1.dt.to_period('M') # 月
print(df3, df3.dtypes)

df3 = df1.dt.to_period('Q') # 季度？
print(df3, df3.dtypes)

df3 = df1.dt.to_period('A') # 年
print(df3, df3.dtypes)

df3 = df1.dt.to_period('D') # 日
print(df3, df3.dtypes)


# In[]:



# In[]:
# ===================================异常处理=================================
# In[]:
min_day = "0001-01-01" # datetime允许最小日期

cday = datetime.datetime.strptime(min_day, '%Y-%m-%d') 
print(cday, type(cday))
# In[]:
min_day = "0001-01-01"

test1 = pd.to_datetime(min_day) # 直接异常
print(test1, type(test1))
# In[]:
pd_min = pd.Timestamp.min
print(pd_min, type(pd_min))

test1 = pd.to_datetime(pd_min) 
print(test1, type(test1))

test1 = pd.to_datetime("1677-09-21 00:12:43.145225") 
print(test1, type(test1))
# In[]:
pd_min =pd.Timestamp.min.ceil('D')
print(pd_min, type(pd_min))

test1 = pd.to_datetime(pd_min) 
print(test1, type(test1))

test1 = pd.to_datetime("1677-09-22 00:00:00") 
print(test1, type(test1))










