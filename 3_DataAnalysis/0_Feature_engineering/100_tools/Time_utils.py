# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:09:12 2019

@author: dell
"""

import pandas as pd
import time;
import datetime
import timeit
import sys
import re
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
b = time.strptime("Sat Mar 28 22:24:24 2016", "%a %b %d %H:%M:%S %Y")  # str → struct_time
print(b, type(b))
time3 = time.mktime(b)  # struct_time → Timestamp
print(time3, type(time3))

# In[]:
# 6、休眠
time.sleep(3)  # 表示延迟3秒后程序继续运行

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
# 2.1、datetime → timestamp（秒，10位）
df = datetime.datetime(2015, 4, 19, 12, 20).timestamp()
print(df, type(df))
# In[]:
# 2.2、timestamp（秒，10位） → datetime
'''
原来java的date默认精度是毫秒，也就是说生成的时间戳就是13位的，而像c++、php、python生成的时间戳默认就是10位的，因为其精度是秒。
'''
t = 1476086345
print(datetime.datetime.fromtimestamp(t), type(datetime.datetime.fromtimestamp(t)))
print(datetime.datetime.utcfromtimestamp(t), type(datetime.datetime.utcfromtimestamp(t)))
# 使用utcfromtimestamp + timedelta的意义在于 避开系统本地时间的干扰，都可以准确转换到 东八区时间。
print(datetime.datetime.utcfromtimestamp(int(str(t)[0:10])) + datetime.timedelta(hours=8))

# In[]:
# 3.1、str → datetime
# 时间字符串转成时间格式时： 匹配格式化字符串 需要与 时间字符串 格式相同
cday = datetime.datetime.strptime('2015-6-1 18:19:59', '%Y-%m-%d %H:%M:%S')
print(cday, type(cday))
cday = datetime.datetime.strptime("0001-01-01", '%Y-%m-%d')  # datetime允许最小日期
print(cday, type(cday))
# In[]:
# 3.2、datetime → str
now = datetime.datetime.now()
print(now.strftime('%a, %b %d %H:%M'))
print(now.strftime("%Y-%m-%d %H:%M:%S"))

# In[]
# 4.1、datetime时间加减
now = datetime.datetime.now()
result = now + datetime.timedelta(hours=8)
print(result, type(result))

result = now + datetime.timedelta(days=2, hours=6)
print(result, type(result))

# In[]:
# 三、Pandas时间模块
print(pd.Timestamp.now())
print(pd.Timestamp.today())
# In[]:
# pd.to_datetime  arg : integer, float, string, datetime, list, tuple, 1-d array, Series
# 1.1、str → PD时间格式（直接转换）
# 时间字符串 必须是 日期时间格式（如下格式都可行）； PD时间格式为： 2017-06-10 00:00:00
str1 = "6/10/2017"
# str1 = "2017/06/10"
# str1 = "2017-06-10"

time1 = pd.Timestamp(str1)
print(time1, type(time1))

test2 = pd.to_datetime(str1)
print(test2, type(test2))

# 1.1.1、数字 → PD时间格式
print(pd.Timestamp(2016, 6, 30))

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
test1 = pd.to_datetime(str1, format="%m/%d/%Y", errors='coerce')
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
# str1 = "2017/06/10"
# str1 = "2017-06-10"

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
test1 = pd.to_datetime(1490195805, unit='s')  # 秒
print(test1, type(test1))

test1 = pd.to_datetime(1490195805433502912, unit='ns')  # 纳秒
print(test1, type(test1))

# In[]:
# 4、PD时间格式 → PD时间格式（意义？）
time1 = pd.Timestamp('1997-07-01')
print(time1, type(time1))

test1 = pd.to_datetime(time1, unit='D')  # unit='D'看不出有什么作用
print(test1, type(test1))

# In[]:
# 5、datetime → PD时间格式
str1 = "6/10/2017"
# str1 = "2017/06/10"
# str1 = "2017-06-10"

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
print(df1.dt.date, df1.dt.date.dtypes, df1.dt.date[0], type(df1.dt.date[0]))  # <class 'datetime.date'>
print(df1.dt.year)
print(df1.dt.month)
print(df1.dt.day)
print(df1.dt.minute)
print(df1.dt.second)
print(df1.dt.quarter)
# In[]:
# 方法 1
df2 = df1.map(lambda x: x.strftime('%Y-%m'))
print(df2, df2.dtypes)
# 方法 2
df3 = df1.dt.to_period('M')  # 月
print(df3, df3.dtypes)

df3 = df1.dt.to_period('Q')  # 季度？
print(df3, df3.dtypes)

df3 = df1.dt.to_period('A')  # 年
print(df3, df3.dtypes)

df3 = df1.dt.to_period('D')  # 日
print(df3, df3.dtypes)

# In[]:
# 函数执行时间：
time_start = timeit.default_timer()
time_end = timeit.default_timer()
print(time_end - time_start)

# In[]:
# ===================================异常处理=================================
# In[]:
min_day = "0001-01-01"  # datetime允许最小日期

cday = datetime.datetime.strptime(min_day, '%Y-%m-%d')
print(cday, type(cday))
# In[]:
min_day = "0001-01-01"

test1 = pd.to_datetime(min_day)  # 直接异常
print(test1, type(test1))
# In[]:
pd_min = pd.Timestamp.min
print(pd_min, type(pd_min))

test1 = pd.to_datetime(pd_min)
print(test1, type(test1))

test1 = pd.to_datetime("1677-09-21 00:12:43.145225")
print(test1, type(test1))
# In[]:
pd_min = pd.Timestamp.min.ceil('D')
print(pd_min, type(pd_min))

test1 = pd.to_datetime(pd_min)
print(test1, type(test1))

test1 = pd.to_datetime("1677-09-22 00:00:00")
print(test1, type(test1))
# In[]:
strs = ["0001-01-01", "0001-1-1", "0-0-0", "0000-00-00", "2019-01-01", "1997-1-1", "1995-9-8", "1992-08-28",
        "1933-9-30", "1990-3-0", "1990-3-19"]

for i in strs:
    print(i)
    if (re.match('^(19|20)\d{2}-\d{1,2}-[1-9]\d{0,1}', i) and '-00' not in i):
        cday = datetime.datetime.strptime(i, '%Y-%m-%d')
        print(cday, type(cday))
    else:
        print("没抓到：", i)

# In[]:
# ===================================时间间隔=================================
# In[]:
# 两 datatime 之间的时间差异
'''
td.days	天 [-999999999, 999999999]
td.seconds	秒 [0, 86399] 只计算 从小时开始 的时间差异（不涉及天的差异）
td.microseconds	微秒 [0, 999999]
td.total_seconds()	时间差中包含的总秒数，等价于: td / timedelta(seconds=1)
'''
cday = datetime.datetime.strptime('2019-11-28 18:19:59', '%Y-%m-%d %H:%M:%S')
print(cday, type(cday))

cday2 = datetime.datetime.strptime("2019-11-21 16:19:59", '%Y-%m-%d %H:%M:%S')  # datetime允许最小日期
print(cday2, type(cday2))

diff = cday - cday2
print(diff, type(diff), diff.days, diff.seconds, diff.total_seconds())

# In[]:
# 两 PD时间格式 之间的时间差异
time1 = pd.Timestamp("2019-11-28 18:19:59")
print(time1, type(time1))

time2 = pd.to_datetime("2019-11-21 16:19:59")
print(time2, type(time2))

diff = time1 - time2
print(diff, type(diff), diff.days, diff.seconds, diff.total_seconds())
# In[]:
# PD时间格式 - datatime  或  datatime - PD时间格式： 最后都会转换为<class 'pandas._libs.tslib.Timedelta'>，PD时间格式级别高。
time1 = pd.Timestamp("2019-11-28 18:19:59")
print(time1, type(time1))

cday2 = datetime.datetime.strptime("2019-11-21 16:19:59", '%Y-%m-%d %H:%M:%S')  # datetime允许最小日期
print(cday2, type(cday2))

diff = time1 - cday2
print(diff, type(diff), diff.days, diff.seconds, diff.total_seconds())

diff2 = cday2 - time1
print(diff2, type(diff2), diff2.days, diff2.seconds, diff2.total_seconds())

# In[]:
# PD时间格式 == datatime
time1 = pd.Timestamp("2019-11-28 18:19:59")
print(time1, type(time1))

time2 = datetime.datetime.strptime("2019-11-28 18:19:59", '%Y-%m-%d %H:%M:%S')
print(time2, type(time2))

print(time1 == time2, time1 - time2)

# In[]:
# ==================================时间大小比较=================================
# In[]:
# 使用 Timestamp 与 字符串时间格式 直接比较选择数据
# 使用 Series.isnull()找到 pd.lib.NaT <class 'pandas._libs.tslib.NaTType'> 数据
# off_train[((off_train.date>='20160315')&(off_train.date<='20160630'))|((off_train["date"].isnull())&(off_train.date_received>='20160315')&(off_train.date_received<='20160630'))]

'''
将分组之后的多个Timestamp 拼接成字符串
t2 = t2.groupby(['user_id','coupon_id'], as_index=False)['date_received'].agg(lambda x:':'.join(x))
print(t2["date_received"].dtypes, type(t2.loc[6,"date_received"])) # object， 已经不是Timestamp
t2['receive_number'] = t2.date_received.apply(lambda s:len(s.split(':'))) # 分组中有几个值

max()函数求 Timestamp 的最大值
t2['max_date_received'] = t2.date_received.apply(lambda s:max([pd.Timestamp(d) for d in s.split(':')]))
t2['min_date_received'] = t2.date_received.apply(lambda s:min([pd.Timestamp(d) for d in s.split(':')]))
'''




