# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:41:32 2020

@author: dell
"""

import pandas as pd
import Apriori as apri  # 就用这个类，我已经改过（不使用 myscripts/apriori.py）
import matplotlib.pyplot as plt
import os
import csv
import time;

os.chdir(
    r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\1_TianshanCollege\10_RecommendedSystem_AssociationRule\15Association")

# In[]:
'''
【白话机器学习】算法理论+实战之关联规则
https://mp.weixin.qq.com/s/KXoKE0cY7hiJIA2hE86mDw
'''
# 一、直接可用数据集
data = [('牛奶', '面包', '尿布'),
        ('可乐', '面包', '尿布', '啤酒'),
        ('牛奶', '尿布', '啤酒', '鸡蛋'),
        ('面包', '牛奶', '尿布', '啤酒'),
        ('面包', '牛奶', '尿布', '可乐')]

# 1.1、第三方库：
from mlxtend.frequent_patterns import apriori as mlxtend_apriori, association_rules as mlxtend_association_rules
from mlxtend.preprocessing import TransactionEncoder

# TransactionEncoder是进行数据转换中的，需要先将上面的data数据转成宽表的形式，何谓宽表，下面的这种：
"""数据转换"""
transEn = TransactionEncoder()
oht_ary = transEn.fit_transform(data)
new_data = pd.DataFrame(oht_ary, columns=transEn.columns_)
# In[]:
print(new_data.iloc[0][0])
print(type(new_data.iloc[0][0]))
# In[]:
# 第一步：计算频繁项集，在这里可以定义最小支持度进行筛选频繁项集：
"""计算频繁项集"""
frequent_itemset = mlxtend_apriori(new_data, min_support=0.5, use_colnames=True)
frequent_itemset
# In[]:
# 第二步：挖取关联规则， 这里的 准则 可以使用 置信度(confidence) 或 提升度（lift）
rules = mlxtend_association_rules(frequent_itemset, metric='confidence', min_threshold=1)  # lift
rules
'''
这种方式一般会用三个函数：
TransactionEncoder: 需要把数据转成宽表的形式
apriori(): 这里面需要指定最小支持度
association_rules(): 这里面指定筛选准则（置信度或者提升度或者支持度都可以）
优点：最后显示的关联规则中，支持度，置信度，提升度等信息非常详细，一目了然。
缺点：数据有特殊的规则要求，处理起来比较麻烦（转换成宽表非常麻烦），并且用关联规则那块两个函数分开，用起来麻烦一些。
'''

# In[]:
# 1.2、第三方库：
from efficient_apriori import apriori

'''
其中 data 是我们要提供的数据集，它是一个 list 数组类型。min_support 参数为最小支持度，
在 efficient-apriori 工具包中用 0 到 1 的数值代表百分比，
比如 0.5 代表最小支持度为 50%。min_confidence 是最小置信度，数值也代表百分比，比如 1 代表 100%。
'''
# 挖掘频繁项集和频繁规则
itemsets, rules = apriori(data, min_support=0.5, min_confidence=0.1)
print(itemsets)
print()
print(rules)
'''
这个的优点是使用起来简单，并且efficient-apriori 工具包把每一条数据集里的项式都放到了一个集合中进行运算，
并没有考虑它们之间的先后顺序。因为实际情况下，同一个购物篮中的物品也不需要考虑购买的先后顺序。
而其他的 Apriori 算法可能会因为考虑了先后顺序，出现计算频繁项集结果不对的情况。
所以这里采用的是 efficient-apriori 这个工具包。
'''

# In[]:
# 2、自定义库：
ress = apri.arules(data, minSupport=0.5, minConf=0.1, minlen=1, maxlen=4)  # DataFrame
print(type(ress))

# In[]:
# *****************************************************************************


# In[]:
# 二、实例： 数据库格式数据集：
inverted = pd.read_csv("Transactions.csv")
inverted.head()


# In[]:
def encode_unit(x):
    if x <= 0:
        return False
    if x >= 1:
        return True


# In[]:
# 1.1、第三方库：
# xxx.groupby(['f1', 'f2'], as_index=False).size()： 当直接使用.size()统计时： as_index关键词失效（无论True/False），分组特征f1、f2都将作为索引。
# .reset_index().set_index('OrderNumber').fillna(0)
new_data2 = inverted.groupby(['OrderNumber', 'Model'], as_index=False).size().unstack().reset_index(drop=True).fillna(0)
# In[]:
new_data2 = new_data2.applymap(encode_unit)  # 这个函数见上面
# In[]:
print(new_data2.iloc[0][0])
print(type(new_data2.iloc[0][0]))
# In[]:
# 第一步：计算频繁项集，在这里可以定义最小支持度进行筛选频繁项集：
"""计算频繁项集"""
itemset1 = mlxtend_apriori(new_data2, min_support=0.01, use_colnames=True)
# In[]:
# 第二步：挖取关联规则， 这里的 准则 可以使用 置信度(confidence) 或 提升度（lift）
rules2 = mlxtend_association_rules(itemset1, metric='confidence', min_threshold=1)  # lift
rules2

# In[]:
# 直接使用自定义apriori的转换功能：
idataset, idataset_dict = apri.dataconvert(inverted, tidvar='OrderNumber', itemvar='Model', data_type='inverted')

# In[]:
# 1.2、第三方库：
itemset2, rules3 = apriori(idataset, min_support=0.01, min_confidence=0.1)
print(itemset2)
print()
print(rules3)

# In[]:
# 2、自定义库： 只会单条 推荐 多条（不会 多条 推荐 单条）
res = apri.arules(idataset, minSupport=0.01, minConf=0.1, minlen=2, maxlen=4)  # DataFrame
print(type(res))
# In[ ]:
res[res.lift > 1].sort_values('support', ascending=False).head(20)
# In[ ]:
# 互补品
res.loc[res.lift > 1, ['lhs', 'rhs', 'lift']].sort_values('lift', ascending=False).head(20)
# In[ ]:
# 互斥品
res.loc[res.lift < 1, ['lhs', 'rhs', 'lift']].sort_values('lift', ascending=True).head(20)
# In[ ]:
# 如果一个新客户刚刚下单了Mountain-200这个产品,如果希望获得最高的营销响应率,那在他的付费成功页面上最应该推荐什么产品?
# 1、如果只是为了提高 营销响应率： 则选择 置信度高的
Mountain_200 = res.loc[res.lhs == frozenset({'Mountain-200'}), :]  # 左手规则
res.loc[res.lhs == frozenset({'Mountain-200'}), ['lhs', 'rhs', 'support', 'confidence', 'lift']].sort_values(
    'confidence', ascending=False).head(20)
# In[ ]:
# 如果一个新客户刚刚下单了Mountain-200这个产品,如果希望最大化提升总体的产品销售额,那在他的付费成功页面上最应该推荐什么产品?
# 2、但有时 置信度 高的 已经是畅销产品，不用推荐。 这时我们推荐 提升度 高的产品： 相对小众一些的产品（相对于 置信度高 的产品而言） 对整体的销售有利。
Mountain_200 = res.loc[res.lhs == frozenset({'Mountain-200'}), :]
res.loc[res.lhs == frozenset({'Mountain-200'}), ['lhs', 'rhs', 'support', 'confidence', 'lift']].sort_values('lift',
                                                                                                             ascending=False).head(
    20)
# In[ ]:
# 如果希望推荐Sport-100自行车，应该如何制定营销策略？
'''
置信度 或 提升度 按倒序选择都可以， 因为被推荐的产品已定（右手规则已定），那么：
1、置信度（条件概率）是定值； 
2、提升度也是定值，因为 提示度 公式中的 无条件概率（被推荐商品概率）已定。
3、所以 置信度 和 提升度 的值有相同的趋势。
'''
Sport_100 = res.loc[res.rhs == frozenset({'Sport-100'}), :]  # 右手规则
res.loc[res.rhs == frozenset({'Sport-100'}), ['lhs', 'rhs', 'support', 'confidence', 'lift']].sort_values('lift',
                                                                                                          ascending=False).head(
    20)

# In[]:
# *****************************************************************************


# In[]:
# 三、 项目实战： 导演是如何选择演员的
# 关于这个数据，需要使用爬虫技术，去https://movie.douban.com搜索框中输入导演姓名，比如“宁浩”。

"""下载某个导演的电影数据集"""


def dowloaddata(director):
    from selenium import webdriver
    from lxml import etree

    # 浏览器模拟
    driver = webdriver.Chrome()

    # 写入csv文件
    file_name = './' + director + '.csv'
    out = open(file_name, 'w', newline='', encoding='utf-8-sig')
    csv_write = csv.writer(out, dialect='excel')
    flags = []
    """下载某个指定页面的数据"""

    def download(request_url):

        driver.get(request_url)
        time.sleep(1)

        html = driver.find_element_by_xpath("//*").get_attribute("outerHTML")
        html = etree.HTML(html)

        # 设置电影名称，导演演员的XPATH
        movie_lists = html.xpath(
            "/html/body/div[@id='wrapper']/div[@id='root']/div[1]//div[@class='item-root']/div[@class='detail']/div[@class='title']/a[@class='title-text']")
        name_lists = html.xpath(
            "/html/body/div[@id='wrapper']/div[@id='root']/div[1]//div[@class='item-root']/div[@class='detail']/div[@class='meta abstract_2']")  # 获取返回的数据个数

        # 获取返回的数据个数
        num = len(movie_lists)
        if num > 15:  # 第一页会有16条数据, 第一条是导演的介绍
            # 默认第一个不是，所以需要去掉
            movie_lists = movie_lists[1:]
            name_lists = name_lists[1:]

        for (movie, name_list) in zip(movie_lists, name_lists):
            # 会存在数据为空的情况
            if name_list.text is None:
                continue
            print(name_list.text)
            names = name_list.text.split('/')

            # 判断导演是否为指定的director
            if names[0].strip() == director and movie.text not in flags:
                # 将第一个字段设置为电影名称
                names[0] = movie.text
                flags.append(movie.text)
                csv_write.writerow(names)

        if num >= 14:  # 有可能一页会有14个电影
            # 继续下一页
            return True
        else:
            # 没有下一页
            return False

    # 开始的ID为0， 每页增加15个
    base_url = 'https://movie.douban.com/subject_search?search_text=' + director + '&cat=1002&start='
    start = 0
    while start < 10000:  # 最多抽取10000部电影
        request_url = base_url + str(start)

        # 下载数据，并返回是否有下一页
        flag = download(request_url)
        if flag:
            start = start + 15
        else:
            break
    out.close()
    print('finished')


"""调用上面的函数"""
directorname = '宁浩'
dowloaddata(directorname)

# In[]:
# 使用 1.2、第三方库：
director = '宁浩'
file_name = './' + director + '.csv'
lists = csv.reader(open(file_name, 'r', encoding='utf-8-sig'))

# 数据加载
data = []
for names in lists:
    name_new = []
    for name in names:
        # 去掉演员数据中的空格
        name_new.append(name.strip())
    data.append(name_new[1:])

# 挖掘频繁项集和关联规则
itemsets, rules = apriori(data, min_support=0.3, min_confidence=0.8)
print(itemsets)
print(rules)
