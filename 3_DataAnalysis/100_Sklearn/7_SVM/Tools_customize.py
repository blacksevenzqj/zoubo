# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:16:16 2019

@author: dell
"""

import numpy as np
import pandas as pd


# In[]:
# 一、表连接
# https://blog.csdn.net/gdkyxy2013/article/details/80785361
def merge_test():
    dataDf1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'],
                            'value': [1, 2, 3, 4]})
    dataDf2 = pd.DataFrame({'rkey': ['foo', 'bar', 'qux', 'bar'],
                            'value': [5, 6, 7, 8]})
    print(dataDf1)
    print(dataDf2)

    # 内连接： 两DataFrame都必须有字段中的该类别 （两者都必须为DataFrame）
    dataIn = dataDf1.merge(dataDf2, left_on='lkey', right_on='rkey')

    # 右连接： 以右边DataFrame字段为准
    dataR = dataDf1.merge(dataDf2, left_on='lkey', right_on='rkey', how='right')

    # 左连接： 以左边DataFrame字段为准
    dataL = dataDf1.merge(dataDf2, left_on='lkey', right_on='rkey', how='left')

    # 全链接： 两边都统计
    dataQ = dataDf1.merge(dataDf2, left_on='lkey', right_on='rkey', how='outer')

    # 全链接： 两边都统计
    # on：指的是用于连接的列索引名称，必须存在于左右两个DataFrame中，如果没有指定且其他参数也没有指定，则以两个DataFrame列名交集作为连接键；


#    dataQ_On = dataDf1.merge(dataDf2, on=["lkey","rkey"], how='outer')

# 以下两行代码效果相同（索引要相同）
#    pd.merge(pd.DataFrame(age_cut_grouped_good), pd.DataFrame(age_cut_grouped_bad), left_index=True, right_index=True)
#    pd.concat([age_cut_grouped_good, age_cut_grouped_bad], axis=1)


# https://www.cnblogs.com/nxf-rabbit75/p/10475320.html
def concat(objs):
    # ignore_index： 默认False，当为True时： 1、合并之后重置索引； 2、如果遇到两张表的列字段本来就不一样，但又想将两个表合并，其中无效的值用nan来表示。那么可以使用ignore_index来实现。
    # keys： 默认None，用来给合并后的表增加key来区分不同的表数据来源 （索引会多出一列 数据来源列）
    return pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False, keys=None, levels=None, names=None,
                     verify_integrity=False)


# 集合去重合并为一维列表：
# 列表排序： https://www.cnblogs.com/huchong/p/8296025.html
def set_union(seriers1, seriers2, reverse=False):
    if type(seriers1) is not pd.Series or type(seriers2) is not pd.Series:
        raise Exception('seriers1/seriers2 Type is Error, must Series')
    if (len(seriers1) != len(seriers2)):
        raise Exception('seriers1 len != seriers2 len')

    #    return sorted(set(pd.concat([seriers1, seriers2], axis=0)), reverse=reverse)
    return sorted(set(seriers1).union(seriers2), reverse=reverse)


# In[]:
# 直接用agg函数求指标
# dat0.price.agg(['mean','median','std']) # 查看price的均值、中位数和标准差等更多信息

# 二、groupby
# https://www.jianshu.com/p/a18fa2074ca4
# https://www.jianshu.com/p/42f1d2909bb6
# https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html

# groupby() 之后直接 sum()： 得到分组后各特征求和的数据矩阵（sum只是一个示例），可数据是非标准的DataFrame，必须进行reset_index()变为标准的DataFrame
def groupby_sum(data, group_cols, as_index=True):
    if type(group_cols) != list:
        raise Exception('group_cols Type is Error, must list')
    return data.groupby(group_cols, as_index=as_index).sum().reset_index()  # 需重置索引： 因是完整的数据


# 1、传统groupby（不会统计 np.nan）
'''
aggs = {'3_total_fee' : [np.min, np.max, np.mean, np.sum], '4_total_fee' : np.sum}
data_group = tc.groupby_agg(data[0:10], ["1_total_fee", "2_total_fee"], aggs)
'''


def groupby_agg(data, group_cols, aggs, as_index=True):  # group_keys在普通groupby中不生效
    if type(group_cols) != list:
        raise Exception('group_cols Type is Error, must list')
    elif type(aggs) != dict:
        raise Exception('aggs Type is Error, must dict')

    # 例子： aggs = {'3_total_fee' : [np.min, np.max, np.mean, np.sum], '4_total_fee' : np.sum}
    # aggs中必须是 DataFrame中存在的、 待统计的特征。
    data_group = data.groupby(group_cols, as_index=as_index).agg(aggs)
    # '_'.join(col).strip()连接列名： ('item_id', '') 和 ('rating', 'mean') 得到： item_id_, rating_mean
    data_group.columns = ['_'.join(col).strip() for col in data_group.columns.values]
    return data_group


# statistical_col统计特征为单个特征（速度近10倍于groupby_apply）
'''
agg = {'bankcard_count':lambda x:len(set(x)), 'bank_phone_num':lambda x:x.nunique(), '放款前账单金额为负数':lambda x:x.where(x<0).count(), 'rating_mean':lambda x:x.count() if x.count() == 2 else 0}
agg = lambda x:':'.join(x) 将分组之后的多个统计值 拼接成字符串
agg = {'sum':np.sum, "mean":np.mean, "len":len}  注意：不要写成 aggs = {'rating_mean' : [np.mean, np.sum]} 列名是混乱的
data_group = tc.groupby_agg_oneCol(data, ["1_total_fee", "2_total_fee"], "3_total_fee", agg)

agg = {'price_mean1':lambda x: x.mean(), 'price_mean2':np.mean} # 两种方式指定函数，一样效果
tc.groupby_agg_oneCol(dat0, ['dist'], 'price', agg, as_index=True)
'''


def groupby_agg_oneCol(data, group_cols, statistical_col, agg, as_index=True):
    if type(group_cols) != list:
        raise Exception('group_col Type is Error, must list')
    if type(statistical_col) != str:
        raise Exception('statistical_col Type is Error, must str')
    if type(agg) != dict:
        raise Exception('aggs Type is Error, must dict')

    data_group = data.groupby(group_cols, as_index=as_index)[statistical_col].agg(agg)
    return data_group


'''
最原始、简单 的统计方式：
train_gb = train_data_1.groupby("brand")
all_info = {}
for kind, kind_data in train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['brand_amount'] = len(kind_data)
    info['brand_price_max'] = kind_data.price.max()
    info['brand_price_median'] = kind_data.price.median()
    info['brand_price_min'] = kind_data.price.min()
    info['brand_price_sum'] = kind_data.price.sum()
    info['brand_price_std'] = kind_data.price.std()
    info['brand_price_average'] = round(kind_data.price.sum() / (len(kind_data) + 1), 2)
    all_info[kind] = info
# all_info字典 转 DataFrame： 字典的key就是DataFrame的行索引index，所以要 .T置转 → .reset_index()重置行索引 → .rename(columns={"index": "brand"})跟换列名： 将all_info的key的原行索引名index（置转后现在列名）更新名称
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "brand"})

相当于

temp_data = train_data_1[train_data_1['price'] > 0] # 数据 条件 提前做好
agg = {'brand_amount':len, "brand_price_max":np.max, "brand_price_median":np.median, "brand_price_min":np.min, "brand_price_sum":np.sum, "brand_price_std":np.std, "brand_price_average":np.mean}
brand_fe2 = tc.groupby_agg_oneCol(temp_data, ['brand'], 'price', agg, as_index=False)
'''


# groupby的value_counts结果使用unstack()来将树状结构变成表状结构（相当于statistical_col=1或0时做两次groupby）
# 分组中某一组统计数量为0时，value_counts维度报错（改为使用分别groupby的方式，保险一些）
def groupby_value_counts_unstack(data, group_col, statistical_col):
    return data.groupby(group_col)[statistical_col].value_counts().unstack()


# 按count()统计，并将结果展开为DataFrame
def groupby_size(data, group_cols):
    if type(group_cols) == list:
        return data.groupby(
            group_cols).size().reset_index()  # groupby_result.size() == groupby_result["X"].count()；但.count()的.reset_index()麻烦
    else:
        raise Exception('group_cols Type is Error, must list')


# 将groupby的结果 转换为 dict
def groupby_to_dict(df):
    from collections import defaultdict

    user_count = df.groupby("user")["event"].count()

    user_count_index = user_count[user_count > 2].index.tolist()

    u = df["user"].isin(user_count_index)
    # df_train["user"].map(lambda x : x in user_count_index)

    w = df.loc[u, ["user", "event"]].sort_values(by="user")

    z = w.groupby("user")

    eventsForUser = defaultdict(set)

    i = 0
    for key, val in z:
        print(key)
        print(val)  # 分组标签（元组） → DataFrame的原索引 → DataFrame的原值
        val.apply(lambda x: eventsForUser[x[0]].add(x[1]), axis=1)  # axis=1按列统计； 默认axis=0按行统计
        i = i + 1
        if i == 2:
            break


# 分组后 按指定特征进行排序
# groupby 和 apply 配合使用，只有group_keys关键字生效，as_index不适用。
# result = tc.groupby_apply_sort(result, ["Feature_1"], ["Correlation_Coefficient"], [False], False)
'''
使用分组排序：外层排序特征Feature_1_sort没用。 因为apply函数是按每个分组标签划分之后，再按该组内的特征进行排序，控制不了分组标签排序。
且 每个分组标签 对应的 外层排序特征Feature_1_sort 都相同，没有意义。 但奇怪的是单独使用Feature_1_sort排序时，会带动其他数值类型特征进行排序...
'''


def groupby_apply_sort(data, group_cols, sort_cols, ascendings, group_keys=False):  # group_keys默认True
    if type(group_cols) != list:
        raise Exception('group_cols Type is Error, must list')
    elif type(sort_cols) != list:
        raise Exception('sort_cols Type is Error, must list')
    elif type(ascendings) != list:
        raise Exception('ascendings Type is Error, must list')

    return data.groupby(group_cols, group_keys=group_keys).apply(
        lambda x: x.sort_values(by=sort_cols, ascending=ascendings))


# 源码： dat0.price.groupby(dat0.dist).mean().sort_values(ascending= True).plot(kind = 'barh') # 先 .price 看似不合理，是否能提高性能？
# tc.groupby_apply_statistics_sort(dat0, ['dist'], 'price')
'''
可以使用 def groupby_agg_oneCol(...)方法代替，效率更高： （显示指定 单统计变量 可以不用apply，效率更高）
agg = {'price_mean1':lambda x: x.mean(), 'price_mean2':np.mean} # 两种方式指定函数，一样效果
tc.groupby_agg_oneCol(dat0, ['dist'], 'price', agg, as_index=True)
'''


def groupby_apply_statistics_sort(data, group_cols, statistic_col, statistic_type=1, ascending=True,
                                  group_keys=False):  # group_keys默认True
    if type(group_cols) != list:
        raise Exception('group_cols Type is Error, must list')

    if statistic_type == 1:
        return data.groupby(group_cols, group_keys=group_keys).apply(lambda x: x[statistic_col].mean()).sort_values(
            ascending=ascending)
    elif statistic_type == 2:
        return data.groupby(group_cols, group_keys=group_keys).apply(lambda x: x[statistic_col].min()).sort_values(
            ascending=ascending)
    elif statistic_type == 3:
        return data.groupby(group_cols, group_keys=group_keys).apply(lambda x: x[statistic_col].max()).sort_values(
            ascending=ascending)
    elif statistic_type == 4:
        return data.groupby(group_cols, group_keys=group_keys).apply(lambda x: x[statistic_col].std()).sort_values(
            ascending=ascending)


# tmp3 = tmp.groupby(["user_id"], group_keys=False).apply(lambda x: x["rating"].count() if x["rating"].count() == 2 else 0)
# tmp_train_order.groupby(by=['id']).apply(lambda x:x['type_pay'][(x['type_pay']=='在线支付').values].count()).reset_index(name = 'type_pay_zaixian')
# tc.groupby_apply_count(train_order, ["id"], 'type_pay', '在线支付', 'type_pay_zaixian')
def groupby_apply_conditionCount(data, group_cols, statistics_col, condition, reset_index_name, group_keys=False,
                                 inplace=False):
    if type(group_cols) != list:
        raise Exception('group_cols Type is Error, must list')

    # .reset_index(name=reset_index_name) 其中的 name=reset_index_name 是重命名 apply新生成的统计列的名称
    return data.groupby(group_cols, group_keys=group_keys).apply(
        lambda x: x[x[statistics_col] == condition][statistics_col].count()).reset_index(inplace=inplace,
                                                                                         name=reset_index_name)


# Series.nunique()返回去重后数量； np.unique(Series).size返回去重后数量；
# unique_label, counts_label = np.unique(Series, return_counts=return_counts)
def groupby_apply_nunique(data, group_cols, statistics_cols, group_keys=False):
    if type(group_cols) != list:
        raise Exception('group_cols Type is Error, must list')
    elif type(statistics_cols) != list:
        raise Exception('statistics_cols Type is Error, must list')

    return data.groupby(group_cols, group_keys=group_keys).apply(lambda x: x[statistics_cols].nunique())


# 特殊例子
'''
1、
def special_groupby_example(data):
    kkk = data[0:10].groupby(["1_total_fee", "2_total_fee"])[["3_total_fee","4_total_fee"]]

    # 冒号前sum是现在新加字段名， np.sum求和 分别作用于 "3_total_fee","4_total_fee"。
    # sum新加字段名 在 原始字段名之上 形成： ('sum', '3_total_fee')、 ('sum', '4_total_fee')
    #kkk = kkk.agg({'sum':np.sum})

    # 冒号前sum是现在新加字段名， np.sum求和、np.mean均值 都分别作用于 "3_total_fee","4_total_fee"。
    # sum新加字段名 在 原始字段名之上 形成： ('sum', '3_total_fee')、 ('sum', '4_total_fee')、('mean', '3_total_fee')、 ('mean', '4_total_fee')
    #kkk = kkk.agg({'sum':np.sum,"mean":np.mean})

    # 冒号前是原始字段名， 则字段各自执行自身的聚合操作
    kkk = kkk.agg({'3_total_fee':np.sum,"4_total_fee":np.mean})


    kkk.columns = ['_'.join(col).strip() for col in kkk.columns.values]
    print(kkk)


2、同一个用户 同一种优惠卷 中 优惠券自身 离其他同一种优惠卷 收货时间 的差值（代码在：extract_feature.py）
'''


# In[]:
# 三、透视表(pivotTab)
# https://blog.csdn.net/bqw18744018044/article/details/80015840
# index相当于分组key； margins总计
def pivot_table_statistical(df, statistical_cols, index=None, columns=None, aggfunc='mean', margins=True):
    #    pd.pivot_table(df, index=['产地','类别'], values=['价格', '数量'], aggfunc=np.mean) # values 相当于 statistical_cols 待统计字段； index行名称； columns列名称
    return df.pivot_table(statistical_cols, index=index, columns=columns, aggfunc=aggfunc, margins=margins)


# 交叉表(crossTab)： 相当于 df.groupby([col1, col2])[X].count() 展开显示
'''
sub_sch=pd.crosstab(dat0.subway,dat0.school)
sub_sch1 = sub_sch.div(sub_sch.sum(axis=1), axis = 0) # sub_sch.sum(axis=1) 以 行索引subway 是否有地铁为分母
sub_sch2 = sub_sch.div(sub_sch.sum(axis=0), axis = 1) # sub_sch.sum(axis=0) 以 列索引school 是否是学区房为分母
'''


def crossTab_statistical(df, col1, col2, margins=True):
    return pd.crosstab(df[col1], df[col2], margins=margins)


# In[]:
'''
# 四、单apply
t2['receive_number'] = t2.date_received.apply(lambda s:len(s.split(':')))
t2['max_date_received'] = t2.date_received.apply(lambda s:max([pd.Timestamp(d) for d in s.split(':')]))

# 细看：
def testr(x, feature_dict):
    output_list = [0] * len(feature_dict)
    if x in feature_dict:
        index = feature_dict[x]
        output_list[index] = 1
    return ",".join([str(ele) for ele in output_list])

# 1、apply用在Series上，元素级别的操作：传入的是元素，使用args关键字
test_data["workclass"].apply(testr, args=(output_dict,))
# 2、apply用在Series上，元素级别的操作：传入的是元素，传统调用方式
test_data["workclass"].apply(lambda x : testr(x, output_dict))
# 3、map用在Series上，元素级别的操作：传入的是元素，传统调用方式
test_data["workclass"].map(lambda x : testr(x, output_dict))

# 4、apply用在DataFrame上，Series级别的操作：使用x["workclass"]传入的是元素，传统调用方式
test_data.apply(lambda x : testr(x["workclass"], output_dict), axis=1)

# 5、apply用在DataFrame上，Series级别的操作：传入的是一行数据，使用args关键字
def testr2(x, feature_dict):
    output_list = [0] * len(feature_dict)
    print(x) # 一行数据
    if x["workclass"] in feature_dict:
        index = feature_dict[x["workclass"]]
        output_list[index] = 1
    return ",".join([str(ele) for ele in output_list])
test_data.apply(testr2, args=(output_dict,), axis=1) # 列向： 传入一行数据
'''

# In[]:
'''
五、单map
error_list1 = ["后", "null", "?", "？"]
# 找索引
for i in error_list1:
    a = train_user1[train_user1['birthday'].map(lambda x: i in str(x))]["birthday"].index.tolist()
    train_user1.loc[a, 'birthday'] = pd.lib.NaT
# 直接赋值
for i in error_list1:
    train_user1['birthday'] = train_user1['birthday'].map(lambda x: pd.lib.NaT if i in str(x) else x)

# In[]:
# re.match 返回 匹配对象 或 None
re_list = ["^(19|20)\d{2}-\d{1,2}-0", "^0-", "^-", "^(19|20)\d{2}-\d{1,2}-$"]
# 1、找索引
for i in re_list:
    a = train_user1[train_user1['birthday'].map(lambda x: re.match(i, str(x)) != None)]["birthday"].index.tolist()
    train_user1.loc[a, 'birthday'] = pd.lib.NaT
# 2、直接赋值
for i in re_list:
    train_user1['birthday'] = train_user1['birthday'].map(lambda x: pd.lib.NaT if (re.match(i, str(x))) else x)

# In[]:
# 显示指定unicode编码
dict1 = {
        u'chaoyang' : "朝阳",
        u'dongcheng' : "东城",
        u'fengtai' :  "丰台",
        u'haidian' : "海淀",
        u'shijingshan' : "石景山",
        u'xicheng': "西城"
        }  
dat0.dist = dat0.dist.map(lambda x : dict1[x])    
'''

# In[]:
'''
六、列表：
1、列表展开式
1.1、[x for x in zip(user_ids,item_ids)]
user_ids、item_ids为Series长度需相等，组成列表（元素为元组）： [(15, 539), ..., (15, 73)]； x元素为元组。
1.2、items = [x[1] for x in zip(user_ids,item_ids) if x[0]==user_id]
if条件判断在 列表展开式 之后， 返回值在 列表展开式 之前。

2、列表降维：
网址： https://segmentfault.com/a/1190000018903731
sum(recom_result["1"][cate], [])
'''


# In[]:
# 字典排序
def dict_sorted(dict_data, position=1, reverse=True):
    import operator
    if type(dict_data) is not dict:
        raise Exception('dict_data Type is Error, must dict')
    return sorted(dict_data.items(), key=operator.itemgetter(position), reverse=reverse)


# 列表（元组元素）排序
def list_tuple_sorted(list_data, position=1, reverse=True):
    if type(list_data) is not list:
        raise Exception('list_data Type is Error, must list')
    return sorted(list_data, key=lambda element: element[position], reverse=reverse)


# numpy.ndarray排序
def ndarray_sort(data, axis=0):
    if (type(data) != np.ndarray):
        raise Exception('data Type is Error, must numpy.ndarray')
    return np.sort(data, axis=axis)  # axis=0 按列跨行排序（行向： 每一行的元素进行排序）； axis=1 按行跨列排序（列向： 每一列的元素进行排序）


'''
kkk = {}
kkk["a"] = {}
kkk["a"]["aaa"] = 11
kkk["b"] = {}
kkk["b"]["bbb"] = 22
{point:0 for point in kkk} # 迭代的是dict的key
'''


# In[]:
# 笛卡尔积
# product([0], sales.shop_id.unique(), sales.item_id.unique()) 代码在： 2_myLearn_simple2.ipynb
def cartesian_product(data):
    from itertools import product
    # 每个入参类型都必须是iterable：可迭代集合


#    return product([0], sales.shop_id.unique(), sales.item_id.unique())


# 限定矩阵值范围： DataFrame的方法
def clip(data, lower=None, upper=None, axis=None):
    aa = np.array([[0.335232, -1.256177],
                   [-1.367855, 0.746646],
                   [0.027753, -1.176076],
                   [0.230930, -0.679613],
                   [1.261967, 0.570967]
                   ])

    print(aa.clip(-1.0, 0.5))  # ndarrary可以使用，没有axis参数
    df_temp = pd.DataFrame(aa)
    df_temp
    print(df_temp.clip(-1.0, 0.5, axis=0))  # 貌似都是一样的
    print(df_temp.clip(-1.0, 0.5, axis=1))

    return data.clip(lower=lower, upper=upper, axis=axis)


# In[]:
'''
# 两DataFrame相减：
# 1、使用drop：相当于删除
dt = {'date_time': ['2018-03-11', '2018-03-12', '2018-03-16', '2018-03-17'], 'code': ['000000', '000001', '000002', '000003']}
dateframe1 = pd.DataFrame(data=dt)

dateframe2 = dateframe1[dateframe1.date_time < '2018-03-15']

# DateFrame.axes[0]获取行轴的标签名。 删除行轴标签名相同的行数据
dateframe3 = dateframe1.drop(labels=dateframe2.axes[0]) 

# 2、两Seriers相减 求差值：行索引必须相同，否则就是Shit
# 先恢复索引，再相减
resid = pd.DataFrame((y['avg_exp'] - predict["Pred"]), columns=['resid'])
resid = pd.DataFrame(y['avg_exp'].sub(predict["Pred"]), columns=['resid'])
'''


