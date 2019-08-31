# coding: utf-8

# In[1]:
"""
dist-所在区
roomnum-室的数量
halls-厅的数量
AREA-房屋面积
floor-楼层
subway-是否临近地铁
school-是否学区房
price-平米单价
"""
# In[1]:
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import statsmodels.api as sm
from numpy import corrcoef, array
# from IPython.display import HTML, display
from statsmodels.formula.api import ols

import os

os.chdir(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\1_TianshanCollege\4_HousingPrice_analysis\homework")

# 1 描述
# In[17]:
datall = pd.read_csv("sndHsPr.csv")  # 读入清洗过后的数据
print("%d", datall.shape[0])  # 样本量

# %%
dat0 = datall
dat0.describe(include="all").T  # 查看数据基本描述

# In[18]:
dat0.price = dat0.price / 10000  # 价格单位转换成万元

# In[19]:
# 将城区的水平由拼音改成中文，以便作图输出美观
dict1 = {
    u'chaoyang': "朝阳",
    u'dongcheng': "东城",
    u'fengtai': "丰台",
    u'haidian': "海淀",
    u'shijingshan': "石景山",
    u'xicheng': "西城"
}
# dat0.dist = dat0.dist.apply(lambda x : dict1[x])
dat0.dist = dat0.dist.map(lambda x: dict1[x])
dat0.head()

# 1.1 因变量
# price
# In[20]:
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体

# 因变量直方图
fig = plt.figure(figsize=(12, 4))
# 1、A图
ax1 = fig.add_subplot(1, 2, 1)
dat0.price.hist(bins=20)
plt.xlabel("单位面积房价（万元/平方米）")
plt.ylabel("频数")

# 2、B图
ax2 = fig.add_subplot(1, 2, 2)
dat0.price.plot(kind='kde', style='k--', grid=True, title='标准化单位面积房价密度曲线', ax=ax2)
dat0.price.hist(bins=20, align='mid', orientation='vertical', color='r', linestyle="--", alpha=0.8, normed=True,
                ax=ax2)  # 没有hold参数
plt.xlabel("标准化单位面积房价")
plt.ylabel("频数")

# In[21]:
print(dat0.price.agg(['mean', 'median', 'std']))  # 查看price的均值、中位数和标准差等更多信息
print(dat0.price.quantile([0.25, 0.5, 0.75]))

# In[22]:
# 查看房价最高和最低的两条观测
pd.concat([(dat0[dat0.price == min(dat0.price)]), (dat0[dat0.price == max(dat0.price)])])

# 1.2 自变量：
# dist+roomnum+halls+floor+subway+school+AREA
# In[23]:
# 整体来看
for i in range(7):
    if i != 3:
        print(dat0.columns.values[i], ":")
        print(dat0[dat0.columns.values[i]].agg(['value_counts']).T)
        print("=======================================================================")
    else:
        continue
print('AREA:')
print(dat0.AREA.agg(['min', 'mean', 'median', 'max', 'std']).T)

# 1.2.1 dist
# In[24]:
# 频次统计
dat0.dist.value_counts().plot(kind='pie')  # 绘制柱柱形图
dat0.dist.agg(['value_counts'])
# dat0.dist.value_counts()

# In[25]:
# 不同城区的单位房价面积均值情况
dat0.price.groupby(dat0.dist).mean().sort_values(ascending=True).plot(kind='barh')

# %%
dat1 = dat0[['dist', 'price']]
dat1.dist = dat1.dist.astype("category")  # 设置为分类变量
dat1.dist.cat.set_categories(["石景山", "丰台", "朝阳", "海淀", "东城", "西城"], inplace=True)
# dat1.sort_values(by=['dist'],inplace=True)
sns.boxplot(x='dist', y='price', data=dat1)
# dat1.boxplot(by='dist',patch_artist=True)
plt.ylabel("单位面积房价(万元/平方米)")
plt.xlabel("城区")
plt.title("城区对房价的分组箱线图")

# 1.2.2 roomnum
# In[27]:
# 不同卧室数的单位面积房价差异不大
dat4 = dat0[['roomnum', 'price']]
# print(type(dat4.price.groupby(dat4.roomnum).mean())) # Series
dat4.price.groupby(dat4.roomnum).mean().plot(kind='bar')
dat4.boxplot(by='roomnum', patch_artist=True)

# 1.2.3 halls
# In[28]:
# 厅数对单位面积房价有轻微影响
dat5 = dat0[['halls', 'price']]
dat5.price.groupby(dat5.halls).mean().plot(kind='bar')
dat5.boxplot(by='halls', patch_artist=True)

# 1.2.4 floor
# In[31]:
# 不同楼层的单位面积房价差异不明显
dat6 = dat0[['floor', 'price']]
print(dat6[0:3])
dat6.floor = dat6.floor.astype("category")
dat6.floor.cat.set_categories(["low", "middle", "high"], inplace=True)
dat6.sort_values(by=['floor'], inplace=True)
dat6.boxplot(by='floor', patch_artist=True)
# dat6.price.groupby(dat6.floor).mean().plot(kind='bar')

# 1.2.5 subway+school
# In[32]:
# 交叉表是用于统计分组频率的特殊透视表
sub_sch = pd.crosstab(dat0.subway, dat0.school)
# print(sub_sch)
# print(sub_sch.sum(axis=1))
sub_sch = sub_sch.div(sub_sch.sum(axis=1), axis=0)  # 第一行除以第一行...
sub_sch


# In[33]:
def stack2dim(raw, i, j, rotation=0, location='upper left'):
    '''
    此函数是为了画两个维度标准化的堆积柱状图
    要求是目标变量j是二分类的
    raw为pandas的DataFrame数据框
    i、j为两个分类变量的变量名称，要求带引号，比如"school"
    rotation：水平标签旋转角度，默认水平方向，如标签过长，可设置一定角度，比如设置rotation = 40
    location：分类标签的位置，如果被主体图形挡住，可更改为'upper left'

    '''
    import math
    data_raw = pd.crosstab(raw[i], raw[j])
    data = data_raw.div(data_raw.sum(1), axis=0)  # 交叉表转换成比率，为得到标准化堆积柱状图

    # 计算x坐标，及bar宽度
    createVar = locals()
    x = [0]  # 每个bar的中心x轴坐标
    width = []  # bar的宽度
    k = 0
    for n in range(len(data)):
        # 根据频数计算每一列bar的宽度
        createVar['width' + str(n)] = data_raw.sum(axis=1)[n] / sum(data_raw.sum(axis=1))
        width.append(createVar['width' + str(n)])
        if n == 0:
            continue
        else:
            k += createVar['width' + str(n - 1)] / 2 + createVar['width' + str(n)] / 2 + 0.05
            x.append(k)

            # 以下是通过频率交叉表矩阵生成一列对应堆积图每一块位置数据的数组，再把数组转化为矩阵
    y_mat = []
    n = 0
    for p in range(data.shape[0]):
        for q in range(data.shape[1]):
            n += 1
            y_mat.append(data.iloc[p, q])
            if n == data.shape[0] * 2:
                break
            elif n % 2 == 1:
                y_mat.extend([0] * (len(data) - 1))
            elif n % 2 == 0:
                y_mat.extend([0] * len(data))

    y_mat = np.array(y_mat).reshape(len(data) * 2, len(data))
    y_mat = pd.DataFrame(y_mat)  # bar图中的y变量矩阵，每一行是一个y变量

    # 通过x，y_mat中的每一行y，依次绘制每一块堆积图中的每一块图
    createVar = locals()
    for row in range(len(y_mat)):
        createVar['a' + str(row)] = y_mat.iloc[row, :]
        if row % 2 == 0:
            if math.floor(row / 2) == 0:
                label = data.columns.name + ': ' + str(data.columns[row])
                plt.bar(x, createVar['a' + str(row)],
                        width=width[math.floor(row / 2)], label='0', color='#5F9EA0')
            else:
                plt.bar(x, createVar['a' + str(row)],
                        width=width[math.floor(row / 2)], color='#5F9EA0')
        elif row % 2 == 1:
            if math.floor(row / 2) == 0:
                label = data.columns.name + ': ' + str(data.columns[row])
                plt.bar(x, createVar['a' + str(row)], bottom=createVar['a' + str(row - 1)],
                        width=width[math.floor(row / 2)], label='1', color='#8FBC8F')
            else:
                plt.bar(x, createVar['a' + str(row)], bottom=createVar['a' + str(row - 1)],
                        width=width[math.floor(row / 2)], color='#8FBC8F')

    plt.title(j + ' vs ' + i)
    group_labels = [data.index.name + ': ' + str(name) for name in data.index]
    plt.xticks(x, group_labels, rotation=rotation)
    plt.ylabel(j)
    plt.legend(shadow=True, loc=location)
    plt.show()


# In[34]:
stack2dim(dat0, i="subway", j="school")

# In[35]:
# 地铁、学区的分组箱线图
dat2 = dat0[['subway', 'price']]
dat3 = dat0[['school', 'price']]
dat2.boxplot(by='subway', patch_artist=True)  # 中位数有差异
dat3.boxplot(by='school', patch_artist=True)  # 中位数有差异

# In[35]:
# 1.2.6 AREA
# %%
datA = dat0[['AREA', 'price']]
plt.scatter(datA.AREA, datA.price, marker='.')
# 求AREA和price的相关系数矩阵
data1 = array(datA['price'])
data2 = array(datA['AREA'])
datB = array([data1, data2])
# 1、np.corrcoef(矩阵A) 只能进行 两个特征之间 的皮尔森相关度计算
# 2、np.corrcoef(矩阵A)中 矩阵A 必须是：行是特征，列是样本
print(corrcoef(datB))  # 基于numpy的皮尔森相关系数矩阵
# min_periods : int, optional，指定每列所需的最小观察数，可选，目前只适合用在pearson和spearman方法。
print(datA.corr(method='pearson', min_periods=1))  # 基于DataFrame的皮尔森相关系数矩阵

# In[58]:看到从左至右逐渐稀疏的散点图,第一反应是对Y取对数
# 房屋面积和单位面积房价（取对数后）的散点图
datA['price_ln'] = np.log(datA['price'])  # 对price取对数
plt.figure(figsize=(8, 8))
plt.scatter(datA.AREA, datA.price_ln, marker='.')
plt.xlabel("面积（平方米）")
plt.ylabel("单位面积房价（取对数后）")

# 求AREA和price_ln的相关系数矩阵
data1 = array(datA['price_ln'])
data2 = array(datA['AREA'])
datB = array([data1, data2])
# 1、np.corrcoef(矩阵A) 只能进行 两个特征之间 的皮尔森相关度计算
# 2、np.corrcoef(矩阵A)中 矩阵A 必须是：行是特征，列是样本
corrcoef(datB)

# In[58]:
# 房屋面积和单位面积房价（取对数后）的散点图
datA['price_ln'] = np.log(datA['price'])  # 对price取对数
datA['AREA_ln'] = np.log(datA['AREA'])  # 对price取对数
plt.figure(figsize=(8, 8))
plt.scatter(datA.AREA_ln, datA.price_ln, marker='.')
plt.xlabel("面积（平方米）")
plt.ylabel("单位面积房价（取对数后）")

# 求AREA_ln和price_ln的相关系数矩阵
data1 = array(datA['price_ln'])
print(data1)
data2 = array(datA['AREA_ln'])
datB = array([data1, data2])
# 1、np.corrcoef(矩阵A) 只能进行 两个特征之间 的皮尔森相关度计算
# 2、np.corrcoef(矩阵A)中 矩阵A 必须是：行是特征，列是样本
print(corrcoef(datB))  # 基于numpy的皮尔森相关系数矩阵
# min_periods : int, optional，指定每列所需的最小观察数，可选，目前只适合用在pearson和spearman方法。
print(datA[['AREA_ln', 'price_ln']].corr(method='pearson', min_periods=1))  # 基于DataFrame的皮尔森相关系数矩阵


#########################################################################################
# 2 建模
# In[38]:
# 1、首先检验每个解释变量是否和被解释变量独立
# %%由于原始样本量太大，无法使用基于P值的构建模型的方案，因此按照区进行分层抽样
# dat0 = datall.sample(n=2000, random_state=1234).copy()
def get_sample(df, sampling="simple_random", k=1, stratified_col=None):
    """
    对输入的 dataframe 进行抽样的函数

    参数:
        - df: 输入的数据框 pandas.dataframe 对象

        - sampling:抽样方法 str
            可选值有 ["simple_random", "stratified", "systematic"]
            按顺序分别为: 简单随机抽样、分层抽样、系统抽样

        - k: 抽样个数或抽样比例 int or float
            (int, 则必须大于0; float, 则必须在区间(0,1)中)
            如果 0 < k < 1 , 则 k 表示抽样对于总体的比例
            如果 k >= 1 , 则 k 表示抽样的个数；当为分层抽样时，代表每层的样本量

        - stratified_col: 需要分层的列名的列表 list
            只有在分层抽样时才生效

    返回值:
        pandas.dataframe 对象, 抽样结果
    """
    import random
    import pandas as pd
    from functools import reduce
    import numpy as np
    import math

    len_df = len(df)
    if k <= 0:
        raise AssertionError("k不能为负数")
    elif k >= 1:
        assert isinstance(k, int), "选择抽样个数时, k必须为正整数"
        sample_by_n = True
        if sampling is "stratified":
            alln = k * df.groupby(by=stratified_col)[stratified_col[0]].count().count()  # 有问题的
            # alln=k*df[stratified_col].value_counts().count()
            if alln >= len_df:
                raise AssertionError("请确认k乘以层数不能超过总样本量")
    else:
        sample_by_n = False
        if sampling in ("simple_random", "systematic"):
            k = math.ceil(len_df * k)

    # print(k)

    if sampling is "simple_random":
        print("使用简单随机抽样")
        idx = random.sample(range(len_df), k)
        res_df = df.iloc[idx, :].copy()
        return res_df

    elif sampling is "systematic":
        print("使用系统抽样")
        step = len_df // k + 1  # step=len_df//k-1
        start = 0  # start=0
        idx = range(len_df)[start::step]  # idx=range(len_df+1)[start::step]
        res_df = df.iloc[idx, :].copy()
        # print("k=%d,step=%d,idx=%d"%(k,step,len(idx)))
        return res_df

    elif sampling is "stratified":
        assert stratified_col is not None, "请传入包含需要分层的列名的列表"
        assert all(np.in1d(stratified_col, df.columns)), "请检查输入的列名"

        grouped = df.groupby(by=stratified_col)[stratified_col[0]].count()
        if sample_by_n == True:
            group_k = grouped.map(lambda x: k)
        else:
            group_k = grouped.map(lambda x: math.ceil(x * k))

        res_df = df.head(0)
        for df_idx in group_k.index:
            df1 = df
            if len(stratified_col) == 1:
                df1 = df1[df1[stratified_col[0]] == df_idx]
            else:
                for i in range(len(df_idx)):
                    df1 = df1[df1[stratified_col[i]] == df_idx[i]]
            idx = random.sample(range(len(df1)), group_k[df_idx])
            group_df = df1.iloc[idx, :].copy()
            res_df = res_df.append(group_df)
        return res_df

    else:
        raise AssertionError("sampling is illegal")


# In[62]:
dat01 = get_sample(dat0, sampling="stratified", k=400, stratified_col=['dist'])

# %%
# 逐个检验变量的解释力度：方差分析
"""
不同卧室数的单位面积房价差异不大
客厅数越多，单位面积房价递减
不同楼层的单位面积房价差异不明显
地铁房单价高
学区房单价高
"""
"""大致原则如下（自然科学取值偏小、社会科学取值偏大）：
n<100 alfa取值[0.05,0.2]之间
100<n<500 alfa取值[0.01,0.1]之间
500<n<3000 alfa取值[0.001,0.05]之间
"""

import statsmodels.api as sm
from statsmodels.formula.api import ols

# 连续变量在前，分类变量在后： 因变量Y（连续）~自变量X(分类)
# PR(>F)
print("dist的P值为:%.4f" % sm.stats.anova_lm(ols('price ~ C(dist)', data=dat01).fit())._values[0][4])
print("roomnum的P值为:%.4f" % sm.stats.anova_lm(ols('price ~ C(roomnum)', data=dat01).fit())._values[0][
    4])  # 明显高于0.001->不显著->独立
print("halls的P值为:%.4f" % sm.stats.anova_lm(ols('price ~ C(halls)', data=dat01).fit())._values[0][
    4])  # 高于0.001->边际显著->暂时考虑
print("floor的P值为:%.4f" % sm.stats.anova_lm(ols('price ~ C(floor)', data=dat01).fit())._values[0][
    4])  # 高于0.001->边际显著->暂时考虑
print("subway的P值为:%.4f" % sm.stats.anova_lm(ols('price ~ C(subway)', data=dat01).fit())._values[0][4])
print("school的P值为:%.4f" % sm.stats.anova_lm(ols('price ~ C(school)', data=dat01).fit())._values[0][4])

# %%
# 厅数不太显著，考虑做因子化处理，变成二分变量，使得建模有更好的解读
# 将是否有厅bind到已有数据集
dat01['style_new'] = dat01.halls
dat01.style_new[dat01.style_new > 0] = '有厅'
dat01.style_new[dat01.style_new == 0] = '无厅'
dat01.head()

# In[39]:
# 哑变量编码
# 对于多分类变量，生成哑变量，并设置基准--完全可以在ols函数中使用C参数来处理虚拟变量
data = pd.get_dummies(dat01[['dist', 'floor']])
data.head()

# In[40]:
# 哑变量 种类为 k-1，所以 每个分类字段 都要删除掉1个类别。 减去的两个类别，都是各自字段中类别值最小的。
# 这两个是参照组-在线性回归中使用C函数也可以
data.drop(['dist_石景山', 'floor_high'], axis=1, inplace=True)
data.head()

# In[41]:
# 生成的哑变量与其他所需变量合并成新的数据框
dat1 = pd.concat([data, dat01[['school', 'subway', 'style_new', 'roomnum', 'AREA', 'price']]], axis=1)
dat1.head()

# 2.1 线性回归模型
# In[42]:
# 线性回归模型:手动将分类特征 做的分解
# lm1 = ols("price ~ dist_丰台+dist_朝阳+dist_东城+dist_海淀+dist_西城+school+subway+floor_middle+floor_low+style_new+roomnum+AREA", data=dat1).fit()
lm1 = ols("price ~ dist_丰台+dist_朝阳+dist_东城+dist_海淀+dist_西城+school+subway+floor_middle+floor_low+AREA", data=dat1).fit()
lm1_summary = lm1.summary()
lm1_summary  # 回归结果展示
# %%
# 线性回归模型: 让函数自定进行分类特征分解，分类变量 必须加 C(xxx)。和上面效果相同
lm1 = ols("price ~ C(dist)+school+subway+C(floor)+AREA", data=dat01).fit()
lm1_summary = lm1.summary()
lm1_summary  # 回归结果展示，等同于上面的程序

# In[43]:
dat1['pred1'] = lm1.predict(dat1)
dat1['resid1'] = lm1.resid  # 残差
# 随着 预测值pred1的增加，残差resid呈现喇叭口形状（增加）
dat1.plot('pred1', 'resid1', kind='scatter')  # 模型诊断图，存在异方差现象，对因变量取对数

# 2.2 对数线性模型
# In[44]:
# 对数线性模型
dat1['price_ln'] = np.log(dat1['price'])  # 对price取对数
dat1['AREA_ln'] = np.log(dat1['AREA'])  # 对AREA取对数

# In[45]:
lm2 = ols("price_ln ~ dist_丰台+dist_朝阳+dist_东城+dist_海淀+dist_西城+school+subway+floor_middle+floor_low+AREA",
          data=dat1).fit()
lm2_summary = lm2.summary()
lm2_summary  # 回归结果展示
# In[45]:
lm2 = ols("price_ln ~ dist_丰台+dist_朝阳+dist_东城+dist_海淀+dist_西城+school+subway+floor_middle+floor_low+AREA_ln",
          data=dat1).fit()
lm2_summary = lm2.summary()
lm2_summary  # 回归结果展示

# In[46]:
dat1['pred2'] = lm2.predict(dat1)
dat1['resid2'] = lm2.resid  # 残差
# 随着 预测值pred1的增加，残差resid没有呈现 喇叭状形态。
dat1.plot('pred2', 'resid2', kind='scatter')  # 模型诊断图，异方差现象得到消除

# 2.3 有交互项的对数线性模型，城区和学区之间的交互作用
# In[50]:
# 交互作用的解释： round()方法返回浮点数x的四舍五入值
schools = ['丰台', '朝阳', '东城', '海淀', '西城']
print('石景山非学区房\t', round(dat0[(dat0['dist'] == '石景山') & (dat0['school'] == 0)]['price'].mean(), 2), '万元/平方米\t',
      '石景山学区房\t', round(dat0[(dat0['dist'] == '石景山') & (dat0['school'] == 1)]['price'].mean(), 2), '万元/平方米')
print('-------------------------------------------------------------------------')
for i in schools:
    print(i + '非学区房\t', round(dat1[(dat1['dist_' + i] == 1) & (dat1['school'] == 0)]['price'].mean(), 2), '万元/平方米\t',
          i + '学区房\t', round(dat1[(dat1['dist_' + i] == 1) & (dat1['school'] == 1)]['price'].mean(), 2), '万元/平方米')

# In[51]:
# 探索石景山学区房价格比较低的原因，是否是样本量的问题？
print('石景山非学区房\t', dat0[(dat0['dist'] == '石景山') & (dat0['school'] == 0)].shape[0], '\t',
      '石景山学区房\t', dat0[(dat0['dist'] == '石景山') & (dat0['school'] == 1)].shape[0], '\t', '石景山学区房仅占石景山所有二手房的0.92%')

# In[52]:
# 构造图形揭示不同城区是否学区房的价格问题
df = pd.DataFrame()
dist = ['石景山', '丰台', '朝阳', '东城', '海淀', '西城']
Noschool = []
school = []
for i in dist:
    Noschool.append(dat0[(dat0['dist'] == i) & (dat0['school'] == 0)]['price'].mean())
    school.append(dat0[(dat0['dist'] == i) & (dat0['school'] == 1)]['price'].mean())

df['dist'] = pd.Series(dist)
df['Noschool'] = pd.Series(Noschool)
df['school'] = pd.Series(school)
df

# In[53]:
df1 = df['Noschool'].T.values
df2 = df['school'].T.values
plt.figure(figsize=(10, 6))
x1 = range(0, len(df))
x2 = [i + 0.3 for i in x1]
plt.bar(x1, df1, color='b', width=0.3, alpha=0.6, label='非学区房')
plt.bar(x2, df2, color='r', width=0.3, alpha=0.6, label='学区房')
plt.xlabel('城区')
plt.ylabel('单位面积价格')
plt.title('分城区、是否学区的房屋价格')
plt.legend(loc='upper left')
plt.xticks(range(0, 6), dist)
plt.show()

# In[54]:
# 分城区的学区房分组箱线图
school = ['石景山', '丰台', '朝阳', '东城', '海淀', '西城']
for i in school:
    dat0[dat0.dist == i][['school', 'price']].boxplot(by='school', patch_artist=True)
    plt.xlabel(i + '学区房')

# In[55]:
# 有交互项的对数线性模型，城区和学区之间的交互作用： 直接放到模型中看 显著性α
lm3 = ols("price_ln ~ (dist_丰台+dist_朝阳+dist_东城+dist_海淀+dist_西城)*school+subway+floor_middle+floor_low+AREA_ln",
          data=dat1).fit()
lm3_summary = lm3.summary()
lm3_summary  # 回归结果展示

# In[56]:
# 有交互项的对数线性模型，3个显著特征的交互作用： 直接放到模型中看 显著性α
'''
1、地区*是否学区房 dist_xxx:school 的 p>|t|值 都是显著的，对线性模型有用。
2、地区*是否地铁 dist_xxx:subway 的 p>|t|值 一般显著，对线性模型一般有用（可有可无）
3、是否学区房*是否地铁 school:subway 的 p>|t|值 0.822，对线性模型无用（删除本交互项）
'''
lm4 = ols("price_ln ~ (dist_丰台+dist_朝阳+dist_东城+dist_海淀+dist_西城)*school+ \
                      (dist_丰台+dist_朝阳+dist_东城+dist_海淀+dist_西城)*subway+ \
                      school*subway+ \
                      +subway+floor_middle+floor_low+AREA_ln", data=dat1).fit()
lm4_summary = lm4.summary()
lm4_summary  # 回归结果展示

# 2.4 最终的预测：
# In[55]:
# 假想情形，做预测，x_new是新的自变量
x_new1 = dat1.head(1)
x_new1
# %%
x_new1['dist_朝阳'] = 0
x_new1['dist_东城'] = 1
x_new1['roomnum'] = 2
x_new1['halls'] = 1
x_new1['AREA_ln'] = np.log(70)
x_new1['subway'] = 1
x_new1['school'] = 1
x_new1['style_new'] = "有厅"

# 预测值：
# round()方法返回浮点数x的四舍五入值
# exp()方法返回x的指数： e^x。
print("单位面积房价：", round(math.exp(lm3.predict(x_new1)), 2), "万元/平方米")
print("总价：", round(math.exp(lm3.predict(x_new1)) * 70, 2), "万元")

# %%
