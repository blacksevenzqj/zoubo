# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 20:41:50 2019

@author: dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import Tools_customize as tc
import FeatureTools as ft

# In[]:
'''
概念：
1、一个特征一个分箱方式的： 每个分箱区间（每个箱子）有一个WOE值
2、一个特征一个分箱方式的： 整个箱体有一个IV值（整个箱体的多个分箱区间的WOE值计算得到一个IV值）
3、WOE指标评价特征分箱好坏； IV指标评价特征重要性，做特征选择（一个特征分箱，其中每个箱子都可以计算一个iv，但是箱子单独的iv没有意义）


终极目标： 将WOE值 映射 到所有特征元素上（所有特征数据都 映射 为WOE值） 进模型
1、使用qcut=20得到每个特征的最初分箱区间num_bins，并调用chi_test_merge_boxes_IV_curve函数将num_bins的分箱区间进行合箱计算：看其IV值变化曲线（卡方、斯皮尔曼：我自己定义的）
并选出每个特征的最优分箱数。
2、还是使用qcut=20得到每个特征的最初分箱区间num_bins，然后调用chi_test_merge_boxes_IV_curve函数将num_bins的分箱区间进行合箱计算 直到 “1中得到的每个特征最优分箱数”；
得到bins_of_col字典：存储了每个特征num_bins的分箱区间进行合箱计算 直到 指定最优箱数 的分箱区间、以及该分箱方式的IV值。
3、通过各种可视化方法验证上述分箱结果（特征分箱区间WOE值曲线、特征IV值柱状图）
4.1、通过到bins_of_col存储的每个特征的分箱区间，计算每个特征每个分箱区间的WOE值（使用cut分箱）；虽然WOE值和上面计算结果相同，但目的是得到分箱区间索引（为了后续WOE数据映射）；
4.2、训练集/测试集 WOE数据 映射： 将每个原始特征数据 按bins_of_col存储的每个特征的分箱区间 分箱后， 再按分箱的结果把WOE结构用map函数映射到数据中；
并将标签Y补充到数据中。
5、所有 原始特征数据 现在都映射成WOE值，进模型训练。
'''

# In[]:
"""
pd.qcut，基于分位数的分箱函数，本质是将连续型变量离散化
只能够处理一维数据。返回箱子的上限和下限
参数q： 要分箱的个数
参数retbins=True： 返回箱子上下限数一维组
箱边缘必须是唯一的，设置duplicates="drop"删除重复的边缘
"""


# model_data["qcut"], updown = pd.qcut(model_data["age"], retbins=True, q=20)
def qcut(df, feature_name, q_num=20, retbins=False):
    return pd.qcut(df[feature_name], q=q_num, retbins=retbins, duplicates="drop")  # 等频分箱


# 统计每个分箱中0和1的数量。 updown：箱子上下限数一维组
def qcut_per_bin_twoClass_num(df, y_name, groupby_col, updown):
    coount_y0 = df[df[y_name] == 0].groupby(by=groupby_col)[y_name].count()
    coount_y1 = df[df[y_name] == 1].groupby(by=groupby_col)[y_name].count()

    # num_bins值分别为每个区间的上界，下界，0出现的次数，1出现的次数
    num_bins = [*zip(updown, updown[1:], coount_y0, coount_y1)]
    # 注意zip会按照最短列来进行结合
    return num_bins


# 计算WOE和BAD RATE
def get_num_bins(data, col, y_name, bins):
    df = data[[col, y_name]].copy()
    # 注意： 使用pd.cut函数之前，确保分箱区间bins的首尾分别为： -np.inf, np.inf
    df["cut"], updown = pd.cut(df[col], bins, retbins=True)  # 可以指定labels=[0,1,2,3]参数
    coount_y0 = df[df[y_name] == 0].groupby(by="cut")[y_name].count()
    coount_y1 = df[df[y_name] == 1].groupby(by="cut")[y_name].count()
    num_bins = [*zip(updown, updown[1:], coount_y0, coount_y1)]
    return num_bins, coount_y0.index


# 分解 num_bins 数据结构：
def break_down_num_bins(num_bins, columns=["min", "max", "count_0", "count_1"]):
    df_temp = pd.DataFrame(num_bins, columns=columns)
    bin_list = tc.set_union(df_temp[columns[0]], df_temp[columns[1]])
    # 注意： 使用pd.cut函数之前，确保分箱区间bins的首尾分别为： -np.inf, np.inf
    bin_list[0], bin_list[-1] = -np.inf, np.inf
    return bin_list


'''
******重点******
希望每组的bad_rate相差越大越好；
woe差异越大越好，应该具有单调性，随着箱的增加，要么由正到负，要么由负到正，只能有一个转折过程；
如果woe值大小变化是有两个转折，比如呈现w型，证明分箱过程有问题
num_bins保留的信息越多越好
'''


# BAD RATE是一个箱中，坏的样本 占 一个箱子里所有样本数的比例 (bad/total)
# 而bad%是一个箱中的坏样本占整个特征中的坏样本的比例
# https://stackoverflow.com/questions/38125319/python-divide-by-zero-encountered-in-log-logistic-regression
def get_woe(num_bins):
    # 通过 num_bins 数据计算 woe
    columns = ["min", "max", "count_0", "count_1"]
    df = pd.DataFrame(num_bins, columns=columns)

    df["total"] = df.count_0 + df.count_1  # 一个箱子当中所有的样本数： 按列相加
    df["percentage"] = df.total / df.total.sum()  # 一个箱子里的样本数 占 所有样本 的比例
    df["bad_rate"] = df.count_1 / df.total  # 一个箱子坏样本的数量 占 一个箱子里所有样本数的比例
    df["good%"] = df.count_0 / df.count_0.sum()  # 一个箱子 好样本 的数量 占 所有箱子里 好样本 的比例
    df["bad%"] = df.count_1 / df.count_1.sum()  # 一个箱子 坏样本 的数量 占 所有箱子里 坏样本 的比例
    df["good_cumsum"] = df["good%"].cumsum()
    df["bad_cumsum"] = df["bad%"].cumsum()
    df["woe"] = np.log(df["bad%"] / df["good%"])
    return df


# 单独计算出woe： 因为测试集映射数据时使用的是训练集的WOE值（测试集不能使用Y值的） 和上面get_num_bins+get_woe函数是一样的，只是简便点
# 本方法有个弊端： 分箱中某一箱统计数量为0时，value_counts维度报错（改为使用上面分别groupby的方式，保险一些）
def get_woe_only(data, col, y_name, bins):
    df = data[[col, y_name]].copy()
    # 注意： 使用pd.cut函数之前，确保分箱区间bins的首尾分别为： -np.inf, np.inf
    df["cut"] = pd.cut(df[col], bins)
    bins_df = tc.groupby_value_counts_unstack(df, "cut", y_name)
    # 返回 特征分箱的WOE值，数据类型时Series（Index索引为分箱区间，这样才能做WOE映射）
    return np.log((bins_df[1] / bins_df[1].sum()) / (bins_df[0] / bins_df[0].sum()))


# 计算IV值
def get_iv(df):
    #    rate = df["good%"] - df["bad%"]
    rate = df["bad%"] - df["good%"]
    iv = np.sum(rate * df.woe)
    return iv


# 计算分箱的斯皮尔曼系数
def spearmanr_bins(df, col, y_name, bin_list):
    X = df[col]
    Y = df[y_name]
    # 注意： 使用pd.cut函数之前，确保分箱区间bins的首尾分别为： -np.inf, np.inf
    d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.cut(X, bin_list)})
    d2 = d1.groupby('Bucket', as_index=True)  # 按照分箱结果进行分组聚合
    # 源码中 以斯皮尔曼系数作为分箱终止条件 while np.abs(r) < 1:
    r, p = scipy.stats.spearmanr(d2.mean().X, d2.mean().Y)  # d2.mean()得到每个箱子的均值
    return r, p


# 斯皮尔曼分箱： 连续自变量X 与 因变量Y（二分类因变量Y 或 连续因变量Y） 之间的相似度
def spearmanr_auto_bins(X, Y, n=10):
    r = 0  # 设定斯皮尔曼 初始值
    r_list = list()
    while np.abs(r) < 1 and n > 1:
        # 用pd.qcut实现最优分箱，Bucket：将X分为n段，n由斯皮尔曼系数决定
        temp_cut_data, updown = pd.qcut(X, n, retbins=True, duplicates="drop")
        d1 = pd.DataFrame({X.name: X, Y.name: Y, "Bucket": temp_cut_data})
        d2 = d1.groupby('Bucket', as_index=True)  # 按照分箱结果进行分组聚合
        r, p = scipy.stats.spearmanr(d2.mean()[X.name], d2.mean()[Y.name])  # 以斯皮尔曼系数作为分箱终止条件
        r_list.append(r)
        n = n - 1
    return d1, updown, n + 1, r_list


# 确保每个箱中都有0和1
def makeSure_zero_one_in_eachBox(num_bins, q_num=20):
    for i in range(q_num):  # 20个箱子
        #    print("第一处i", i)
        # 如果第一个组没有包含正样本或负样本，向后合并
        if 0 in num_bins[0][2:]:
            print("第一处合并", i)
            num_bins[0:2] = [(
                num_bins[0][0],  # 第一行/桶 下界
                num_bins[1][1],  # 第二行/桶 上界
                num_bins[0][2] + num_bins[1][2],
                num_bins[0][3] + num_bins[1][3])]
            continue

        """
        合并了之后，第一行的组是否一定有两种样本了呢？不一定
        如果原本的第一组和第二组都没有包含正样本，或者都没有包含负样本，那即便合并之后，第一行的组也还是没有
        包含两种样本
        所以我们在每次合并完毕之后，还需要再检查，第一组是否已经包含了两种样本
        这里使用continue跳出了本次循环，开始下一次循环，所以回到了最开始的for i in range(20), 让i+1
        这就跳过了下面的代码，又从头开始检查，第一组是否包含了两种样本
        如果第一组中依然没有包含两种样本，则if通过，继续合并，每合并一次就会循环检查一次，最多合并20次
        如果第一组中已经包含两种样本，则if不通过，就开始执行下面的代码
        """
        # 已经确认第一组中肯定包含两种样本了，如果其他组没有包含两种样本，就向前合并
        # 此时的num_bins已经被上面的代码处理过，可能被合并过，也可能没有被合并
        # 但无论如何，我们要在num_bins中遍历，所以写成in range(len(num_bins))
        #    print("2")
        for i in range(len(num_bins)):
            #        print("第二处i", i)
            if 0 in num_bins[i][2:]:
                #                print("第二处合并", i)
                num_bins[i - 1:i + 1] = [(
                    num_bins[i - 1][0],
                    num_bins[i][1],
                    num_bins[i - 1][2] + num_bins[i][2],
                    num_bins[i - 1][3] + num_bins[i][3])]
                break  # 跳出当前这里的循环， 不执行下面的 else， 直接跳到开始for i in range(20)处
                """
                第一个break解释：
                这个break，只有在if被满足的条件下才会被触发
                也就是说，只有发生了合并，才会打断for i in range(len(num_bins))这个循环
                为什么要打断这个循环？因为我们是在range(len(num_bins))中遍历
                但合并发生后，len(num_bins)发生了改变，但循环却不会重新开始
                举个例子，本来num_bins是5组，for i in range(len(num_bins))在第一次运行的时候就等于for i in 
                range(5)
                range中输入的变量会被转换为数字，不会跟着num_bins的变化而变化，所以i会永远在[0,1,2,3,4]中遍历
                进行合并后，num_bins变成了4组，已经不存在=4的索引了，但i却依然会取到4，循环就会报错
                因此在这里，一旦if被触发，即一旦合并发生，我们就让循环被破坏，使用break跳出当前循环
                循环就会回到最开始的for i in range(20)处，for i in range(len(num_bins))却会被重新运行
                这样就更新了i的取值，循环就不会报错了
                """
        else:  # 这个 else: 是单独的，没有和 开头的 if 是一组的，真TM坑啊
            #        print("3")
            # 如果对第一组和对后面所有组的判断中，都没有进入if去合并，则提前结束所有的循环
            # 顺序执行下来 就进这里的 else， break结束循环
            break

    return num_bins


# 卡方检验，合并箱体，画出IV曲线
'''
******重点******
希望每组的bad_rate相差越大越好；
woe差异越大越好，应该具有单调性，随着箱的增加，要么由正到负，要么由负到正，只能有一个转折过程；
如果woe值大小变化是有两个转折，比如呈现w型，证明分箱过程有问题
num_bins保留的信息越多越好
'''


def chi_test_merge_boxes_IV_curve(num_bins, data, x_name, y_name, min_bins=2, pv_limit=0.00001, is_spearmanr=False,
                                  spearmanr_limit=1, graph=True):
    num_bins_ = num_bins.copy()
    IV = []
    axisx = []
    PV = []
    pv_state = True
    bins_pv = np.nan
    bins_woe_pv = np.nan
    bins_iv_pv = np.nan
    spearmanr_state = True
    bins_spearmanr = np.nan
    bins_woe_spearmanr = np.nan
    bins_iv_spearmanr = np.nan

    while len(num_bins_) > min_bins:  # 大于设置的最低分箱个数
        pvs = []
        # 获取 num_bins_两两之间的卡方检验的置信度（或卡方值）
        for i in range(len(num_bins_) - 1):
            x1 = num_bins_[i][2:]
            x2 = num_bins_[i + 1][2:]
            # 0 返回 chi2 值，1 返回 p 值。
            pv = scipy.stats.chi2_contingency([x1, x2])[1]  # p值
            # chi2 = scipy.stats.chi2_contingency([x1,x2])[0] # 计算卡方值
            pvs.append(pv)

        # 通过 卡方p值 进行处理。 合并 卡方p值 最大的两组
        '''
         2、独立性检验： 详细解释在代码： 2_Scorecard_model_case_My.py 和 笔记“1、卡方分布” 中
        '''
        if max(pvs) < pv_limit and pv_state:
            # pv最大值都 < 0.00001， 拒绝原假设，接受备选假设： 特征（两个箱子/类别） 与 因变量Y 相关， 箱子不需要合并
            bins_pv = num_bins_.copy()
            pv_state = False
            bins_woe_pv = get_woe(bins_pv)
            bins_iv_pv = get_iv(bins_woe_pv)
        #           break

        # 斯皮尔曼相关系数选择分箱
        if is_spearmanr and spearmanr_state:
            bin_list = break_down_num_bins(num_bins_)
            r, p = spearmanr_bins(data, x_name, y_name, bin_list)
            if abs(r) == spearmanr_limit:
                bins_spearmanr = num_bins_.copy()
                spearmanr_state = False
                bins_woe_spearmanr = get_woe(bins_spearmanr)
                bins_iv_spearmanr = get_iv(bins_woe_spearmanr)

        i = pvs.index(max(pvs))
        num_bins_[i:i + 2] = [(
            num_bins_[i][0],
            num_bins_[i + 1][1],
            num_bins_[i][2] + num_bins_[i + 1][2],
            num_bins_[i][3] + num_bins_[i + 1][3])]

        bins_woe = get_woe(num_bins_)
        axisx.append(len(num_bins_))
        bins_iv = get_iv(bins_woe)
        IV.append(bins_iv)
        PV.append(max(pvs))  # 卡方p值， 没用到

    if graph:
        plt.figure()
        plt.plot(axisx, IV)
        # plt.plot(axisx,PV)
        plt.xticks(axisx)
        plt.xlabel(x_name + " number of box")
        plt.ylabel("IV")
        plt.show()
        # 选择转折点处，也就是下坠最快的折线点，6→5折线点最陡峭，所以这里对于age来说选择箱数为6

    # 注意： 返回的分箱区间在使用pd.cut函数之前，确保分箱区间bins的首尾分别为： -np.inf, np.inf
    return num_bins_, bins_woe, bins_iv, bins_pv, bins_woe_pv, bins_iv_pv, bins_spearmanr, bins_woe_spearmanr, bins_iv_spearmanr


# 1、自动分箱可视化（画IV曲线：选择最优分箱个数）； 2、根据给定最优分箱个数，得到分箱区间、整个箱体IV值
def graphforbestbin(data, x_name, y_name, min_bins=2, q_num=20, qcut_name="qcut", pv_limit=0.00001, is_spearmanr=False,
                    spearmanr_limit=1, graph=True):
    df = data[[x_name, y_name]]

    df[qcut_name], updown = qcut(df, x_name, q_num, retbins=True)

    num_bins = qcut_per_bin_twoClass_num(df, y_name, qcut_name, updown)

    num_bins = makeSure_zero_one_in_eachBox(num_bins, q_num)

    # 注意： 返回的分箱区间在使用pd.cut函数之前，确保分箱区间bins的首尾分别为： -np.inf, np.inf
    return chi_test_merge_boxes_IV_curve(num_bins, df, x_name, y_name, min_bins, pv_limit, is_spearmanr,
                                         spearmanr_limit, graph)


# 手动指定分箱区间（不能使用自动分箱的变量（稀疏数据））
'''
保证区间覆盖使用 np.inf替换最大值，用-np.inf替换最小值 
原因：比如一些新的值出现，例如家庭人数为30，以前没出现过，改成范围为极大值之后，这些新值就都能分到箱里边了
'''


def hand_bins_customize(hand_bins):
    # return {k:[-np.inf,*v[:-1],np.inf] for k,v in hand_bins.items()} # 1维数组
    return {k: [[-np.inf, *v[:-1], np.inf]] for k, v in hand_bins.items()}  # 扩为2维数组


# 得到 特征分箱区间 以及 整个箱体IV值：
# 1、自动分箱： 已知最优分箱个数，填充 分箱区间、IV值。
# 2、手动分箱： 已知最优分箱区间，填充 IV值（分箱区间已知，不用填充）
'''
auto_col_bins = {"RevolvingUtilizationOfUnsecuredLines":6,
                "age":6,
                "DebtRatio":4,
                "MonthlyIncome":3,
                "NumberOfOpenCreditLinesAndLoans":5}

hand_bins = {"NumberOfTime30-59DaysPastDueNotWorse":[0,1,2,13]
            ,"NumberOfTimes90DaysLate":[0,1,2,17]
            ,"NumberRealEstateLoansOrLines":[0,1,2,4,54]
            ,"NumberOfTime60-89DaysPastDueNotWorse":[0,1,2,8]
            ,"NumberOfDependents":[0,1,2,3]}
'''


def automatic_hand_binning_all(df, y_name, auto_col_bins, hand_bins, q_num=20):
    bins_of_col = {}

    # 1、自动分箱的分箱区间和分箱后的 IV值
    for col in auto_col_bins:
        afterbins, bins_woe, bins_iv, bins_pv, bins_woe_pv, bins_iv_pv, \
        bins_spearmanr, bins_woe_spearmanr, bins_iv_spearmanr = graphforbestbin(
            df, col, y_name,
            min_bins=auto_col_bins[col], q_num=q_num
        )

        bins_list = tc.set_union(bins_woe["min"], bins_woe["max"])
        # 保证区间覆盖使用 np.inf 替换最大值 -np.inf 替换最小值
        bins_list[0], bins_list[-1] = -np.inf, np.inf
        bins_of_col[col] = [bins_list, bins_iv]

    # 2、手动分箱的分箱区间和分箱后的 IV值
    hand_bins = hand_bins_customize(hand_bins)  # 首位分箱界线换为：-np.inf、np.inf
    for col in hand_bins:
        # 手动分箱区间已给定，使用cut函数指定分箱后，求WOE及其IV值。
        num_bins_temp, bin_index = get_num_bins(df, col, y_name, hand_bins[col][0])
        iv_temp = get_iv(get_woe(num_bins_temp))
        hand_bins[col].append(iv_temp)

    # 3、合并手动分箱数据    
    bins_of_col.update(hand_bins)
    return bins_of_col


# 分箱指标可视化（重点）
'''
******重点******
希望每组的bad_rate相差越大越好；
woe差异越大越好，应该具有单调性，随着箱的增加，要么由正到负，要么由负到正，只能有一个转折过程；
如果woe值大小变化是有两个转折，比如呈现w型，证明分箱过程有问题
num_bins保留的信息越多越好
'''


# 1、WOE曲线
def box_indicator_visualization(df, feature_name, y_name, bins_of_col=None, bin_list=None):
    # 分箱区间
    if bins_of_col is not None and type(bins_of_col) is dict:
        bin_interval = bins_of_col[feature_name][0]
    elif bin_list is not None and type(bin_list) is list:
        bin_interval = bin_list
    else:
        raise Exception('bins_of_col Type is Error')

    num_bins, bin_index = get_num_bins(df, feature_name, y_name, bin_interval)
    bins_woe = get_woe(num_bins)
    bins_woe["bin_index"] = bin_index  # 列值为索引，type为Index
    bins_iv = get_iv(bins_woe)

    import matplotlib as mpl
    mpl.rcParams['font.sans-serif'] = 'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False

    # 柱状图：
    #    axe1 = bins_woe[["good%","bad%"]].plot.bar()
    #    axe1.set_xticklabels(bin_index,rotation=15)
    #    axe1.set_ylabel("Num")
    #    axe1.set_title("bar of " + feature_name)

    fig, axe = plt.subplots(1, 1, figsize=(10, 10))
    # 柱状图：
    bar_index = np.arange(len(bin_index))
    axe.bar(bar_index - 0.2 / 2, bins_woe["good%"], width=.2, label='good%')
    axe.bar(bar_index + 0.2 / 2, bins_woe["bad%"], width=.2, label='bad%')
    axe.legend(fontsize=16)  # 图例
    axe.set_xlabel("共%d个箱体：差第一个箱体%s" % (len(bin_index), bin_index[0]), fontsize=16)  # x轴标签
    axe.set_xticklabels(bin_index, rotation=15, fontsize=12)
    axe.set_title("bar of " + feature_name, fontsize=16)

    # 折线图：
    fig, axe = plt.subplots(1, 1, figsize=(10, 10))
    axe.plot(bins_woe["bad_rate"], color='blue', label='bad_rate')
    axe.plot(bins_woe["good%"], color='black', label='good%')
    axe.plot(bins_woe["bad%"], color='green', label='bad%')
    axe.plot(bins_woe["woe"], color='red', label='woe')
    axe.legend(fontsize=16)  # 图例
    axe.set_xlabel("共%d个箱体：差第一个箱体%s，IV=%.4f" % (len(bin_index), bin_index[0], bins_iv), fontsize=16)  # x轴标签
    axe.set_xticklabels(bin_index, rotation=50, fontsize=12)
    axe.set_title(feature_name + ' WOE cure', fontsize=16)  # 图名

    fig, axe = plt.subplots(1, 1, figsize=(10, 10))
    axe.plot(bins_woe["good_cumsum"], bins_woe["bad_cumsum"], color='purple', label='ROC曲线')
    axe.plot((0, 1), (0, 1), c='b', lw=1.5, ls='--', alpha=0.7)  # 横轴fprs2：0→1范围；竖轴tprs2：0→1范围
    axe.plot((0, bins_woe["good_cumsum"][0]), (0, bins_woe["bad_cumsum"][0]), c='r', lw=1.5, ls='--', alpha=0.7)
    axe.legend(loc="top left", fontsize=16)  # 图例
    axe.grid(b=True)
    axe.set_xlabel('FPR: good_cumsum', fontsize=16)  # x轴标签
    axe.set_ylabel('TPR: bad_cumsum', fontsize=16)  # y轴标签
    axe.set_title('good/bad-ROC曲线', fontsize=16)  # 图名

    return bins_woe, bins_iv


# 2、斯皮尔曼相关系数[-1,1]： （对之前的 分箱结果 进行 检测验证）
def spearmanr_visualization(df, y_name, bins_of_col):
    rlist = []
    index = []  # x轴的标签
    collist = []
    for i, col in enumerate(bins_of_col):
        print("x" + str(i + 1), col, bins_of_col[col][0])
        r, p = spearmanr_bins(df, col, y_name, bins_of_col[col][0])
        rlist.append(r)
        index.append("x" + str(i + 1))
        collist.append(col)

    fig1 = plt.figure(1, figsize=(8, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    x = np.arange(len(index)) + 1
    ax1.bar(x, rlist, width=.4)
    ax1.plot(x, rlist)
    ax1.set_xticks(x)
    ax1.set_xticklabels(index, rotation=0, fontsize=15)
    ax1.set_ylabel('R', fontsize=16)  # IV(Information Value),
    # 在柱状图上添加数字标签
    for a, b in zip(x, rlist):
        plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=12)
    plt.show()

    a = np.array(index)
    b = np.array(collist)
    c = np.array(rlist)
    d = np.vstack([a, b, c])
    df_ = pd.DataFrame(d).T
    ft.df_change_colname(df_, {0: "x_axis", 1: "feature", 2: "spearmanr"})
    df_ = ft.data_sort(df_, ["spearmanr"], [False])
    return df_


# 3、IV可视化（特征选择）
'''
IV值 取值区间如下：
1、0 --- 0.02 弱
2、0.02 --- 0.1 有价值
3、0.1 --- 0.4 很有价值
4、0.4 --- 0.6 非常强
5、0.6 以上 单独将变量拿出来，如果是信用评级，单独做一条规则。
'''


def iv_visualization(bins_of_col):
    ivlist = []  # 各变量IV
    index = []  # x轴的标签
    collist = []
    for i, col in enumerate(bins_of_col):
        print("x" + str(i + 1), col, bins_of_col[col][1])
        ivlist.append(bins_of_col[col][1])
        index.append("x" + str(i + 1))
        collist.append(col)

    fig1 = plt.figure(1, figsize=(8, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    x = np.arange(len(index)) + 1
    ax1.bar(x, ivlist, width=.4)  # ax1.bar(range(len(index)),ivlist, width=0.4)#生成柱状图  #ax1.bar(x,ivlist,width=.04)
    ax1.set_xticks(x)
    ax1.set_xticklabels(index, rotation=0, fontsize=15)
    ax1.set_ylabel('IV', fontsize=16)  # IV(Information Value),
    # 在柱状图上添加数字标签
    for a, b in zip(x, ivlist):
        plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=12)
    plt.show()

    a = np.array(index)
    b = np.array(collist)
    c = np.array(ivlist)
    d = np.vstack([a, b, c])
    df_ = pd.DataFrame(d).T
    ft.df_change_colname(df_, {0: "x_axis", 1: "feature", 2: "iv"})
    df_ = ft.data_sort(df_, ["iv"], [False])
    return df_


# 盒须图分箱： 根据Y=1的盒须图区间 对 整体数据进行分箱：
def box_whisker_diagram(df, feature_name, y_name, y_value=1):
    df_temp = df[df[y_name] == y_value][feature_name]
    val_list = ft.box_whisker_diagram_Interval(df_temp)
    # 注意： 使用pd.cut函数之前，确保分箱区间bins的首尾分别为： -np.inf, np.inf
    val_list[0], val_list[-1] = -np.inf, np.inf
    return val_list


# =============================================================================
# 将WOE值 映射 到所有特征元素上（所有特征数据都 映射 为WOE值）
# 1、将 所有特征的 WOE值 存储到 字典 中
def storage_woe_dict(data, y_name, bins_of_col):
    woeall = {}
    for col in bins_of_col:
        # woeall字典中每个元素（特征）的value为： 特征分箱的WOE值，数据类型时Series（Index索引为分箱区间，这样才能做WOE映射）
        woeall[col] = get_woe_only(data, col, y_name, bins_of_col[col][0])
    return woeall


# 2.1、训练集/测试集 WOE数据 映射：
def woe_mapping(data, y_name, bins_of_col, woeall, is_validation=True, save_path=None, encoding="UTF-8"):
    model_woe = pd.DataFrame(index=data.index)

    # 将每个原始特征数据 按bins_of_col存储的每个特征的分箱区间 分箱后， 再按分箱的结果把WOE结构用map函数映射到数据中
    for col in bins_of_col:
        # 注意： 使用pd.cut函数之前，确保分箱区间bins的首尾分别为： -np.inf, np.inf
        model_woe[col] = pd.cut(data[col], bins_of_col[col][0]).map(woeall[col])

    # 将标签补充到数据中（只有训练集、验证集/测试集 有标签Y， 真实提交测试集是没有标签Y的）
    if is_validation:
        model_woe[y_name] = data[y_name]  # 这就是建模数据了

    # 保存 建模数据
    if save_path is not None:
        ft.writeFile_outData(model_woe, save_path, encoding)

    return model_woe


# 2.2、单特征 WOE数据 映射：
# feature_woe_series： 特征分箱的WOE值，数据类型为Series（索引Index为分箱区间，这样才能做WOE映射）
def woe_mapping_simple(data, col, y_name, bin_list, feature_woe_series):
    if type(bin_list) is not list:
        raise Exception('bin_list Type is Error, must list')
    if type(feature_woe_series) is not pd.Series:
        raise Exception('feature_woe_series Type is Error, must Series')

    return pd.cut(data[col], bin_list).map(feature_woe_series)


# In[]
# =============================================================================
# In[]:
# 二、分类特征“概化”： 对分类水平过多的变量进行合并（概化）： 每个箱子中的 样本数量 接近
# 分类特征类别过多时，一般 “概化”（分箱） 到10个类别，最多到20个类别。
# 注意： 没有像 连续特征 一样 使用 WOE分箱 → IV值 的方式选取最优分箱区间，之后可以自行测试对比一下。
def category_feature_generalization(data, var_d, y_name, bin_num=10):
    # 1、统计每个水平的对应目标变量的均值，和每个水平数量
    '''
    注意： 因为 因变量Y 是二分类，取值区间[0,1]，所以groupby后求均值mean，就是求 分类特征中的每个类别 在y==1时的频次（概率）
    将这些类别尽量以 大致均等的方式， 以 响应率（y==1） 为序 归结为bin_num个大类。 也就是说 响应率（y==1） 相近的类别会被分为一箱。
    '''
    grp = data[[var_d, y_name]].groupby(var_d, as_index=False)  # 注意： 必须使用 as_index=False
    demc = grp[y_name].agg({'mean': 'mean', 'count': 'count'}).sort_values("mean")  # 以 响应率（y==1） 为序

    # 2、将这些类别尽量以 大致均等的方式， 以 响应率（y==1） 为序 归结为bin_num个大类。
    demc["count_cumsum"] = demc["count"].cumsum()
    # 按 值的累加和 分箱
    new_feature_name = "new_" + var_d
    demc[new_feature_name] = demc["count_cumsum"].apply(lambda x: x // (len(data) / bin_num))  # float64
    demc[new_feature_name] = demc[new_feature_name].astype(int)

    # 3、查看 分箱特征 中 每个类别 的 count样本数 是否接近。
    print(demc.groupby(new_feature_name)["count"].sum())

    # 4、将 分类特征 “概化”（分箱）结果 映射到 原数据集，替换 分类特征原始值
    demc_new = demc[[var_d, new_feature_name]].set_index(var_d)  # DataFrame
    data[var_d] = data[var_d].map(demc_new[new_feature_name])


# In[]:
# =============================================================================
# 连续特征分箱选择过程记录：
'''
代码：04094_my.py
# In[]:
# 4.1.2.3、评分的分箱
# 4.1.2.3.1.1、自动分箱可视化（画IV曲线：选择最优分箱个数）
afterbins, bins_woe, bins_iv, bins_pv, bins_woe_pv, bins_iv_pv, \
bins_spearmanr, bins_woe_spearmanr, bins_iv_spearmanr = \
bt.graphforbestbin(tmp_train_credit, "credit_score", "target", is_spearmanr=True) # 注意： 返回的分箱区间在使用pd.cut函数之前，确保分箱区间bins的首尾分别为： -np.inf, np.inf
# In[]:
# 给出的卡方分箱区间 画WOE曲线
bin_list = bt.break_down_num_bins(bins_pv)
# kf_bins_woe == bins_woe_pv、 kf_bins_iv == bins_iv_pv（分箱区间相同，只是从算一遍而已）
kf_bins_woe, kf_bins_iv = bt.box_indicator_visualization(tmp_train_credit, "credit_score", "target", bin_list=bin_list)
print(bt.spearmanr_bins(tmp_train_credit, "credit_score", "target", bin_list)) # 斯皮尔曼分箱系数
# In[]:
# 给出的斯皮尔曼分箱区间 画WOE曲线
bin_list2 = bt.break_down_num_bins(bins_spearmanr)
# sp_bins_woe == bins_woe_spearmanr、 sp_bins_iv == bins_iv_spearmanr（分箱区间相同，只是从算一遍而已）
sp_bins_woe, sp_bins_iv = bt.box_indicator_visualization(tmp_train_credit, "credit_score", "target", bin_list=bin_list2)
print(bt.spearmanr_bins(tmp_train_credit, "credit_score", "target", bin_list2)) # 斯皮尔曼分箱系数

# In[]:
# 先看credit_score的直方图、盒须图（平台信用评分在200-400之间的违约率最高，并非信用评分越低就表征用户越可能逾期还款，所以需要做WOE分箱）
f, axes = plt.subplots(2, 2, figsize=(20, 18))
ft.class_data_distribution(tmp_train_credit, "credit_score", "target", axes)
# In[]:
# 4.1.2.3.1.2、盒须图分箱： 根据Y=1的盒须图区间 画WOE曲线
val_list = bt.box_whisker_diagram(tmp_train_credit, "credit_score", "target")
cs_bins_woe, cs_bins_iv = bt.box_indicator_visualization(tmp_train_credit, "credit_score", "target", bin_list=val_list)
print(bt.spearmanr_bins(tmp_train_credit, "credit_score", "target", val_list)) # 斯皮尔曼分箱系数

# In[]:
# 4.1.2.3.2、训练集 WOE数据 映射： 
# 选择 斯皮尔曼分箱区间 因为从 WOE图 和 sp_bins_woe（bins_woe_spearmanr）统计表格中看出，各项指标都相对比较好。
bin_woe_map = sp_bins_woe["woe"]
ft.seriers_change_index(bin_woe_map, sp_bins_woe["bin_index"])
tmp_train_credit["credit_score_woe"] = bt.woe_mapping_simple(tmp_train_credit, "credit_score", "target", bin_list, bin_woe_map)
'''









