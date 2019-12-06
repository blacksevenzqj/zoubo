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
    df["cut"], updown = pd.cut(df[col], bins, retbins=True)
    coount_y0 = df[df[y_name] == 0].groupby(by="cut")[y_name].count()
    coount_y1 = df[df[y_name] == 1].groupby(by="cut")[y_name].count()
    num_bins = [*zip(updown, updown[1:], coount_y0, coount_y1)]
    return num_bins, coount_y0.index


# 分解 num_bins 数据结构：
def break_down_num_bins(num_bins, columns, col1, col2):
    df_temp = pd.DataFrame(num_bins, columns=columns)
    bin_list = tc.set_union(df_temp[col1], df_temp[col2])
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
    df["woe"] = np.log(df["good%"] / df["bad%"])
    return df


# 单独计算出woe： 因为测试集映射数据时使用的是训练集的WOE值（测试集不能使用Y值的） 和上面get_num_bins+get_woe函数是一样的，只是简便点
def get_woe_only(data, col, y_name, bins):
    df = data[[col, y_name]].copy()
    df["cut"] = pd.cut(df[col], bins)
    bins_df = tc.groupby_value_counts_unstack(df, "cut", y_name)
    woe = bins_df["woe"] = np.log((bins_df[0] / bins_df[0].sum()) / (bins_df[1] / bins_df[1].sum()))
    return woe


# 计算IV值
def get_iv(df):
    rate = df["good%"] - df["bad%"]
    iv = np.sum(rate * df.woe)
    return iv


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
    columns = ["min", "max", "count_0", "count_1"]
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
         2、独立性检验
         可以看成：一个特征中的多个类别/分桶  与  另一个特征中多个类别/分桶  的一个条件（类别数量）观测值 与 期望值 的计算。
         原假设： X与Y不相关   特征（两个箱子/类别） 与 因变量Y 不相关， 箱子需要合并
         备选假设： X与Y相关   特征（两个箱子/类别） 与 因变量Y 相关， 箱子不需要合并
         理论上应该这样做，但这里不是
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
            bin_list = break_down_num_bins(num_bins_, columns, "min", "max")
            X = data[x_name]
            Y = data[y_name]
            d1 = pd.DataFrame(
                {"X": X, "Y": Y, "Bucket": pd.cut(X, bin_list)})  # 用pd.qcut实现最优分箱，Bucket：将X分为n段，n由斯皮尔曼系数决定
            d2 = d1.groupby('Bucket', as_index=True)  # 按照分箱结果进行分组聚合
            r, p = scipy.stats.spearmanr(d2.mean().X, d2.mean().Y)  # 以斯皮尔曼系数作为分箱终止条件
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
        plt.xlabel("number of box")
        plt.ylabel("IV")
        plt.show()
        # 选择转折点处，也就是下坠最快的折线点，6→5折线点最陡峭，所以这里对于age来说选择箱数为6

    return num_bins_, bins_woe, bins_iv, bins_pv, bins_woe_pv, bins_iv_pv, bins_spearmanr, bins_woe_spearmanr, bins_iv_spearmanr


def graphforbestbin(data, x_name, y_name, min_bins=2, q_num=20, qcut_name="qcut", pv_limit=0.00001, is_spearmanr=False,
                    spearmanr_limit=1, graph=True):
    df = data[[x_name, y_name]].copy()

    df[qcut_name], updown = qcut(df, x_name, q_num, retbins=True)

    num_bins = qcut_per_bin_twoClass_num(df, y_name, qcut_name, updown)

    num_bins = makeSure_zero_one_in_eachBox(num_bins, q_num)

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


# 自动分箱、手动分箱
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

    # 1、自动分箱的分箱区间和分箱后的 IV 值
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

    # 2、手动分箱的分箱区间和分箱后的 IV 值
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
def box_indicator_visualization(df, feature_name, y_name, bins_of_col):
    num_bins, bin_index = get_num_bins(df, feature_name, y_name, bins_of_col[feature_name][0])
    bins_woe = get_woe(num_bins)

    print(bin_index.tolist())
    import matplotlib as mpl
    mpl.rcParams['font.sans-serif'] = 'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False

    axe1 = bins_woe[["good%", "bad%"]].plot.bar()
    axe1.set_xticklabels(bin_index, rotation=15)
    axe1.set_ylabel("Num")
    axe1.set_title("bar of " + feature_name)

    fig, axe = plt.subplots(1, 1, figsize=(10, 10))
    # 柱状图：
    # bar_index = np.arange(len(bin_index))
    # axe.bar(bar_index-0.3/2, bins_woe["count_0"], width=.3)
    # axe.bar(bar_index+0.3/2, bins_woe["count_1"], width=.3)
    # axe.set_xticklabels(bin_index, rotation=15)
    # axe.set_title("Normal distribution for skew")

    axe.plot(bins_woe["bad_rate"], color='blue', label='bad_rate')
    axe.plot(bins_woe["good%"], color='black', label='good%')
    axe.plot(bins_woe["bad%"], color='green', label='bad%')
    axe.plot(bins_woe["woe"], color='red', label='woe')
    axe.legend(fontsize=16)  # 图例
    axe.set_xlabel("共%d个箱体：差第一个箱体%s" % (len(bin_index), bin_index[0]), fontsize=16)  # x轴标签
    axe.set_xticklabels(bin_index, rotation=50, fontsize=12)
    axe.set_title(feature_name + ' WOE cure', fontsize=16)  # 图名


# 2、斯皮尔曼相关系数[-1,1]： （对之前的 分箱结果 进行 检测验证）
def spearmanr_visualization(df, y_name, bins_of_col):
    rlist = []
    index = []  # x轴的标签
    collist = []
    for i, col in enumerate(bins_of_col):
        print("x" + str(i + 1), col, bins_of_col[col][0])
        X = df[col]
        Y = df[y_name]
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.cut(X, bins_of_col[col][0])})
        d2 = d1.groupby('Bucket', as_index=True)  # 按照分箱结果进行分组聚合
        r, p = scipy.stats.spearmanr(d2.mean().X, d2.mean().Y)  # 源码中 以斯皮尔曼系数作为分箱终止条件 while np.abs(r) < 1:
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


# 3、IV可视化
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


# =============================================================================
# 将WOE值 映射 到所有特征元素上（所有特征数据都 映射 为WOE值）
# 1、将 所有特征的 WOE值 存储到 字典 中
def storage_woe_dict(data, y_name, bins_of_col):
    woeall = {}
    for col in bins_of_col:
        woeall[col] = get_woe_only(data, col, y_name, bins_of_col[col][0])
    return woeall


# 2、训练集/测试集 WOE数据 映射：
def woe_mapping(data, y_name, bins_of_col, woeall, is_validation=True, save_path=None, encoding="UTF-8"):
    model_woe = pd.DataFrame(index=data.index)

    # 将每个原始特征数据 按bins_of_col存储的每个特征的分箱区间 分箱后， 再按分箱的结果把WOE结构用map函数映射到数据中
    for col in bins_of_col:
        model_woe[col] = pd.cut(data[col], bins_of_col[col][0]).map(woeall[col])

    # 将标签补充到数据中（只有训练集、验证集/测试集 有标签Y， 真实提交测试集是没有标签Y的）
    if is_validation:
        model_woe[y_name] = data[y_name]  # 这就是建模数据了

    # 保存 建模数据
    if save_path is not None:
        ft.writeFile_outData(model_woe, save_path, encoding)

    return model_woe

