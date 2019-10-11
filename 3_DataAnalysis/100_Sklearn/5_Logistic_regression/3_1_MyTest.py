import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

os.chdir(r"E:\code\python_workSpace\idea_space\zoubo\3_DataAnalysis\0_Feature_engineering\101_My_Learn\98_My_Model_Statistical_indicators\100_Inferential_statistics\2_Chi-square_test")

# 准备数据：
model_data = pd.read_csv(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\5_Logistic_regression\model_data.csv")
vali_data = pd.read_csv(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\5_Logistic_regression\vali_data.csv")
print(model_data.shape) # (195008, 12)
print(vali_data.shape) # (83576, 12)
model_data.drop('Unnamed: 0', inplace=True, axis=1)
vali_data.drop('Unnamed: 0', inplace=True, axis=1)
print(model_data.shape) # (195008, 11)
print(vali_data.shape) # (83576, 11)

model_data_My = model_data.copy()
vali_data_My = vali_data.copy()
q = 20


# In[]:
# 一、自动分桶：
# 1、分类变量 两两分桶 与 因变量Y 做 卡方检验
model_data_My['qcut'], updown_age = pd.qcut(model_data_My.age, retbins=True, q=q)
print(model_data_My['SeriousDlqin2yrs'].astype('int64').groupby(model_data_My['qcut']).agg(['count', 'mean']))
#print(model_data_My['qcut'].value_counts())

cut=[]    #  cut 存放箱段节点
cut.append(float('-inf'))    # 在列表前加-inf
for i in range(1,q):        # n是前面的分箱的分割数，所以分成n+1份
    qua=model_data_My.age.quantile(i/(q))         #quantile 分为数  得到分箱的节点
    cut.append(round(qua,4))    # 保留4位小数       #返回cut
cut.append(float('inf')) 

# In[]:
# 1.1、使用 crosstab 交叉表快捷， 但是 不能计算其他指标。
bins_stats_crosstab_age = pd.crosstab(model_data_My['qcut'], model_data_My['SeriousDlqin2yrs'], margins=True)
print(bins_stats_crosstab_age.iloc[0:2, :-1])
print('''chisq = %6.4f
p-value = %6.4fs
dof = %i
expected_freq = %s'''  % stats.chi2_contingency(bins_stats_crosstab_age.iloc[0:2, :-1]))

# In[]:
# 1.2、使用 groupby(分桶特征)： 卡方检验
age_coount_y0 = model_data_My[model_data_My["SeriousDlqin2yrs"] == 0].groupby(by="qcut").count()["SeriousDlqin2yrs"]
age_coount_y1 = model_data_My[model_data_My["SeriousDlqin2yrs"] == 1].groupby(by="qcut").count()["SeriousDlqin2yrs"]

# num_bins值分别为每个区间的上界，下界，0出现的次数，1出现的次数
num_bins_age = [*zip(updown_age,updown_age[1:],age_coount_y0,age_coount_y1)] # 左开右闭（除了第一个是左闭）
#num_bins_age = [*zip(cut,cut[1:],age_coount_y0,age_coount_y1)]
# 注意zip会按照最短列来进行结合
print(num_bins_age)
# In[]:
def get_num_bins(data,col,y,bins):
    df = data[[col,y]].copy()
    df["cut"],bins = pd.cut(df[col],bins,retbins=True)
    coount_y0 = df.loc[df[y]==0].groupby(by="cut").count()[y]
    coount_y1 = df.loc[df[y]==1].groupby(by="cut").count()[y]
    num_bins = [*zip(bins,bins[1:],coount_y0,coount_y1)]
    return num_bins
    
def get_woe(num_bins):
    # 通过 num_bins 数据计算 woe
    columns = ["min","max","count_0","count_1"]  # 左开右闭（除了第一个是左闭）
    df = pd.DataFrame(num_bins,columns=columns)

    df["total"] = df.count_0 + df.count_1 # 一个箱子当中所有的样本数： 按列相加
    df["percentage"] = df.total / df.total.sum() # 一个箱子里的样本数，占所有样本的比例
    df["bad_rate"] = df.count_1 / df.total # 一个箱子坏样本的数量占一个箱子里边所有样本数的比例
    df["good%"] = df.count_0/df.count_0.sum()
    df["bad%"] = df.count_1/df.count_1.sum()
    df["woe"] = np.log(df["good%"] / df["bad%"])
    return df
 
#计算IV值
def get_iv(df):
    rate = df["good%"] - df["bad%"]
    iv = np.sum(rate * df.woe)
    return iv

def graphforbestbin_My(num_bins, n, graph=True):
    IV = []
    axisx = []
    PV = []
    pv_state = True
    bins_df = np.nan
    num_bins_pv = np.nan
    
    while len(num_bins) > n:
        pvs = []
        for i in range(len(num_bins)-1):
            x1 = num_bins[i][2:]
            x2 = num_bins[i+1][2:]
            pv = stats.chi2_contingency([x1,x2])[1]
            pvs.append(pv)

        # 通过 卡方p值 进行处理。 合并 卡方p值 最大的两组
        '''
         2、独立性检验
         可以看成：一个特征中的多个类别/分桶  与  另一个特征中多个类别/分桶  的一个条件（类别数量）观测值 与 期望值 的计算。
         原假设： X与Y不相关   特征（两个箱子/类别） 与 因变量Y 不相关， 箱子需要合并
         备选假设： X与Y相关   特征（两个箱子/类别） 与 因变量Y 相关， 箱子不需要合并
         理论上应该这样做，但这里不是
        '''
        if max(pvs) < 0.001 and pv_state:
            # pv最大值都 < 0.001， 拒绝原假设，接受备选假设： 特征（两个箱子/类别） 与 因变量Y 相关， 箱子不需要合并
           num_bins_pv = num_bins.copy()
           pv_state = False
#           break 
            
        i = pvs.index(max(pvs))
        num_bins[i:i+2] = [(
            num_bins[i][0],
            num_bins[i+1][1],
            num_bins[i][2]+num_bins[i+1][2],
            num_bins[i][3]+num_bins[i+1][3])]

        bins_df = pd.DataFrame(get_woe(num_bins)) # 左开右闭（除了第一个是左闭）
        axisx.append(len(num_bins))
        iv = get_iv(bins_df)
        IV.append(iv)
        PV.append(max(pvs)) # 卡方p值
        
    if graph:
        fig, axs = plt.subplots(1,2,figsize=(14,7))
        axs[0].plot(axisx,IV,color="r")
        axs[0].set_xticks(axisx)
        axs[0].set_title("number of box for IV")
        axs[0].set_xlabel("number of box")
        axs[0].set_ylabel("IV")
        
        axs[1].plot(axisx,PV,color="b")
        axs[1].set_xticks(axisx)
        axs[1].set_title("number of box for PV")
        axs[1].set_xlabel("number of box")
        axs[1].set_ylabel("PV")
        plt.show()
        
    return bins_df, num_bins_pv, iv

# In[]:
num_bins_age_ = num_bins_age.copy()
bins_df_age_, num_bins_pv_age_, iv_age_ = graphforbestbin_My(num_bins_age_, 2)
print(len(bins_df_age_), len(num_bins_pv_age_))


# In[]:
# 2、分类变量 两两分桶 与 因变量Y 做 斯皮尔曼相关系数 
X = model_data_My.age
Y = model_data_My['SeriousDlqin2yrs']
r = 0
ns = q

while np.abs(r) < 1:  
    d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, ns)}) # 用pd.qcut实现最优分箱，Bucket：将X分为ns段，ns由斯皮尔曼系数决定    
    d2 = d1.groupby('Bucket', as_index = True) # 按照分箱结果进行分组聚合        
    r, p = stats.spearmanr(d2.mean().X, d2.mean().Y) # 以斯皮尔曼系数作为分箱终止条件
    ns = ns - 1
    print(r, p)
#    print(d2.mean())

cut_s = []    #  cut 存放箱段节点
cut_s.append(float('-inf'))    # 在列表前加-inf
for i in range(1,ns+1):        # n是前面的分箱的分割数，所以分成n+1份
    qua=model_data_My.age.quantile(i/(ns+1))         #quantile 分为数  得到分箱的节点
    cut_s.append(round(qua,4))    # 保留4位小数       #返回cut
cut_s.append(float('inf')) 



# In[]:
# 一、特征选择：
num_bins_age_fin = num_bins_age.copy()
bins_of_col = {}

bins_df_age_fin, num_bins_pv_age_fin, iv_age_fin = graphforbestbin_My(num_bins_age_fin, 6, False)
print(len(bins_df_age_fin), len(num_bins_pv_age_fin))
bins_list = sorted(set(bins_df_age_fin["min"]).union(bins_df_age_fin["max"])) # 左开右闭（除了第一个是左闭）
#保证区间覆盖使用 np.inf 替换最大值 -np.inf 替换最小值
bins_list[0],bins_list[-1] = -np.inf,np.inf
bins_of_col['age'] = [bins_list, iv_age_fin]

num_bins_age_fin2 = get_num_bins(model_data_My, 'age', 'SeriousDlqin2yrs', bins_of_col['age'][0])
iv_temp = get_iv(get_woe(num_bins_age_fin2))

# In[]:
# 1、皮尔森相似度：
def corrFunction(data_corr):
    corr = data_corr.corr()#计算各变量的相关性系数
    # 皮尔森相似度 绝对值 排序
    df_all_corr_abs = corr.abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_all_corr_abs.rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: 'Correlation_Coefficient'}, inplace=True)
    print(df_all_corr_abs[(df_all_corr_abs["Feature_1"] != 'SeriousDlqin2yrs') & (df_all_corr_abs['Feature_2'] == 'SeriousDlqin2yrs')])
    print()
    # 皮尔森相似度 排序
    df_all_corr = corr.unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_all_corr.rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: 'Correlation_Coefficient'}, inplace=True)
    print(df_all_corr[(df_all_corr["Feature_1"] != 'SeriousDlqin2yrs') & (df_all_corr['Feature_2'] == 'SeriousDlqin2yrs')])
    
    
    xticks = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']#x轴标签
    yticks = list(corr.index)#y轴标签
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(1, 1, 1)
    sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax1, annot_kws={'size': 12, 'weight': 'bold', 'color': 'black'})#绘制相关性系数热力图
    ax1.set_xticklabels(xticks, rotation=0, fontsize=14)
    ax1.set_yticklabels(yticks, rotation=0, fontsize=14)
    plt.show()

corrFunction(model_data_My[['age','SeriousDlqin2yrs']])

# In[]:
# 2、卡方值比较：
'''
2、独立性检验
可以看成：一个特征中的多个类别/分桶  与  另一个特征中多个类别/分桶  的一个条件（类别数量）观测值 与 期望值 的计算。
原假设：X与Y不相关
备选假设：X与Y相关
'''
model_data_My['cut'] = pd.cut(model_data_My['age'],bins_of_col['age'][0])
print(model_data_My['SeriousDlqin2yrs'].astype('int64').groupby(model_data_My['cut']).agg(['count', 'mean']))
#print(model_data_My['cut'].value_counts())

bins_stats_crosstab_age = pd.crosstab(model_data_My['cut'],model_data_My['SeriousDlqin2yrs'], margins=True)
print(bins_stats_crosstab_age.iloc[:-1, :-1])
print('''chisq = %6.4f
p-value = %6.4fs
dof = %i
expected_freq = %s'''  % stats.chi2_contingency(bins_stats_crosstab_age.iloc[:-1, :-1]))
# chisq = 14033.0747， p-value = 0.0000s

# In[]:
# 3、IV值 比较选择：
ivlist = [] # 各变量IV
index = [] # x轴的标签
for i, col in enumerate(bins_of_col):
    ivlist.append(bins_of_col[col][1])
    index.append("x" + str(i+1))

fig1 = plt.figure(1,figsize=(8,5))
ax1 = fig1.add_subplot(1, 1, 1)
x = np.arange(len(index))+1 
ax1.bar(x,ivlist,width=.4) #  ax1.bar(range(len(index)),ivlist, width=0.4)#生成柱状图  #ax1.bar(x,ivlist,width=.04)
ax1.set_xticks(x)
ax1.set_xticklabels(index, rotation=0, fontsize=15)
ax1.set_ylabel('IV', fontsize=16)   # IV(Information Value),
#在柱状图上添加数字标签
for a, b in zip(x, ivlist):
    plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=12)
plt.show()









