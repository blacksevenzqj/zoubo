# -*- coding: utf-8 -*-

"""
    作者:     梁斌
    版本:     1.0
    日期:     2017/02/13
    项目名称： 世界高峰数据可视化 (World's Highest Mountains)
    参考：    https://www.kaggle.com/alex64/d/abcsds/highest-mountains/let-s-climb
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')     # 设置图片显示的主题样式

# 解决matplotlib显示中文问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

dataset_path = './dataset/Mountains.csv'


def preview_data(data):
    """
        数据预览
    """
    # 数据预览
    print(data.head())

    # 数据信息
    print(data.info())


def proc_success(val):
    """
        处理 'Ascents bef. 2004' 列中的数据
    """
    if '>' in str(val):
        return 200
    elif 'Many' in str(val):
        return 160
    else:
        return val


def run_main():
    """
        主函数
    """
    data = pd.read_csv(dataset_path)

    preview_data(data)

    # 数据重构
    # 重命名列名
    data.rename(columns={'Height (m)': 'Height', 'Ascents bef. 2004': 'Success',
                         'Failed attempts bef. 2004': 'Failed'}, inplace=True)

    # 数据清洗
    data['Failed'] = data['Failed'].fillna(0).astype(int)
    data['Success'] = data['Success'].apply(proc_success)
    data['Success'] = data['Success'].fillna(0).astype(int)
    data = data[data['First ascent'] != 'unclimbed']
    data['First ascent'] = data['First ascent'].astype(int)

    # 可视化数据
    # 1. 登顶次数 vs 年份

    plt.hist(data['First ascent'].astype(int), bins=20)
    plt.ylabel('高峰数量')
    plt.xlabel('年份')
    plt.title('登顶次数')
    plt.savefig('./first_ascent_vs_year.png')
    plt.show()

    # 2. 高峰vs海拔
    data['Height'].plot.hist(color='steelblue', bins=20)
    plt.bar(data['Height'],
            (data['Height'] - data['Height'].min()) / (data['Height'].max() - data['Height'].min()) * 23,   # 按比例缩放
            color='red',
            width=30, alpha=0.2)
    plt.ylabel('高峰数量')
    plt.xlabel('海拔')
    plt.text(8750, 20, "海拔", color='red')
    plt.title('高峰vs海拔')
    plt.savefig('./mountain_vs_height.png')
    plt.show()

    # 3. 首次登顶
    data['Attempts'] = data['Failed'] + data['Success']  # 攀登尝试次数
    fig = plt.figure(figsize=(13, 7))
    fig.add_subplot(211)
    plt.scatter(data['First ascent'], data['Height'], c=data['Attempts'], alpha=0.8, s=50)
    plt.ylabel('海拔')
    plt.xlabel('登顶')

    fig.add_subplot(212)
    plt.scatter(data['First ascent'], data['Rank'].max() - data['Rank'], c=data['Attempts'], alpha=0.8, s=50)
    plt.ylabel('排名')
    plt.xlabel('登顶')
    plt.savefig('./mountain_vs_attempts.png')
    plt.show()

    # 课后练习，尝试使用seaborn或者bokeh重现上述显示的结果

if __name__ == '__main__':
    run_main()
