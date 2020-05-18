# -*- coding: utf-8 -*-

"""
    作者:     梁斌
    版本:     1.0
    日期:     2017/02/18
    项目名称： 股票数据分析
"""
import pandas as pd
import pandas_datareader
import datetime
import matplotlib.pylab as plt
from matplotlib.pylab import style
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

style.use('ggplot')     # 设置图片显示的主题样式

# 解决matplotlib显示中文问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def run_main():
    """
        主函数
    """
    # 1. 准备数据
    # 指定股票分析开始日期
    start_date = datetime.datetime(2007, 1, 1)
    # 指定股票分析截止日期
    end_date = datetime.datetime(2017, 3, 1)
    # 股票代码
    stock_code = '600519.SS'    # 沪市贵州茅台

    stock_df = pandas_datareader.data.DataReader(
                        stock_code, 'yahoo', start_date, end_date
                )
    # 预览数据
    print(stock_df.head())

    # 2. 可视化数据
    plt.plot(stock_df['Close'])
    plt.title('股票每日收盘价')
    plt.show()

    # 按周重采样
    stock_s = stock_df['Close'].resample('W-MON').mean()
    stock_train = stock_s['2014':'2016']
    plt.plot(stock_train)
    plt.title('股票周收盘价均值')
    plt.show()

    # 分析 ACF
    acf = plot_acf(stock_train, lags=20)
    plt.title("股票指数的 ACF")
    acf.show()

    # 分析 PACF
    pacf = plot_pacf(stock_train, lags=20)
    plt.title("股票指数的 PACF")
    pacf.show()

    # 3. 处理数据，平稳化数据
    # 这里只是简单第做了一节差分，还有其他平稳化时间序列的方法
    # 可以查询资料后改进这里的平稳化效果
    stock_diff = stock_train.diff()
    diff = stock_diff.dropna()
    print(diff.head())
    print(diff.dtypes)

    plt.figure()
    plt.plot(diff)
    plt.title('一阶差分')
    plt.show()

    acf_diff = plot_acf(diff, lags=20)
    plt.title("一阶差分的 ACF")
    acf_diff.show()

    pacf_diff = plot_pacf(diff, lags=20)
    plt.title("一阶差分的 PACF")
    pacf_diff.show()

    # 4. 根据ACF和PACF定阶并建立模型
    model = ARIMA(stock_train, order=(1, 1, 1), freq='W-MON')
    # 拟合模型
    arima_result = model.fit()
    print(arima_result.summary())

    # 5. 预测
    pred_vals = arima_result.predict('20170102', '20170301',
                                     dynamic=True, typ='levels')
    print(pred_vals)

    # 6. 可视化预测结果
    stock_forcast = pd.concat([stock_s, pred_vals], axis=1, keys=['original', 'predicted'])

    plt.figure()
    plt.plot(stock_forcast)
    plt.title('真实值vs预测值')
    plt.savefig('./stock_pred.png', format='png')
    plt.show()


if __name__ == '__main__':
    run_main()