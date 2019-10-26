# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:55:59 2019

@author: seven
"""
import numpy as np
import matplotlib as mpl
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score

'''
默认是以 目标变量（因变量Y）== 1 为基准：
'''


def TP(y_true, y_predict, state=1):
    assert len(y_true) == len(y_predict)
    if state == 1:
        return np.sum((y_true == 1) & (y_predict == 1))
    else:
        return np.sum((y_true == 0) & (y_predict == 0))


def TN(y_true, y_predict, state=1):
    assert len(y_true) == len(y_predict)
    if state == 1:
        return np.sum((y_true == 0) & (y_predict == 0))
    else:
        return np.sum((y_true == 1) & (y_predict == 1))


def FP(y_true, y_predict, state=1):
    assert len(y_true) == len(y_predict)
    if state == 1:
        return np.sum((y_true == 0) & (y_predict == 1))
    else:
        return np.sum((y_true == 1) & (y_predict == 0))


def FN(y_true, y_predict, state=1):
    assert len(y_true) == len(y_predict)
    if state == 1:
        return np.sum((y_true == 1) & (y_predict == 0))
    else:
        return np.sum((y_true == 0) & (y_predict == 1))


# 混淆矩阵
def confusion_matrix_customize(y_true, y_predict, state=1):
    return np.array([
        [TN(y_true, y_predict, state), FP(y_true, y_predict, state)],
        [FN(y_true, y_predict, state), TP(y_true, y_predict, state)],
    ])


# 准确率
def precision_scoreAll_customize(y_true, y_predict, state=1):
    tp = TP(y_true, y_predict, state)
    fp = FP(y_true, y_predict, state)
    tn = TN(y_true, y_predict, state)
    fn = FN(y_true, y_predict, state)
    try:
        return (tp + tn) / (tp + fp + tn + fn)
    except:
        return 0.0


# 精准率
def precision_score_customize(y_true, y_predict, state=1):
    tp = TP(y_true, y_predict, state)
    fp = FP(y_true, y_predict, state)
    try:
        return tp / (tp + fp)
    except:
        return 0.0


# 召回率
def recall_score_customize(y_true, y_predict, state=1):
    tp = TP(y_true, y_predict, state)
    fn = FN(y_true, y_predict, state)
    try:
        return tp / (tp + fn)
    except:
        return 0.0


# 调和平均值
# F1分数的公式为 = 2*查准率*查全率 / (查准率 + 查全率)
def f1_score_customize(y_true, y_predict, state=1):
    precisionScore = precision_score_customize(y_true, y_predict, state)
    recallScore = recall_score_customize(y_true, y_predict, state)
    try:
        return 2 * precisionScore * recallScore / (precisionScore + recallScore)
    except:
        return 0.0


# TPR：就是召回率
def TPR(y_true, y_predict, state=1):
    return recall_score_customize(y_true, y_predict, state)


# 特异度： 1-FPR
def SPE(y_true, y_predict, state=1):
    tn = TN(y_true, y_predict, state)
    fp = FP(y_true, y_predict, state)
    try:
        return tn / (tn + fp)
    except:
        return 0.0


# FPR： 1-特异度
def FPR(y_true, y_predict, state=1):
    fp = FP(y_true, y_predict, state)
    tn = TN(y_true, y_predict, state)
    try:
        return fp / (fp + tn)
    except:
        return 0.0


def thresholdSelection(y_true, decision_scores, decision_start, decision_stop, score, score_type=1, state=1):
    thresholds = np.arange(decision_start, decision_stop, 0.1)
    for threshold in thresholds:
        my_predict = np.array(decision_scores >= threshold, dtype='int')
        if score_type == 1:
            if precision_score_customize(y_true, my_predict, state) > score:
                return threshold
            else:
                continue
        elif score_type == 2:
            if recall_score_customize(y_true, my_predict, state) > score:
                return threshold
            else:
                continue
        elif score_type == 3:
            if f1_score_customize(y_true, my_predict, state) > score:
                return threshold
            else:
                continue
        else:
            if FPR(y_true, my_predict, state) < score:
                return threshold
            else:
                continue

    return np.nan


# 手动综合指标：
def ComprehensiveIndicator(y_true, decision_scores, state=1):
    # 1.1、手动循环 创建TPR、FPR：
    precisions = []
    recalls = []
    f1Scores = []
    #    tprs = []
    fprs = []
    thresholds = np.arange(np.min(decision_scores), np.max(decision_scores), 0.1)
    for threshold in thresholds:
        # 重点：将 decision_scores（线性回归𝜃x+b的结果） 大于（正预测）  其自身区间中的某一个阈值 的结果 设置为 预测结果
        my_predict = np.array(decision_scores >= threshold, dtype='int')
        precisions.append(precision_score_customize(y_true, my_predict, state))
        recalls.append(recall_score_customize(y_true, my_predict, state))  # TPR
        f1Scores.append(f1_score_customize(y_true, my_predict, state))
        #        tprs.append(TPR(y_true, my_predict, state)) # recalls
        fprs.append(FPR(y_true, my_predict, state))

    # 1.1.1、计算KS值及其阈值：KS=max(TPR-FPR)
    print("手动计算长度：", len(thresholds), len(recalls), len(fprs))
    maxKsValue = abs(np.array(recalls) - np.array(fprs)).max()
    maxKsIndex = abs(np.array(recalls) - np.array(fprs)).tolist().index(abs(np.array(recalls) - np.array(fprs)).max())
    maxKsThresholds = thresholds[maxKsIndex]
    recallsValue = recalls[maxKsIndex]
    fprsValue = fprs[maxKsIndex]
    print('max(TPR-FPR) = %.4f, 最大阈值 = %.4f, 召回率/TPR = %.4f, FPR = %.4f' % (
    abs(np.array(recalls) - np.array(fprs)).max(), thresholds[maxKsIndex], recalls[maxKsIndex], fprs[maxKsIndex]))
    Ks_PValue = precisions[maxKsIndex]  # KS最大时 精准率值
    Ks_f1Value = f1Scores[maxKsIndex]  # KS最大时 F1分数值
    print('KS最大阈值时：精准率 = %.4f, F1分数 = %.4f' % (Ks_PValue, Ks_f1Value))

    # 1.1.2、计算F1分数最大值及其阈值：
    maxF1ScoresValue = max(f1Scores)
    maxF1ScoresIndex = f1Scores.index(max(f1Scores))  # 从左到右：第一个出现的最大数的索引
    maxF1Thresholds = thresholds[maxF1ScoresIndex]
    print('F1最大值 = %.4f，' % maxF1ScoresValue, '最大阈值 = %.4f' % maxF1Thresholds)
    # 1.1.3、计算F1分数最大值阈值 与 KS值阈值 距离：
    diffValue = maxF1Thresholds - maxKsThresholds
    print('F1最大值阈值 - KS阈值 = %.4f' % diffValue)

    # 默认阈值
    my_predict_def = np.array(decision_scores >= 0, dtype='int')
    # 1.1.3、计算precisions默认值及其阈值：
    defPValue = precision_score_customize(y_true, my_predict_def, state)
    print('Precisions默认值 = %.4f，' % defPValue, '默认阈值 = 0')

    # 1.1.4、计算recalls/TPR默认值及其阈值：
    defRValue = recall_score_customize(y_true, my_predict_def, state)
    print('Recalls默认值 = %.4f，' % defRValue, '默认阈值 = 0')

    # 1.1.5、计算FPR默认值及其阈值：
    defFValue = FPR(y_true, my_predict_def, state)
    print('FPR默认值 = %.4f，' % defFValue, '默认阈值 = 0')

    return thresholds, precisions, recalls, fprs, f1Scores, recallsValue, fprsValue, maxKsThresholds, maxKsValue, maxF1Thresholds, maxF1ScoresValue, diffValue, defPValue, defRValue, defFValue, Ks_PValue, Ks_f1Value


# 手动综合指标图：
def ComprehensiveIndicatorFigure(y_true, decision_scores, axe, state=1):
    mpl.rcParams['font.sans-serif'] = 'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False

    # 1、A1图
    # 以阈值为横坐标，分别以TPR和FPR的值为纵坐标，就可以画出两个曲线，这就是K-S曲线
    '''
    在阈值从最低处-85.68：意味着 几乎所有样本都被预测为正例，所以：
    1、TP预测为1的相对很高，FN漏预测为1（错误预测为0）的=0，TPR = 1；
    2、FP错误预测为1（漏预测为0）的也相对很高，TN预测为0=0，FPR = 1；
    3、也就是说 在 阈值最低点 的模型 TPR = FPR = 1，KS = max(TPR-FPR) = 0，预测没有什么意义。
    '''
    thresholds, precisions, recalls, fprs, f1Scores, recallsValue, fprsValue, maxKsThresholds, maxKsValue, maxF1Thresholds, maxF1ScoresValue, diffValue, defPValue, defRValue, defFValue, Ks_PValue, Ks_f1Value = ComprehensiveIndicator(
        y_true, decision_scores, state)
    axe[0].plot(thresholds, precisions, color='blue', label='精准率')
    axe[0].scatter(0, defPValue, c='#0000FF', s=30, cmap="rainbow")  # 蓝色
    axe[0].plot(thresholds, recalls, color='black', label='召回率/TPR')
    axe[0].scatter(0, defRValue, c='#000000', s=30, cmap="rainbow")  # 黑色
    axe[0].plot(thresholds, f1Scores, color='green',
                label='F1阈值 = %.4f，F1 = %.4f' % (maxF1Thresholds, maxF1ScoresValue))
    axe[0].plot(thresholds, fprs, color='pink', label='FPR')
    axe[0].scatter(0, defFValue, c='#FFC0CB', s=30, cmap="rainbow")  # 粉色
    axe[0].plot((maxKsThresholds, maxKsThresholds), (recallsValue, fprsValue), c='r', lw=1.5, ls='--', alpha=0.7,
                label='KS阈值 = %.4f，KS = %.4f' % (maxKsThresholds, maxKsValue))
    axe[0].plot((maxKsThresholds, maxF1Thresholds), (maxF1ScoresValue, maxF1ScoresValue), c='purple', lw=1.5, ls='-',
                alpha=0.7, label='(F1-KS)的阈值差 = %.4f' % diffValue)
    axe[0].scatter(maxKsThresholds, Ks_PValue, c='#0000FF', s=30, cmap="rainbow")  # 蓝色
    axe[0].scatter(maxKsThresholds, Ks_f1Value, c='#006400', s=30, cmap="rainbow")  # 绿色
    axe[0].legend(loc="center left")  # 图例
    axe[0].set_xlabel('阈值')  # x轴标签
    axe[0].set_ylabel('精准率、召回率/TPR、F1分数、FPR、KS')  # y轴标签
    axe[0].set_title('手动-阈值与精准率、召回率/TPR、F1分数、FPR、KS=max(TPR-FPR)')  # 图名
    axe[0].text(0, defPValue - 0.03, '默认阈值0：%.4f' % defPValue, ha='center', va='bottom')  # fontsize=8
    axe[0].text(0, defRValue + 0.03, '默认阈值0：%.4f' % defRValue, ha='center', va='bottom')
    axe[0].text(0, defFValue - 0.03, '默认阈值0：%.4f' % defFValue, ha='center', va='bottom')
    axe[0].text(maxKsThresholds, recallsValue + 0.01, '%.4f' % recallsValue, ha='center', va='bottom')
    axe[0].text(maxKsThresholds, fprsValue - 0.05, '%.4f' % fprsValue, ha='center', va='bottom')
    axe[0].text(maxKsThresholds, Ks_PValue - 0.03, 'KS阈值：%.4f' % Ks_PValue, ha='center', va='bottom')
    axe[0].text(maxKsThresholds, Ks_f1Value - 0.03, 'KS阈值：%.4f' % Ks_f1Value, ha='center', va='bottom')

    '''
    从上面 手动阈值 从低到高 计算出的TPR和FPR值的趋势是 从高到低的；而在曲线图中TPR和FPR值被默认设置为 从低到高显示
    '''
    axe[1].plot(fprs, recalls, color='purple', label='ROC曲线')
    axe[1].plot((0, 1), (0, 1), c='b', lw=1.5, ls='--', alpha=0.7)  # 横轴fprs2：0→1范围；竖轴tprs2：0→1范围
    axe[1].scatter(fprsValue, recallsValue, c=1, s=30, cmap="rainbow")
    axe[1].plot((fprsValue, fprsValue), (recallsValue, fprsValue), c='r', lw=1.5, ls='--', alpha=0.7)
    axe[1].legend(loc="center right")  # 图例
    axe[1].set_xlabel('FPR')  # x轴标签
    axe[1].set_ylabel('TPR')  # y轴标签
    axe[1].set_title('手动-ROC曲线')  # 图名
    axe[1].grid(b=True)
    axe[1].text(fprsValue, recallsValue + 0.01, '%.4f' % recallsValue, ha='center', va='bottom')  # fontsize=8
    axe[1].text(fprsValue, fprsValue - 0.05, '%.4f' % fprsValue, ha='center', va='bottom')
    axe[1].text(fprsValue + 0.05, (fprsValue + recallsValue) / 2, '%.4f' % maxKsValue, ha='center', va='bottom')


# 自动
'''
默认是以 目标变量（因变量Y）== 1 为基准：
'''


def ComprehensiveIndicatorSkLib(y_true, decision_scores):
    '''
    1、roc_curve 函数使用 decision_scores 和 y_log_predict_proba 计算得到的 fprs 和 tprs 相同，而 阈值不同；y_log_predict_proba 计算的阈值 不能使用。
    2、precision_recall_curve 函数使用 decision_scores 和 y_log_predict_proba 计算得到的 精准率、召回率、F1 相同，而 阈值不同；y_log_predict_proba 计算的阈值 不能使用。
    '''
    # 1.2、自动 创建TPR、FPR 得到 ROC：
    fprs2, tprs2, thresholds2 = roc_curve(y_true, decision_scores)  # 自动-thresholds2 和 手动-thresholds 是不完全相同的，非常类似

    # 1.2.1、自动 计算KS值及其阈值：KS=max(TPR-FPR)
    maxKsValue_auto = abs(np.array(tprs2) - np.array(fprs2)).max()
    maxindex_auto = abs(np.array(tprs2) - np.array(fprs2)).tolist().index(abs(np.array(tprs2) - np.array(fprs2)).max())
    maxKsThresholds_auto = thresholds2[maxindex_auto]
    recallsValue_auto = tprs2[maxindex_auto]
    fprsValue_auto = fprs2[maxindex_auto]
    print('max(TPR-FPR) = %.4f, 最大阈值 = %.4f, 召回率/TPR = %.4f, FPR = %.4f' % (
    abs(np.array(tprs2) - np.array(fprs2)).max(), thresholds2[maxindex_auto], tprs2[maxindex_auto],
    fprs2[maxindex_auto]))

    # 1.3、自动 创建AUC面积：Area Under Curve
    rocAucScore = roc_auc_score(y_true, decision_scores)

    # 默认阈值
    my_predict_def = np.array(decision_scores >= 0, dtype='int')
    # 计算recalls/TPR默认值及其阈值：
    defRValue = recall_score(y_true, my_predict_def)
    print('Recalls默认值 = %.4f，' % defRValue, '默认阈值 = 0')
    # 计算FPR默认值及其阈值：
    defFValue = FPR(y_true, my_predict_def)
    print('FPR默认值 = %.4f，' % defFValue, '默认阈值 = 0')

    return thresholds2, tprs2, fprs2, recallsValue_auto, fprsValue_auto, maxKsThresholds_auto, maxKsValue_auto, rocAucScore, defRValue, defFValue


# 自动图
def ComprehensiveIndicatorSkLibFigure(y_true, decision_scores, axe):
    mpl.rcParams['font.sans-serif'] = 'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False

    thresholds2, tprs2, fprs2, recallsValue_auto, fprsValue_auto, maxKsThresholds_auto, maxKsValue_auto, rocAucScore, defRValue, defFValue = ComprehensiveIndicatorSkLib(
        y_true, decision_scores)
    axe[0].plot(thresholds2, tprs2, color='black', label='召回率/TPR')
    axe[0].plot(thresholds2, fprs2, color='pink', label='FPR')
    axe[0].plot((maxKsThresholds_auto, maxKsThresholds_auto), (recallsValue_auto, fprsValue_auto), c='r', lw=1.5,
                ls='--', alpha=0.7, label='KS = %.4f，KS阈值 = %.4f' % (maxKsValue_auto, maxKsThresholds_auto))
    axe[0].scatter(0, defRValue, c='#000000', s=30, cmap="rainbow")  # 黑色
    axe[0].scatter(0, defFValue, c='#FFC0CB', s=30, cmap="rainbow")  # 粉色
    axe[0].legend(loc='lower left')  # 图例
    axe[0].set_xlabel('阈值')  # x轴标签
    axe[0].set_ylabel('召回率/TPR、FPR、KS')  # y轴标签
    axe[0].set_title('自动-阈值与召回率/TPR、FPR、KS=max(TPR-FPR)')  # 图名
    axe[0].text(0, defRValue + 0.03, '默认阈值0：%.4f' % defRValue, ha='center', va='bottom')
    axe[0].text(0, defFValue - 0.03, '默认阈值0：%.4f' % defFValue, ha='center', va='bottom')
    axe[0].text(maxKsThresholds_auto, recallsValue_auto + 0.01, '%.4f' % recallsValue_auto, ha='center',
                va='bottom')  # fontsize=8
    axe[0].text(maxKsThresholds_auto, fprsValue_auto - 0.05, '%.4f' % fprsValue_auto, ha='center', va='bottom')

    axe[1].plot(fprs2, tprs2, color='purple', label='AUC=%.3f' % rocAucScore)
    axe[1].plot((0, 1), (0, 1), c='b', lw=1.5, ls='--', alpha=0.7)  # 横轴fprs2：0→1范围；竖轴tprs2：0→1范围
    axe[1].scatter(fprsValue_auto, recallsValue_auto, c=1, s=30, cmap="rainbow")
    axe[1].plot((fprsValue_auto, fprsValue_auto), (recallsValue_auto, fprsValue_auto), c='r', lw=1.5, ls='--',
                alpha=0.7)
    axe[1].set_xlabel('FPR')  # x轴标签
    axe[1].set_ylabel('TPR')  # y轴标签
    axe[1].grid(b=True)
    axe[1].legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=14)
    axe[1].set_title('自动-ROC曲线和AUC值')  # fontsize=17
    axe[1].text(fprsValue_auto, recallsValue_auto + 0.01, '%.4f' % recallsValue_auto, ha='center',
                va='bottom')  # fontsize=8
    axe[1].text(fprsValue_auto, fprsValue_auto - 0.05, '%.4f' % fprsValue_auto, ha='center', va='bottom')
    axe[1].text(fprsValue_auto + 0.05, (fprsValue_auto + recallsValue_auto) / 2, '%.4f' % maxKsValue_auto, ha='center',
                va='bottom')


# 训练集 与 测试集 ROC比较
def comparedRoc(y_true_train, decision_scores_train, y_true_test, decision_scores_test, axe):
    mpl.rcParams['font.sans-serif'] = 'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False

    fpr_train, tpr_train, th_train = roc_curve(y_true_train, decision_scores_train)
    rocAucScore_train = roc_auc_score(y_true_train, decision_scores_train)

    fpr_test, tpr_test, th_test = roc_curve(y_true_test, decision_scores_test)
    rocAucScore_test = roc_auc_score(y_true_test, decision_scores_test)

    axe.plot(fpr_train, tpr_train, color='red', label='AUC_Train = %.3f' % rocAucScore_train)
    axe.plot(fpr_test, tpr_test, color='blue', label='AUC_Test = %.3f' % rocAucScore_test)
    axe.plot((0, 1), (0, 1), c='b', lw=1.5, ls='--', alpha=0.7)  # 横轴fprs2：0→1范围；竖轴tprs2：0→1范围
    axe.set_xlabel('FPR')  # x轴标签
    axe.set_ylabel('TPR')  # y轴标签
    axe.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=14)
    axe.set_title('自动-ROC曲线Train与Test对比')




