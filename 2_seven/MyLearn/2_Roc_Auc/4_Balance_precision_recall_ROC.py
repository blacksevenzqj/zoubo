import numpy as np
from sklearn import datasets
import matplotlib as mpl
import matplotlib.pyplot as plt


digits = datasets.load_digits()
x = digits.data
y = digits.target.copy()
print('x的长度%i' % len(x), 'y的长度%i:' % len(y))

# 使数据集的样本比例的严重偏斜，变为2分类
y[digits.target == 9] = 1
y[digits.target != 9] = 0

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=666)

from sklearn.linear_model import LogisticRegression

# 没有使用交叉验证选正则项值（roc_auc评分标准）  或  直接使用 LogisticRegressionCV 类自带的交叉验证功能
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train) # 模型训练时：是不用阈值的，具体细看损失函数及其求偏导的过程。

score = log_reg.score(x_test, y_test) # 直接求 准确率
# print(score)

y_log_predict = log_reg.predict(x_test) # 求 预测值

y_log_predict_proba = log_reg.predict_proba(x_test)[:,1] # 求 预测值 概率
y_log_predict_proba_predict = np.array(y_log_predict_proba >= 0.5, dtype='int')
# print(len(y_test), np.sum(y_log_predict == y_log_predict_proba_predict)) # 可见 默认概率为 0.5

decision_scores = log_reg.decision_function(x_test) # 理解为：线性回归 𝜃x+b 的结果


def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))

# predictTP = TP(y_test, y_log_predict)
# print(predictTP)

def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 0))

# predictTN = TN(y_test, y_log_predict)
# print(predictTN)

def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 1))

# predictFP = FP(y_test, y_log_predict)
# print(predictFP)

def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 0))

# predictFN = FN(y_test, y_log_predict)
# print(predictFN)

# 混淆矩阵
def confusion_matrix(y_true, y_predict):
    return np.array([
        [TN(y_true, y_predict), FP(y_true, y_predict)],
        [FN(y_true, y_predict), TP(y_true, y_predict)],
    ])

# matrix = confusion_matrix(y_test, y_log_predict)
# print(matrix)

# 准确率
def precision_scoreAll(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)
    tn = TN(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return (tp + tn) / (tp + fp + tn + fn)
    except:
        return 0.0

# precisionScoreAll = precision_scoreAll(y_test, y_log_predict)
# print(precisionScoreAll)

# 精准率
def precision_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)
    try:
        return tp / (tp + fp)
    except:
        return 0.0

# precisionScore = precision_score(y_test, y_log_predict)
# print(precisionScore)

# 召回率
def recall_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0

# recallScore = recall_score(y_test, y_log_predict)
# print(recallScore)

# 调和平均值
# F1分数的公式为 = 2*查准率*查全率 / (查准率 + 查全率)
def f1_score_my(precisionScore, recallScore):
    try:
        return 2 * precisionScore * recallScore / (precisionScore + recallScore)
    except:
        return 0.0

# f1Score = f1_score_my(precisionScore, recallScore)
# print(f1Score)

# TPR：就是就是召回率
def TPR(y_true, y_predict):
    return recall_score(y_true, y_predict)

# tprScore = TPR(y_test, y_log_predict)
# print(tprScore)

# FPR：
def FPR(y_true, y_predict):
    fp = FP(y_true, y_predict)
    tn = TN(y_true, y_predict)
    try:
        return fp / (fp + tn)
    except:
        return 0.0

# fprScore = FPR(y_test, y_log_predict)
# print(fprScore)


print("=============================================================================================")


# 混淆矩阵
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(y_test, y_log_predict)
# print(confusionMatrix)

# 准确率
from sklearn.metrics import accuracy_score
accuracyScore = accuracy_score(y_test, y_log_predict)
# print(accuracyScore)

# 精准率
from sklearn.metrics import precision_score
precisionScore = precision_score(y_test, y_log_predict)
# print(precisionScore)

# 召回率
from sklearn.metrics import recall_score
recallScore = recall_score(y_test, y_log_predict)
# print(recallScore)

# F1分数
from sklearn.metrics import f1_score
f1Score = f1_score(y_test, y_log_predict)
# print(f1Score)

# 多分类综合指标：只有这个指标能计算多分类，以上的都是计算二分类的
from sklearn.metrics import classification_report
classificationReport = classification_report(y_test, y_log_predict)
# print(classificationReport)


print("=============================================================================================")


# matplotlib 图表中文显示
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

# 1.1、手动循环 创建TPR、FPR：
precisions = []
recalls = []
f1Scores = []
tprs = []
fprs = []
thresholds = np.arange(np.min(decision_scores), np.max(decision_scores), 0.1)
for threshold in thresholds:
    # 重点：将 decision_scores（线性回归𝜃x+b的结果） 大于（正预测）  其自身区间中的某一个阈值 的结果 设置为 预测结果
    my_predict = np.array(decision_scores >= threshold, dtype='int')
    precisions.append(precision_score(y_test, my_predict))
    recalls.append(recall_score(y_test, my_predict)) # TPR
    f1Scores.append(f1_score(y_test, my_predict))
    tprs.append(TPR(y_test, my_predict)) # recalls
    fprs.append(FPR(y_test, my_predict))
    # print(confusion_matrix(y_test, my_predict))

# print("阈值：", thresholds[0:10], '维度：', len(thresholds)) # 1056
# print("准确率：", precisions[0:10], '维度：', len(precisions)) # 1056
# print("召回率：", recalls[0:10], '维度：', len(recalls)) # 1056
# print("F1分数：", f1Scores[0:10], '维度：', len(f1Scores)) # 1056
# print("TPR：", tprs[0:10], '维度：', len(tprs))
# print("FPR：", fprs[0:10], '维度：', len(fprs))


# 1.2、自动 创建TPR、FPR 得到 ROC：
from sklearn.metrics import roc_curve
fprs2, tprs2, thresholds2 = roc_curve(y_test, decision_scores) # 自动-thresholds2 和 手动-thresholds 是不完全相同的
fprs3, tprs3, thresholds3 = roc_curve(y_test, y_log_predict_proba)
# print(len(fprs2), sum(fprs2 == fprs3)) # 两种方式结果相同

# 1.3、自动 创建AUC面积：Area Under Curve
from sklearn.metrics import roc_auc_score
rocAucScore = roc_auc_score(y_test, decision_scores)
rocAucScore3 = roc_auc_score(y_test, y_log_predict_proba)
# print(rocAucScore, rocAucScore3) # 两种方式结果相同

# 1.4、计算KS值及其阈值：KS=max(TPR-FPR)
print(len(thresholds), len(recalls), len(fprs))
tempValue = 0.00
maxKsValue = 0.00
maxKsThresholds = 0.00
recallsValue = 0.00
fprsValue = 0.00
for i in np.arange(len(thresholds)):
    tempValue = abs(recalls[i] - fprs[i])
    if tempValue > maxKsValue:
        maxKsValue = tempValue
        maxKsThresholds = thresholds[i]
        recallsValue = recalls[i]
        fprsValue = fprs[i]
print('max(TPR-FPR) = %.6f，' % maxKsValue, '最大阈值 = %.6f' % maxKsThresholds)
print('max(TPR-FPR) = %.6f' % abs(np.array(recalls) - np.array(fprs)).max())

# 1.5、计算F1分数最大值及其阈值：
maxF1ScoresValue = max(f1Scores)
maxF1ScoresIndex = f1Scores.index(max(f1Scores)) # 从左到右：第一个出现的最大数的索引
maxF1Thresholds = thresholds[maxF1ScoresIndex]
print('F1最大值 = %.6f，' % maxF1ScoresValue, '最大阈值 = %.6f' % maxF1Thresholds)
# 1.5.1、计算F1分数最大值阈值 与 KS值阈值 距离：
diffValue = maxF1Thresholds - maxKsThresholds
print('F1最大值阈值 - KS阈值 = %.6f' % diffValue)


fig = plt.figure(figsize = (24,12))
# 1、A1图
# 以阈值为横坐标，分别以TPR和FPR的值为纵坐标，就可以画出两个曲线，这就是K-S曲线
ax1 = fig.add_subplot(2,2,1)
plt.plot(thresholds, precisions, color = 'blue', label='精准率')
plt.plot(thresholds, recalls, color='black', label='召回率/TPR')
plt.plot(thresholds, f1Scores, color='green', label='F1分数阈值 = %.6f' % maxF1Thresholds)
plt.plot(thresholds, fprs, color='pink', label='FPR')
plt.plot((maxKsThresholds,maxKsThresholds), (recallsValue,fprsValue), c='r', lw=1.5, ls='--', alpha=0.7, label='KS阈值 = %.6f' % maxKsThresholds)
plt.plot((maxKsThresholds,maxF1Thresholds), (maxF1ScoresValue,maxF1ScoresValue), c='purple', lw=1.5, ls='-', alpha=0.7, label='(F1-KS)的阈值差 = %.4f' % diffValue)
plt.legend()  # 图例
plt.xlabel('阈值')  # x轴标签
plt.ylabel('精准率、召回率/TPR、F1分数、FPR、KS') # y轴标签
plt.title('手动-阈值与精准率、召回率/TPR、F1分数、FPR、KS=max(TPR-FPR)')  # 图名

# 2、A2图
ax2 = fig.add_subplot(2,2,2)
plt.plot(precisions, recalls, color='purple', label='P-R曲线')
plt.legend()  # 图例
plt.xlabel('精准率')  # x轴标签
plt.ylabel('召回率') # y轴标签
plt.title('手动-P-R曲线')  # 图名

# 3、B1图
ax4 = fig.add_subplot(2,2,3)
plt.plot(fprs, tprs, color='purple', label='ROC曲线')
plt.legend()  # 图例
plt.xlabel('FPR')  # x轴标签
plt.ylabel('TPR') # y轴标签
plt.title('手动-ROC曲线')  # 图名

# 4、B2图
ax4 = fig.add_subplot(2,2,4)
plt.plot(fprs2, tprs2, color='purple', label='AUC=%.3f' % rocAucScore)
plt.plot((0, 1), (0, 1), c='b', lw=1.5, ls='--', alpha=0.7) # 横轴fprs2：0→1范围；竖轴tprs2：0→1范围
plt.xlabel('FPR')  # x轴标签
plt.ylabel('TPR') # y轴标签
plt.grid(b=True)
plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=14)
plt.title('自动-ROC曲线和AUC值', fontsize=17)

plt.show()