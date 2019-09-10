import numpy as np
from sklearn import datasets
import matplotlib as mpl
import matplotlib.pyplot as plt


digits = datasets.load_digits()
x = digits.data
y = digits.target.copy()
print('x的长度:%i' % len(x), 'y的长度:%i' % len(y))

# 使数据集的样本比例的严重偏斜
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

'''
1、predict函数：默认阈值为0，大于0的为一类。（根据线性回归 𝜃x+b 的结果判断，教程上说的！！！）
2、decision_function 函数：是线性回归 𝜃x+b 的结果。
'''
decision_scores = log_reg.decision_function(x_test)
# print(decision_scores[:10])
# print(np.min(decision_scores), np.max(decision_scores))


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

# 召回率
from sklearn.metrics import recall_score
recallScore = recall_score(y_test, y_log_predict)

# f1分数
from sklearn.metrics import f1_score
f1Score = f1_score(y_test, y_log_predict)

# 多分类综合指标：只有这个指标能计算多分类，以上的都是计算二分类的（以每个类别为基准，分别计算 每个类别各自的 精准率、召回率、F1 等指标）
from sklearn.metrics import classification_report
classificationReport = classification_report(y_test, y_log_predict)


print("=============================================================================================")


# matplotlib 图表中文显示
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

# '''
# 1、直接用 decision_function 函数的 线性回归 𝜃x+b 的结果 判断：
# y_predict_2 = np.array(decision_scores >= 0, dtype='int')
# print(confusion_matrix(y_test, y_predict_2))

# 1.1.1、手动循环 找阈值：自定义阀值： P-R曲线、F1分数
precisions = []
recalls = []
f1Scores = []
thresholds = np.arange(np.min(decision_scores), np.max(decision_scores), 0.1)
for threshold in thresholds:
    my_predict = np.array(decision_scores >= threshold, dtype='int')
    precisions.append(precision_score(y_test, my_predict))
    recalls.append(recall_score(y_test, my_predict))
    f1Scores.append(f1_score(y_test, my_predict))
    # print(confusion_matrix(my_predict, y_test))

# print("阈值：", thresholds[0:10], '维度：', len(thresholds)) # 1056
# print("准确率：", precisions[0:10], '维度：', len(precisions)) # 1056
# print("召回率：", recalls[0:10], '维度：', len(recalls)) # 1056
# print("F1分数：", f1Scores[0:10], '维度：', len(f1Scores)) # 1056


'''
precision_recall_curve 函数使用 decision_scores 和 y_log_predict_proba 计算得到的 精准率、召回率、F1 相同，而 阈值不同；y_log_predict_proba 计算的阈值 不能使用。  
'''
# 1.1.2、sklearn的函数 找阈值：
from sklearn.metrics import precision_recall_curve # P-R曲线

# 调和平均值 F1分数的公式为 = 2*查准率*查全率 / (查准率 + 查全率)
# 因 precision_recall_curve 没有返回F1分数，所以写F1的自定义函数，和sklearn的f1_score函数相同。
def f1_score_my(precisionScore, recallScore):
    try:
        return 2 * precisionScore * recallScore / (precisionScore + recallScore)
    except:
        return 0.0

# 1、 precisions2、recalls2 和 thresholds 的shape维度 是由函数 precision_recall_curve 自定义步长决定的。
#  thresholds的维度 比 precisions2、recalls2 少1个维度。所以 precisions2[:-1], recalls2[:-1] 进行计算。
#  原文如下：
#  The last precision and recall values are 1. and 0. respectively and do not have a corresponding threshold.
#  This ensures that the graph starts on the y axis.
#  最大的threshold对应的 精准率为1，召回率为0，所以没有保留最大的threshold。
# 2、precision_recall_curve 函数计算时，没有从 decision_scores 中的最小值开始计算，从函数自认为重要的值开始计算，所以
# 返回的 precisions2, recalls2, thresholds2 也没有对应到 decision_scores 中的最小值。
precisions2, recalls2, thresholds2 = precision_recall_curve(y_test, decision_scores)
f1Scores2 = f1_score_my(precisions2[:-1], recalls2[:-1])

# print("阈值：", thresholds2[0:10], '维度：', thresholds2.shape) # 144
# print("准确率：", precisions2[0:10], '维度：', precisions2.shape) # 145
# print("召回率：", recalls2[0:10], '维度：', recalls2.shape) # 145
# print("F1分数：", f1Scores2[0:10], '维度：', f1Scores2.shape) # 144


fig = plt.figure(figsize = (24,12))
# 1、A1图
ax1 = fig.add_subplot(2,2,1)
plt.plot(thresholds, precisions, color = 'blue', label='精准率')
plt.plot(thresholds, recalls, color='black', label='召回率')
plt.plot(thresholds, f1Scores, color='green', label='F1分数')
plt.legend()  # 图例
plt.xlabel('阈值')  # x轴标签
plt.ylabel('精准率、召回率、F1分数') # y轴标签
plt.title('手动-阈值与精准率、召回率、F1分数')  # 图名

# 2、A2图
ax2 = fig.add_subplot(2,2,2)
plt.plot(precisions, recalls, color='purple', label='P-R曲线')
plt.legend()  # 图例
plt.xlabel('精准率')  # x轴标签
plt.ylabel('召回率') # y轴标签
plt.title('手动-P-R曲线')  # 图名

# 3、B1图
ax3 = fig.add_subplot(2,2,3)
plt.plot(thresholds2, precisions2[:-1], color = 'blue', label='精准率')
plt.plot(thresholds2, recalls2[:-1], color='black', label='召回率')
plt.plot(thresholds2, f1Scores2, color='green', label='F1分数')
plt.legend()  # 图例
plt.xlabel('阈值')  # x轴标签
plt.ylabel('精准率、召回率、F1分数') # y轴标签
plt.title('自动-阈值与精准率、召回率、F1分数')  # 图名

# 4、B2图
ax4 = fig.add_subplot(2,2,4)
plt.plot(precisions2[:-1], recalls2[:-1], color='purple', label='P-R曲线')
plt.legend()  # 图例
plt.xlabel('精准率')  # x轴标签
plt.ylabel('召回率') # y轴标签
plt.title('自动-P-R曲线')  # 图名

plt.show()
# '''



print("============================================================================================================")



# 以下两种计算方式 结果非常类似：
'''
# 2.1、转换为概率判断：（使用decision_scores，有问题）
def mySigmoid(z): # 计算概率p
    return 1 / (1 + np.exp(-z))

def myPredict(z, threshold): # 根据概率p判断分类
    return [1 if p >= threshold else 0 for p in mySigmoid(z)]

# my_predict = myPredict(decision_scores, 0.5)
# print(y_log_predict[:10]) # ndarray
# print(my_predict[:10]) # list

# 2.1.1、循环找阈值：
precisions = []
recalls = []
f1Scores = []
thresholds = np.arange(0.02, 0.9, 0.01)
for threshold in thresholds:
    my_predict = myPredict(decision_scores, threshold)
    precisions.append(precision_score(y_test, my_predict))
    recalls.append(recall_score(y_test, my_predict))
    f1Scores.append(f1_score(y_test, my_predict))
    # print(confusion_matrix(my_predict, y_test))

print("阈值：", thresholds)
print("准确率：", precisions)
print("召回率：", recalls)
print("F1分数：", f1Scores)

fig = plt.figure(figsize = (12,4))
# 1、A图
ax1 = fig.add_subplot(1,2,1)
plt.plot(thresholds, precisions, color = 'blue', label='精准率')
plt.plot(thresholds, recalls, color='black', label='召回率')
plt.plot(thresholds, f1Scores, color='green', label='F1分数')
plt.legend()  # 图例
plt.xlabel('阈值')  # x轴标签
plt.ylabel('精准率、召回率、F1分数') # y轴标签

# 2、B图
ax2 = fig.add_subplot(1,2,2)
plt.plot(precisions, recalls, color='purple', label='P-R曲线')
plt.legend()  # 图例
plt.xlabel('精准率')  # x轴标签
plt.ylabel('召回率') # y轴标签

plt.show()


print("--------------------------------------------------------------------------------------------------------")


# 2.2、转换为概率判断：（使用y_log_predict_proba，有问题）
# 2.2.1、循环找阈值：
precisions = []
recalls = []
f1Scores = []
thresholds = np.arange(0.01, 0.99, 0.01)
for threshold in thresholds:
    my_predict = np.array(y_log_predict_proba >= threshold, dtype='int')
    precisions.append(precision_score(y_test, my_predict))
    recalls.append(recall_score(y_test, my_predict))
    f1Scores.append(f1_score(y_test, my_predict))
    # print(confusion_matrix(my_predict, y_test))

print("阈值：", thresholds)
print("准确率：", precisions)
print("召回率：", recalls)
print("F1分数：", f1Scores)

fig = plt.figure(figsize = (12,4))
# 1、A图
ax1 = fig.add_subplot(1,2,1)
plt.plot(thresholds, precisions, color = 'blue', label='精准率')
plt.plot(thresholds, recalls, color='black', label='召回率')
plt.plot(thresholds, f1Scores, color='green', label='F1分数')
plt.legend()  # 图例
plt.xlabel('阈值')  # x轴标签
plt.ylabel('精准率、召回率、F1分数') # y轴标签

# 2、B图
ax2 = fig.add_subplot(1,2,2)
plt.plot(precisions, recalls, color='purple', label='P-R曲线')
plt.legend()  # 图例
plt.xlabel('精准率')  # x轴标签
plt.ylabel('召回率') # y轴标签

plt.show()
'''