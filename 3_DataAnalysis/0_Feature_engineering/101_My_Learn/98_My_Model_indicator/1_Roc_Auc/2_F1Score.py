import numpy as np
from sklearn import datasets

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

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

score = log_reg.score(x_test, y_test) # 直接求 准确率
print(score)
y_log_predict = log_reg.predict(x_test) # 求 预测值


'''
默认是以 目标变量（因变量Y）== 1 为基准：
'''
# 混淆矩阵
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(y_test, y_log_predict)

# 准确率
from sklearn.metrics import accuracy_score
accuracyScore = accuracy_score(y_test, y_log_predict)
print(accuracyScore)

# 精准率
from sklearn.metrics import precision_score
precisionScore = precision_score(y_test, y_log_predict)
print(precisionScore)

# 召回率
from sklearn.metrics import recall_score
recallScore = recall_score(y_test, y_log_predict)

# 多分类综合指标：只有这个指标能计算多分类，以上的都是计算二分类的（以每个类别为基准，分别计算 每个类别各自的 精准率、召回率、F1 等指标）
from sklearn.metrics import classification_report
classificationReport = classification_report(y_test, y_log_predict)


print("=============================================================================================")


# 调和平均值
# F1分数的公式为 = 2*查准率*查全率 / (查准率 + 查全率)
def f1_score_my(precisionScore, recallScore):
    try:
        return 2 * precisionScore * recallScore / (precisionScore + recallScore)
    except:
        return 0.0

print(f1_score_my(0.5, 0.5), f1_score_my(0.1, 0.9), f1_score_my(0.0, 1.0)) # 不会像算数平均数一样

print(f1_score_my(precisionScore, recallScore))

from sklearn.metrics import f1_score

print(f1_score(y_test, y_log_predict))