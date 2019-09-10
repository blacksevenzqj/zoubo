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
print("准确率为：", score)

y_log_predict = log_reg.predict(x_test) # 求 预测值
print("准确率为：", np.sum(y_log_predict == y_test) / len(y_test))

y_log_predict_proba = log_reg.predict_proba(x_test)[:, 1]
y_log_predict_proba_predict = np.array(y_log_predict_proba >= 0.5, dtype='int')
print(len(y_test), np.sum(y_log_predict == y_log_predict_proba_predict)) # 可见 默认概率为 0.5


def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))

predictTP = TP(y_test, y_log_predict)
print(predictTP)

def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 0))

predictTN = TN(y_test, y_log_predict)
print(predictTN)

def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 1))

predictFP = FP(y_test, y_log_predict)
print(predictFP)

def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 0))

predictFN = FN(y_test, y_log_predict)
print(predictFN)

# 混淆矩阵
def confusion_matrix(y_true, y_predict):
    return np.array([
        [TN(y_true, y_predict), FP(y_true, y_predict)],
        [FN(y_true, y_predict), TP(y_true, y_predict)],
    ])

matrix = confusion_matrix(y_test, y_log_predict)
print(matrix)

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

precisionScoreAll = precision_scoreAll(y_test, y_log_predict)
print("准确率为：", precisionScoreAll)

# 精准率
def precision_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)
    try:
        return tp / (tp + fp)
    except:
        return 0.0

precisionScore = precision_score(y_test, y_log_predict)
print("精准率为：", precisionScore)

# 召回率
def recall_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0

recallScore = recall_score(y_test, y_log_predict)
print("召回率：", recallScore)

# TPR：就是就是召回率
def TPR(y_true, y_predict):
    return recall_score(y_true, y_predict)

tprScore = TPR(y_test, y_log_predict)
print(tprScore)

# FPR：
def FPR(y_true, y_predict):
    fp = FP(y_true, y_predict)
    tn = TN(y_true, y_predict)
    try:
        return fp / (fp + tn)
    except:
        return 0.0

fprScore = FPR(y_test, y_log_predict)
print(fprScore)


print("=============================================================================================")


# 混淆矩阵
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(y_test, y_log_predict)
print(confusionMatrix)

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
print(recallScore)

# 多分类综合指标：只有这个指标能计算多分类，以上的都是计算二分类的（以每个类别为基准，分别计算 每个类别各自的 精准率、召回率、F1 等指标）
from sklearn.metrics import classification_report
classificationReport = classification_report(y_test, y_log_predict)
print(classificationReport)

