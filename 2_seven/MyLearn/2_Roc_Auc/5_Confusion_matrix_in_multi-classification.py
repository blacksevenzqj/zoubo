import numpy as np
from sklearn import datasets
import matplotlib as mpl
import matplotlib.pyplot as plt


# 多分类问题：
digits = datasets.load_digits()
x = digits.data
y = digits.target.copy()
print('x的长度%i' % len(x), 'y的长度%i:' % len(y))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=666)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

score = log_reg.score(x_test, y_test) # 直接求 准确率
print(score)
y_log_predict = log_reg.predict(x_test) # 求 预测值

decision_scores = log_reg.decision_function(x_test) # 线性回归 𝜃x+b 的结果


# 多分类问题 的 精准率
from sklearn.metrics import precision_score

precisionScore = precision_score(y_test, y_log_predict, average='micro')
print(precisionScore)


# 多分类综合指标：只有这个指标能计算多分类，以上的都是计算二分类的（以每个类别为基准，分别计算 每个类别各自的 精准率、召回率、F1 等指标）
from sklearn.metrics import classification_report
classificationReport = classification_report(y_test, y_log_predict)
print(classificationReport)


# 多分类问题 的 混淆矩阵
from sklearn.metrics import confusion_matrix

confusionMatrix = confusion_matrix(y_test, y_log_predict)
print(confusionMatrix)
plt.matshow(confusionMatrix, cmap=plt.cm.gray)
plt.show()

row_sums = np.sum(confusionMatrix, axis=1)
err_matrix = confusionMatrix / row_sums
np.fill_diagonal(err_matrix, 0) # 将对角线的值覆盖为0
print(err_matrix)
plt.matshow(err_matrix, cmap=plt.cm.gray)
plt.show()
