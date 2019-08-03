import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)

plt.scatter(X[y==0,0], X[y==0,1]) # X[y==0,0]取随机数：X[y==0]得到一行，再取第0个元素
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_text, y_train, y_test = train_test_split(X, y, random_state=42)
print(pd.value_counts(y_train, sort=True))


from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)
print(log_clf.score(X_text, y_test))


from sklearn.svm import SVC

svm_clf = SVC()
svm_clf.fit(X_train, y_train)
print(svm_clf.score(X_text, y_test))


from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
print(dt_clf.score(X_text, y_test))


# 1、手动实现
# 少数服从多少表决
y_predict1 = log_clf.predict(X_text)
y_predict2 = svm_clf.predict(X_text)
y_predict3 = dt_clf.predict(X_text)
# 至少有2票，至多有3票
y_predict = np.array((y_predict1 + y_predict2 + y_predict3) >= 2, dtype='int')
print(y_predict[0:10])

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_predict))


print("-----------------------------------------------------------------------------------------------")


# 2、使用sklearn包：
from sklearn.ensemble import VotingClassifier

# hard：少数服从多数
voting_clf = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()),
    ('svm_clf', SVC()),
    ('dt_clf', DecisionTreeClassifier())
], voting='hard')

voting_clf.fit(X_train, y_train)
print(voting_clf.score(X_text, y_test))