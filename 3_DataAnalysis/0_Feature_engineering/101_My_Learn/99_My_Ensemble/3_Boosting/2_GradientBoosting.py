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


from sklearn.ensemble import GradientBoostingClassifier

# 默认就是使用决策树模型
gb_clf = GradientBoostingClassifier(max_depth=2, n_estimators=30)
gb_clf.fit(X_train, y_train)
print(gb_clf)
print(gb_clf.score(X_text, y_test))