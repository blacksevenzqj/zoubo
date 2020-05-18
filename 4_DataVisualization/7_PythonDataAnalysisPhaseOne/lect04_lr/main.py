# -*- coding: utf-8 -*-

"""
    作者:     梁斌
    版本:     1.0
    日期:     2017/02/13
    项目名称： Logistic Regression 模型的手工实现
    参考：    https://github.com/willemolding/LogisticRegressionPython
"""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from lr_tools import LogisticRegression, cal_acc


def run_main():
    """
        主函数
    """
    X, y = make_classification(
            n_samples=2000,
            n_features=100,
            n_classes=2,
            random_state=17)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=17)

    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    y_pred = lr_model.predict(X_test)

    print('真实值：', y_test)
    print('预测值：', y_pred)
    acc = cal_acc(y_test, y_pred)
    print('准确率：{:.2%}'.format(acc))


if __name__ == '__main__':
    run_main()
