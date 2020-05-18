# -*- coding: utf-8 -*-

"""
    作者:     梁斌
    版本:     1.0
    日期:     2017/02/13
    项目名称： Logistic Regression 模型的手工实现
    参考：    https://github.com/willemolding/LogisticRegressionPython
"""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b


class LogisticRegression(object):
    """
        Logistic Regression 类
    """
    def __init__(self, c=1.):
        self.c = c

    def fit(self, X, y):
        """
            训练模型
        """
        self._beta = np.zeros((X.shape[1] + 1, 1))

        # 使用L-BFGS-B求最优化
        result = fmin_l_bfgs_b(cost_func,               # 损失函数
                               self._beta,              # 初始值
                               args=(X, y, self.c))     # 损失函数的参数

        self._beta = result[0]
        return self

    def predict(self, X):
        """
            预测，返回标签
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        """
            预测，返回概率
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        XBeta = np.dot(X, self._beta).reshape((-1, 1))

        probs = 1. / (1. + np.exp(-XBeta))
        return np.hstack((1 - probs, probs))


def cost_func(beta, X, y, C):
    """
        损失函数/目标函数
        返回 正则化的负对数似然值 及 梯度值
    """

    # 给X加一列1，便于计算
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    # 转成列向量
    y = y.reshape((-1, 1))

    # 预先计算XBeta
    XBeta = np.dot(X, beta).reshape((-1, 1))

    # 预先计算Xbeta的exp值
    exp_XBeta = np.exp(XBeta)

    # 负对数似然值
    # neg_ll = C*np.sum(np.log(1. + exp_XBeta) - y*XBeta, axis=0) + 0.5*np.inner(beta, beta)
    neg_ll = C * np.sum(np.log(1. + exp_XBeta) - y * XBeta, axis=0)

    # 负对数似然值得梯度
    grad_neg_ll = C*np.sum((1. / (1. + exp_XBeta))*exp_XBeta*X - y*X, axis=0) + beta

    return neg_ll, grad_neg_ll


def cal_acc(true_labels, pred_labels):
    """
        计算准确率
    """
    n_total = len(true_labels)
    correct_list = [true_labels[i] == pred_labels[i] for i in range(n_total)]

    acc = sum(correct_list) / n_total
    return acc
