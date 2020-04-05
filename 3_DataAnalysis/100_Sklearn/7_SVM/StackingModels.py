# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 22:57:36 2020

@author: dell
"""
import pandas as pd
import numpy as np
import datetime
import os
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error as MSE, r2_score, mean_absolute_error as MAE

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor as XGBR
from lightgbm import LGBMRegressor as LGBMR
import xgboost as xgb
import joblib

from math import isnan
import FeatureTools as ft
import Tools_customize as tc
import Binning_tools as bt

path = r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\100_Data_analysis_competition\3_TianChi\1_Used_car_transaction_price_prediction\models"


# In[]:
# =============================Stacking models：堆叠模型=========================
# In[]:
# 1、自定义Stacking：
# 1.1、回归模型： 都是Sklearn库
# 暂时只能用来 调参测试， 不能直接用来预测真实的预测集（因要保存训练好的模型）
def Stacking_Regressor_customize(clfs, train_X, train_y, n_splits=3, random_state=0):
    dayTime = datetime.datetime.now().strftime('%Y%m%d')
    temp_path = path + "\\" + dayTime
    isExists = os.path.exists(temp_path)
    if not isExists:
        os.makedirs(temp_path)

    X, X_predict, y, y_predict = train_test_split(train_X, train_y, test_size=0.3, random_state=random_state)

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs)))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for j, value in enumerate(clfs.items()):
        clf_name = value[0]
        clf = value[1]
        print(j, clf_name, clf)
        # 依次训练各个单模型
        dataset_blend_test_j = np.zeros((X_predict.shape[0], n_splits))
        for i, (train, test) in enumerate(kf.split(X, y)):
            print(train[0:10], test[0:10])
            # 5-Fold交叉训练，使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。
            X_train, y_train, X_test, y_test = X.iloc[train], y.iloc[train], X.iloc[test], y.iloc[test]
            clf.fit(X_train, y_train)

            savepath = temp_path + "\\" + clf_name + "_level1_" + str(i) + ".dat"
            print(savepath)
            joblib.dump(clf, savepath)

            y_submission = clf.predict(X_test)
            dataset_blend_train[
                test, j] = y_submission  # 每个模型（j）、每一折验证集(test索引) → 每个模型（j）、5次交叉验证：共5折验证集：最后的行维度就是 训练集X的行维度。
            dataset_blend_test_j[:, i] = clf.predict(X_predict)
        # 对于测试集，直接用这k个模型的预测值均值作为新的特征。
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
        print("MAE Score: %f" % MAE(y_predict, dataset_blend_test[:, j]))

    clf = LinearRegression()
    clf.fit(dataset_blend_train, y)
    savepath = temp_path + "\\LR_level2.dat"
    joblib.dump(clf, savepath)
    y_submission = clf.predict(dataset_blend_test)

    print("MAE Score of Stacking: %f" % (MAE(y_predict, y_submission)))


def Stacking_Regressor_customize_load(clfs, X_test, n_splits=3, date_path=None):
    if date_path is None:
        dayTime = datetime.datetime.now().strftime('%Y%m%d')
        temp_path = path + "\\" + dayTime
    else:
        temp_path = path + "\\" + date_path

    dataset_blend_test = np.zeros((X_test.shape[0], len(clfs)))

    for j, value in enumerate(clfs.items()):
        clf_name = value[0]
        clf = value[1]
        print(j, clf_name, clf)
        # 依次训练各个单模型
        dataset_blend_test_j = np.zeros((X_test.shape[0], n_splits))
        for i in range(n_splits):
            loadpath = temp_path + "\\" + clf_name + "_level1_" + str(i) + ".dat"
            print(loadpath)
            clf = joblib.load(loadpath)
            dataset_blend_test_j[:, i] = clf.predict(X_test)
        # 对于测试集，直接用这k个模型的预测值均值作为新的特征。
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    loadpath = temp_path + "\\LR_level2.dat"
    print(loadpath)
    clf = joblib.load(loadpath)
    y_submission = clf.predict(dataset_blend_test)

    return y_submission


# In[]:
# 1、平均基本模型（类似bagging）
'''
最简单的堆叠方法：平均基本模型:Simplest Stacking approach : Averaging base models（类似bagging）
我们从平均模型的简单方法开始。 我们建立了一个新类，以通过模型扩展scikit-learn，并进行封装和代码重用（继承inheritance）。
https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)

理解：
本堆叠模型进交叉验证时，每一折交叉验证所有模型都计算一次，并求所有模型对这一折交叉验证数据的预测结果指标均值。
如 5折交叉验证时，每一折都是 所有模型的预测结果指标均值； 最后再求 5折交叉验证的均值。
'''


# Averaged base models class 平均基本模型
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

    # 2、堆叠平均模型类（类似boosting）


# 入参 X、y 都是矩阵格式
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True)  # random_state=156

        '''
        重点：
        如果调用者使用的是 交叉验证： 如下代码就是 交叉验证 中的 交叉验证。
        如，外层5折交叉验证： 这里传入的X,y就是4折的总训练集： 总训练集*4/5。
        1、for i, model in enumerate(self.base_models) 是循环 堆叠模型第一层中每一个基模型。
        2、for train_index, holdout_index in kfold.split(X, y) 堆叠模型第一层中每一个基模型进行 内层手动交叉验证；
        内层手动交叉验证 也是5折，那么 内层手动交叉验证 的训练集数量是： 总训练集*4/5*4/5， 测试集数量是： 总训练集*4/5*1/5。
        3、所以 self.base_models_ 的shape为： 3行（堆叠模型第一层3个基模型）， 5列（内层5折手动交叉验证） 也就是 每个基模型 有5个 内层手动交叉验证 的训练模型。   
        4、堆叠模型第一层中每一个基模型的 内层5折手动交叉验证 结果存入 out_of_fold_predictions 矩阵中，shape为： X.shape[0]行， 3列（堆叠模型第一层3个基模型）
        5、self.meta_model_.fit(out_of_fold_predictions, y) 
        将 堆叠模型第一层的结果out_of_fold_predictions 和 y 传入 堆叠模型第二层模型进行训练。
        '''
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])  # 矩阵的行索引： 取矩阵的一整行
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred.ravel()

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # 看 “StackingAveragedModels代码拆分解析” 中 self.base_models_ 的结构。
    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)

# =============================Stacking models：堆叠模型=========================