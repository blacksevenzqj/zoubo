
# coding: utf-8

# In[ ]:
#宽带营销的数据"broadband.csv"
from sklearn.model_selection import train_test_split
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV

import os

os.chdir(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\TianshanCollege\7_BankAnti-fraud_model\homework")

# In[ ]:
model_data = pd.read_csv("broadband.csv")
model_data.head()

# In[ ]:
target = model_data["BROADBAND"]
orgData1 = model_data.ix[ :,1:-2]
print(pd.value_counts(target, sort=True)) # 0:908 1:206 样本不平衡

# In[ ]:
train_data, test_data, train_target, test_target = train_test_split(
    orgData1, target, test_size=0.4, train_size=0.6, random_state=12345)  #划分训练集和测试集

# In[ ]:
#决策树算法
param_grid = {
    'criterion':['entropy','gini'],
    'max_depth':[2,3,4,5,6,7,8],
    'min_samples_split':[4,8,12,16,20,24,28] 
}
clf = tree.DecisionTreeClassifier()
clfcv = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc', cv=4)
clfcv.fit(train_data, train_target)

test_est = clfcv.predict(test_data)

print("decision tree accuracy:") # 在不知道阈值的情况下看 精准率 和 召回率 没有意义
print(metrics.classification_report(test_target,test_est)) 

print("decision tree AUC:") # 主要看 AUC面积
fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est)
print('AUC = %.4f' % metrics.auc(fpr_test, tpr_test))


################################################################################################
# In[ ]:
#随机森林
param_grid = {
    'criterion':['entropy','gini'],
    'max_depth':[7,8,10,12], # [5,6,7,8] 初次参数，需要反复调整。
    'n_estimators':[11,13,15],  #决策树个数-随机森林特有参数
    'max_features':[0.2,0.3,0.4,0.5], #每棵决策树使用的变量占比-随机森林特有参数
    'min_samples_split':[4,8,12,16] 
}

rfc = ensemble.RandomForestClassifier()
rfccv = GridSearchCV(estimator=rfc, param_grid=param_grid, scoring='roc_auc', cv=4)
rfccv.fit(train_data, train_target)
test_est = rfccv.predict(test_data)
print("random forest accuracy:")
print(metrics.classification_report(test_target,test_est))
print("random forest AUC:")
fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est)
print('AUC = %.4f' % metrics.auc(fpr_test, tpr_test))

#%%
rfccv.best_params_
# 由于一般缺乏对网格搜索参数的经验，建议把最优参数打印出来，看看取值是否在边届上，如果在边界上，就需要扩大搜索范围；
# 如果最优值在 左边界，则搜索参数向小方向调整： min_samples_split最优值为4，则将左边界调小，并适当缩小右边界，再次计算 直到最优值为中间值。
# 如果最优值在 右边界，则搜索参数向大方向调整： max_depth最优值为8，则将右边界调大，并适当缩小左边界，再次计算 直到最优值为中间值。
# 网格搜索需要有宽到细多进行几次。



#%%
################################################################################################
#Adaboost算法： 就是一层的二叉树
param_grid = {
    #'base_estimator':['DecisionTreeClassifier'],
    'learning_rate':[0.1,0.3,0.5,0.7,1]
}
abc = ensemble.AdaBoostClassifier(n_estimators=100,algorithm='SAMME')
abccv = GridSearchCV(estimator=abc, param_grid=param_grid, scoring='roc_auc', cv=4)
abccv.fit(train_data, train_target)
test_est = abccv.predict(test_data)
print("abc classifier accuracy:")
print(metrics.classification_report(test_target,test_est))
print("abc classifier AUC:")
fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est)
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))

#%%
abccv.best_params_



# In[ ]:
################################################################################################
#GBDT
param_grid = {
    'loss':['deviance','exponential'],
    'learning_rate':[0.1,0.3,0.5,0.7,1],
    'n_estimators':[10,15,20,30],  #决策树个数-GBDT特有参数
    'max_depth':[1,2,3],  #单棵树最大深度-GBDT特有参数
    'min_samples_split':[2,4,8,12,16,20] # 内部节点再划分所需最小样本数
    
}

gbc = ensemble.GradientBoostingClassifier()
gbccv = GridSearchCV(estimator=gbc, param_grid=param_grid, scoring='roc_auc', cv=4)
gbccv.fit(train_data, train_target)
test_est = gbccv.predict(test_data)
print("gradient boosting accuracy:")
print(metrics.classification_report(test_target,test_est))
print("gradient boosting AUC:")
fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est)
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))

#%%
gbccv.best_params_
#为什么一定要打印模型参数看一下?
# In[ ]:




#%%
