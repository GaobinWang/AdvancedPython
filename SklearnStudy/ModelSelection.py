#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 23:35:36 2017

@author: Lesile
"""

"""
Part1:Cross-validation
机器学习方法论中将数据分为三部分:训练集(train set)、验证集(validation set)和测试集(test set).
我们在训练集中训练参数,在验证集中验证模型效果,在测试集上进行预测.
然而,上述方法减少了用于训练模型的样本数,并且模型的结果依赖于训练集合测试集的选取.
为了使模型具有尽可能多的训练样本,同时使模型具有较好的泛化能力,我们可以采取交叉验证(cross-validation)的方法.
k-折交叉验证(k-fold CV)将样本随机分为k组,进行k次循环,每次循环中用(k-1)组样本训练模型,并在剩余的
一组中进行测试,并计算评价指标.模型最终的评价指标是k次循环得到的评价指标的平均.
k-折交叉验证虽然增加了计算量,但是提高了样本的使用效率.
"""
###将数据集分为训练集和测试集,在训练集上训练模型,在测试集上进行测试.
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
iris.data.shape, iris.target.shape
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
X_train.shape, y_train.shape
X_test.shape, y_test.shape

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)


###计算交叉验证后的评价指标
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
scores
#平均得分
scores.mean()
#95%置信区间
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))                                              

##更改评价指标
from sklearn import metrics
scores = cross_val_score(
    clf, iris.data, iris.target, cv=5, scoring='f1_macro')
scores   

##其他的交叉验证方法
from sklearn.model_selection import ShuffleSplit
n_samples = iris.data.shape[0]
cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
cross_val_score(clf, iris.data, iris.target, cv=cv)


