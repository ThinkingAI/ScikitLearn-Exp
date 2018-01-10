# -*- coding: utf-8 -*-
# -*- author: knock -*-

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split  #old version is cross_validation module
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
'''
本实验应用标准Sklearn库，并采用标准化机器学习步骤进行分类实验;
数据集为鸢尾花数据集；
这里应用SGDClassifier分类器进行分类，适合做概率估计，即对分配给每个类别一个估计概率；
这里采用CrossValidation进行交叉训练检验模型;
'''
#1. 从自带sklearn.datasets数据集加载数据
iris = load_iris()

#2. DESCR数据
print(iris.keys())
print(iris.data.shape, iris.target.shape,iris.feature_names,iris.target_names,iris.DESCR)
print(iris.data[:2])

#3. 定义训练集和测试集
X_iris, y_iris = iris.data, iris.target
# 30%的训练数据作为测试集
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.3, random_state = 33)
#4. 数据标准化预处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#5. 训练模型选择并训练，这里使用SGD分类器，适合大规模数据，随机梯度下降方法估计参数
clf = SGDClassifier()
clf.fit(X_train, y_train)

#6. 利用训练好的模型来进行预测
y_train_predict = clf.predict(X_train)

#7. 内测，使用训练样本进行准确性能评估
score = metrics.accuracy_score(y_train, y_train_predict)
print( score)

#8. 标准外测，利用测试集进行准确性评估
y_predict = clf.predict(X_test)
score = metrics.accuracy_score(y_test, y_predict)
print( score)

#9. 应用pipeline进行多步骤流水线作业（预处理，学习）
clf = Pipeline([('scaler', StandardScaler()), ('sgd_classifier', SGDClassifier())])

#10. 5折交叉验证整个数据集合
cv = KFold(5, shuffle=True, random_state = 33)
scores = cross_val_score(clf, X_iris, y_iris, cv=cv)

print( scores)
#计算均值和标准差
print( scores.mean(),scores.std())


'''
scores output:
0.847619047619
0.822222222222
[ 0.9         0.86666667  0.8         1.          0.93333333]
0.9 0.0666666666667
'''