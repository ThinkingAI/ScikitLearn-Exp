# -*- coding: utf-8 -*-
# -*- author: knock -*-

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split  #old version is cross_validation module
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
'''
本实验应用标准Sklearn库，并采用标准化机器学习步骤进行分类实验；
数据集为 AT&T 400张人脸；
这里应用svm分类器进行分类，强分类器，直接做分类，基于线性假设，将特征映射到更加高维度，甚至非线性的空间上，从而使数据空间变得更加可分；
这里采用CrossValidation进行交叉训练检验模型;
'''
#1. 从自带sklearn.datasets数据集加载数据
faces = fetch_olivetti_faces()

#2. DESCR数据
print(faces.keys())
print(faces.data.shape, faces.target.shape, faces.DESCR)
print(faces.data[:2])

#3. 定义训练集和测试集
X_face, y_face = faces.data, faces.target
# 30%的训练数据作为测试集
X_train, X_test, y_train, y_test = train_test_split(X_face, y_face, test_size=0.3, random_state = 33)
#4. 数据标准化预处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#5. 训练模型选择并训练，这里使用SVC分类器
clf = SVC( kernel='linear')
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
clf = Pipeline([('scaler', StandardScaler()), ('sgd_classifier', SVC( kernel='linear'))])

#10. 5折交叉验证整个数据集合
cv = KFold(5, shuffle=True, random_state = 33)
scores = cross_val_score(clf, X_face, y_face, cv=cv)

print( scores)
#计算均值和标准差
print( scores.mean(),scores.std())


'''
scores output:
[[ 0.30991736  0.36776859  0.41735536 ...,  0.15289256  0.16115703
   0.1570248 ]
 [ 0.45454547  0.47107437  0.51239669 ...,  0.15289256  0.15289256
   0.15289256]]
1.0
0.966666666667
[ 0.9875  0.9375  0.95    0.9625  0.95  ]
0.9575 0.0169558249578
'''