# -*- coding: utf-8 -*-
# -*- author: knock -*-

from sklearn.datasets import  fetch_20newsgroups
from sklearn.model_selection import train_test_split  #old version is cross_validation module
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from scipy.stats import sem
# 许多原始数据无法直接被分类器所使用，图像可以直接使用pixel信息，文本则需要进一步处理成数值化的信息
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
'''
本实验应用标准Sklearn库，并采用标准化机器学习步骤进行分类实验；
数据集为 20类新闻数据；
这里应用Native Bayes分类器进行分类，适用于文本分类，邮件过滤等场景；
这里采用CrossValidation进行交叉训练检验模型;
'''
#1. 从自带sklearn.datasets数据集加载数据
news = fetch_20newsgroups()

#2. DESCR数据
print(news.keys())
print(news.target_names, news.target_names.__len__())# 检索出新闻类别和种类
print(len(news.data), len(news.target))
print(news.data[:2])
#exit(0)
#3. 定义训练集和测试集
X_news, y_news = news.data, news.target
# 30%的训练数据作为测试集
X_train, X_test, y_train, y_test = train_test_split(X_news, y_news, test_size=0.3, random_state = 33)
#4. 特征工程，选取特征，应用多个特征提取方法
#5. 训练模型选择并训练，这里使用xxx分类器
#9. 应用pipeline进行多步骤流水线作业（预处理，学习）
# 许多原始数据无法直接被分类器所使用，图像可以直接使用pixel信息，文本则需要进一步处理成数值化的信息
# 我们在NB_Classifier的基础上，对比几种特征抽取方法的性能。并且使用Pipline简化构建训练流程
clf_1 = Pipeline([('count_vec', CountVectorizer()), ('mnb', MultinomialNB())])
#clf_2 = Pipeline([('hash_vec', HashingVectorizer()), ('mnb', MultinomialNB())])
clf_3 = Pipeline([('tfidf_vec', TfidfVectorizer()), ('mnb', MultinomialNB())])



#6. 利用训练好的模型来进行预测
#7. 内测，使用训练样本进行准确性能评估
#8. 标准外测，利用测试集进行准确性评估
#10. 5折交叉验证整个数据集合
# 构造一个便于交叉验证模型性能的函数（模块）
def evaluate_cross_validation(clf, X, y, K):
    # KFold 函数需要如下参数，数据量, K,是否洗牌
    cv = KFold(K, shuffle=True, random_state = 0)
    # 采用上述的分隔方式进行交叉验证，测试模型性能，对于分类问题，这些得分默认是accuracy，也可以修改为别的
    scores = cross_val_score(clf, X, y, cv=cv)
    print (scores)
    print ('\nMean score: %.3f (+/-%.3f)\n' % (scores.mean(), sem(scores)))


clfs = [clf_1, clf_3]
for clf in clfs:
    evaluate_cross_validation(clf, X_train, y_train, 5)

# 从上述结果中，我们发现常用的两个特征提取方法得到的性能相当。 让我们选取其中之一，进一步靠特征的精细筛选提升性能。
clf_4 = Pipeline([('tfidf_vec_adv', TfidfVectorizer(stop_words='english')), ('mnb', MultinomialNB())])
evaluate_cross_validation(clf_4, X_train, y_train, 5)

# 如果再尝试修改贝叶斯分类器的平滑参数，调优性能
clf_5 = Pipeline([('tfidf_vec_adv', TfidfVectorizer(stop_words='english')), ('mnb', MultinomialNB(alpha=0.01))])
evaluate_cross_validation(clf_5, X_train, y_train, 5)

'''

'''