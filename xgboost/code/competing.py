# coding:utf-8
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomTreesEmbedding, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
import csv
import os
import time
import pdb
import log


def s(arg):
    # print str(arg)
    return str(arg).split(", 'Hy'")[0]


def nanReplace(df):
    df['drug_MolProperty_TPSA'] = df['drug_MolProperty_TPSA'].apply(
        s)
    for col in df.columns[3:]:
        print '替换成float', col
        df[[col]] = df[[col]].astype(float)
    df.replace(np.inf, np.nan, inplace=True)  # 先把inf替换了
    empty = df.isnull().any()
    empty = empty[empty == True]
    mean = df.mean()
    for col in empty.index:
        print "替换nan,inf : ", col, mean[col] if col in mean else 0
        df[col].replace(np.nan, mean[col] if col in mean else 0, inplace=True)

# rf = RandomForestClassifier(n_estimators=100)
# rf = LogisticRegression()
# rf = KNeighborsClassifier()
# rf = SVC(kernel='linear', C=0.025)  # Linear SVM
# rf = SVC(gamma=2, C=1)  # RBF SVM
# rf = GaussianNB()  # Naive Bayes
# rf = MLPClassifier()  # Neural Net

meths = {
    "rf":RandomForestClassifier(n_estimators=100),
    "LR":LogisticRegression(),
    "knn":KNeighborsClassifier(),
    "svm":SVC(kernel='linear', C=0.025),  # Linear SVM
    # rf = SVC(gamma=2, C=1)  # RBF SVM
    "NB":GaussianNB(),  # Naive Bayes
    "NN":MLPClassifier()  # Neural Net  
}

def start(filename, logFile, method_name="rf"):
    model = meths[method_name]
    logger = log.get_logger(__name__, logFile)
    df = pd.read_csv('./data/{0}.csv'.format(filename))
    nanReplace(df)
    # pdb.set_trace()
    train, test = train_test_split(
        df, test_size=0.2)
    # 记录日志
    logger.info('训练方法:%s' % method_name)
    logger.info('训练样本采用五折交叉检验，数据样本是 %s' % filename)
    logger.info('样本维度为：%d %d' % (df.shape[0],df.shape[1]))
    logger.info('训练集 train 的样本数：%d ,%d  正样本数：%d   负样本数:%d' % (
        train.shape[0], train.shape[1], (train['label'] == 1).sum(axis=0), (train['label'] == 0).sum(axis=0)))
    logger.info('测试集 test 的样本数：%d , %d   正样本数：%d   负样本数:%d' % (
        test.shape[0], test.shape[1], (test['label'] == 1).sum(axis=0), (test['label'] == 0).sum(axis=0)))

    model.fit(train.drop(['label', 'proteinId', 'drugId'],
                      axis=1), train['label'])

    if hasattr(model, "predict_proba"):
        y_prop = model.predict_proba(
            test.drop(['label', 'proteinId', 'drugId'], axis=1))[:, 1]
    else:  # use decision function
        prob_pos = model.decision_function(
            test.drop(['label', 'proteinId', 'drugId'], axis=1))
        y_prop = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    # y_prop = model.predict_proba(
    #     test.drop(['label', 'proteinId', 'drugId'], axis=1))[:, 1]

    test_result = test[['proteinId', 'drugId', 'label']]
    test_result['score'] = y_prop
    test_result.to_csv(
        "./predict/compete_{0}_original.csv".format(method_name))

    test_result['Int'] = test_result.score
    test_result.Int[test_result.Int < 0.5] = 0
    test_result.Int[test_result.Int >= 0.5] = 1

    a = float((test_result['Int'] == test_result['label']).sum(
        axis=0)) / float(test_result.shape[0])
    logger.info('准确率%f' % a)

    aucVal = roc_auc_score(test.label, y_prop)
    logger.info('AUC: %.5f\n\n\n' % aucVal)

    # pipeline(0, boost_round, gamma, max_depth, lambd, subsample,
    #          colsample_bytree, min_child_weight, eta, seed, train0, test, logger, filename, keepId)
