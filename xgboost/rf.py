# coding:utf-8
# 使用随机森林进行预测
# author chenjiyun(1211169311@qq.com)
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import csv
import os
import numpy as np
from code import log

logger = log.get_logger(__name__, 'train.log')
filename = 'drug_protein'
boost_round = 100  # [10,100,1000,10000]
# dropCols = ['label', 'proteinId', 'drugId']
dropCols = ['label']


def convertToNumberAndRemoveNan(table, len=0):
    try:
        if len:
            colTypes = table.dtypes[len:]
        else:
            colTypes = table.dtypes
        for col in colTypes.index:
            if colTypes[col] != 'float64' and colTypes[col] != 'int64':
                print colTypes[col], "下标", col
                table[[col]] = table[[col]].astype(float)
    except Exception, e:
        print '出错啦:', e
    return table.fillna(-1)


if __name__ == "__main__":

    table = pd.read_csv('./data/{0}.csv'.format(filename))
    table = convertToNumberAndRemoveNan(table, 3)

    train, test = train_test_split(table, test_size=0.2, random_state=1)
    testIds = test[['proteinId', 'drugId', 'label']]
    train_y = train['label'].values
    train_x = train.drop(['proteinId', 'drugId'], axis=1).values
    test_x = test.drop(['proteinId', 'drugId'], axis=1).values

    try:
        for i in range(len(train_x)):
            sample = train_x[i]
            for j in range(len(sample)):
                if np.isnan(sample[j]):
                    print 'train_x', i, j, sample[j]
    except Exception, e:
        print '出错啦:', e

    try:
        for i in range(len(train_y)):
            if np.isnan(train_y[i]):
                print 'train_y', i, train_y[i]
    except Exception, e:
        print '出错啦:', e

    # 随机森林进行预测
    RF = RandomForestClassifier()
    RF = RF.fit(train_x, train_y)

    # 获得预测结果概率
    predicted_proba = RF.predict_proba(test_x)
    # 转为DataFrame类型
    predicted_proba = pd.DataFrame(predicted_proba)

    # 概率对的结果加入进用户商品对中
    predicted = pd.concat(
        [testIds, predicted_proba], axis=1)
    predicted.to_csv('./rfPredict/%s.csv' % filename, index=None)
