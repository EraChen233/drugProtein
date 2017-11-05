# coding:utf-8
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.cross_validation import train_test_split
import csv
import os
import time
import pdb
import log
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from subprocess import check_output
from sklearn.preprocessing import OneHotEncoder


def pipeline(iteration, boost_round, gamma, max_depth, lambd, subsample, colsample_bytree, min_child_weight, eta, seed, train0, test, logger, filename, keepId):

    dropCols = ['label'] if keepId else ['label', 'proteinId', 'drugId']
    train1, val = train_test_split(train0, test_size=0.1, random_state=1)
    y = train1.label

    dtrain = xgb.DMatrix(train1.drop(dropCols, axis=1), label=train1.label)
    dtest = xgb.DMatrix(test.drop(dropCols, axis=1))  # use local test data
    dval = xgb.DMatrix(val.drop(dropCols, axis=1), label=val.label)
    recordFileName = "{0}_{1}_{2}_{3}".format(
        filename, "keepId" if keepId else "withoutId", boost_round, time.strftime('%H%M%S'))

    params = {
        'booster': 'gbtree',  # linear booster 和tree booster。linear booster没有tree booster好，因此很少用到。
        # 默认reg:linear，这个参数定义需要被最小化的损失函数。binary:logistic 二分类的逻辑回归，返回预测的概率(不是类别)
        'objective': 'binary:logistic',
        # 默认1 在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。
        'scale_pos_weight': float(len(y) - sum(y)) / float(sum(y)),
        'eval_metric': 'auc',
        'gamma': gamma,                     # 默认0。用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': max_depth,             # 默认6，典型值3-10，构建树的深度，越大越容易过拟合
        'lambda': lambd,                    # 默认1 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        # 默认1 随机采样训练样本，这个参数控制对于每棵树，随机采样的比例。 减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。 典型值：0.5-1
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,    # 默认1 生成树时进行的列采样
        # 默认1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言,
        'min_child_weight': min_child_weight,
        # 假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        'eta': eta,                         # 默认0.3 如同学习率,通过减少每一步的权重，可以提高模型的鲁棒性。 典型值为0.01-0.2。
        'seed': seed,                  # 默认0，随机数的种子 设置它可以复现随机数据的结果，也可以用于调整参数，
        'early_stopping_rounds': 100
    }

    logger.info('迭代次数 %d' % boost_round)

    watchlist = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(
        params, dtrain, num_boost_round=boost_round, evals=watchlist)
    model.save_model(
        './model/{0}.model'.format(recordFileName))
    # model.dump_model(
    #     './model/{0}.txt'.format(recordFileName))

    # --------------------------- 用xgboot的predict(dtrain, pred_leaf=True)生成新的特征 ------------------------
    train_new_feature = model.predict(dtrain, pred_leaf=True)
    test_new_feature = model.predict(dtest, pred_leaf=True)
    train_new_feature1 = pd.DataFrame(train_new_feature)
    test_new_feature1 = pd.DataFrame(test_new_feature)

    pdb.set_trace()

    logger.info("原始train: %d  %d，原始test %d  %d",
                train1.shape[0], train1.shape[1], test.shape[0], test.shape[1])
    logger.info("训练train: %d  %d，训练test %d  %d",
                train_new_feature1.shape[0], train_new_feature1.shape[1], test_new_feature1.shape[0], test_new_feature1.shape[1])
    lr = LogisticRegression(n_jobs=-1, C=0.1, penalty='l1')
    lr.fit(train_new_feature1, y)
    # joblib.dump(lr, 'gbdt_lr_test/model/lr_orgin.m')
    # 预测及AUC评测
    y_pred_test = lr.predict_proba(test_new_feature1)[:, 1]
    # lr_test_auc = roc_auc_score(y_test_origin, y_pred_test)
    # print('基于原有特征的LR AUC: %.5f' % lr_test_auc)
    pd.DataFrame({"proteinId": test['proteinId'].values, "drugId": test['drugId'].values, "label": test['label'].values,  "predict": y_pred_test}).to_csv(
        'gbdt_proto_feature_lr.csv', index=False)

    # ---------------------------------------------------------------------------------------------------------

    print "best best_ntree_limit", model.best_ntree_limit  # did not save the best,why?

    # ---------------------------------- gbdt+lr test1 --------------------------------------------------------
    # 定义模型
    # xgboost = xgb.XGBClassifier(nthread=4, learning_rate=0.08,
    #                             n_estimators=50, max_depth=5, gamma=0, subsample=0.9, colsample_bytree=0.5)
    # # 训练学习
    # xgboost.fit(train1.drop(dropCols, axis=1), train1.label)
    # X_train_leaves = xgboost.apply(train1.drop(dropCols, axis=1))  # 70 50
    # pd.DataFrame(X_train_leaves).to_csv('X_train_leaves.csv', index=None)

    # print "train1:", train1.shape
    # print "X_train_leaves.shape:", X_train_leaves.shape
    # ----------------------------------------------------------------------------------------------------------

    print "best best_ntree_limit", model.best_ntree_limit  # did not save the best,why?

    test_y = model.predict(dtest)
    test_result = test[['proteinId', 'drugId', 'label']]
    test_result['score'] = test_y

    test_result['Int'] = test_result.score
    test_result.Int[test_result.Int < 0.5] = 0
    test_result.Int[test_result.Int >= 0.5] = 1

    # remember to edit xgb.csv , add ""
    test_result.to_csv(
        "./predict/{0}.csv".format(recordFileName))

    # save feature score
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(),
                           key=lambda x: x[1], reverse=True)
    fs = []
    for (key, value) in feature_score:
        fs.append("{0},{1}\n".format(key, value))

    with open('./featurescore/{0}.csv'.format(recordFileName), 'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)
    a = float((test_result['Int'] == test_result['label']).sum(
        axis=0)) / float(test_result.shape[0])
    logger.info('准确率%f\n\n\n' % a)


def start(filename, boost_round, keepId, logFile):

    logger = log.get_logger(__name__, logFile)

    # if not os.path.exists("./predict/%s" % filename):
    #     os.mkdir("./predict/%s" % filename)

    train = pd.read_csv('./data/{0}.csv'.format(filename))
    train0, test = train_test_split(train, test_size=0.2, random_state=1)

    # 记录日志
    logger.info('训练样本采用五折交叉检验，保留protein_id、drug_id，数据样本是 %s' % filename)
    logger.info('训练集 train0 的样本数：%d ,%d  正样本数：%d   负样本数:%d' % (
        train0.shape[0], train0.shape[1], (train0['label'] == 1).sum(axis=0), (train0['label'] == 0).sum(axis=0)))
    logger.info('测试集 test 的样本数：%d , %d   正样本数：%d   负样本数:%d' % (
        test.shape[0], test.shape[1], (test['label'] == 1).sum(axis=0), (test['label'] == 0).sum(axis=0)))

    max_depth = 10                  # range(3,15,1)
    min_child_weight = 4.7         # [i/10.0 for i in range(0,50,1)]
    gamma = 6
    subsample = 0.88
    colsample_bytree = 0.22
    eta = 0.01
    lambd = 0.31
    seed = 27

    # with open('./params/params.csv','w') as f:
    #     writer = csv.writer(f)
    # paramsName = ['iteration', 'accuracy', 'boost_round', 'gamma', 'max_depth',
    #               'lambda', 'subsample', 'colsample_bytree', 'min_child_weight', 'eta', 'seed']
    #     writer.writerow(paramsName)
    #     # for i in range(len(subsample1)):
    #     #     for j in range(len(colsample_bytree1)):
    #     #         index = i*len(subsample1) +j;
    #     #         subsample = subsample1[i]
    #     #         colsample_bytree = colsample_bytree1[j]
    #     #         result = pipeline(index,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight,eta,seed)
    #     #         writer.writerow([index,result,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight,eta,seed])
    #     for i in range(len(boost_round1)):
    #         boost_round = boost_round1[i]
    #         result = pipeline(i,boost_round,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight,eta,seed)
    #         writer.writerow([i,result,boost_round,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight,eta,seed])
    # f.close

    pipeline(0, boost_round, gamma, max_depth, lambd, subsample,
             colsample_bytree, min_child_weight, eta, seed, train0, test, logger, filename, keepId)