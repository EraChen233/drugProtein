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
# from sklearn.cross_validation import StratifiedKFold
# from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import normalize
# from sklearn.decomposition import PCA
# from subprocess import check_output
# from sklearn.preprocessing import OneHotEncoder


class Train():
    def __init__(self, train_method, filename, boost_round, keepId, logFile):
        self.logger = log.get_logger(__name__, logFile)
        self.train_method, self.filename, self.keepId, self.boost_round = train_method, filename, keepId, boost_round
        self.data = pd.read_csv('./data/{0}.csv'.format(filename))
        self.nanReplace(self.data)
        self.train0, self.test = train_test_split(
            self.data, test_size=0.2)
        self.dropCols = ['label'] if keepId else [
            'label', 'proteinId', 'drugId']
        self.gbdt_args = {
            'max_depth': 5,                  # range(3,15,1)
            'min_child_weight': 4.7,         # [i/10.0 for i in range(0,50,1)]
            'gamma': 6,
            'subsample': 0.88,
            'colsample_bytree': 0.22,
            'eta': 0.01,
            'lambd': 0.31,
            'seed': 27
        }
        self.gbdt()

    def gbdt(self):
        train1, val = train_test_split(
            self.train0, test_size=0.1, random_state=1)
        trainY = train1.label
        train1 = train1.drop(self.dropCols, axis=1)

        dtrain = xgb.DMatrix(train1, label=trainY)
        dtest = xgb.DMatrix(self.test.drop(
            self.dropCols, axis=1))  # use local test data
        dval = xgb.DMatrix(val.drop(self.dropCols, axis=1), label=val.label)

        params = {
            'booster': 'gbtree',  # linear booster 和tree booster。linear booster没有tree booster好，因此很少用到。
            # 默认reg:linear，这个参数定义需要被最小化的损失函数。binary:logistic 二分类的逻辑回归，返回预测的概率(不是类别)
            'objective': 'binary:logistic',
            # 默认1 在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。
            'scale_pos_weight': float(len(trainY) - sum(trainY)) / float(sum(trainY)),
            'eval_metric': 'auc',
            # 'gamma': gamma,                     # 默认0。用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
            # 'max_depth': max_depth,             # 默认6，典型值3-10，构建树的深度，越大越容易过拟合
            # 'lambda': lambd,                    # 默认1 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            # 默认1 随机采样训练样本，这个参数控制对于每棵树，随机采样的比例。 减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。 典型值：0.5-1
            # 'subsample': subsample,
            # 'colsample_bytree': colsample_bytree,    # 默认1 生成树时进行的列采样
            # 默认1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言,
            # 'min_child_weight': min_child_weight,
            # 假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
            # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
            # 'eta': eta,                         # 默认0.3 如同学习率,通过减少每一步的权重，可以提高模型的鲁棒性。 典型值为0.01-0.2。
            # 'seed': seed,                  # 默认0，随机数的种子 设置它可以复现随机数据的结果，也可以用于调整参数，
            'early_stopping_rounds': 100
        }
        params.update(self.gbdt_args)

        self.logDataInfo()
        self.logger.info('迭代次数 %d' % self.boost_round)

        watchlist = [(dtrain, 'train'), (dval, 'val')]
        self.gbdt_model = xgb.train(
            params, dtrain, num_boost_round=self.boost_round, evals=watchlist)
        # self.gbdt_model.save_model(
            # './model/{0}.model'.format(self.currFilenam()))
        # self.gbdt_model.dump_model(
        #     './model/{0}.txt'.format(self.currFilenam()))

        # ----------------- 以下使用GBDT的输出作为LR的输入 -----------------
        if self.train_method == "GBDT_LR":
            lrTrain, lrTest = self.constructLrInput(train1, dtrain, dtest)
            self.logger.info("原始train: %d  %d，原始test %d  %d",
                             train1.shape[0], train1.shape[1], self.test.shape[0], self.test.shape[1])
            self.logger.info("训练train: %d  %d，训练test %d  %d",
                             lrTrain.shape[0], lrTrain.shape[1], lrTest.shape[0], lrTest.shape[1])
            self.lr(lrTrain, lrTest, trainY)

        # did not save the best,why?
        print "best best_ntree_limit", self.gbdt_model.best_ntree_limit

        test_y = self.gbdt_model.predict(dtest)
        self.arrangeResult(self.test, test_y,
                           "./predict/{0}.csv".format(self.currFilenam()))

        # save feature score
        feature_score = self.gbdt_model.get_fscore()
        feature_score = sorted(feature_score.items(),
                               key=lambda x: x[1], reverse=True)
        fs = []
        for (key, value) in feature_score:
            fs.append("{0},{1}\n".format(key, value))

        with open('./featurescore/{0}.csv'.format(self.currFilenam()), 'w') as f:
            f.writelines("feature,score\n")
            f.writelines(fs)

    def constructLrInput(self, train1, dtrain, dtest):
        # --------------------------- 用xgboot的predict(dtrain, pred_leaf=True)生成新的特征 ------------------------
        train_new_feature = self.gbdt_model.predict(dtrain, pred_leaf=True)
        test_new_feature = self.gbdt_model.predict(dtest, pred_leaf=True)
        train_new_feature1 = pd.DataFrame(train_new_feature)
        test_new_feature1 = pd.DataFrame(test_new_feature)

        # ********************* 增加原始特征训练 ************************
        train_new_feature1 = pd.concat(
            [train1.reset_index(drop=True), train_new_feature1], axis=1)
        test_new_feature1 = pd.concat(
            [self.test.drop(self.dropCols, axis=1).reset_index(drop=True), test_new_feature1], axis=1)
        # ************************************************************
        train_new_feature1.to_csv(
            './gbdt_lr_test/train_data/gbdt_lr_%s.csv' % self.currFilenam(), index=False)
        test_new_feature1.to_csv(
            './gbdt_lr_test/test_data/gbdt_lr_%s.csv' % self.currFilenam(), index=False)
        return train_new_feature1, test_new_feature1

    def nanReplace(self, df):
        df.replace(np.inf, np.nan, inplace=True)  # 先把inf替换了
        empty = df.isnull().any()
        empty = empty[empty == True]
        mean = df.mean()
        for col in empty.index:
            df[col].replace(np.nan, mean[col], inplace=True)

    def lr(self, train, test, trainY):

        # pdb.set_trace()

        lr = LogisticRegression(n_jobs=-1, C=0.1, penalty='l1')
        lr.fit(train, trainY)
        # joblib.dump(lr, 'gbdt_lr_test/model/lr_orgin.m')
        # 预测及AUC评测

        y_pred_test = lr.predict_proba(test)[:, 1]
        self.arrangeResult(
            self.test, y_pred_test, './gbdt_lr_test/predict/gbdt_lr_%s.csv' % self.currFilenam())
        # ---------------------------------- gbdt+lr test1 --------------------------------------------------------
        # 定义模型
        # xgboost = xgb.XGBClassifier(nthread=4, learning_rate=0.08,
        #                             n_estimators=50, max_depth=5, gamma=0, subsample=0.9, colsample_bytree=0.5)
        # # 训练学习
        # xgboost.fit(train1, trainY)
        # X_train_leaves = xgboost.apply(train1)  # 70 50
        # pd.DataFrame(X_train_leaves).to_csv('X_train_leaves.csv', index=None)

        # print "train1:", train1.shape
        # print "X_train_leaves.shape:", X_train_leaves.shape
        # ----------------------------------------------------------------------------------------------------------

    def logDataInfo(self):
        # 记录日志
        self.logger.info(
            '训练样本采用五折交叉检验，数据样本是 %s' % self.filename)
        self.logger.info('样本维度为：%d %d' % (self.data.shape[0],self.data.shape[1]))
        self.logger.info('训练集 train0 的样本数：%d ,%d  正样本数：%d   负样本数:%d' % (
            self.train0.shape[0], self.train0.shape[1], (self.train0['label'] == 1).sum(axis=0), (self.train0['label'] == 0).sum(axis=0)))
        self.logger.info('测试集 test 的样本数：%d , %d   正样本数：%d   负样本数:%d' % (
            self.test.shape[0], self.test.shape[1], (self.test['label'] == 1).sum(axis=0), (self.test['label'] == 0).sum(axis=0)))

    def arrangeResult(self, originalDF, y, filename):
        result = originalDF[['proteinId', 'drugId', 'label']]
        result['score'] = y

        result['Int'] = result.score
        result.Int[result.Int < 0.5] = 0
        result.Int[result.Int >= 0.5] = 1
        result.to_csv(filename, index=None)

        a = float((result['Int'] == result['label']).sum(
            axis=0)) / float(result.shape[0])
        self.logger.info('准确率%f' % a)

        aucVal = roc_auc_score(originalDF.label, y)
        self.logger.info('AUC: %.5f\n\n\n' % aucVal)

    def currFilenam(self):
        return "{0}_{1}_{2}_{3}".format(
            self.filename, "keepId" if self.keepId else "withoutId", self.boost_round, time.strftime('%H%M%S'))
