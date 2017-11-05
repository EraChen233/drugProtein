# coding:utf-8
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.cross_validation import train_test_split
# import sys,random
import csv
import os
import re
import time
import pdb
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
from code import log
from sklearn.preprocessing import OneHotEncoder

logger = log.get_logger(__name__, 'train1.log')
filename = 'drug_protein'
boost_round = 100  # [10,100,1000,10000]
dropCols = ['label', 'proteinId', 'drugId']
# dropCols = ['label']


if not os.path.exists("./predict/%s" % filename):
    os.mkdir("./predict/%s" % filename)
if not os.path.exists("./featurescore/%s" % filename):
    os.mkdir("./featurescore/%s" % filename)
if not os.path.exists("./model/%s" % filename):
    os.mkdir("./model/%s" % filename)

train = pd.read_csv('./data/{0}.csv'.format(filename))
train0, test = train_test_split(train, test_size=0.2, random_state=1)

# 记录日志
logger.info('训练样本采用五折交叉检验，保留protein_id、drug_id，数据样本是 %s' % filename)
logger.info('训练集 train0 的样本数：%d ,%d  正样本数：%d   负样本数:%d' % (
    train0.shape[0], train0.shape[1], (train0['label'] == 1).sum(axis=0), (train0['label'] == 0).sum(axis=0)))
logger.info('测试集 test 的样本数：%d , %d   正样本数：%d   负样本数:%d' % (
    test.shape[0], test.shape[1], (test['label'] == 1).sum(axis=0), (test['label'] == 0).sum(axis=0)))

train1, val = train_test_split(train0, test_size=0.1, random_state=1)

y = train1.label

dtrain = xgb.DMatrix(train1.drop(dropCols, axis=1), label=train1.label)
dtest = xgb.DMatrix(test.drop(dropCols, axis=1))  # use local test data
dval = xgb.DMatrix(val.drop(dropCols, axis=1), label=val.label)

paramsName = ['iteration', 'accuracy', 'boost_round', 'gamma', 'max_depth',
              'lambda', 'subsample', 'colsample_bytree', 'min_child_weight', 'eta', 'seed']


def pipeline(iteration, boost_round, gamma, max_depth, lambd, subsample, colsample_bytree, min_child_weight, eta, seed):
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
        #'alpha':alpha
        # 'nthread':8,                       #缺省值是当前系统可以获得的最大线程数
        # 'silent':1
        'early_stopping_rounds': 100
    }

    logger.info('迭代次数 %d' % boost_round)

    # The early stopping is based on last set in the evallist
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    # watchlist  = [(dtrain,'train')]#The early stopping is based on last set in the evallist
    model = xgb.train(
        params, dtrain, num_boost_round=boost_round, evals=watchlist)
    model.save_model('./model/{0}/xgb{1}.model'.format(filename, iteration))
    # model.dump_model(
    #     './model/{0}/xgb_featrue{1}.txt'.format(filename, iteration))

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

    test_y = model.predict(dtest)
    test_result = pd.DataFrame()
    test_result['proteinId'] = test.proteinId
    test_result['drugId'] = test.drugId
    test_result['label'] = test.label
    test_result['score'] = test_y

    test_result['Int'] = test_result.score
    test_result.Int[test_result.Int < 0.5] = 0
    test_result.Int[test_result.Int >= 0.5] = 1

    # remember to edit xgb.csv , add ""
    test_result.to_csv(
        "./predict/{0}/test{1}.csv".format(filename, iteration), encoding='utf-8', index=None)

    # save feature score
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(),
                           key=lambda x: x[1], reverse=True)
    fs = []
    for (key, value) in feature_score:
        fs.append("{0},{1}\n".format(key, value))

    with open('./featurescore/{0}/feature_score_{1}.csv'.format(filename, iteration), 'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)
    # print (test_result['Int'] == test_result['label'])
    # print "\n-----------------------------\n"
    # print ((test_result['Int'] == test_result['label']).sum(axis=0))
    # print (test_result.shape[0])
    a = float((test_result['Int'] == test_result['label']).sum(
        axis=0)) / float(test_result.shape[0])
    logger.info('准确率%f\n\n\n' % a)


def oneHotEncoderBetweenTwoTwoTable(df1, df2):
    for i in df1.columns:
        arr = np.unique(list(df1[i]) + list(df2[i]))
        ohe = OneHotEncoder()


def lr(GBDT_FILE, train, test):
    check_time = time.time()
    print "GBDT_FILE", GBDT_FILE
    f = open(GBDT_FILE, 'r')
    feature_set = set()
    origin_cols = train.columns[3:]
    for line in f.readlines():
        if '<' in line:  # feature line
            line = line.split(':')[1].strip()
            feature_re = re.match('\[(.*)?\]', line)
            info = feature_re.group(0)  # should be only one group
            info = re.sub('\[|\]', '', info)
            feature_set.add(info)

    # feature encoder

    logger.info('GBDT feature set length: %d\n\n\n' % len(feature_set))

    i = 1
    for info in feature_set:
        print i, " feature_set:", info
        key = info.split("<")[0].strip()
        value = float(info.split("<")[1].strip())
        train[info] = train[key].apply(lambda x: 1 if x < value else 0)
        test[info] = test[key].apply(lambda x: 1 if x < value else 0)
        i += 1

    print "begin drop columns"
    # remove original feature
    train = train.drop(origin_cols, axis=1)
    test = test.drop(origin_cols, axis=1)
    train.to_csv('train_after_gbdt.csv', index=None)
    test.to_csv('test_after_gbdt.csv', index=None)
    logger.info('feature generated base on gbdt sub-model, <<<<%s feature generated, time spend:<<<<%s' %
                (test.shape[1] - 1, round(((time.time() - check_time) / 60), 2)))
    check_time = time.time()

    #--------------------------------------------------------UNDERSAMPLE-------------------------------------------------------------------
    train1 = train[train['label'] == 1]  # positive train samples
    train2 = train[train['label'] == 0]  # negative train samples
    # train2 suppose to be the majority type, if not change it
    if train2.shape[0] < train1.shape[0]:
        train1 = train[train['label'] == 0]  # positive train samples
        train2 = train[train['label'] == 1]  # negative train samples
    train1 = train1.reset_index(drop=True)
    train2 = train2.reset_index(drop=True)
    fold = train2.shape[0] / train1.shape[0]
    # folds = int(fold / 4.)
    folds = 5
    skf1 = StratifiedKFold(train1.label.values,
                           n_folds=folds,
                           shuffle=False,
                           random_state=1580)
    skf2 = StratifiedKFold(train2.label.values,
                           n_folds=folds,
                           shuffle=False,
                           random_state=1580)
    #------------------------------------------------------LVL1 Logistic Regression--------------------------------------------------------
    clf = LogisticRegression(penalty='l1', C=0.25,
                             random_state=1580, n_jobs=-1)
    features = list(train.columns)
    for tmp in dropCols:
        features.remove(tmp)

    df_train_pred = []
    df_pred = []
    result = []
    # pdb.set_trace()
    for i, (a, neg_index) in enumerate(skf2):
        print "LVL1----i:", i, "  a:", a, "  neg_index:", neg_index
        fold_tag = "fold_%s" % i
        pos_index = list(skf1)[i][0]
        train_pos = train1.loc[pos_index, :]
        trainner = pd.concat(
            (train_pos, train2.loc[neg_index, :]), axis=0, ignore_index=True)
        y = trainner.label.values
        X = trainner[features]
        clf.fit(X, y)
        train_pred = clf.predict_proba(train[features])[:, 1]
        y_pred = clf.predict_proba(test[features])[:, 1]
        df_train_pred.append(train_pred)
        df_pred.append(y_pred)
        result.append(clf.coef_[0])
    # average results
    train_preds = np.average(np.array(df_train_pred), axis=0)
    test_preds = np.average(np.array(df_pred), axis=0)

    # output feature weights
    coef = np.average(np.array(result), axis=0)
    IMPORTANCE = 3
    result = sorted(
        zip(map(lambda x: round(x, IMPORTANCE), coef), features), reverse=True)
    o = open('Logistic_coef.txt', "w")
    o.write("len of feature: %s\n" % len(result))
    for (value, key) in result:
        if 0 == value:
            pass
        else:
            writeln = "%s: %.3f\n" % (key, value)
            o.write(writeln)
    o.close()

    logger.info('LVL1 Logistic Model trained, Average AUC: <<<<%s, time spend:<<<<%s' % (
        roc_auc_score(train.label.values, train_preds), round(((time.time() - check_time) / 60), 2)))
    check_time = time.time()
    #-----------------------------------------------------OUTPUT----------------------------------------------------------------------------
    pd.DataFrame({"proteinId": test['proteinId'].values, "drugId": test['drugId'].values, "label": test['label'].values,  "predict": test_preds}).to_csv(
        'gbdt_lr_baseline.csv', index=False)
    logger.info("end time: %s\n\n\n" % time.ctime())


if __name__ == "__main__":
    # random_seed = range(1000,2000,10)                          # 1000，1010，1020，1030。。。2000
    # gamma = [i/1000.0 for i in range(100,200,1)]               # 0.010，0.011，0.012。。。0.02
    # max_depth = [3, 4, 5, 6,7,8]                                        # 6，7，8
    # lambd = range(1,200,2)                                   # 100，101，102。。。200
    # subsample = [i/1000.0 for i in range(500,1000,5)]
    # colsample_bytree = [i/1000.0 for i in range(250,750,1)]
    # min_child_weight = [i/1000.0 for i in range(200,300,1)]

    #     random_seed = range(1000,2000,10)
    #     random.shuffle(random_seed)
    #     params = {
    #         'objective':'binary:logistic',
    #         'early_stopping_round':100,
    #         'scale_pos_weight':float(len(y)-sum(y))/float(sum(y)),
    #         'gamma':0.2,
    #         'max_depth':5,
    #         'lambda':1,
    #         'alpha':1,
    #         'subsample':0.8,
    #         'colsample_bytree':0.5,
    #         'min_child_weight':2.5,
    #         'eta':0.01,
    #         'eval_metric': 'auc',
    #         'seed':random_seed[0],
    #         'nthread':8
    #     }

    max_depth = 10                  # range(3,15,1)
    min_child_weight = 4.7         # [i/10.0 for i in range(0,50,1)]
    # range(0,100,1)  #[i/10.0 for i in range(0,100,1)]
    gamma = 6
    # [i/100.0 for i in range(80,100,2)]        # [i/10.0 for i in range(0,11)]
    subsample = 0.88
    # [i/100.0 for i in range(10,30,2)]   #[i/10.0 for i in range(1,11)]
    colsample_bytree = 0.22
    # alpha1 =                      # [1e-5, 1e-2, 0.1, 1, 100]
    # [i/1000.0 for i in range(0,101,2)]         # [0.00001,0.0001,0.001,0.01,0.1,0,1,10]
    eta = 0.01
    # [i/100.0 for i in range(20,40,1) ]  # [i/10000.0 for i in range(0,1000,20)]       # [0.0001,0.001,0.01,0.1,0,1,100,1000]
    lambd = 0.31
    # range(0,100)  #[-10000,-1000,-100,-10,-1,0,1,10,100,1000,10000,100000]
    seed = 27

    # with open('./params/params.csv','w') as f:
    #     writer = csv.writer(f)
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
             colsample_bytree, min_child_weight, eta, seed)
    # lr('./model/{0}/xgb_featrue{1}.txt'.format(filename, 0), train0, test)
