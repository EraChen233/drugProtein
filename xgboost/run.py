# coding:utf-8
from code import gbdt
from code import gbdt_lr
from code import competing

train_method = "GBDT"           # ["GBDT","GBDT_LR"]
# logFile = './before_mangshen.log'
logFile = './before_mangshen1.log'
# filename = 'drug_protein_original'
filename = 'drug_protein*2*800_feature_score_links*encode_id'
# logFile = './K_Featrues.log'
# filename = 'drug_protein*encode_id*500Features'
boost_round = 10000
keepId = True

# gbdt.start(filename, boost_round, keepId, logFile)
# 一般都用下面这个进行GBDT，通过 train_method 控制模型的办法
# gbdt_lr.Train(train_method, filename, boost_round, keepId, logFile)

# 与各个方法对比
logFile = 'competing.log'
filename = 'drug_protein'
# filename = 'drug_protein*2*800_feature_score_links*encode_id'
method_name='svm'         # rf、LR、knn、svm、NB、NN
competing.start(filename, logFile,method_name)

