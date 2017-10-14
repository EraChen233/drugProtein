#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
''' example by red
    python dpi_predict.py --algo=xgboost --iter=10000
'''
import subprocess, sys, os, time
import argparse
import pandas as pd
import platform
import logging
import datetime
#from converters.common import *
from mytool import  *
import re
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation,metrics
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split

import pdb
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import operator
import copy
sys.path.append("/data1/red/deepdetect/clients/python")
from dd_client import  DD
hive_cli='/data1/red/hcat_udb_v13/bin/hive'
random_seed = 1225
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                #datefmt='%a, %d %b %Y %H:%M:%S',
                datefmt='%Y-%m-%d %H:%M:%S',
                filename='dpi_predict.log',
                filemode='w')
#定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
console = logging.StreamHandler()
console.setLevel(logging.INFO)
#formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] : %(levelname)-4s %(message)s')
console.setFormatter(formatter)
logging.getLogger(__name__).addHandler(console)
logger = logging.getLogger(__name__)



def saveImage(df,key,val,outfile,kind='barh',title='tile',xlabel='xlabel'):
    plt.figure()  
    df.plot(kind=kind, x=key, y=val, legend=False, figsize=(24, 20))  
    plt.title(title)  
    plt.xlabel(xlabel)  
    plt.savefig(outfile)
    return 

def pct_rank_qcut(series, n):
    edges = pd.Series([float(i) / n for i in range(n + 1)])
    f = lambda x: (edges >= x).argmax()
    return series.rank(pct=1).apply(f)           

class gbdt_ffm_model:
    def __init__(self,dt,thread):
        self.file=''
        self.file_test=''
        self.NR_THREAD=thread
        self.tree_num=3
        self.head= "Id,Label,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,\
        I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,I14,I15,I16,I17\
        ,I18,I19,I20,I21,I22,I23"
        self.head=re.sub(' ','',self.head)
        self.cnum=self.head.count('C')
        self.inum=self.head.count('I')-1
        self.model=None
        self.sql_head="\
    money_sum ,   \
    uid ,               \
    anchorid ,          \
    sid ,               \
    consume_last_date , \
    free_last_date ,    \
    share_last_date ,   \
    level ,                \
    state_level ,       \
    roler_num ,            \
    message_last_date , \
    is_firstpay ,          \
    is_fans,              \
    view_sum ,             \
    view_count ,           \
    live_duration ,        \
    live_count ,           \
    effective_time ,    \
    effective_count ,   \
    view_days ,         \
    view_date ,         \
    is_view_days ,         \
    consume_last_day ,  \
    giftpay_days ,         \
    flower_sum ,           \
    flower_count ,         \
    free_last_day ,     \
    freelog_days ,         \
    share_count ,          \
    share_last_day ,    \
    is_share_days ,        \
    leve_days ,         \
    message_count ,        \
    message_last_day ,  \
    is_message_days  \
"
       #dictionary {feature:type}
        if 'xgboost' in FLAGS.algo:
            features = pd.read_csv("dti_type.csv")
            self.fea_col=list(features['feature'].values)
            #pdb.set_trace()
            self.fea_col_num=list(features.loc[features['type'].str.strip()=='numeric','feature'].values)
                            
            self.sql_head=','.join(self.fea_col)
            self.head = 'Id,Label,'+self.sql_head
            
            self.features=features
            
        else:
            self.sql_head=re.sub(' ','',self.sql_head)
            
        if FLAGS.debug:
            self.sql="select %s from %s where dt='%s' and money_sum>0 limit %d  union all \
            select %s from %s where dt='%s' and money_sum<=0 limit %d "%(self.sql_head,'datamine.datamine_uid_anchor_feature_day_30',dt,FLAGS.limit,self.sql_head,'datamine.datamine_uid_anchor_feature_day_30',dt,FLAGS.limit)
        else:
            self.sql="select %s from %s where dt='%s' and money_sum>0 union all select %s from %s where dt='%s' and money_sum<=0 limit %d "\
            %(self.sql_head,'datamine.datamine_uid_anchor_feature_day_30',dt,self.sql_head,'datamine.datamine_uid_anchor_feature_day_30',dt,FLAGS.neg_sample)
       
 
        return
    def loadData(self,file='uid_aid_feature.csv'):
        if platform.system() == 'Windows':
            self.file='D:/YY/data/uid_aid_feature10.csv'
            self.file_test='D:/YY/data/uid_aid_feature10.test.csv'
        else:
            self.file='/data1/red/learn/dpi/fin_data.csv'
            self.file_test='/data1/red/learn/dpi/fin_data.csv.test'
            logger.info("parse file %s"%(self.file))
            #self.parseFile(self.file)
                
        self.file_test_out=self.file_test +'.out'
        self.file_out=self.file

        
        return
    def parseFile(self,file):
        self.file_out=file+'.out'
        with open(file,'r') as input , open(self.file_out,'w') as output, open(self.file_out_test,'w') as output_test:
            sizehint = 104857600   # 100M
            pos=0
            lines=input.readlines(sizehint)
            lines=lines[1:]#过滤head行  
            head=self.head+'\n'
            output.write(head)
            output_test.write(head)
            i=1
            #while not input.tell()-pos <= 0:
            while lines:
                out=''
                out_test=''
                for line in lines:
                    line=','.join(line.split())
                    
                    val=float(line.split(',')[2])#label标签
                    if 'class'  in FLAGS.aim :
                        if val>0:
                            label=1
                        else:
                            label=0
                    elif 'rank'  in FLAGS.aim :
                        if val>0:
                            label=1
                        else:
                            label=0                  
                    else:
                        label=val#目标值直接回归

                    out+='%d,%d,%s\n'%(i,label,line)             
                               
                    i+=1
                output.write(out)
                lines=input.readlines(sizehint)
     
    def pandasRead(self,bigfile):
        
        reader=pd.read_csv(bigfile,iterator=True)
        chunkSize=1000000
        chunks=[]
        loop=True
        while loop:
            try:
                chunk=reader.get_chunk(chunkSize)
                chunks.append(chunk)
            except StopIteration:
                loop=False
                print("Iteration stoped!")
        df = pd.concat(chunks,ignore_index=True)
        #train,val = train_test_split(df, test_size = 0.2,random_state=1)#random_state is of big influence for val-auc
        train,val = train_test_split(df, test_size = 0.2,random_state=1)#random_state is of big influence for val-auc
        y = train.label
        X=train[self.fea_col]
        #pdb.set_trace()
        #X = train.drop(['Label','Id','money_sum'],axis=1)
        X = X.drop(['ID','Drug_ID','label'],axis=1,errors='ignore')
        val_y = val.label
        
        val_X=val[self.fea_col]
        val_X = val_X.drop(['ID','Drug_ID','label'],axis=1,errors='ignore')
        
        feature_info = {}
        features = list(train.columns)
        #pdb.set_trace()
        features.remove('ID')
        features.remove('Drug_ID')
        
        for feature in features:
            max_ = train[feature].max()
            min_ = train[feature].min()
            n_null = len(train[train[feature]<0])  #number of null
            n_gt1w = len(train[train[feature]>10000])  #greater than 10000
            feature_info[feature] = [min_,max_,n_null,n_gt1w]
            
        return X,y,val_X,val_y,feature_info

    def pandasRead1(self,bigfile):
        df=self.prePareXgboost(bigfile)        
        train,val = train_test_split(df, test_size = 0.25,random_state=1)#random_state is of big influence for val-auc
        y = train.label
        #X=train[self.fea_col]
        X=train
        print("feature len=%d,feature=%s"%(len(X.keys()),X.keys()))
        X = X.drop(['ID','Drug_ID','label'],axis=1,errors='ignore')
        val_y = val.label
        
        #val_X=val[self.fea_col]
        val_X=val
        val_X = val_X.drop(['ID','Drug_ID','label'],axis=1,errors='ignore')
        
        feature_info = {}
        features = list(train.columns)
        features.remove('ID')
        features.remove('Drug_ID')
        
        for feature in features:
            max_ = train[feature].max()
            min_ = train[feature].min()
            n_null = len(train[train[feature]<0])  #number of null
            n_gt1w = len(train[train[feature]>10000])  #greater than 10000
            feature_info[feature] = [min_,max_,n_null,n_gt1w]
            
        return X,y,val_X,val_y,feature_info

    def pandasReadTest(self,bigfile):
        reader=pd.read_csv(bigfile,iterator=True)
        chunkSize=1000000
        chunks=[]
        loop=True
        while loop:
            try:
                chunk=reader.get_chunk(chunkSize)
                chunks.append(chunk)
            except StopIteration:
                loop=False
                print("Iteration stoped!")
        df = pd.concat(chunks,ignore_index=True)
        X=df.drop(['Label'],axis=1)
        return X

    def feaBucket(self,data_df,bucket_num=10):
        bucket={}
        for key in data_df.keys():
            
            min_val=data_df[key].min()
            max_val=data_df[key].max()
            bucket[key]=list(np.linspace(min_val,max_val, bucket_num+1))
            
        return bucket           
    
    def getSeq(self,val,sort_lst):
        pos=0
        length=len(sort_lst)
        while pos+1<length:
            if val>sort_lst[pos+1]:
                pos+=1
            else:
                break
        return pos   
     
    def prePareXgboost(self,infile):
        data=pd.read_csv(infile)
        if FLAGS.bucket:
            #bucket=self.feaBucket(data[self.fea_col_num],FLAGS.bucket)
            #pdb.set_trace()
#             for key in self.fea_col_num:
#                 data[key+'_b']=data[key].map(lambda x:self.getSeq(x,bucket[key]))
            for key in self.fea_col_num:
                data[key+'_cut']=pd.cut(data[key],FLAGS.bucket,labels=False)
        if FLAGS.bucket_qcut:
            for key in self.fea_col_num:
                data[key+'_qcut']=pd.qcut(data[key].rank(method='first'),FLAGS.bucket,labels=False)
                
        return data
    def setRatio(self,label):
        ratio=1.0       
        if label is not None:
            ratio=float(np.sum(label == 0)) / (0.00001+np.sum(label==1))
        print("ratio=%.f"%(ratio))
        return ratio
    
    def setObj(self):
        obj='reg:linear'
        if "class" in FLAGS.aim:
            obj='binary:logistic'
        elif "regress" in FLAGS.aim:
            #obj='reg:logistic'
            obj='reg:linear'
        elif "rank" in FLAGS.aim:
            obj='rank:pairwise'
        return obj
    def setParams(self,label):
        params={
            'booster':'gbtree',
            'objective': self.setObj(),
            'early_stopping_rounds':100,
            'scale_pos_weight': self.setRatio(label),#1500.0/13458.0,
            #'eval_metric': ['auc','rmse','error','logloss'],
            'gamma':0.2,#0.1
            'max_depth':5,#8
            'lambda':1,
            'alpha':1,
            'subsample':0.8,#0.5,#0.7,
            'colsample_bytree':0.5,
            'min_child_weight':2.5, 
            'eta': 0.01,
            'seed':random_seed,
            'nthread':FLAGS.thread
            }
        if 'class' in FLAGS.aim:
            params['eval_metric'] = ['auc','rmse','error','logloss','map']
        elif 'rank' in FLAGS.aim:
            params['eval_metric'] = ['rmse','auc','map']
        elif 'regress' in FLAGS.aim:
            params['eval_metric'] = ['rmse']
        else:
            params['eval_metric'] = ['rmse']
        return params
    
    def trainXgboost(self):  
        #X,y,val_X,val_y,feature_info=self.pandasRead1(self.file_out)
        X,y,val_X,val_y,feature_info=self.pandasRead1(self.file_out)
        
        dval = xgb.DMatrix(val_X,label=val_y)
        dtrain = xgb.DMatrix(X, label=y)
        params=self.setParams(y)
        print("xgboost param=%s"%params)
        watchlist  = [(dtrain,'train'),(dval,'val')]#The early stopping is based on last set in the evallist
        model = xgb.train(params,dtrain,num_boost_round=FLAGS.iter,evals=watchlist)
        model.save_model('./model/xgb.model')
        self.model=model
        print ("best best_ntree_limit",model.best_ntree_limit)        

        #save feature score and feature information:  feature,score,min,max,n_null,n_gt1w
        feature_score = model.get_fscore()
        feature_score_new=copy.deepcopy(feature_score)
        for key in feature_score:
            #feature_score[key] = [feature_score[key]]+feature_info[key]+[features_type[key]]
            feature_score_new[key] = [feature_score[key]]+feature_info[key]
        
        feature_score_new = sorted(feature_score_new.items(), key=lambda x:x[1],reverse=True)
        print("feature_score size=%d, first=%s"%(len(feature_score_new),feature_score_new[0]))
        #pdb.set_trace()
        fs = []
        for (key,value) in feature_score_new:
            #fs.append("{0},{1},{2},{3},{4},{5},{6}\n".format(key,value[0],value[1],value[2],value[3],value[4],value[5]))
            fs.append("{0},{1},{2},{3},{4},{5}\n".format(key,value[0],value[1],value[2],value[3],value[4]))
        
        with open('feature_score.csv','w') as f:
            f.writelines("feature,score,min,max,n_null,n_gt1w\n")
            f.writelines(fs)
        
        feature_score = sorted(feature_score.items(), key=operator.itemgetter(1))          
        feature_score = pd.DataFrame(feature_score, columns=['feature', 'fscore'])
        #pdb.set_trace()  
        feature_score['fscore_new'] = feature_score['fscore'] / feature_score['fscore'].sum()  

        saveImage(feature_score,'feature','fscore_new','feature_score.png',title='XGBoost Feature Importance',xlabel='relative importance')

        return              
    
    def predictXgboost(self):
        test_x=self.pandasReadTest(self.file_test)
        dtest = xgb.DMatrix(test_x)
        test_y = self.model.predict(dtest,ntree_limit=self.model.best_ntree_limit)
        test_result = pd.DataFrame(columns=["uid","score"])
        #test_result.uid = test_uid
        test_result.score = test_y
        test_result.to_csv("xgb.csv",index=None,encoding='utf-8')  #remember to edit xgb.csv , add ""
        
        return
           
    def prePare(self):        
        cmd = 'utils/count.py %s > fc.trva.t10.txt'%(self.file)
        logger.info('cmd1=%s'%(cmd))
        subprocess.call(cmd, shell=True)

        gbdt_dense=self.file+'.gbdt_dense'
        gbdt_sparse=self.file+'.gbdt_sparse'
        cmd = 'converters/parallelizer-a.py -s {nr_thread} converters/pre-a.py {infile} {gbdt_dense} {gbdt_sparse}'\
        .format(nr_thread=self.NR_THREAD,infile=self.file,gbdt_dense=gbdt_dense,gbdt_sparse=gbdt_sparse)
        logger.info('cmd2=%s'%(cmd))
        subprocess.call(cmd, shell=True)

        gbdt_dense_test=self.file_test+'.gbdt_dense'
        gbdt_sparse_test=self.file_test+'.gbdt_sparse'
        cmd = 'converters/parallelizer-a.py -s {nr_thread} converters/pre-a.py {infile} {gbdt_dense} {gbdt_sparse}'\
        .format(nr_thread=self.NR_THREAD,infile=self.file_test,gbdt_dense=gbdt_dense_test,gbdt_sparse=gbdt_sparse_test)
        logger.info('cmd3=%s'%(cmd))
        subprocess.call(cmd, shell=True)
        
        gbdt_test_out=self.file_test +'.gbdt_out'
        gbdt_out=self.file +'.gbdt_out'
        cmd = './gbdt -t {tree_num} -s {nr_thread} {gbdt_dense_test} {gbdt_sparse_test} {gbdt_dense} {gbdt_sparse} {gbdt_test_out} {gbdt_out} '\
        .format(tree_num=self.tree_num,nr_thread=self.NR_THREAD,gbdt_dense_test=gbdt_dense_test,gbdt_sparse_test=gbdt_sparse_test,gbdt_dense=gbdt_dense,gbdt_sparse=gbdt_sparse,gbdt_test_out=gbdt_test_out,gbdt_out=gbdt_out)
        logger.info('cmd4=%s'%(cmd))
        subprocess.call(cmd, shell=True)
        
#         cmd = 'rm -f te.gbdt.dense te.gbdt.sparse tr.gbdt.dense tr.gbdt.sparse'
#         subprocess.call(cmd, shell=True)
        self.ffm_out=self.file+'.ffm_out'
        self.ffm_out_test=self.file_test+'.ffm_out'
        cmd = 'converters/parallelizer-b.py -s {nr_thread} converters/pre-b.py {input} {gbdt_out} {ffm_out} '\
        .format(nr_thread=self.NR_THREAD,input=self.file,gbdt_out=gbdt_out,ffm_out=self.ffm_out)
        logger.info('cmd5=%s'%(cmd))
        subprocess.call(cmd, shell=True)

        cmd = 'converters/parallelizer-b.py -s {nr_thread} converters/pre-b.py {input} {gbdt_out} {ffm_out} '\
        .format(nr_thread=self.NR_THREAD,input=self.file_test,gbdt_out=gbdt_test_out,ffm_out=self.ffm_out_test)
        logger.info('cmd6=%s'%(cmd))
        subprocess.call(cmd, shell=True)
        
#         cmd = 'rm -f te.gbdt.out tr.gbdt.out'
#         subprocess.call(cmd, shell=True)
       
        return
            
    def train(self):
        self.model='model'
        cmd = './ffm-train -k 4 -t 18 -s {nr_thread} -p {ffm_out_test} {ffm_out} {model}'\
        .format(nr_thread=self.NR_THREAD,ffm_out_test=self.ffm_out_test,ffm_out=self.ffm_out,model=self.model)
        logger.info("train: %s"%(cmd))
        subprocess.call(cmd, shell=True)
        return
    
    def eval(self):
        
        cmd = './ffm-predict {ffm_test} {model} {test_out} '\
        .format(nr_thread=self.NR_THREAD,ffm_test=self.ffm_out_test,model=self.model,test_out=self.file_test_out)
        logger.info("predict: %s"%(cmd))
        subprocess.call(cmd, shell=True)
        
        test_out_cal=self.file_test_out+'.cal'
        cmd = './utils/calibrate.py {test_out} {test_out_cal} '\
        .format(nr_thread=self.NR_THREAD,test_out=self.file_test_out,test_out_cal=test_out_cal)
        logger.info("calibrate: %s"%(cmd))
        subprocess.call(cmd, shell=True)
        
        submit=self.file_test+'.sub.csv'
        cmd = './utils/make_submission.py {test_out_cal} {submit} '\
        .format(nr_thread=self.NR_THREAD,test_out_cal=test_out_cal,submit=submit)
        logger.info("make_submission: %s"%(cmd))
        subprocess.call(cmd, shell=True)
        return
    def predict(self):
        return
        
    
def main(FLAGS):

    dt=FLAGS.dt
    if dt=='yesterday':
        fmtdt=datetostr(getYesterday(),'%Y-%m-%d')
        dt=fmtdt[0:4]+fmtdt[5:7]+fmtdt[8:10]
    else: 
        fmtdt=dt[0:4]+'-'+dt[4:6]+'-'+dt[6:8]
    start= datetime.datetime.now()
    logger.info("rs_predict begin: dt=%s,algo=%s,aim=%s,debug=%d,iter=%d,thread=%d,neg_sample=%d"%(dt,FLAGS.algo,FLAGS.aim,FLAGS.debug,FLAGS.iter,FLAGS.thread,FLAGS.neg_sample))
    rs_model = gbdt_ffm_model(dt,FLAGS.thread)
    rs_model.loadData()
    if 'xgboost' in FLAGS.algo:
        logger.info("trainXgboost")
        rs_model.trainXgboost()
#         logger.info("predictXgboost")
#         rs_model.predictXgboost()
    else:        
        rs_model.prePare()
        logger.info("train")
        rs_model.train()
        logger.info("eval")
        rs_model.eval()
    
    end= datetime.datetime.now()
    logger.info("Done!,start=%s,end=%s,time=%s"%(start,end,end-start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--spark_mode",
        type=str,
        default="stand",
        help="Base spark_mode yarn, stand ."
    )
    parser.add_argument(
        "--dt",
        type=str,
        default='yesterday',
        help=""
    )
    parser.add_argument(
        "--watch_train",
        type=int,
        default=-30,
        help=""
    )
    parser.add_argument(
        "--watch_test",
        type=int,
        default=-1,
        help=""
    )
    parser.add_argument(
        "--train_sample",
        type=float,
        default=0.01,
        help=""
    )
    parser.add_argument(
        "--test_sample",
        type=float,
        default=0.0001,
        help=""
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=1,
        help=""
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help=""
    )
    parser.add_argument(
        "--like_dr",
        type=int,
        default=60,
        help=""
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=100,
        help="top n"
    )
    parser.add_argument(
        "--als_sample",
        type=float,
        default=0.05,
        help=""
    )
    parser.add_argument(
        "--algo",
        type=str,
        default='gbdt_ffm',
        help="gbdt_ffm,xgboost"
    )
    parser.add_argument(
        "--aim",
        type=str,
        default='class',
        help="class,regress,rank"
    )
    
    parser.add_argument(
        "--rank",
        type=int,
        default=30,
        help=""
    )
    parser.add_argument(
        "--regu",
        type=float,
        default=0.1,
        help=""
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="als"
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=100,
        help=""
    )
    parser.add_argument(
        "--clear",
        type=int,
        default=0,
        help=""
    )
    parser.add_argument(
        "--train",
        type=str,
        default='base',
        help=""
    )
    parser.add_argument(
        "--del_bigaid",
        type=int,
        default=0,
        help=""
    )
    parser.add_argument(
        "--is_eval",
        type=int,
        default=1,
        help=""
    )
    parser.add_argument(
        "--like_pay",
        type=int,
        default=0,
        help=""
    )
    parser.add_argument(
        "--thread",
        type=int,
        default=20,
        help=""
    )
    parser.add_argument(
        "--neg_sample",
        type=int,
        default=8000000,
        help=""
    )
    parser.add_argument(
        "--fea_file",
        type=str,
        default="feature_type.csv",
        help=""
    )
    parser.add_argument(
        "--bucket_qcut",
        type=int,
        default=10,
        help="bucket_qcut"
    )    
    parser.add_argument(
        "--bucket",
        type=int,
        default=10,
        help="bucket num"
    )    
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
