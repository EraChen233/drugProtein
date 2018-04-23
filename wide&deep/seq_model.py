# coding=utf-8
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import tempfile
import pdb
import collections
import log

    
def build_estimator(df_b, model_dir, model_type):
    continuous_col = list(df_b.columns[3:])
    print(continuous_col)
    categorical_col =  ["proteinId","drugId"]
    print(categorical_col)
    con_features = []
    sps_features = []
    emb_features = []
    bkt_features = []
    bkt_features_1 = []
    cross_features = []
    cross_features_1 = []
    fre_bkt_features = []


    for i in continuous_col:
        con_features.append(tf.contrib.layers.real_valued_column(i)) 
    print("continuous feature")  

    for i in categorical_col:
        sps_features.append(tf.contrib.layers.sparse_column_with_hash_bucket(
            i, hash_bucket_size=10000))
    print("categorical feature")
  
    for i in sps_features:
        emb_features.append(tf.contrib.layers.embedding_column(i, dimension=8))   
    print("embedding feature")
    ''' 
    for i in con_features:
        b = list(np.linspace(min_val, max_val, 10))
        bkt_features.append(tf.contrib.layers.bucketized_column(i, boundaries=b))   
    print("") 
    
    k = 0
    for i in range(len(bkt_features)):
        if i == k: continue
        cross_features.append(tf.contrib.layers.crossed_column(
                          [bkt_features[i],bkt_features[k]], hash_bucket_size=10000))
        k += 1
    print("绛夎窛妗跺寲鐨勪氦鍙夌壒寰�")

    for i in qcut_col:        
        fre_bkt_features.append(tf.contrib.layers.sparse_column_with_integerized_feature(i, bucket_size=100))
    '''    
    '''
    length=len(bkt_features)
    i=0;j=1
    while i<length:
        while j+i<length:
            var=tf.contrib.layers.crossed_column([fre_bkt_features[i], fre_bkt_features[j]],hash_bucket_size=int(1000))
            cross_features_1.append(var) 
            j+=1
        i+=1    
    '''    
    wide_columns = con_features  + sps_features

    deep_columns = con_features + emb_features

    if model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
    elif model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[50, 50])
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[200,200,200],
        #dnn_dropout=0.4,
        #linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.01),
           # dnn_optimizer=tf.train.AdagradOptimizer(learning_rate=0.01),
            #dnn_activation_fn=tf.nn.sigmoid,
        fix_global_step_increment_bug=True)
    return m


def input_fn(df):
    LABEL_COLUMN = 'label'
    # CATEGORICAL_COLUMNS = ["protein_drug"]
    CATEGORICAL_COLUMNS = ["proteinId","drugId"]
    CONTINUOUS_COLUMNS = list(df.columns[3:])
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i,0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size,1]
            )
        for k in CATEGORICAL_COLUMNS
    }
    
    continuous_cols = {}
    for k in CONTINUOUS_COLUMNS:
        continuous_cols[k] = tf.constant(df[k].values)
    '''continuous_cols = {k: tf.constant(df[k].values)
                        for k in CONTINUOUS_COLUMNS}'''

    
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)    
    label = tf.constant(df[LABEL_COLUMN].values)
    return feature_cols,label

def train_and_eval():
    
    '''
    df_train = pd.read_csv('./train_data.csv')
    df_test = pd.read_csv('./test_data.csv')    
    '''
    df = pd.read_csv("./pro_norm.csv")  
    # df = pd.read_csv("./fin_data.csv")
    # df = pd.read_csv("./2qcut_data.csv")  
    df_train =  df[df.columns[1:]]
    df_target = df['label']
    train_X, test_X, train_y, test_y = train_test_split(
        df_train, df_target, test_size=0.2)
    train_X.insert(0,'label',train_y) 
    test_X.insert(0,'label',test_y)    
    print("Success")
    print ("number of positive: %d "  % list(train_X['label']).count(1))
    
    model_dir = tempfile.mkdtemp()
    print("model directory = %s"%model_dir)
    m = build_estimator(train_X, model_dir,"wide_deep")
    m.fit(input_fn=lambda: input_fn(train_X),steps=10000)
    results = m.evaluate(input_fn=lambda: input_fn(test_X),steps=1)
    predition = m.predict(input_fn = lambda: input_fn(test_X),as_iterable=True)
    #predition.to_csv('./result.csv')
    logger = log.get_logger(__name__, 'train.log')    
    for key in sorted(results):
        print(" %s: %s" % (key,results[key]))
        logger.info(" %s: %s" % (key,results[key]))

def main(_):
    logger = log.get_logger(__name__, 'train.log')
    logger.info('wide and deep,归一化，10000次')
    train_and_eval()
    


if __name__ == "__main__":
  tf.app.run()
         
    