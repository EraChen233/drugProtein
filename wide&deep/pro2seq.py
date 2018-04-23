
import pandas as pd 

import random
import pdb
import numpy as np

def splitfeature():
    columns = ["ID","Drug_ID","feature1"]
    df = pd.read_csv("./seq2vec/protVec_100d_3grams.csv",names=columns,skipinitialspace=True,skiprows=1)
    target = df['ID']
    target_lst=list(target)
    drug = df['Drug_ID']
    drug_lst = list(drug)
    feature1 = df["feature1"] 
    feature_lst = list(feature1)    # 全部特征数据
    
    keys = eval(feature1[0]).keys()
    res = eval(feature1[0])
    df_feature = pd.DataFrame()

    for k in range(len(feature1)):
        try:            
            res=eval(feature1[k])
            df = pd.DataFrame(res,index=[k])
            df_feature=df_feature.append(df)      
        except:
            df = pd.DataFrame(index=[k])
            df_feature=df_feature.append(df)   

    df_feature.insert(0,'ID',target_lst)
    df_feature.insert(1,'Drug_ID',drug_lst)
    df_feature.to_csv('./neg_data.csv',index=False)
    
    print("Success")
    
    def main():    
    neg_data()

if __name__=="__main__":
    main()
  
    