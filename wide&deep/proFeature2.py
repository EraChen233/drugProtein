# -*-coding:utf-8-*-
#import pydpi
#from pydpi import pydpi
import pandas as pd 

import random
import pdb
import numpy as np

from datetime import datetime

'''
def loadfeature_2(df): 
    label_lst=list(df['label'])
    target = df['proteinId']
    target_lst=list(target)
    target = df['proteinId']
    drug_lst=list(df['drugId'])        
    DPIfeature1 = df['sequence']
    for k in range(340,33455):     
        if pd.isnull(DPIfeature1[k]) :
            try:            
                dpi=pydpi.PyDPI()
                print(k)
                print(target_lst[k])
                DPIfeature1[k]=dpi.GetProteinSequenceFromID(target_lst[k]) # 从uniport下载一个蛋白质序列
            except:
                DPIfeature1[k]="" # 计算蛋白质-配体相互作用特征 
                #print("特征为空")
        data = pd.DataFrame()
        data.insert(0,'label',label_lst)
        data.insert(1,'drugId',drug_lst)
        data.insert(2,'proteinId',target_lst)
        data.insert(3,'sequence',DPIfeature1)
        data = data[['label','proteinId','drugId','sequence']]
        data.to_csv('./seq.csv',index=False)    
        print("Success")
'''  

def  pro_vec():
    df = pd.read_csv('./seq.csv')
    gram = pd.read_csv('./3gram_T.csv')
    seq_vec = pd.DataFrame()
    seq= df['sequence']        
    print(seq.count())
    for k in range(0,5000):
        #pdb.set_trace()
        print(k)
        seq_vec[k] = gram[seq[k][0:3]]  
        cnt = 0   
        for i in range(0,3):
            for j in range(1,len(seq[k])-3):
                word_1 = seq[k]
                word=word_1[i+j:i+j+3]
                if(len(word)==3):
                    cnt+=1
                    try:
                        word_vec = gram[word]
                        seq_vec[k] = word_vec+seq_vec[k]
                    except:
                        seq_vec[k] = '' 
                        #k+=1
        seq_vec.T.to_csv('./meanvec.csv')               
        seq_vec[k]/=cnt       
    seq_vec.T.to_csv('./meanvec.csv')       
    print("Success")
            
def norm():
    df = pd.read_csv("./pro_vec.csv")
    CONTINUOUS_COLUMNS = list(df.columns[0:])
    for scale_col in CONTINUOUS_COLUMNS:
        df[scale_col]=(df[scale_col]-df[scale_col].mean())/(df[scale_col].max()-df[scale_col].min())
    df.to_csv("./pro_norm.csv",index=False)

def main():    
    norm()

if __name__=="__main__":
    main()
  
    
