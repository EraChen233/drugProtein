# -*-coding:utf-8-*-
#import pydpi
#from pydpi import pydpi
import pandas as pd 

import random
import pdb
import numpy as np

from datetime import datetime

def  pro_vec():
    df = pd.read_csv('./seq.csv')
    gram = pd.read_csv('./3gram_T.csv')
    seq_vec = pd.DataFrame()
    seq= df['sequence']        
    print(seq.count())
    for k in range(2463,5000):
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
        
        seq_vec[k]/=cnt   
        seq_vec.T.to_csv('./meanvec.csv')     
            
    seq_vec.T.to_csv('./meanvec.csv')       
    print("Success")
            
def norm():
    df = pd.read_csv("./del_links.csv")
    CONTINUOUS_COLUMNS = list(df.columns[5:])
    for scale_col in CONTINUOUS_COLUMNS:
        df[scale_col]=(df[scale_col]-df[scale_col].mean())/(df[scale_col].max()-df[scale_col].min())
    df.to_csv("./pro_norm.csv",index=False)

def del_duplicate():
    df = pd.read_csv("./data_links.csv")
    for i in list(df.columns):
        if(len(df[i].value_counts()) == 1):
            df=df.drop(i,axis=1)
    df.to_csv("./del_links.csv",index=False)

def main():    
    norm()

if __name__=="__main__":
    main()
  
    
