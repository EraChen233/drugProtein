# -*-coding:utf-8-*-

import pydpi
from pydpi import pydrug,pypro,pydpi
import pandas as pd 

import random
import pdb
import numpy as np

from datetime import datetime

def dropNonhuman():
    columns = ["ID","Species","Drug_ID"]
    df = pd.read_csv("./target_drug.csv",names=columns,skipinitialspace=True,skiprows=1)
    pdb.set_trace()
    data = df[df['Species']=="Human"]
    data.to_csv('./testdata.csv',index=False)
    print("Success")

# 下载feature1
def loadfeature(df,filename): 
    target = df['ID']
    target_lst=list(target)
    drug = df['Drug_ID']
    drug_lst = list(drug)
        
    AAComp = []
    DPIfeature1 = []
    DPIfeature2 = []
    # pdb.set_trace()
    k = 0
    a = datetime.now()
    for i in target_lst:        
        try:
            b = datetime.now()
            if (b-a).hours < 1.5:
                dpi=pydpi.PyDPI()
                ps=dpi.GetProteinSequenceFromID(i) # 从uniport下载一个蛋白质序列
                print i
                dpi.ReadProteinSequence(ps) # 读取蛋白质序列
                pdict=dpi.GetAAComp() #计算amino acid composition描述符
                smi=dpi.GetMolFromDrugbank(drug_lst[k])  # 从DrugBank下载一个药物
                print drug_lst[k]
                dpi.ReadMolFromSmile(smi) # 读取分子
                ddict=dpi.GetConnectivity() # connectivity药物描述符
                ddict.update(dpi.GetKappa()) # kappa药物描述符
                pdict.update(dpi.GetAPAAC()) # 蛋白描述符
                DPIfeature1.append(dpi.GetDPIFeature1(ddict,pdict)) # 计算蛋白质-配体相互作用特征 
                # combining drug features and protein features.(nd+np)
                # DPIfeature2.append(dpi.GetDPIFeature2(ddict,pdict))
                # by  the tensor product.  (nd*np)
            else :
                break
        except:
            DPIfeature1.append('') # 计算蛋白质-配体相互作用特征 
            # combining drug features and protein features.(nd+np)
            # DPIfeature2.append('')
        k += 1    
    
    data = pd.DataFrame()
    data.insert(0,'ID',target_lst)
    data.insert(1,'Drug_ID',drug_lst)
    data.insert(2,'feature1',DPIfeature1)
    data = data[["ID","Drug_ID","feature1"]]
    data.to_csv(filename,index=False)
    
    print("Success")
    
def splitfeature():
    columns = ["ID","Drug_ID","feature1"]
    df = pd.read_csv("./neg_data1.csv",names=columns,skipinitialspace=True,skiprows=1)
    pdb.set_trace()
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
    
def neg_data():
    df = pd.read_csv("./data/Human_only.csv")
    drug = df['Drug_ID'].drop_duplicates() # 11352
    target = df['ID'].drop_duplicates() # 11386
    drug_lst=list(drug)
    target_lst=list(target)
    
    drug_neg=[]
    target_neg=[]
    for i in range(0,114000):
        drug_neg.append(random.choice(drug_lst))
        target_neg.append(random.choice(target_lst))
        
    data = pd.DataFrame()
    data.insert(0,'ID',target_neg)
    data.insert(1,'Drug_ID',drug_neg)
    
    print("data success")
    data.to_csv('./neg_data.csv',index=False)    
    loadfeature(data,'./neg_data.csv')
    print("feature success")
 
def loadfeature_1(df,filename): 
    target = df['ID']
    target_lst=list(target)
    drug = df['Drug_ID']
    drug_lst = list(drug)        
    DPIfeature1 = df['feature1']    
    k = 0
    a = datetime.now()
    for i in target_lst:        
        b = datetime.now()
        print((b-a).seconds)
        if (b-a).seconds < 120:           
            if pd.isnull(DPIfeature1[k]) :
                try:            
                    dpi=pydpi.PyDPI()
                    ps=dpi.GetProteinSequenceFromID(i) # 从uniport下载一个蛋白质序列
                    print i
                    dpi.ReadProteinSequence(ps) # 读取蛋白质序列
                    pdict=dpi.GetAAComp() #计算amino acid composition描述符
                    smi=dpi.GetMolFromDrugbank(drug_lst[k])  # 从DrugBank下载一个药物
                    print drug_lst[k]
                    dpi.ReadMolFromSmile(smi) # 读取分子
                    ddict=dpi.GetConnectivity() # connectivity药物描述符
                    ddict.update(dpi.GetKappa()) # kappa药物描述符
                    pdict.update(dpi.GetAPAAC()) # 蛋白描述符
                    DPIfeature1[k]=dpi.GetDPIFeature1(ddict,pdict) # 计算蛋白质-配体相互作用特征 
                except:
                    DPIfeature1[k]="" # 计算蛋白质-配体相互作用特征 
        k += 1    
    print("输出")
    data = pd.DataFrame()
    data.insert(0,'ID',target_lst)
    data.insert(1,'Drug_ID',drug_lst)
    data.insert(2,'feature1',DPIfeature1)
    data = data[["ID","Drug_ID","feature1"]]
    print("数据")
    data.to_csv(filename,index=False)
    
    print("Success")
    
def loadfeature_2(df): 
    target = df['ID']
    target_lst=list(target)
    drug = df['Drug_ID']
    drug_lst = list(drug)
        
    DPIfeature1 = df['feature1']
    for k in range(18,100):     
        if pd.isnull(DPIfeature1[k]) :
            try:            
                dpi=pydpi.PyDPI()
                ps=dpi.GetProteinSequenceFromID(target_lst[k]) # 从uniport下载一个蛋白质序列
                print target_lst[k]
                dpi.ReadProteinSequence(ps) # 读取蛋白质序列
                pdict=dpi.GetAAComp() #计算amino acid composition描述符
                smi=dpi.GetMolFromDrugbank(drug_lst[k])  # 从DrugBank下载一个药物
                print drug_lst[k] 
                dpi.ReadMolFromSmile(smi) # 读取分子
                ddict=dpi.GetConnectivity() # connectivity药物描述符
                ddict.update(dpi.GetKappa()) # kappa药物描述符
                pdict.update(dpi.GetAPAAC()) # 蛋白描述符
                DPIfeature1[k]=dpi.GetDPIFeature1(ddict,pdict) # 计算蛋白质-配体相互作用特征 
            except:
                DPIfeature1[k]="" # 计算蛋白质-配体相互作用特征 
                print("特征为空")
    data = pd.DataFrame()

    data.insert(0,'ID',target._lst)
    data.insert(1,'Drug_ID',drug_lst)
    data.insert(2,'feature1',DPIfeature1)
    data = data[["ID","Drug_ID","feature1"]]  
    print(data) 
    data.to_csv('./neg_feature.csv',index=False)
    
    print("Success")

def dropNAN(df,filename):
    df = df.dropna(axis=0)
    df.to_csv(filename,index=False)
    print("Success")
        
def addLabel(df, filename, label):
    df.insert(2, 'label', label)
    df.to_csv(filename, index=False)
    print("Success")

def mergeData():
    df_neg = pd.read_csv('./neg.csv')
    df_pos = pd.read_csv('./pos.csv')
    data = pd.merge(df_pos, df_neg, how='outer')
    data.to_csv('alldata.csv', index=False)
    print("Success")


def dropWrongNeg():
    df_neg = pd.read_csv('./neg.csv')
    df_pos = pd.read_csv('./pos.csv')
    df_outer = pd.merge(df_pos, df_neg, how='outer')
    df_inner = pd.merge(df_pos, df_neg, how='inner',on=['ID','Drug_ID'])
    for i in range(len(df_inner['ID'])):
        for j in range(len(df_outer['ID'])):
            if df_inner['ID'][i] == df_outer['ID'][j] and \
            df_inner['Drug_ID'][i] == df_outer['Drug_ID'][j] and \
            df_outer['label'][j] == 0:
                df_outer.drop(j)
                print j
    df_outer.to_csv('./fin_data.csv',index=False)
    print("Success")

    
    
def main():    
    neg_data()

if __name__=="__main__":
    main()
  
    
