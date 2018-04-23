# -*-coding:utf-8-*-

import pandas as pd 

import random
import numpy as np

def addLinksToTable():
    data = pd.read_csv('./drug_link.csv' )
    drug_links = pd.read_csv('./drug_links.csv')
    protein_links = pd.read_csv('./protein_links.csv')
    drugs={}
    for index, row in drug_links.iterrows():   # 获取每行的index、row
        drugs[row['drug']] = row['num']
    proteins={}
    for index, row in protein_links.iterrows():   # 获取每行的index、row
        proteins[row['protein']] = row['num']
    data['drugLinks']=0
    data['proteinLinks']=0
    for i in range(data.shape[0]):
        data.loc[i,'drugLinks'] = drugs[data.loc[i,'drugId']]
        data.loc[i,'proteinLinks'] = proteins[data.loc[i,'proteinId']]
    data.to_csv('./data_links.csv' , index=None)
    
def main():    
    addLinksToTable()

if __name__=="__main__":
    main()
  
    
