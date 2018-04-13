# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.utils import shuffle 

def nanReplace(df):
    df.replace(np.inf, np.nan, inplace=True)  # 先把inf替换了
    empty = df.isnull().any()
    empty = empty[empty == True]
    mean = df.mean()
    for col in empty.index:
        df[col].replace(np.nan, mean[col], inplace=True)

# 选择K个特征
K_features = 500

def SelectKBestFun():
    data = pd.read_csv('./data/drug_protein*encode_id.csv')

    #然后，SelectKBest类，通过回归的方法，以及要选择多少个特征值，
    #新建一个 SelectKBest对象，
    # selectKBest = SelectKBest(
    #     f_regression,k=data.shape[1]
    # )
    selectKBest = SelectKBest(
        f_regression,k=K_features
    )
    nanReplace(data)
    #接着，把自变量选择出来，然后调用fit_transform方法，
    #把自变量和因变量传入，即可选出相关度最高的两个变量。
    features = data.columns-['label']
    bestFeature =selectKBest.fit_transform(
        data[features],
        data['label']
    )

    #我们想要知道这两个自变量的名字，使用get_support方法即可得到相应的列名
    features = data.columns[selectKBest.get_support()]
    result = data[features]
    result['label'] = data['label']
    result['drugId'] = data['drugId']
    result['proteinId'] = data['proteinId']
    result.to_csv('./data/drug_protein*encode_id*{0}Features.csv'.format(K_features), index=None)

def SelectFeatureScore(k,filename):
    data = pd.read_csv('./data/%s.csv' % filename)
    fs = pd.read_csv('./featurescore/drug_protein_original_withoutId_10000.csv')
    # data[list(fs['feature'][:k])+['label','drugId','proteinId']].to_csv('./data/{0}*{1}_feature_score.csv'.format(filename,k), index=None)
    data[['drugId','proteinId']+list(fs['feature'][:k])].to_csv('./data/{0}*{1}_feature_score.csv'.format(filename,k), index=None)

def findLinks():
    data1 = pd.read_csv('./data/drug&protein/positive.csv')
    proteins = [item.split('_')[0] for item in data1['protein_drug']]
    drugs = [item.split('_')[1] for item in data1['protein_drug']]
    drugArr=[]
    proteinArr = []
    for drug in set(drugs):
        drugArr.append([drug,drugs.count(drug)])
    for protein in set(proteins):
        proteinArr.append([protein,proteins.count(protein)])
    pd.DataFrame(drugArr,columns=['drug','num']).to_csv('./data/drug&protein/drug_links.csv'.format(K_features), index=None)
    pd.DataFrame(proteinArr,columns=['protein','num']).to_csv('./data/drug&protein/protein_links.csv'.format(K_features), index=None)

def addLinksToTable(filename):
    data = pd.read_csv('./data/%s.csv' % filename)
    drug_links = pd.read_csv('./data/drug&protein/drug_links.csv')
    protein_links = pd.read_csv('./data/drug&protein/protein_links.csv')
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
    data.to_csv('./data/%s_links.csv' % filename, index=None)

def spliteProtein(arg):
    return arg.split('_')[0]
def spliteDrug(arg):
    return arg.split('_')[1]

def joinProteinOrDrug(arg):
    data = pd.read_csv('./data/drug&protein/{0}.csv'.format(arg))
    drug = pd.read_csv('./data/drug&protein/drug_format.csv')
    protein = pd.read_csv('./data/drug&protein/protein_format.csv')
    drug.columns = drug.columns.map(lambda x: "drug_"+x)
    drug.rename(columns={'drug_drugId': 'drugId'}, inplace=True) 
    protein.columns = protein.columns.map(lambda x: "protein_"+x)
    protein.rename(columns={'protein_proteinId': 'proteinId'}, inplace=True) 
    data['proteinId'] = data['protein_drug'].apply(spliteProtein)
    data['drugId'] = data['protein_drug'].apply(spliteDrug)
    data = data.drop(['protein_drug'],axis = 1)
    data = pd.merge(data,drug,'left',on='drugId',)
    data = pd.merge(data,protein,'left',on='proteinId')
    data.to_csv('./data/drug_protein_{0}.csv'.format(arg),index=None)

# def SelectOnlyLinks(filename):
#     data = pd.read_csv('./data/%s.csv' % filename)
#     features = list(data.columns[:20])+['drugId','proteinId','drugLinks','proteinLinks','label']
#     data[features].to_csv('./data/drug_protein*2_onlylinks.csv', index=None)
# SelectOnlyLinks('drug_protein*2*800_feature_score_links')


# 构造正负样本，正样本为1，负样本为n
def connectTwoTable(n):
    data1 = pd.read_csv('./data/drug_protein_positive*800_feature_score_links.csv')
    data1['label' ]=1
    data2 = pd.read_csv('./data/drug_protein_negetive*%d*800_feature_score_links.csv' % n)
    data2['label']=0
    result = pd.concat([data1,data2])
    shuffle(result).to_csv('./data/drug_protein*1%d_800_feature_score_links.csv' % n,index=None)


def getNegetiveData(len):
    positiveData = pd.read_csv('./data/drug&protein/positive.csv').iloc[:,0];
    proteinCsv = pd.read_csv('./data/drug&protein/protein.csv');
    drugCsv = pd.read_csv('./data/drug&protein/drug.csv');
    positiveData = list(set(positiveData))
    negetiveData = []
    for i in range(len*12314) :
        temp = ""
        while (temp=="") or (temp in positiveData) or (temp in negetiveData):
            a = random.randint(0,4949)
            b = random.randint(0,2311)
            temp = proteinCsv.iat[b,0]+"_"+drugCsv.iat[a,0]
        negetiveData.append(temp)
    negetiveData = ['protein_drug']+negetiveData  
    pd.Series(negetiveData).to_csv('./data/drug&protein/negetive*%s.csv'% len,index=None)


# getNegetiveData(5)
# joinProteinOrDrug('negetive*5')

# SelectFeatureScore(800,"drug_protein_negetive*5")
# addLinksToTable('drug_protein_negetive*5*800_feature_score')

connectTwoTable(5)


