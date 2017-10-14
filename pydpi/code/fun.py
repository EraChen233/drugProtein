# -*- coding: UTF-8 -*-  
#import pydrug, pypro, pydpi
import pandas as pd
import random
import json
import math
import numpy as np 
import matplotlib.pylab as plt


def splitData():
    originalData = pd.read_csv('./originalData/all.csv');

    human = originalData[originalData['Species']=="Human"];
    protein = human['ID']
    protein = list(set(protein))
    protein = ["proteinId"] + protein
    pd.Series(protein).to_csv('./originalData/protein.csv',index=None)
    
    bigStr = ";".join(human['Drug IDs'].values)
    arr = bigStr.split(";")
    arr = [item.strip() for item in arr]
    arr = list(set(arr))
    arr = ["drugId"] + arr
    pd.Series(arr).to_csv('./originalData/drug.csv',index=None)

def getPositiveSample() : 
    originalData = pd.read_csv('./originalData/all.csv');
    originalData = originalData[originalData['Species']=="Human"]
    arr = []
    for i in range(0,originalData.shape[0]) :
        proteinId = originalData.iat[i,0].strip()
        drugIds = originalData.iat[i,12].split(";")
        drugIds = [item.strip() for item in drugIds]
        for drug in drugIds :
            arr.append(proteinId+"_"+drug)
    arr = ['protein_drug']+arr
    pd.Series(arr).to_csv('./originalData/positive.csv',index=None)


def getNegetiveData(len):
    positiveData = pd.read_csv('./originalData/positive.csv').iloc[:,0];
    proteinCsv = pd.read_csv('./originalData/protein.csv');
    drugCsv = pd.read_csv('./originalData/drug.csv');
    positiveData = list(set(positiveData))
    negetiveData = []
    for i in range(len) :
        temp = ""
        while (temp=="") or (temp in positiveData) or (temp in negetiveData):
            a = random.randint(0,4949)
            b = random.randint(0,2311)
            temp = proteinCsv.iat[b,0]+"_"+drugCsv.iat[a,0]
        negetiveData.append(temp)
    negetiveData = ['protein_drug']+negetiveData  
    pd.Series(negetiveData).to_csv('./originalData/negetive*10.csv',index=None)

def dp1split():
    head = ["Chi3ch","knotp","dchi3","dchi2","dchi1","dchi0","Chi5ch","V","Y","dchi4","Chiv4pc","Chiv3c","Chi3c","APAAC17","W","APAAC20","D","F","H","L","N","P","R","kappa1","T","kappa3","kappa2","Chiv6ch","APAAC15","PAAC40","Chiv4ch","APAAC13","Chiv9","Chiv8","Chiv5","Chiv4","Chiv7","Chiv6","Chiv1","Chiv0","Chiv3","Chiv2","Chi4c","kappam1","kappam3","kappam2","Chi8","Chi9","PAAC28","Chi2","Chi3","Chi0","Chi1","Chi6","Chi7","Chi4","Chi5","PAAC32","PAAC33","PAAC30","PAAC31","PAAC36","PAAC37","PAAC34","PAAC35","PAAC25","PAAC38","PAAC39","PAAC24","A","Chiv4c","phi","E","G","I","K","M","Chi4pc","Q","APAAC14","S","PAAC29","APAAC8","APAAC9","APAAC6","APAAC7","APAAC4","APAAC5","APAAC2","APAAC3","APAAC1","knotpv","Chiv5ch","C","Chiv3ch","Chiv10","PAAC21","PAAC23","PAAC22","APAAC18","APAAC19","PAAC27","PAAC26","Chi10","Chi4ch","APAAC16","mChi1","APAAC10","APAAC11","APAAC12","Chi6ch"]
    table = pd.read_csv('./dataset/dpi1_positive.csv')    
    datas = table['dpi1']
    arr =[]
    for i in range(0,len(datas)):
        if(isinstance(datas[i], str)):
            dpi1Split =datas[i]
            dpi1Split = dpi1Split.split('{')[1]
            dpi1Split = dpi1Split.split('}')[0]
            dpi1Split = dpi1Split.split(',')
            tempArr = []
            for x in dpi1Split:
                tempArr.append(x.split(':')[1])
            arr.append(tempArr)
        else:
            print i
            print datas[i]
            print "\n"
    result = pd.DataFrame(arr,columns=head)
    result = pd.concat([table, result], axis=1)
    result  = result.drop(['dpi1'],axis = 1)
    result.insert(0,'proteinId',result['protein_drug'].apply(spliteProtein))
    result.insert(1,'drugId',result['protein_drug'].apply(spliteDrug))
    result = result.drop(['protein_drug'],axis = 1)
    result.to_csv('./dataset/dpi1_format_positive.csv',index=None)


# type:0 不用处理
# type:1 用“_”链接
# type:2 分割
# type:3 直接链接
def proteinsplit(arg,type,init=''):
    head = []
    # if init == 'init':
    #     table = pd.read_csv('./dataset/protein/protein_0.csv')
    # else:
    #     table = pd.read_csv('./dataset/protein_format.csv')
    if init == 'init':
        table = pd.read_csv('./dataset/protein.csv')
    else:
        table = pd.read_csv('./dataset/protein_format.csv')
    
    datas = table[arg]
    arr =[]
    print arg
    for item in datas[0].split('{')[1].split('}')[0].split(','):
        sub = item.split(':')[0].split("'")[1]
        if type == 0:
            head.append(sub)
        elif type == 1:
            head.append(arg+'_'+sub)
        elif type == 2:
            head.append(sub.replace(arg,arg+"_"))
        elif type == 3:
            head.append(arg+sub)
    print head
    print '\n'
    for i in range(0,len(datas)):
        if(isinstance(datas[i], str)):
            dpi1Split =datas[i]
            dpi1Split = dpi1Split.split('{')[1]
            dpi1Split = dpi1Split.split('}')[0]
            dpi1Split = dpi1Split.split(',')
            tempArr = []
            for x in dpi1Split:
                tempArr.append(x.split(':')[1])
            arr.append(tempArr)
        else:
            print i
            print datas[i]
            print "\n"
    result = pd.DataFrame(arr,columns=head)
    result = pd.concat([table, result], axis=1)
    result  = result.drop([arg],axis = 1)
    result.to_csv('./dataset/protein_format.csv',index=None)
    
def drugsplit(arg,init=''):
    head = []
    if init == 'init':
       table = pd.read_csv('./dataset/drug.csv')
    else:
       table = pd.read_csv('./dataset/drug_format.csv')
    
    datas = table[arg]
    arr =[]
    # print arg
    if init == ';':
        entery = datas[0].split('{')[1].split('}')[0].split(';')
    else:
        entery = datas[0].split('{')[1].split('}')[0].split(',')
    for item in entery:
        # print item
        sub = item.split(':')[0].split("'")[1]
        head.append(arg+'_'+sub)
        
    # print head
    # print len(head)
    # print '\n'
    for i in range(0,len(datas)):
        if(isinstance(datas[i], str)):
            print arg, i ,table.iat[i,0]
            print datas[i]
            print '\n\n'
            dpi1Split =datas[i]
            dpi1Split = dpi1Split.split('{')[1]
            dpi1Split = dpi1Split.split('}')[0]
            if init ==';':
                dpi1Split = dpi1Split.split(';')
            else:
                dpi1Split = dpi1Split.split(',')
            tempArr = []
            for j in range(0,len(dpi1Split)):
                if len(dpi1Split[j])>0 :
                    tempArr.append(dpi1Split[j].split(':')[1])
                else:
                    print arg,"  ",i
                    print dpi1Split
                    print "\n"
            
            arr.append(tempArr)
        else:
            print i,"不是str"
            print datas[i]
            print "\n"
    
    result = pd.DataFrame(arr,columns=head)
    result = pd.concat([table, result], axis=1)
    result  = result.drop([arg],axis = 1)
    result.to_csv('./dataset/drug_format.csv',index=None)

def spliteProtein(arg):
    return arg.split('_')[0]
def spliteDrug(arg):
    return arg.split('_')[1]

def joinProteinOrDrug(arg):
    data = pd.read_csv('./originalData/{0}.csv'.format(arg))
    drug = pd.read_csv('./dataset/drug_format.csv')
    protein = pd.read_csv('./dataset/protein_format.csv')
    drug.columns = drug.columns.map(lambda x: "drug_"+x)
    drug.rename(columns={'drug_drugId': 'drugId'}, inplace=True) 
    protein.columns = protein.columns.map(lambda x: "protein_"+x)
    protein.rename(columns={'protein_proteinId': 'proteinId'}, inplace=True) 
    data['proteinId'] = data['protein_drug'].apply(spliteProtein)
    data['drugId'] = data['protein_drug'].apply(spliteDrug)
    data = data.drop(['protein_drug'],axis = 1)
    data = pd.merge(data,drug,'left',on='drugId',)
    data = pd.merge(data,protein,'left',on='proteinId')
    data.to_csv('./dataset/drug_protein_{0}.csv'.format(arg),index=None)

def splitdpi2():
    data = pd.read_csv('./dataset/dpi2.csv')
    data.insert(0,'proteinId',data['protein_drug'].apply(spliteProtein))
    data.insert(1,'drugId',data['protein_drug'].apply(spliteDrug))
    data = data.drop(['protein_drug'],axis = 1)
    data.to_csv('./dataset/dpi2.csv',index=None)


def drawEmpty(filename):
    origin = pd.read_csv('./dataset/{0}.csv'.format(filename))
    charactersLen = origin.shape[1]-1
    print origin.shape[0],origin.shape[1]
    origin['not_null'] = (origin>0).sum(axis=1)
    table = origin[origin['not_null']<600]
    # table = origin
    table = table.sort(columns='not_null')
    t = table['not_null'].values
    x = range(len(t))
    plt.scatter(x,t,c='k')
    plt.title('feature of less than 600')
    # plt.title('feature of empty values')
    plt.show();
    table = table[table['not_null']>50]
    table[['proteinId','drugId','not_null']].to_csv('./dataset/{0}_缺失值.csv'.format(filename),index=None)
    origin = origin[origin['not_null']>50]
    origin.drop(['not_null'],axis =1).to_csv('./dataset/{0}_1.csv'.format(filename),index=None)    

def continuousOrDiscrete(filename):
    table = pd.read_csv('./dataset/{0}.csv'.format(filename))
    # where_are_nan = math.isnan(table.apply(float))
    # table[where_are_nan] = 0
    continuousOrDiscrete = pd.DataFrame(columns=['featrueName','len','values']) 
    cols = table.columns[3:]
    continuousOrDiscrete['featrueName'] = cols
    length = []
    values = []
    for i in range(0,len(cols)):
        temp = set(table[cols[i]].values)
        length.append(len(temp))
        if len(temp)<200:
            values.append(str(list(temp)))
        else:
            values.append('')
    continuousOrDiscrete['len'] = length
    continuousOrDiscrete['values'] = values
    continuousOrDiscrete.to_csv('./dataset/{0}_离散or连续.csv'.format(filename),index=None)