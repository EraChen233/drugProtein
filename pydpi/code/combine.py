# -*- coding: utf-8 -*-
import pandas as pd
# dpi1_positive 1233个文件
# dpi1_negative 1501个文件

def comb(arg,len):
    base_url = './dataset/'+arg
    fileList = []
    for i in range(0,len) :
        print i
        file = pd.read_csv(base_url+"/{0}.csv".format(i))
        col = file['protein_drug']
        fileList.append(file)
    result = pd.concat(fileList)
    result.to_csv("./dataset/{0}.csv".format(arg),index=None)

def combTwoFile(arg):
    file1 = pd.read_csv("./dataset/{0}_negative.csv".format(arg))
    file1['label'] = 0
    file2 = pd.read_csv("./dataset/{0}_positive.csv".format(arg))
    file2['label'] = 1
    result = pd.concat([file1,file2])
    result.insert(2,'label',result.pop('label'))
    # result = result.drop(['drug_AllDescriptor'],axis =1)
    result.to_csv("./dataset/{0}.csv".format(arg),index=None)

# comb('dpi1_negative',1501)

combTwoFile('drug_protein')