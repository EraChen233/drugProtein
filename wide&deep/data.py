#-*-coding:utf-8-*-

import pandas as pd

import pdb    

def loadCSV(filename):
    columns=["ID", "Name","Gene_Name","GenBank_Protein_ID","GenBank_Gene_ID","UniProt_ID"
                 ,"Uniprot_Title","PDB_ID","GeneCard_ID","GenAtlas_ID","HGNC_ID","Species","Drug_IDs"]
    df = pd.read_csv(filename,names=columns,skipinitialspace=True,skiprows=1)
    targets = df['ID'].drop_duplicates()
    drugs= list(df['Drug_IDs'].str.split(';'))
    flatten = lambda x:[y for i in x for y in flatten(i)] if type(x) is list else [x]
    drugs_all = flatten(drugs)
    drug_lst = list(set(drugs_all))
    target_drug_lst = []
    target_lst = list(targets)
    
    target_val = []
    drug_val = []
        
    k = 0
    for i in target_lst:
        target_key = target_lst.index(i)        
        for j in drugs[k]:
            drug_key = drug_lst.index(j)
            target_val.append(i)
            drug_val.append(j)
            target_drug_lst.append([target_key,drug_key,1])
        k += 1    
    
    # pdb.set_trace()
    
    data = pd.DataFrame()
    data.insert(0,'ID',target_val)
    data.insert(1,'Drug_ID',drug_val)
    data = pd.merge(data,df,how="inner",on="ID")
    data = data[["ID","Species","Drug_ID"]]
    data = data.drop_duplicates()
    data.to_csv('./target_drug.csv',index=False)
    
    print("Success")

def dropNAN(df,filename):
    df = df.dropna(axis=0)
    df.to_csv(filename,index=False)
    print("Success")
        
def addLabel(df, filename, label):
    df.insert(0, 'label', label)
    df.to_csv(filename, index=False)
    print("Success")
    '''
    df = pd.read_csv('./dpi2_negative.csv')
    addLabel(df, 'negative.csv',0)
    df = pd.read_csv('./dpi2_positive.csv')
    addLabel(df, 'positive.csv',1)
    print("success")
    '''

def mergeData():
    df_neg = pd.read_csv('./negative.csv')
    df_pos = pd.read_csv('./positive.csv')
    data = pd.merge(df_pos, df_neg, how='outer')
    data.to_csv('dpi2_data.csv', index=False)
    print("Success")

def qcut_feature(df):
    continuous_col = list(df.columns[3:])    
    k = len(df.columns)
    for i in continuous_col:        
        df[i+'_qcut']=pd.qcut(df[i].rank(method='first'),10,labels=False)                   
        k += 1       
    df.to_csv('./dpi2_qcut.csv',index=False)
    print("success")
    
def oneHotProfile(infile,label='label',sepin=',',sepout=',',hasheader=True,onehot_cols=[],drop_cols=[],train_sample=1.0,b_scale=True):
    #read data file
    start = datetime.datetime.now()
    outfile=infile+'.onehot'  
    if hasheader:
        df=pd.read_csv(infile,sep=sepin)
    else:
        df=pd.read_csv(infile,header=None,sep=sepin)
    df.drop(drop_cols,axis=1,inplace=True,errors='ignore')
    if b_scale:
        scala_col=list(set(df.columns)-set(onehot_cols)-set([label]))
        #df_scala=normalize(df[scala_col].values, norm = 'l2')#L2正则
        #df_scala = StandardScaler().fit(df[scala_col].values)#Z标准化
        df_scala = scale(df[scala_col].values)#Z标准化
        if set(onehot_cols).issubset(df.columns):    
            df = pd.get_dummies(df, columns = onehot_cols)
        else:
            print("onehot %s not exist,ignore"%(onehot_cols))
        df.drop(onehot_cols,axis=1,inplace=True,errors='ignore')
        cols=list(df)
        cols.insert(0, cols.pop(cols.index(label)))#label移到第1位
        df=df.ix[:,cols]
        df=pd.concat([df.drop(scala_col,axis=1),pd.DataFrame(df_scala,columns=scala_col)],join='inner',axis=1)
    else:
        if set(onehot_cols).issubset(df.columns):    
            df = pd.get_dummies(df, columns = onehot_cols)
        else:
            print("onehot %s not exist,ignore"%(onehot_cols))
        df.drop(onehot_cols,axis=1,inplace=True,errors='ignore')
        cols=list(df)
        cols.insert(0, cols.pop(cols.index(label)))#label移到第1位
        df=df.ix[:,cols]
    if hasheader:
        df.to_csv(outfile,index=False,header=True,sep=sepout)
    else:
        df.to_csv(outfile,index=False,header=False,sep=sepout)
    print("oneHotProfile: outfile=%s,start=%s,end=%s,time=%s"%(outfile,start,datetime.datetime.now(),datetime.datetime.now()-start)) 
    return outfile , len(list(df))-1


def main():    
    df = pd.read_csv('./dpi2.csv')
    label = df.pop('label')
    df.insert(0,'label',label)
    print("success")
    qcut_feature(df)
    


if __name__=="__main__":
    main()
