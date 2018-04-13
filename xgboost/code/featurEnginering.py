# coding:utf-8

import pandas as pd
import numpy as np
import json
import cols
import matplotlib.pylab as plt


def s(arg):
    print str(arg)
    return str(arg).split(", 'Hy'")[0]


def featurEnginering(file, arg1="", bucket=True, filePath=''):
    if filePath:
        table = pd.read_csv('../data/{0}.csv'.format(file))
    else:
        table = pd.read_csv('../data/{0}/{0}.csv'.format(file))
    colTypes = cols.get()[file.split('*')[0]]

    # 删除只有一个取值的列 dpi1
    if arg1 == "onlyOneValue":
        table = table.drop(colTypes['unique'], axis=1)

    # 把drug_protein数据的列drug_MolProperty_TPSA，去除每个字符串后面的", 'Hy'"
    if file.find('drug_protein') >= 0:
        table['drug_MolProperty_TPSA'] = table['drug_MolProperty_TPSA'].apply(
            s)
    # 数据填充，把None替换为-1
    # table = table.fillna(-1)
    nanReplace(table)

    if bucket:
        for h in colTypes['discrete']:
            table[[h]] = table[[h]].astype(float)

        round = 10
        unit = table.shape[0] / round + 1

        for h in colTypes['continues']:
            try:
                print h
                # 自己折腾的分桶
                table[h] = table[h].rank(method='first')
                for i in range(0, round):
                    table[h][(table[h] > (unit * i)) &
                             (table[h] <= (unit * (i + 1)))] = i + 1
                table[[h]] = table[[h]].fillna(-1).astype(int)

                # qcut
                # table[h] = pd.qcut(table[h].rank(method='first'),10)

                # table = sorted(table[h].values)
                # x = range(len(table))
                # plt.scatter(x,table,c='k')
                # plt.title('distribution of '+ h)
                # plt.show();

                print table[h].unique()

                # ont hot
                # pd.get_dummies(table[h], prefix=h)
                # dummy_df =pd.get_dummies(table[h]).rename(columns=lambda x: h+"_" + str(x) )
                # table =pd.concat([table,dummy_df],axis=1).drop([h],axis=1)
            except Exception, e:
                print '出错啦:', e

    # 再次检查，把protein_id、drug_id、label列之外的列，不是数字类型的转换为float类型
    try:
        colTypes = table.dtypes[3:]
        for col in colTypes.index:
            if colTypes[col] != 'float64' and colTypes[col] != 'int64':
                print colTypes[col], "下标", col
                table[[col]] = table[[col]].astype(float)
    except Exception, e:
        print '出错啦:', e

    print table.columns
    print "\n\n", table.shape[0], table.shape[1]
    table.to_csv('../data/{0}.csv'.format(file), index=None)


def transferIdToNumber(file):
    table = pd.read_csv('../data/{0}.csv'.format(file))
    table['proteinId'] = table['proteinId'].rank(method='dense')
    table['drugId'] = table['drugId'].rank(method='dense')
    table.to_csv('../data/{0}*encode_id.csv'.format(file), index=None)
    print file, 'id has tranfered to number'
    featurEnginering('%s*encode_id' % file, bucket=False,
                     filePath='drug_protein')


def nanReplace(df):
    df.replace(np.inf, np.nan, inplace=True)  # 先把inf替换了
    empty = df.isnull().any()
    empty = empty[empty == True]
    mean = df.mean()
    for col in empty.index:
        m = mean[col] if col in mean else 0
        print 'nanReplace',m,'           ', col
        df[col].replace(np.nan, m, inplace=True)


# featurEnginering("dpi2","2")
# featurEnginering("dpi1", "onlyOneValue")
transferIdToNumber('drug_protein*15_800_feature_score_links')
# featurEnginering('drug_protein*2*encode_id', bucket=False,
#                  filePath='drug_protein')
