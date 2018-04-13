# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import pandas as pd

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def plot(tables):
    plt.figure()
    for ob in tables:
        print ob
        table = pd.read_csv(
            'predict/compete_%s_drug_protein.csv' % ob['file'])
        fpr = dict()
        tpr = dict()
        fpr, tpr, thresholds = roc_curve(table['label'], table['score'])
        # Compute micro-average ROC curve and ROC area

        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=ob['color'], linestyle=ob['line'],
                 label='%s  ROC = %0.3f ' % (ob['method'], roc_auc))
    plt.plot([0, 1], [0, 1], color='#000000', lw=2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC 曲线')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    # fileName = "gbdt_lr_baseline"
    # fileName = ["best", 'knn', 'lr', 'nb', 'nn', 'rf', 'svm']
    # 英文版
    # tables = [
    #     {
    #         'file': 'best',
    #         'method': 'Our approach',
    #         'color': 'darkorange',
    #         'line': '-'
    #     },
    #     {
    #         'file': 'rf',
    #         'method': 'Ramdom Forest',
    #         'color': '#531460',
    #         'line': '--'
    #     },
    #     {
    #         'file': 'knn',
    #         'method': 'Nearest Neighbors',
    #         'color': '#d40045',
    #         'line': '-'
    #     },
    #     {
    #         'file': 'lr',
    #         'method': 'Logistic Regression',
    #         'color': '#807dba',
    #         'line': ':'
    #     },
    #     {
    #         'file': 'svm',
    #         'method': 'SVM',
    #         'color': '#007a87',
    #         'line': '-.'
    #     },
    #     {
    #         'file': 'nn',
    #         'method': 'Neural Net',
    #         'color': '#99cf15',
    #         'line': '--'
    #     },
    #     {
    #         'file': 'nb',
    #         'method': 'Naive Bayes',
    #         'color': '#ff9914',
    #         'line': ':'
    #     }
    # ]

    tables = [
        {
            'file': 'best',
            'method': '本文的模型',
            'color': 'darkorange',
            'line': '-'
        },
        {
            'file': 'rf',
            'method': '随机森林',
            'color': '#531460',
            'line': '--'
        },
        {
            'file': 'knn',
            'method': '邻近算法',
            'color': '#d40045',
            'line': '-'
        },
        {
            'file': 'lr',
            'method': '逻辑回归',
            'color': '#807dba',
            'line': ':'
        },
        {
            'file': 'svm',
            'method': '支持向量机',
            'color': '#007a87',
            'line': '-.'
        },
        {
            'file': 'nn',
            'method': '神经网络',
            'color': '#99cf15',
            'line': '--'
        },
        {
            'file': 'nb',
            'method': '朴素贝叶斯',
            'color': '#ff9914',
            'line': ':'
        }
    ]
    # fileName = 'predict/drug_protein_original/test0'
    plot(tables)
