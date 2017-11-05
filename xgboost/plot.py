import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import pandas as pd


def plot(y_test, y_score, fileName):
    fpr = dict()
    tpr = dict()
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    # Compute micro-average ROC curve and ROC area

    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, color='darkorange',
             label='ROC  %s (area = %0.3f)  %s' % (fileName, roc_auc, roc_auc_score(y_test.values, y_score.values)))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    # fileName = "gbdt_lr_baseline"
    fileName = "gbdt_proto_feature_lr"
    # fileName = 'predict/drug_protein_original/test0'
    table = pd.read_csv('%s.csv' % fileName)
    plot(table['label'], table['predict'], fileName)
