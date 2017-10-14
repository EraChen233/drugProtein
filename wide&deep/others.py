from sklearn import *
from sklearn.svm import *
from sklearn.linear_model import *
from sklearn.naive_bayes import *
from sklearn.ensemble import * 
from sklearn.calibration import *
from sklearn.metrics import * 
from sklearn.model_selection import train_test_split
from scipy import interp

import pandas as pd
import numpy as np
 
    
def train():
    df = pd.read_csv("./2fill.csv")  
    # df = pd.read_csv("./fin_data.csv")
    df = df.dropna(axis=0)
    df_train =  df.drop(['label','proteinId','drugId'],axis=1)
    df_target = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        df_train, df_target, test_size=0.2)
    # X_train.insert(0,'label',y_train) 
    # X_test.insert(0,'label',y_test)    
    print("Success")
    # print ("number of positive: %d "  % list(X_train['label']).count(1))
    
    # Create classifiers
    lr = LogisticRegression()
    gnb = GaussianNB()
    svc = LinearSVC(C=1.0)
    rfc = RandomForestClassifier(n_estimators=100)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)        
    for clf, name in [(lr, 'Logistic'), (gnb, 'Naive Bayes'),\
                          (svc, 'Support Vector Classification'), (rfc, 'Random Forest')]:
        clf.fit(X_train, y_train)
        y_pred =clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)
        fpr, tpr, thresholds = roc_curve(y_test, prob_pos)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)    
        pre = precision_score(y_test, y_pred)

        print("name=%s,acc=%f,rec=%f,pre=%f,auc=%f,prob_pos=%s"%(name,acc,rec,pre,roc_auc,prob_pos))



if __name__ == "__main__":
    train()
