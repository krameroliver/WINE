import numpy as np
import pandas as pd
from sklearn.datasets import load_wine


from sklearn.metrics import accuracy_score, auc, average_precision_score,balanced_accuracy_score, roc_auc_score,roc_curve
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_log_error, mean_squared_error, median_absolute_error
from sklearn.metrics import label_ranking_average_precision_score,label_ranking_loss,coverage_error
from sklearn.preprocessing import LabelEncoder as LE


def _multi_class(y_true=0,y_pred=0):
    a = label_ranking_average_precision_score(y_true,y_pred)
    b = label_ranking_loss(y_true,y_pred)
    c = coverage_error(y_true,y_pred)
    erg = np.zeros((1, 3))
    erg[0, 0] = a
    erg[0, 1] = b
    erg[0, 2] = c

    res_mclass = pd.DataFrame(data=erg,
                             columns=['label_ranking_average_precision_score', 'label_ranking_loss', 'coverage_error'])

    return res_mclass


def _reg_q(y_true=0,y_pred=0):

    a = max_error(y_pred=y_pred,y_true=y_true)
    b = mean_absolute_error(y_pred=y_pred,y_true=y_true)
    c = mean_squared_log_error(y_pred=y_pred,y_true=y_true)
    d = mean_squared_error(y_pred=y_pred,y_true=y_true)
    e = median_absolute_error(y_pred=y_pred,y_true=y_true)

    erg = np.hstack(( b, c, d, e))
    res_reg = pd.DataFrame(data=erg,columns=['mean_absolute_error', 'mean_squared_log_error', 'mean_squared_error', 'median_absolute_error'])
    return res_reg

def _class_q(y_true=0,y_pred=0):
    labels_en = LE()
    labels_en.fit(y_true)
    y_pred = labels_en.fit(y_pred)
    try:
        a = accuracy_score(y_pred=y_pred, y_true=y_true,pos_label=labels_en.classes_)
    except:
        a = np.nan
    try:

        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        b = auc(fpr, tpr)
    except:
        b = np.nan
    try:
        c = average_precision_score(y_true,y_pred, )
    except:
        c = np.nan
    try:
        d = balanced_accuracy_score(y_pred=y_pred, y_true=y_true)
    except:
        d = np.nan

    #e = roc_auc_score(y_true,y_pred,'micro' )

    erg = np.zeros((1,4))
    erg[0, 0] = a
    erg[0, 1] = b
    erg[0, 2] = c
    erg[0, 3] = d

    res_class = pd.DataFrame(data=erg,
                             columns=['accuracy_score', 'auc', 'average_precision_score', 'balanced_accuracy_score'])

    return res_class

def quality(data_quality_type='classification',y_true=0,y_pred=0):


    if(data_quality_type=='classification'):
        print(_class_q(y_true,y_pred))
    elif(data_quality_type=='regression'):
        print(_reg_q(y_true,y_pred))
    elif(data_quality_type=='multi_label'):
        print(_multi_class(y_true,y_pred))
    elif(data_quality_type=='both'):
        print('classification:')
        print(_class_q(y_true, y_pred))
        print("\n-----------------------------------------------------------------------------------------------------------\n")
        print('regression:')
        print(_reg_q(y_true,y_pred))
