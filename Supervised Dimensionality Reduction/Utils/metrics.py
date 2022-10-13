import numpy as np
from scipy import stats
from easydict import EasyDict as edict
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred) / (y_true + 1e-5)) * 100

def return_regression_result(y_true, y_pred):
    performance = edict()
    performance.corr = stats.pearsonr(y_true, y_pred)[0]
    performance.mape = (1- mean_absolute_percentage_error(y_true=y_true,y_pred=y_pred))
    performance.rmse = (1 - mean_squared_error(y_true=y_true,y_pred=y_pred,squared=False))
    performance.mae = (1- mean_absolute_error(y_true=y_true, y_pred=y_pred))
    return performance

def return_classification_result(y_true, y_pred, y_proba):
    performance = edict()
    performance.accuracy = accuracy_score(y_true, y_pred)
    performance.f1_score = f1_score(y_true, y_pred, average='macro')
    performance.roc_auc_score = roc_auc_score(y_true, y_proba, multi_class='ovr')
    return performance