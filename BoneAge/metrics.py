# Files with several clustering metrics definitions.

from sklearn import metrics
from time import time
import numpy as np
import warnings
from math import sqrt

# The HIGHER the better defined clusters. (From -1 to 1).
# Scores around 0 indicate overlapping clusters.
def Silhouette_Coefficient(data, labels_predicted):
    if len(np.unique(labels_predicted)) > 1:
        return metrics.silhouette_score(data, labels_predicted, metric='euclidean')
    else:
        return 0

# The HIGHER the better defined clusters.
def Calinski_Harabaz_Index(data, labels_predicted):
    if len(np.unique(labels_predicted)) > 1:
        return metrics.calinski_harabaz_score(data, labels_predicted)
    else:
        return 0

# Zero is the lowest possible score.
# Values closer to zero indicate a better partition.
def Davies_Bouldin_Index(data, labels_predicted):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if len(np.unique(labels_predicted)) > 1:
            return metrics.davies_bouldin_score(data, labels_predicted)
        else:
            return 0

def err_min(data_info, labels_predicted, top):
    return min(err_all(data_info, labels_predicted, top))

def rmse(pred, target):
    mse = metrics.mean_squared_error(pred, target)
    return sqrt(mse)

def err_all(data_info, labels_predicted, top):
    err = [0] * top
    for idx, label in enumerate(np.unique(labels_predicted)):
        preds = data_info[labels_predicted == label]['predictions'].values
        labels = data_info[labels_predicted == label]['labels'].values
        err[idx] = rmse(preds, labels)
    return err

def count(data_info, labels_predicted, top):
    acc = [0] * top
    for idx, label in enumerate(np.unique(labels_predicted)):
        acc[idx] = data_info[labels_predicted == label]['predictions'].count()
    return acc
