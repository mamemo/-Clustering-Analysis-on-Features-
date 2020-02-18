# Files with several clustering metrics definitions.

from sklearn import metrics
from time import time
import numpy as np
import warnings

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

def acc_max(data_info, labels_predicted, top):
    return max(acc_all(data_info, labels_predicted, top))

def acc_all(data_info, labels_predicted, top):
    acc = [0] * top
    for idx, label in enumerate(np.unique(labels_predicted)):
        acum = 0
        total = 0
        for pred, label in zip(data_info[labels_predicted == label]['predictions'], data_info[labels_predicted == label]['labels']):
            if pred == label:
                acum += 1
            total += 1
        acc[idx] = acum / total
    return acc

def count(data_info, labels_predicted, top):
    acc = [0] * top
    for idx, label in enumerate(np.unique(labels_predicted)):
        acc[idx] = data_info[labels_predicted == label]['predictions'].count()
    return acc
