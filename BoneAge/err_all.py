# Error calculation for all the data.

import numpy as np
from math import sqrt
from sklearn import metrics

def rmse(pred, target):
    mse = metrics.mean_squared_error(pred, target)
    return sqrt(mse)

def run(data_info):
    preds = data_info['predictions'].values
    labels = data_info['labels'].values
    
    return rmse(preds, labels)