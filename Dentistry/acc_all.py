# Accuracy calculation for all the data.

import numpy as np

def run(data_info):
    # Preprocess data
    # indices = np.random.permutation(len(data_info_original))
    # data_info = data_info_original.iloc[indices]
    # data_info = data_info[['predictions','labels']]

    # Calculate c_index
    acum = 0
    total = 0
    for pred, label in zip(data_info['predictions'], data_info['labels']):
        if pred == label:
            acum += 1
        total += 1

    return acum / total
