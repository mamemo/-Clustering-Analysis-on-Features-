# File to perform PCA on the data

from sklearn.preprocessing import scale
from sklearn import decomposition
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pca_file(filename, folder='./Data/'):
    print("\n\nPCAing file...")
    # Read the data
    data_original = pd.read_csv(folder+filename)

    # Clean input data
    indexes = np.random.permutation(len(data_original))
    data = data_original.iloc[indexes]
    df_pca  = pd.DataFrame() 

    ignore_cols = ['Unnamed: 0','img_ids','predictions', 'labels', 'phase']
    data = data.drop(ignore_cols, axis=1)
    data = scale(data)

    # Performs PCA
    pca = decomposition.PCA(n_components=10)
    pca.fit(data)
    data = pca.transform(data)

    # Save pca features to cluster
    r = pd.DataFrame(data)
    r = r.reindex(indexes)
    r = pd.concat([df_pca, r], axis=1)
    r.to_csv(folder + 'pca_'+filename, index=False)

    # PCA Analysis

    # print("Weights on each Component")
    # weights_comp = pca.explained_variance_ratio_
    # plt.plot(weights_comp)
    # plt.show()
    # print(list(weights_comp), weights_comp.sum())

    # print("Loadings")
    # loadings = pca.components_
    # print(loadings)
