# Implements a class for Fuzzy K-Means using the scikit-learn library

import skfuzzy as fuzz
import numpy as np

class K_fuzzy():

    def __init__(self, k):
        self.k = k
        self.cluster_trained = None

    def train(self, data):
        data = np.transpose(data)
        cntr, _, _, _, _, _, _ = fuzz.cluster.cmeans(
            data, self.k, m=2, error=0.005, maxiter=1000, init=None)
        self.cluster_trained = cntr

    def test(self, data):
        data = np.transpose(data)
        u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
                data, self.cluster_trained, m=2, error=0.005, maxiter=1000, init=None)
        cluster_membership = np.argmax(u, axis=0)
        return cluster_membership
