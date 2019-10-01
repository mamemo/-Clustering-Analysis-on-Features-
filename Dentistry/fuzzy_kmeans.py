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

    # def run_metrics(self, data):
    #     res = []
    #     for cluster in self.clusters_trained:
    #         predictions = self.test(data, cluster)
    #         res.append([Silhouette_Coefficient(data, predictions),
    #                 Calinski_Harabaz_Index(data, predictions),
    #                 Davies_Bouldin_Index(data, predictions)])
    #     return np.array(res)

    # def run_metrics_acc(self, data, data_info):
    #     return np.array([acc_all(data_info, self.test(data, cluster), self.end)
    #                 for cluster in self.clusters_trained])

    # def run_metrics_count(self, data, data_info):
    #     return np.array([count(data_info, self.test(data, cluster), self.end)
    #                 for cluster in self.clusters_trained])
