# Entry file

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

from pca_data import pca_file
from fuzzy_kmeans import K_fuzzy
from metrics import err_all as err_per_cluster
from err_all import run as average_cluster

# Function to manage the order of clusters over several runs
def reset_cluster_labels(err, n_clusters = 4):
    dic = {} # Key: Running label Value: Real label
    sort_err = sorted(err)[::-1]
    for real_idx, a in enumerate(sort_err):
        run_idx = err.index(a)
        dic[run_idx] = real_idx
    if len(dic) < n_clusters or 0 in sort_err:
        return False, False

    return dic, sort_err

# Function to calculate the euclidean distance between two points
def euclidean_distance(a, b):
    return np.linalg.norm(a-b)

# Runs clustering analysis over a dataset
def run(filename, folder="./Data/"):
    times_run = 10
    n_clusters = 4

    # Read the data
    print("\nReading data...")
    # Gets the components to cluster
    data_original = pd.read_csv(folder+ 'pca_' + filename)
    data_info_original = pd.read_csv(folder+filename)

    # Get training info (id, prediction, label, phase(train/test))
    data_summary = pd.concat([data_original, data_info_original[['img_ids', 'predictions','labels', 'phase']]], axis=1)
    amount_features = len(data_original.columns)
    # print("Discovering identification points...")
    # known_points = obtain_known_points(data_original, data_info_original, times_run, n_clusters)

    # Accumulators
    err_per_clusters_test = []
    err_all_test = []
    cluster_size_all = []
    distance_centroid_all = []
    distance_closest_all = []
    distance_points_centroid_all = []

    # Runs clustering algo several times
    while times_run > 0:
        print("\nTraining cluster...")

        # Prepare data to train
        indices = np.random.permutation(len(data_original))
        data = data_original.iloc[indices]
        data_info = data_info_original.iloc[indices]
        data_info = data_info[['img_ids', 'predictions','labels', 'phase']]
        data = scale(data)

        data_train = data[data_info['phase']=='train']
        data_info_train = data_info[data_info['phase']=='train']
        data_test = data[data_info['phase']=='val']
        data_info_test = data_info[data_info['phase']=='val']

        # Fuzzy K Means
        # Training
        clusterer = K_fuzzy(k = n_clusters)
        clusterer.train(data_train)
        pred_train = clusterer.test(data_train)

        # print(acc_per_cluster(data_info_train, pred_train, 4))
        # print(average_cluster(data_info_train))

        # Testing
        pred_test = clusterer.test(data_test)
        errors = err_per_cluster(data_info_test, pred_test, n_clusters)

        # print(average_cluster(data_info_test))

        # Controls the cluster predictions
        label_dic, new_errors = reset_cluster_labels(errors)
        if label_dic:
            times_run -= 1

            # Rearrange cluster predictions
            new_pred_train = [label_dic[pred] for pred in pred_train]
            new_pred_test = [label_dic[pred] for pred in pred_test]

            for idx_prediction, sample in enumerate(data_info_train.iterrows()):
                data_summary.loc[sample[0], 'cluster'+str(times_run)] = new_pred_train[idx_prediction]

            for idx_prediction, sample in enumerate(data_info_test.iterrows()):
                data_summary.loc[sample[0], 'cluster'+str(times_run)] = new_pred_test[idx_prediction]

            print("Cluster trained...")

            # Distances to observe the changes of the clusters

            print("Calculating errors...")
            err_per_clusters_test.append(new_errors)
            err_all_test.append(average_cluster(data_info_test))

            print("Calculating centroid...")
            # Separate the clusters
            clusters = []
            for n in range(n_clusters):
                clusters.append(data_summary[data_summary['cluster'+str(times_run)]==n])

            # Calculate centroids and extract points
            centroids = []
            features = []
            cluster_size = []
            for cluster in clusters:
                feat = cluster[[str(i) for i in range(amount_features)]]
                centroids.append(feat.mean())
                features.append(feat.values)
                cluster_size.append(len(feat))

            print("Calculating amount of people...")
            cluster_size_all.append(cluster_size)

            print("Calculating centroid distance...")
            # Calculate centroid distance
            distance_centroid_all.append([[euclidean_distance(centroid_x, centroid_y) for centroid_y in centroids] for centroid_x in centroids])
            # print(distance_centroid_all)
            # exit()


            print("Calculating point distances...")
            distance_closest = []
            distance_points_centroid = []
            for idf_1, feat_1 in enumerate(features):
                distances = []
                for idf_2, feat_2 in enumerate(features):
                    if idf_1!=idf_2:
                        closest_distance = 1000
                        for row in feat_1:
                            for row2 in feat_2:
                                d = euclidean_distance(row, row2)
                                if d < closest_distance:
                                    closest_distance = d
                        distances.append(closest_distance)
                    else:
                        distances.append(0)
                distance_closest.append(distances)

                # Intra-distances
                centroid = centroids[idf_1]
                centroid_vs_points = np.array([euclidean_distance(row, centroid) for row in feat_1])
                #min,max,mean,std
                distance_points_centroid.append([centroid_vs_points.min(), centroid_vs_points.max(),
                    centroid_vs_points.mean(), np.median(centroid_vs_points), centroid_vs_points.std()])

            distance_closest_all.append(distance_closest)
            distance_points_centroid_all.append(distance_points_centroid)

    # Average the distances over all the runs
    err_all_test = np.array(err_all_test).mean()
    err_per_clusters_test = np.array(err_per_clusters_test).mean(axis=0)
    cluster_size_all = np.array(cluster_size_all).mean(axis=0)
    distance_centroid_all = np.array(distance_centroid_all).mean(axis=0)
    distance_closest_all = np.array(distance_closest_all).mean(axis=0)
    distance_points_centroid_all = np.array(distance_points_centroid_all).mean(axis=0)

    print("\nErrors for test dataset")
    print(err_all_test)
    print("\nErrors per cluster:")
    print(err_per_clusters_test)
    print("\nPeople per cluster:")
    print(cluster_size_all)
    print("\nCentroid Distance:")
    print(distance_centroid_all)
    print("\nClosest Point Distance:")
    print(distance_closest_all)
    print("\nCentroid-Points Distance:")
    print("Row=Clusters, Columns=Min,Max,Mean,Median,Std")
    print(distance_points_centroid_all)

    return (err_all_test, err_per_clusters_test, cluster_size_all, distance_centroid_all, distance_closest_all, distance_points_centroid_all)


# Runs the Clustering Algo over different sample sizes and using K-fold
def main():
    k_fold = 4
    plt.figure()
    plt.title('Errors Reducing Whole Dataset')
    all_errors = []
    sample_sizes = [300,700,2837,5674]

    # For every sample size do:
    for sample, sample_size in zip([300, 700, "half", "all"],sample_sizes):
        print("\n\nSample Size {}".format(sample))

        # Accumulators
        err_all_test_files = []
        err_per_clusters_test_files = []
        cluster_size_all_files = []
        distance_centroid_all_files = []
        distance_closest_all_files = []
        distance_points_centroid_all_files = []

        # For every fold do pca and run the training
        for i in range(k_fold):
            print("\nFile {}".format(i+1))
            filename = 'boneage_features_red_whole_'+str(sample)+'_fold_'+str(i)+'.csv'
            pca_file(filename)
            file_results = run(filename)
            err_all_test_files.append(file_results[0])
            err_per_clusters_test_files.append(file_results[1])
            cluster_size_all_files.append(file_results[2])
            distance_centroid_all_files.append(file_results[3])
            distance_closest_all_files.append(file_results[4])
            distance_points_centroid_all_files.append(file_results[5])

        # Average the distances over all the folds
        err_all_test_files = np.array(err_all_test_files).mean()
        err_per_clusters_test_files = np.array(err_per_clusters_test_files).mean(axis=0)
        cluster_size_all_files = np.array(cluster_size_all_files).mean(axis=0)
        distance_centroid_all_files = np.array(distance_centroid_all_files).mean(axis=0)
        distance_closest_all_files = np.array(distance_closest_all_files).mean(axis=0)
        distance_points_centroid_all_files = np.array(distance_points_centroid_all_files).mean(axis=0)

        print("\n\nFinal Results")
        print("\nErrors for test dataset")
        print(err_all_test_files)
        print("\nErrors per cluster:")
        print(err_per_clusters_test_files)
        print("\nPeople per cluster:")
        print(cluster_size_all_files)
        print("\nCentroid Distance:")
        print(distance_centroid_all_files)
        print("\nClosest Point Distance:")
        print(distance_closest_all_files)
        print("\nCentroid-Points Distance:")
        print("Row=Clusters, Columns=Min,Max,Mean,Median,Std")
        print(distance_points_centroid_all_files)

        all_errors.append(err_all_test_files)
        plt.scatter([sample_size]*len(err_per_clusters_test_files), err_per_clusters_test_files)
    plt.plot(sample_sizes,all_errors)
    plt.show()

if __name__ == "__main__":
    main()
