import pickle
from sklearn_extra.cluster import KMedoids
import scipy.spatial as sp
from scipy.cluster.hierarchy import fcluster, linkage
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib
from sklearn.metrics import silhouette_score
matplotlib.use('GTK3Agg')

# ToDo
# avr dissimilarity
# median dissimilarity, HOW?
# dunn index
# Silhouette Coefficient

# Do for geo_dist and lang_dist

# def _cluster_dist_mat()


def _get_dist_mat(gran, metric, distance_type):
    gran_path = "data/" + gran

    if distance_type == 'lang':
        dist_mat_file = open(gran_path + "/dist_mats/" +
                             metric + "_dist_mat.pickle", "rb")
        dist_mat = pickle.load(dist_mat_file)
    elif distance_type == 'geo':
        dist_mat_file = open(gran_path + "/dist_mats/geo_mat.pickle", "rb")
        dist_mat = pickle.load(dist_mat_file)

    return dist_mat


def _cluster_lables(dist_mat, n_clusters, algo, method):
    if algo == 'hrchy':
        linkage_result = linkage(
            sp.distance.squareform(dist_mat), method=method)
        clustering = fcluster(
            linkage_result, t=n_clusters, criterion='maxclust')
    elif algo == 'kmed':
        clustering = KMedoids(n_clusters=n_clusters,
                              metric='precomputed').fit_predict(dist_mat)

    return clustering


def average_median_dissimilarity(gran, eval_metric, nMin, nMax, algo, distance_type, method='ward', metric='norm'):
    dist_mat = _get_dist_mat(gran, metric, distance_type)
    avr_of_results = []
    for n in range(nMin, nMax):
        labels = list(_cluster_lables(dist_mat, n, algo, method))

        cluster_results = []
        print(n)
        for i in set(labels):
            points_indices = [p for p, x in enumerate(labels) if x == i]
            if len(points_indices) > 1:
                cluster_dist_mat = dist_mat[points_indices][:, points_indices]

                if eval_metric == 'mean':
                    cluster_results.append(
                        np.mean(sp.distance.squareform(cluster_dist_mat)))
                elif eval_metric == 'median':
                    cluster_results.append(
                        np.median(sp.distance.squareform(cluster_dist_mat)))

        avr_of_results.append(sum(cluster_results) / len(cluster_results))

    ax = plt.axes()
    x = range(nMin, nMax)
    ax.plot(x, avr_of_results)
    plt.xticks(range(nMin, nMax + 1))
    ax.xaxis.grid(True)
    plt.show()


def silhouette(gran, nMin, nMax, algo, distance_type, method='ward', metric='norm'):
    dist_mat = _get_dist_mat(gran, metric, distance_type)
    results = []
    for n in range(nMin, nMax):
        labels = _cluster_lables(dist_mat, n, algo, method)
        silhouette_result = silhouette_score(
            X=dist_mat, labels=labels, metric="precomputed", sample_size=None)
        results.append(silhouette_result)

    ax = plt.axes()
    x = range(nMin, nMax)
    ax.plot(x, results)
    plt.xticks(range(nMin, nMax + 1))
    ax.xaxis.grid(True)
    plt.show()


def dunn(gran, nMin, nMax, algo, distance_type, method='ward', metric='norm'):
    dist_mat = _get_dist_mat(gran, metric, distance_type)
    results = []
    for n in range(nMin, nMax):

        labels = _cluster_lables(dist_mat, n, algo, method)
        inter_cluster_all = []
        intra_cluster_all = []
        for i in set(labels):
            # print(i)
            points_indices_i = [p for p, x in enumerate(labels) if x == i]
            if len(points_indices_i) > 1:
                intra_cluster = np.mean(
                    sp.distance.squareform(dist_mat[points_indices_i][:, points_indices_i]))
                intra_cluster_all.append(intra_cluster)

            for j in set(labels):
                if j != i:
                    points_indices_j = [
                        p for p, x in enumerate(labels) if x == j]
                    inter_cluster = np.mean(dist_mat[points_indices_i][:,
                                                                       points_indices_j])
                    inter_cluster_all.append(inter_cluster)

        di = min(inter_cluster_all) / max(intra_cluster_all)

        results.append(di)

    ax = plt.axes()
    x = range(nMin, nMax)
    ax.plot(x, results)
    plt.xticks(range(nMin, nMax + 1))
    ax.xaxis.grid(True)
    plt.show()


# average_median_dissimilarity(gran='cities', eval_metric='mean',
#                              nMin=2, nMax=16, metric='norm', algo='kmed', distance_type='lang')


# silhouette(gran='cities', nMin=2, nMax=20, metric='norm',
#            algo='kmed', distance_type='lang')


dunn(gran='cities', nMin=2, nMax=50, metric='norm',
     algo='kmed', distance_type='lang')
