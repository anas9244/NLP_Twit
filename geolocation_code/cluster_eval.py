import pickle
from sklearn_extra.cluster import KMedoids
import scipy.spatial as sp
from scipy.cluster.hierarchy import fcluster, linkage
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.stats import entropy
matplotlib.use('GTK3Agg')
import random
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


def _get_lang_model(sample):
    """ Generates a directory of word occurrences for subsets in a given sample """
    word_vec = {}
    for tweet in sample:
        for word in tweet.split():
            if word not in word_vec:
                word_vec[word] = 1
            else:
                word_vec[word] += 1
    return word_vec


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

def _resample_clusters(clusters_tweets):

    resample_results = []
    min_cluster = min([len(cluster) for cluster in clusters_tweets])
    max_cluster = max([len(cluster) for cluster in clusters_tweets])
    iters = int(round(max_cluster / min_cluster, 0))
    print("iterations: ", iters)
    for i in range(1, iters + 1):

        print(i, "iter")
        sample_mean_perplexity = 0
        for cluster in clusters_tweets:
            start_index = random.randint(0, len(cluster) - min_cluster)
            end_index = start_index + min_cluster
            sample_tweets = cluster[start_index:end_index]
            sample_corpus = " ".join(sample_tweets)
            count_vectorizer = CountVectorizer(preprocessor=lambda x: x)
            tf_vectorizer = TfidfTransformer(use_idf=False, norm="l1")

            X = count_vectorizer.fit_transform([sample_corpus])
            X_tf = tf_vectorizer.fit_transform(X)

            ent = entropy(X_tf.toarray()[0], base=2)
            sample_perplexity = 2**ent
            sample_mean_perplexity += sample_perplexity

        sample_mean_perplexity /= len(clusters_tweets)

        resample_results.append(sample_mean_perplexity)
    print("Overall perplexity: ", sum(resample_results) / len(resample_results))

    return sum(resample_results) / len(resample_results)


def perplexity(gran, nMin, nMax, algo, distance_type, method='ward', metric='norm'):
    dist_mat = _get_dist_mat(gran, metric, distance_type)
    results = []

    gran_path = "data/" + gran
    dataset_file = open(gran_path + "/dataset.pickle", "rb")
    dataset = pickle.load(dataset_file)
    keys = list(dataset.keys())

    for n in range(nMin, nMax):
        print("###################")
        print("cluster size: ", n)
        print("####################")
        labels = _cluster_lables(dist_mat, n, algo, method)
        # joined_lists=[]
        clusters_tweets = []
        for i in set(labels):
            points_indices_i = [p for p, x in enumerate(labels) if x == i]
            cluster_tweets = [
                tweet for p in points_indices_i for tweet in dataset[keys[p]]]
            clusters_tweets.append(cluster_tweets)
            # # joined_lists= [" ".join(dataset[keys[p]]) for p in points_indices_i]
            # #print(len(cluster_tweets))
            # corpus = " ".join(cluster_tweets)

            # # print("finished joining", keys)
            # count_vectorizer = CountVectorizer(
            #     preprocessor=lambda x: x)
            # tf_vectorizer = TfidfTransformer(use_idf=False, norm="l1")

            # X = count_vectorizer.fit_transform([corpus])
            # X_tf = tf_vectorizer.fit_transform(X)
            # ent = entropy(X_tf.toarray()[0], base=2)
            # print(i," , ",len(cluster_tweets), 2**ent)

        results.append(_resample_clusters(clusters_tweets))

    ax = plt.axes()
    x = range(nMin, nMax)
    ax.plot(x, results)
    plt.xticks(range(nMin, nMax + 1))
    ax.xaxis.grid(True)
    plt.show()

# 1699.0051465005286
# average_median_dissimilarity(gran='cities', eval_metric='mean',
#                              nMin=2, nMax=16, metric='norm', algo='kmed', distance_type='lang')


silhouette(gran='cities', nMin=2, nMax=20, metric='norm',
            algo='kmed', distance_type='lang')


# dunn(gran='cities', nMin=2, nMax=50, metric='norm',
#      algo='kmed', distance_type='lang')

#perplexity(gran='cities', nMin=2, nMax=20, metric='norm',
#           algo='kmed', distance_type='lang')
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from scipy.stats import entropy
# import pickle

# gran_path = "data/states"
# dataset_file = open(gran_path + "/dataset.pickle", "rb")
# dataset = pickle.load(dataset_file)
# print("finished loading data")
# key = list(dataset.keys())[2]
# corpus = " ".join(dataset[key])


# print("finished joining", key)
# # vectorizer = HashingVectorizer(
# #     preprocessor=_clean_string, alternate_sign=False, n_features=9, norm=None)
# count_vectorizer = CountVectorizer(
#     preprocessor=lambda x: x)
# tf_vectorizer = TfidfTransformer(use_idf=False, norm="l1")

# X = count_vectorizer.fit_transform([corpus])
# X_tf = tf_vectorizer.fit_transform(X)
# ent = entropy(X_tf.toarray()[0], base=2)
# print(2**ent)
# # # 1:2.251629167387823
