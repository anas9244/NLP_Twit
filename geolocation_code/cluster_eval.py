import pickle

from sklearn_extra.cluster import KMedoids
import scipy.spatial as sp
from scipy.cluster.hierarchy import fcluster, linkage
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib
import matplotlib.ticker
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import DBSCAN
from scipy.stats import entropy
import time
import random
from collections import Counter
from dbscan import mydunn
matplotlib.use('GTK3Agg')
import plotly.express as px
from nltk import everygrams
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction import DictVectorizer, FeatureHasher


# ToDo
# avr dissimilarity
# median dissimilarity, HOW?
# dunn index
# Silhouette Coefficient

# Do for geo_dist and lang_dist

# def _cluster_dist_mat()

def _get_dist_mat(gran, metric, distance_type, gram_type):
    gran_path = "data_" + gram_type + "/" + gran

    if distance_type == 'lang':

        dist_mat_file = open(gran_path + "/dist_mats/" +
                             metric + "_dist_mat.pickle", "rb")
        dist_mat = pickle.load(dist_mat_file)
    elif distance_type == 'geo':
        dist_mat_file = open(gran_path + "/dist_mats/geo_mat.pickle", "rb")
        dist_mat = pickle.load(dist_mat_file)

    return dist_mat


# def _get_dist_mat(gran, metric, distance_type, gramtype):
#     gran_path = "data_" + gramtype + "/" + gran

#     if distance_type == 'lang':

#         dist_mat_file = open(gran_path + "/dist_mats/" +
#                              metric + "_dist_mat.pickle", "rb")
#         dist_mat = pickle.load(dist_mat_file)
#     elif distance_type == 'geo':
#         dist_mat_file = open(gran_path + "/dist_mats/geo_mat.pickle", "rb")
#         dist_mat = pickle.load(dist_mat_file)

#     return dist_mat


def _cluster_lables(dist_mat, n_clusters, algo, method):
    if algo == 'hrchy':
        linkage_result = linkage(
            sp.distance.squareform(dist_mat), method=method)
        clustering = fcluster(
            linkage_result, t=n_clusters, criterion='maxclust')
    elif algo == 'kmed':
        clustering = KMedoids(n_clusters=n_clusters,
                              metric='precomputed').fit_predict(dist_mat)
    elif algo == "dbs":
        clustering = DBSCAN(n_clusters=n_clusters,
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


# , subsets_per_cluster_means
def plot_eval(eval_metric, gran, algo, distance_type, method, nMin, nMax, results, subsets_per_cluster_means):
    # ion()
    fig, ax1 = plt.subplots(figsize=(18.0, 10.0))  #
    x = range(nMin, nMax)
    color = 'tab:blue'
    ax1.set_xlabel('num of clusters')
    ax1.set_ylabel('mean num. of subsets per cluster', color=color)
    ax1.plot(x, subsets_per_cluster_means, color=color, linewidth=1)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel(eval_metric + " value", color=color)
    ax2.plot(x, results,
             color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()

    ax1.xaxis.grid(True)
    ax1.set_xticklabels(range(nMin, nMax), rotation=45, fontsize=10)
    plt.xticks(range(nMin, nMax), rotation=45)

    title = str(eval_metric) + "_" + str(gran) + "_" + \
        str(algo) + "_" + str(distance_type) + "_" + str(method)
    plt.title(title)
    plt.show()
    #plt.savefig(title + "_2", dpi=800, bbox_inches="tight")

    #ax = plt.axes()

    #ax.plot(x, results)
    # plt.xticks(range(nMin, nMax + 1))
    # ax.xaxis.grid(True)
    # plt.title(str(eval_metric) + ", " + str(gran) + ", " + str(algo) + ", " +
    #           str(distance_type) + ", " + str(method))
    # plt.show()


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
    plot_eval("average_dissimilarity", gran, algo,
              distance_type, method, nMin, nMax, avr_of_results)


def silhouette(gran, nMin, nMax, algo, distance_type, dist_mat, method='ward', metric='norm'):
    #dist_mat = _get_dist_mat(gran, metric, distance_type)
    results = []
    subsets_per_cluster_means = []
    subsets_per_cluster_all = []
    for n in range(nMin, nMax):
        labels = _cluster_lables(dist_mat, n, algo, method)
        subsets_per_cluster = list(Counter(labels).values())
        subsets_per_cluster_all.append(subsets_per_cluster)
        # subsets_per_cluster_means.append(np.mean(subsets_per_cluster))
        silhouette_result = silhouette_score(
            X=dist_mat, labels=labels, metric="precomputed", sample_size=None)
        results.append(silhouette_result)

    # plot_eval("silhouette", gran, algo, distance_type,
    #           method, nMin, nMax, results, subsets_per_cluster_means)

    return results, subsets_per_cluster_all


def dunn(gran, nMin, nMax, algo, distance_type, dist_mat, method='ward', metric='norm'):
    #dist_mat = _get_dist_mat(gran, metric, distance_type)
    results = []
    subsets_per_cluster_means = []
    subsets_per_cluster_all = []
    for n in range(nMin, nMax):
        # print(n)
        #start = time.time()

        labels = _cluster_lables(dist_mat, n, algo, method)

        subsets_per_cluster = list(Counter(labels).values())
        #print("subsets_per_cluster: ", subsets_per_cluster)
        subsets_per_cluster_all.append(subsets_per_cluster)
        # subsets_per_cluster_means.append(np.mean(subsets_per_cluster))

        di = mydunn(labels, dist_mat, diameter_method='mean',
                    cdist_method='mean')

        # labels_set = set(labels)
        # clusters = np.unique(labels)
        # n_clusters = len(clusters)
        # inter_cluster_all = []
        # intra_cluster_all = []
        # diameters = np.zeros(n_clusters)

        # for i in range(0, len(labels) - 1):
        #     for ii in range(i + 1, len(labels)):
        #         if labels[i] == labels[ii]:
        #             diameters[labels[i]] += dist_mat[i, ii]
        # for i in range(len(diameters)):
        #     diameters[i] /= sum(labels == i)

        # for i in labels_set:
        #     # print(i)
        #     points_indices_i = [p for p, x in enumerate(labels) if x == i]

        #     # if len(points_indices_i) > 1:
        #     #     intra_cluster = np.mean(
        #     #         sp.distance.squareform(dist_mat[points_indices_i][:, points_indices_i]))
        #     #     intra_cluster_all.append(intra_cluster)
        #     # else:
        #     #     intra_cluster_all.append(0)

        #     for j in labels_set:
        #         if j != i:
        #             points_indices_j = [
        #                 p for p, x in enumerate(labels) if x == j]
        #             inter_cluster = np.mean(
        #                 dist_mat[points_indices_i][:, points_indices_j])
        #             inter_cluster_all.append(inter_cluster)

        #di = np.min(inter_cluster_all) / np.max(diameters)
        # print(di)
        #print(time.time() - start)
        results.append(di)
    return results, subsets_per_cluster_all

    # #f = px.data.gapminder().query("country=='Canada'")
    # fig = go.Figure(data=go.Scatter(
    #     x=[i for i in range(nMin, nMax)], y=results))
    # plotly.offline.plot(fig, filename='dunn_kmed.html')

    # # plot_eval("dunn", gran, algo, distance_type, method,
    # #           nMin, nMax, results, subsets_per_cluster_means)


def _get_word_vec(sample, analyzer, ngram_range):  #
    """ Generates a directory of word occurrences for subsets in a given sample """
    word_vec = {}
    for tweet in sample:
        #grams = everygrams(tweet.split(), max_len=3)
        # hash_grams = [str(int(sha256(
        #   "".ajoin(gram).encode('utf-8')).hexdigest(), 16) % 10**8) for gram in grams]
        tokens = [c for c in tweet] if analyzer == "char" else tweet.split(
        ) if analyzer == "word" else "error"  # if c != " "
        grams = everygrams(
            tokens, min_len=ngram_range[0], max_len=ngram_range[1])
        for gram in grams:
            # for word in tweet.split():
            joined_gram = "".join(gram)  # if analyzer == "word" else "".join(
            # gram) if analyzer == "char" else gram if ngram_range == (1, 1) else "wtf"

            if joined_gram not in word_vec:
                word_vec[joined_gram] = 1
            else:
                word_vec[joined_gram] += 1

    #     for word in tweet.split():

    #         if word not in word_vec:
    #             word_vec[word] = 1
    #         else:

    #             word_vec[word] += 1
    return word_vec


def _resample_clusters(clusters_tweets):

    iters_results = []
    min_cluster = min([len(cluster) for cluster in clusters_tweets])
    max_cluster = max([len(cluster) for cluster in clusters_tweets])
    iters = int(round(max_cluster / min_cluster, 0)
                ) if int(round(max_cluster / min_cluster, 0)) < 20 else 20
    print("iterations: ", iters)
    for i in range(1, iters + 1):

        #print(i, "iter")
        clusters_perlex = []
        #sample_mean_perplexity = 0
        for cluster in clusters_tweets:
            start_index = random.randint(0, len(cluster) - min_cluster)
            end_index = start_index + min_cluster
            sample_tweets = cluster[start_index:end_index]
            #sample_corpus = " ".join(sample_tweets)

            # print("counting")
            #start_count = time.time()
            count_vec = _get_word_vec(
                sample=sample_tweets, analyzer="char", ngram_range=(3, 3))

            #print("finished counting after: ", time.time() - start_count)

            features = {feature for feature in count_vec}
            v = FeatureHasher(n_features=len(features), alternate_sign=False)

            #count_vectorizer = CountVectorizer(preprocessor=lambda x: x)
            tf_vectorizer = TfidfTransformer(use_idf=False)

            #X = count_vectorizer.fit_transform([sample_corpus])
            X = v.fit_transform([count_vec])

            X_tf = tf_vectorizer.fit_transform(X)
            ent = entropy(X_tf.toarray()[0], base=2)

            cluster_perplexity = 2**ent

            #sample_mean_perplexity += sample_perplexity

        #sample_mean_perplexity /= len(clusters_tweets)
            clusters_perlex.append(cluster_perplexity)
        iters_results.append(clusters_perlex)

    arrays = [np.array(x) for x in iters_results]

    resample_results = [np.mean(k) for k in zip(*arrays)]
    #print("Overall perplexity: ", sum(resample_results) / len(resample_results))

    return resample_results  # sum(resample_results) / len(resample_results)


def perplexity(gran, nMin, nMax, algo, distance_type, dist_mat, gram_type, method='ward', metric='norm'):
    #dist_mat = _get_dist_mat(gran, metric, distance_type)
    results = []

    gran_path = "data_" + gram_type + "/" + gran
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

    return results

    # plot_eval("perplexity", gran, algo, distance_type,
    #           method, nMin, nMax, results)


# average_median_dissimilarity(gran='cities', eval_metric='mean',
#                              nMin=2, nMax=30, metric='norm', algo='kmed', distance_type='lang')
def _translate(value, leftMin, leftMax):
    """ Translates a value in a given range into 0-1 range """
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin

    rightSpan = 1 - 0

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return 0 + (valueScaled * rightSpan)


def eval_plot(n_clusters, gramtype, gran, metric, distance_type):
    dist_mat = _get_dist_mat(gran, metric, distance_type, gramtype)
    print("wtf")
    # sil_results, sil_cluster_dist = silhouette(gran='cities', nMin=n_clusters[0], nMax=n_clusters[1], metric='norm',
    #                                            algo='kmed', distance_type='lang', method='ward', dist_mat=dist_mat)

    # di_results, di_cluster_dist = dunn(gran='cities', nMin=n_clusters[0], nMax=n_clusters[1], metric='norm',
    #                                    algo='kmed', distance_type='lang', method='ward', dist_mat=dist_mat)
    # sil_results_normed = []
    # di_results_normed = []

    # for i in range(len(sil_results)):
    #     sil_results_normed.append(_translate(
    #         sil_results[i], min(sil_results), max(sil_results)))
    #     di_results_normed.append(_translate(
    #         di_results[i], min(di_results), max(di_results)))

    # fig = make_subplots(rows=2, cols=1, subplot_titles=[
    #                     'Dunn+Silhouette', 'Cluster distribution'], vertical_spacing=0.09)

    # fig = go.Figure()

    # fig.add_trace(go.Scatter(x=[i for i in range(n_clusters[0], n_clusters[1])], y=di_results_normed,
    #                          mode='lines',
    #                          name='Dunn'))
    # fig.add_trace(go.Scatter(x=[i for i in range(n_clusters[0], n_clusters[1])], y=sil_results_normed,
    #                          mode='lines',
    #                          name='silhouette'))
    # plotly.offline.plot(fig, filename='cluster_eval.html')

    plex_result = perplexity(gran='cities', nMin=n_clusters[0], nMax=n_clusters[1],
                             metric='norm', algo='kmed', distance_type='lang', method='ward',
                             dist_mat=dist_mat, gram_type="char")

    # fig_box = go.Figure(data=[go.Box(
    #     y=di_cluster_dist[i], name=len(di_cluster_dist[i]),
    #     boxmean=True) for i in range(len(di_cluster_dist))])
    # fig_box.update_layout(showlegend=False)

    # plotly.offline.plot(fig_box, filename='box_eval.html')

    fig_box = go.Figure(data=[go.Box(
        y=plex_result[i], name=len(plex_result[i]),
        boxmean=True) for i in range(len(plex_result))])
    fig_box.update_layout(showlegend=False)

    plotly.offline.plot(fig_box, filename='box_perplex.html')


eval_plot((2, 500), gramtype="char", gran='cities',
          metric='norm', distance_type="lang")

# perplexity(gran='cities', nMin=2, nMax=20, metric='norm',
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
