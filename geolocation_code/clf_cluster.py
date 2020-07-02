from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn import model_selection, naive_bayes, svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pickle
from sklearn_extra.cluster import KMedoids
from operator import itemgetter
from collections import OrderedDict, Counter
#!/usr/bin/python3
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from statistics import mean

import string
from nltk.stem import PorterStemmer
import re
import json
from time import time
from scipy.cluster.hierarchy import fcluster, linkage
import scipy.spatial as sp
from sklearn.model_selection import train_test_split

# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import CondensedNearestNeighbour
# from imblearn.combine import SMOTEENN
from collections import Counter
import build_data as b
from sklearn.cluster import DBSCAN

import numpy as np
from sklearn.dummy import DummyClassifier
import plotly
import plotly.graph_objects as go

DATA = "/mnt/ceph/storage/data-in-progress/wstud-geolocation/internal/Twitter/tweets_for_training/"


files = [file for file in b._get_files(DATA)]


def _get_dist_mat(gran, metric, distance_type, gram_type):
    gran_path = "data_" + gram_type + "/" + gran

    if distance_type == 'lang':

        dist_mat_file = open(gran_path + "/dist_mats/" +
                             metric + "_dist_mat.pickle", "rb")
        dist_mat = pickle.load(dist_mat_file)

    elif distance_type == 'geo':
        dist_mat_file = open(gran_path + "/dist_mats/geo_mat.pickle", "rb")
        dist_mat = pickle.load(dist_mat_file)

    city_ids_file = open(gran_path + "/city_ids.pickle", "rb")
    city_ids = pickle.load(city_ids_file)

    return dist_mat, city_ids


def _filter_noise(lables):

    # n_clusters = len(set(lables))

    if -1 in lables:
        available_lables = np.array(
            [i for i in range(len(lables)) if i not in lables])

        # print(set(lables))
        # print(lables)
        # print(available_lables.shape)

        # print(np.where(lables != -1),lables,available_lables)

        # print(set(lables))
        # lables_filtered = np.zeros(len(lables), dtype="int32")
        current = -1

        # filtered=[]

        # for i in range(len(noise)):
        # lables[i] =

        for i in range(len(lables)):
            if lables[i] == -1:
                # lables_filtered[i] = lables[i]
                # else:
                current += 1
                lables[i] = available_lables[current]
                # del available_lables[0]
        return lables
    else:
        return lables

def training_data(n_cluster, distance_type, algo, dbscan_params):

    msg = "Clustering based on language distance" if distance_type == "lang" else "Clustering based on geographic distance"
    print(msg)
    dist_mat, city_names = _get_dist_mat(
        "cities", "norm", distance_type, "char")

    if algo == "kmed":

        clustering = KMedoids(
            n_clusters=n_cluster, metric='precomputed').fit_predict(dist_mat)
    elif algo == "dbscan":

        clustering = DBSCAN(eps=dbscan_params[0], min_samples=dbscan_params[1],
                            metric='precomputed').fit_predict(dist_mat)
        clustering = _filter_noise(clustering)
    print("n_cluster: ", len(set(clustering)))
    # linkage_result = linkage(sp.distance.squareform(dist_mat), method="ward")
    # clustering = fcluster(
    #     linkage_result, t=3, criterion='maxclust')

    clusters_cities = {}
    for index, subset in enumerate(city_names):
        clusters_cities[subset] = clustering[index]

    # print(len(clusters_names))

    # # print(clusters_names)
    # #tweets_file = open(files[0], "r", encoding="utf-8")
    # tweets = []
    # lables = []

    # n_tweets = 10000

    # starta_size = n_tweets / len((city_names))

    # print(starta_size)

    # city_counts = {}

    # completed = 0

    # for i in city_names:
    #     city_counts[i] = 0
    # for file in files:
    #     opened_file = open(file, "r", encoding="utf-8")
    #     for line in opened_file:
    #         # print(line)
    #         tweet = json.loads(line)
    #         if tweet["country"] == "US" and tweet["place_type"] == "city":
    #             clean_tweet = b._clean_string(tweet["text"])

    #             if tweet['place_name'] in city_names:
    #                 #cluster_label = clusters_names[tweet['place_name']]
    #                 lable = tweet['place_name']
    #                 city_counts[lable] += 1
    #                 if city_counts[lable] <= 1000:
    #                     tweets.append(clean_tweet)

    #                     lables.append(lable)
    #             # print(len(tweets))
    #             # print(len(city_counts))
    #             # print(
    #             #     len([i for i in city_counts if city_counts[i] == intstarta_size]))

    #             if len([i for i in city_counts if city_counts[i] >= 1000]) > completed:
    #                 completed += 1
    #                 print([i for i in city_counts if city_counts[i] >= 1000])
    #                 print(completed)

    #             if completed >= 10:
    #                 break
    #             # print(clean_tweet)

    # top_lables=
    # print(Counter(lables[10:]))

    # pickle.dump( tweets, open( "traning_tweets.pickle", "wb" ) )
    # pickle.dump( lables, open( "traning_lables.pickle", "wb" ) )

    tweets = pickle.load(open("traning_tweets.pickle", "rb"))
    lables = pickle.load(open("traning_lables.pickle", "rb"))

    return tweets, lables, clusters_cities, len(set(clustering))


def classify(n_cluster, distance_type, algo, dbscan_params):

    # print(algo)
    tweets, lables, clusters_cities, dbscan_n_cluster = training_data(
        n_cluster, distance_type, algo, dbscan_params)
    # print(Counter(lables))

    # print("n_tweets: ", str(len(tweets)))

    count_vectorizer = CountVectorizer(
        preprocessor=lambda x: x, ngram_range=(3, 3), analyzer="char")
    X = count_vectorizer.fit_transform(tweets)

    vec = TfidfTransformer(sublinear_tf=True, norm="l2")
    Train_X_Tfidf = vec.fit_transform(X)
    MNB = naive_bayes.MultinomialNB()

    X_train, X_test, y_train, y_test = train_test_split(
        Train_X_Tfidf, lables, test_size=0.2, stratify=lables)

    # print(Counter(y_test))

    MNB.fit(X_train, y_train)

    perdicts = MNB.predict(X_test)

    correct = 0
    for i in range(len(perdicts)):
        if perdicts[i] == y_test[i] or clusters_cities[perdicts[i]] == clusters_cities[y_test[i]]:
            correct += 1

    print("acc", correct / len(perdicts))

    if algo == "kmed":
        return correct / len(perdicts)
    elif algo == "dbscan":
        return correct / len(perdicts), dbscan_n_cluster

    # print(perdicts[0])

    # print("Naive Bayes Accuracy Score -> ", MNB.score(X_test, y_test))

    # # acc_NB = mean(cross_val_score(MNB, Train_X_Tfidf,
    # #                               lables, cv=5, error_score="raise"))

    # # print("Naive Bayes Accuracy Score -> ",
    # #       acc_NB)

    # # SVM = svm.LinearSVC()
    # # #acc_SVM = mean(cross_val_score(SVM, Train_X_Tfidf, lables, cv=5))
    # # SVM.fit(X_train, y_train)
    # # acc_SVM = SVM.score(X_test, y_test)
    # # print("LinearSVC Accuracy Score -> ",
    # #       acc_SVM)

    # # ridge_model = RidgeClassifier()
    # # acc_ridge = mean(cross_val_score(
    # #     ridge_model, Train_X_Tfidf, lables, cv=5))
    # # print("RidgeClassifier Accuracy Score -> ",
    # #       acc_ridge)

    # dummy_clf = DummyClassifier(strategy="stratified")
    # dummy_clf.fit(X_train, y_train)
    # acc__dumm = dummy_clf.score(X_test, y_test)

    # print("Dummy CLF Accuracy Score -> ",
    #       acc__dumm)


lang_results = []
geo_results = []

x_lang = []

dbscan_params_lang = [(0.901, 10), (0.891, 3), (0.881, 3),
                      (0.871, 11), (0.861, 33), (0.851, 14)]

dbscan_params_geo = [(501.600, 39), (501.600, 49), (401.600, 47),
                     (301.600, 30), (101.600, 6), (101.600, 8)]
for i in dbscan_params_lang:
    # print(i)

    acc_lang, n_cluster = classify(1, "lang", "dbscan", i)

    #acc_geo = classify(i, "geo", "dbscan",i)
    lang_results.append(acc_lang)
    x_lang.append(n_cluster)
    # geo_results.append(acc_geo)

x_geo = []
for i in dbscan_params_geo:

    acc_geo, n_cluster = classify(i, "geo", "dbscan", i)
    geo_results.append(acc_geo)
    x_geo.append(n_cluster)

# x_lang = [54, 72, 98, 146, 205]
# x_geo = [54, 72, 99, 150, 206]

# lang_results = [0.5062277444103128, 0.44059827100494653,
#                 0.41162716060703314, 0.32137198864996763, 0.2217790207795251]
# geo_results = [0.5991625021013861, 0.4086521072445606,
#                0.2449324754583569, 0.06983733997626071, 0.06523721465723208]
fig = go.Figure()

fig.add_trace(go.Scatter(x=x_lang, y=lang_results,
                         mode='lines',
                         name='lang_clustering'))

fig.add_trace(go.Scatter(x=x_geo, y=geo_results,
                         mode='lines',
                         name='geo_clustering'))

plotly.offline.plot(fig, filename='clf_results_dbscan.html')


# rfc_model = RandomForestClassifier()
# acc_rfc = mean(cross_val_score(
#     rfc_model, Train_X_Tfidf, lables, cv=4))
# print("RandomForestClassifier Accuracy Score -> ",
#       acc_rfc)


# ps = PorterStemmer()
# punc = set(string.punctuation)


# result_mat_file = open(
#     "iter_results_merged_new.pickle", "rb")
# dist_mat = pickle.load(result_mat_file)
# result_mat_file.close()

# average_mat = sum(dist_mat) / len(dist_mat)


# tweets_dict_file = open("normed_tweets.pickle", "rb")
# tweets_dict = pickle.load(tweets_dict_file)
# tweets_dict_file.close()

# names = [state for state in tweets_dict]
# tweets = []


# for state in tweets_dict:
#     for tweet in tweets_dict[state]:
#         tweets.append(tweet)


# def clusters_tweets(n):
#     cluster_labels = []

#     clustering = KMedoids(
#         n_clusters=n, metric='precomputed').fit_predict(average_mat)

#     clusters_names = {}
#     for index, state in enumerate(names):
#         clusters_names[state] = clustering[index]

#     for state in tweets_dict:
#         for tweet in tweets_dict[state]:
#             # if not tweet.startswith("[mention]"):

#             cluster_labels.append(clusters_names[state])

#     cluster_sizes = {}
#     clutser_counts = {}
#     for c in cluster_labels:
#         clutser_counts[c] = 0
#         if c not in cluster_sizes:
#             cluster_sizes[c] = 1
#         else:
#             cluster_sizes[c] += 1

#     min_cluster = min(Counter(cluster_labels).values())
#     max_cluster = max(Counter(cluster_labels).values())

#     sampled_cluster_labels = []
#     sampled_tweets = []

#     for index, c in enumerate(cluster_labels):
#         if clutser_counts[c] < min_cluster:
#             clutser_counts[c] += 1
#             sampled_cluster_labels.append(c)
#             sampled_tweets.append(tweets[index])

#     return sampled_tweets, sampled_cluster_labels, min_cluster, max_cluster


# #file = open("clf_stats_Z_KMedoids.json", "a")


# # for i in range(7, 8):
# file = open("clf_stats_Z_KMedoids.json", "a")
# start = time()

# #cc = RandomUnderSampler()
# tweets, cluster_labels, min_cluster, max_cluster = clusters_tweets(7)
# #Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(tweets,Corpus['label'],test_size=0.3)


# print(min_cluster)
# print(Counter(cluster_labels))

# vec = TfidfVectorizer(ngram_range=(1, 3))
# Train_X_Tfidf = vec.fit_transform(tweets)

# print("done tfidf")

# # X_res, y_res = cc.fit_resample(Train_X_Tfidf, cluster_labels)

# # print(Counter(y_res))
# # print(time() - start, "finished resampling")

# MNB = naive_bayes.MultinomialNB()
# acc_NB = mean(cross_val_score(MNB, Train_X_Tfidf, cluster_labels, cv=5))
# print(time() - start, "Naive Bayes Accuracy Score -> ",
#       acc_NB)

# SVM = svm.LinearSVC()
# acc_SVM = mean(cross_val_score(SVM, X_res, y_res, cv=5))
# print(time() - start, "LinearSVC Accuracy Score -> ",
#       acc_SVM)

# ridge_model = RidgeClassifier()
# acc_ridge = mean(cross_val_score(
#     ridge_model, X_res, y_res, cv=5))
# print(time() - start, "RidgeClassifier Accuracy Score -> ",
#       acc_ridge)

# rfc_model = RandomForestClassifier()
# acc_rfc = mean(cross_val_score(
#     rfc_model, X_res, y_res, cv=5))
# print(time() - start, "RandomForestClassifier Accuracy Score -> ",
#       acc_rfc)

# record = {"Random_downsampling_1_3grams_n_clusters": 7, "cluster_tweets_num": min_cluster, 'max_cluster': max_cluster,
#           "acc_NB": acc_NB}  # , 'acc_SVM': acc_SVM, 'acc_ridge': acc_ridge, 'acc_rfc': acc_rfc}
# json.dump(record, file)

# file.write("\n")
# file.close()
