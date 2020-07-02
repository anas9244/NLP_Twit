# import plotly.graph_objects as go

# import pandas as pd

# fig = go.Figure()

# cities_text = ['Lewisville, TX', 'Santa Clarita, CA',
#                'Spokane, WA', 'Riverview, FL']


# cities_loc = [(33.035303, -96.988046), (34.403326, -118.527593),
#               (47.668131, -117.398425), (27.866140, -82.326241)]


# for i in range(len(cities_text)):

#     fig.add_trace(go.Scattergeo(
#         locationmode='USA-states',
#         # lon=(cities_loc[i][1], cities_loc[i][1] + 0.1),
#         # lat=(cities_loc[i][0], cities_loc[i][0] + 0.1),
#         lon=cities_loc[i][1],
#         lat=cities_loc[i][0],
#         text=cities_text[i],
#         marker=dict(
#             size=10,
#             line_color='rgb(40,40,40)',
#             line_width=0.5,
#             sizemode='area'
#         )))

# fig.update_layout(
#     title_text='2014 US city populations<br>(Click legend to toggle traces)',
#     showlegend=True,
#     geo=dict(
#         scope='usa',
#         landcolor='rgb(217, 217, 217)',
#     )
# )

# fig.show()


# # df = pd.read_csv(
# #     'https://raw.githubusercontent.com/plotly/datasets/master/2014_us_cities.csv')
# # df.head()

# # df['text'] = df['name'] + '<br>Population ' + \
# #     (df['pop'] / 1e6).astype(str) + ' million'
# # limits = [(0, 2), (3, 10), (11, 20), (21, 50), (50, 3000)]
# # colors = ["royalblue", "crimson", "lightseagreen", "orange", "lightgrey"]
# # cities = []
# # scale = 5000

# # fig = go.Figure()

# # for i in range(len(limits)):
# #     lim = limits[i]
# #     df_sub = df[lim[0]:lim[1]]
# #     fig.add_trace(go.Scattergeo(
# #         locationmode='USA-states',
# #         lon=df_sub['lon'],
# #         lat=df_sub['lat'],
# #         text=df_sub['text'],
# #         marker=dict(
# #             size=df_sub['pop'] / scale,
# #             color=colors[i],
# #             line_color='rgb(40,40,40)',
# #             line_width=0.5,
# #             sizemode='area'
# #         ),
# #         name='{0} - {1}'.format(lim[0], lim[1])))

# # fig.update_layout(
# #     title_text='2014 US city populations<br>(Click legend to toggle traces)',
# #     showlegend=True,
# #     geo=dict(
# #         scope='usa',
# #         landcolor='rgb(217, 217, 217)',
# #     )
# # )

# # fig.show()

# N = len(set(labels))
# HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
# RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
# for (r, g, b) in RGB_tuples:

#     colors.append('#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)))


import plotly.graph_objects as go
import pickle
import pandas as pd
import colorsys
from sklearn_extra.cluster import KMedoids
import scipy.spatial as sp
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN
from scipy.stats import entropy
from nltk import everygrams

import numpy as np

from statistics import mode

from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.feature_extraction import FeatureHasher

import random
import plotly


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


def perplexity(labels):
    #dist_mat = _get_dist_mat(gran, metric, distance_type)
    #results = []

    dataset_file = open("dataset.pickle", "rb")
    dataset = pickle.load(dataset_file)
    keys = list(dataset.keys())

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

    return np.mean(_resample_clusters(clusters_tweets))


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


def box_plt(lables, algo, dbscan_vals=None):

    fig = go.Figure()
    fig.add_trace(go.Box(y=lables, boxmean=True))

    # fig.update_yaxes(title_text="algo +", n_clusters: " + str(len(set(lables)))")
    fig.update_traces(jitter=0.1)

    # gram_title = "Unigram word" if gram_type == "wrod" else "1-4 gram word"if gram_type == "gramword" else "2-5 gram char" if gram_type == "char" else ""
    if algo == "dbscan":
        fig.update_layout(
            title=algo + ", n_clusters: " + str(len(set(lables))) + ", min_samples: " + str(dbscan_vals[0]) + ", eps: " + str(dbscan_vals[1]))
    elif algo == "kmed":
        fig.update_layout(
            title=algo + ", n_clusters: " + str(len(set(lables))))
    plotly.offline.plot(fig, filename="cluster_city_box.html")

    fig.show()


city_coors_file = open("city_coors.pickle", "rb")
city_coors = pickle.load(city_coors_file)

city_id_file = open("city_ids.pickle", "rb")
city_id = pickle.load(city_id_file)

dist_mat_file = open("norm_dist_mat.pickle", "rb")
dist_mat = pickle.load(dist_mat_file)

geo_mat_file = open("geo_mat.pickle", "rb")
geo_mat = pickle.load(dist_mat_file)


ordered_city_coords = []
for city in city_id:
    ordered_city_coords.append(city_coors[city])


def _cluster_lables(dist_mat, n_clusters, algo, method, dbscan_vals=None):
    if algo == 'hrchy':
        linkage_result = linkage(
            sp.distance.squareform(dist_mat), method=method)
        clustering = fcluster(
            linkage_result, t=n_clusters, criterion='maxclust')
    elif algo == 'kmed':
        clustering = KMedoids(n_clusters=n_clusters,
                              metric='precomputed').fit_predict(dist_mat)

    elif algo == "dbscan":

        clustering = DBSCAN(eps=dbscan_vals[0], min_samples=dbscan_vals[1],
                            metric='precomputed').fit_predict(dist_mat)
        clustering = _filter_noise(clustering)

    return clustering


def get_colors(labels):
    colors = []

    N = len(set(labels))
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    for (r, g, b) in RGB_tuples:

        colors.append('#%02x%02x%02x' %
                      (int(r * 255), int(g * 255), int(b * 255)))
    return N, colors


def plot_map(n_clusters, algo, dbscan_vals=None):

    if algo == "kmed":
        labels = _cluster_lables(dist_mat, n_clusters, algo, "ward")
    elif algo == "dbscan":
        labels = _cluster_lables(dist_mat, n_clusters,
                                 algo, "ward", dbscan_vals)

    box_plt(labels, algo, dbscan_vals)

    # subsets_per_cluster = list(Counter(labels).values())
    # print()
    # print(subsets_per_cluster)
    print(mode(labels))

    print(labels)
    print(perplexity(labels))

    N, colors = get_colors(labels)

    # print(labels)

    labels_colors = [colors[i] for i in labels]

    fig = go.Figure(data=go.Scattergeo(
        lon=[lon[1] for lon in ordered_city_coords],
        lat=[lon[0] for lon in ordered_city_coords],
        mode='markers', marker_color=labels_colors

    ))

    if algo == "dbscan":
        fig.update_layout(
            title=algo + ", n_clusters: " + str(len(set(labels))) + ", min_samples: " + str(dbscan_vals[0]) + ", eps: " + str(dbscan_vals[1]))
    elif algo == "kmed":
        fig.update_layout(
            title=algo + ", n_clusters: " + str(len(set(labels))))

    fig.update_layout(

        geo_scope='usa',  # margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    fig.update_layout(

        geo=dict(
            scope='usa',
            landcolor='rgb(217, 217, 217)',
        )
    )

    fig.update_traces(marker=dict(size=10,
                                  ))

    plotly.offline.plot(fig, filename="cluster_map.html")
    # fig.show()


plot_map(15, "dbscan", dbscan_vals=(0.931, 1))
