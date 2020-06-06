import pickle
from sklearn.cluster import DBSCAN
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
from collections import Counter
from sklearn.metrics import silhouette_score
import time
import plotly.express as px


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


def _filter_noise(lables):

    #n_clusters = len(set(lables))

    if -1 in lables:
        available_lables = np.array(
            [i for i in range(len(lables)) if i not in lables])

        # print(set(lables))
        # print(lables)
        # print(available_lables.shape)

        # print(np.where(lables != -1),lables,available_lables)

        # print(set(lables))
        #lables_filtered = np.zeros(len(lables), dtype="int32")
        current = -1

        # filtered=[]

        # for i in range(len(noise)):
        # lables[i] =

        for i in range(len(lables)):
            if lables[i] == -1:
                #lables_filtered[i] = lables[i]
                # else:
                current += 1
                lables[i] = available_lables[current]
                #del available_lables[0]
        return lables
    else:
        return lables

    # cluster_filtered = np.array([i for i in lables if i != -1])
    # noise_points = [p for p, x in enumerate(lables) if x == -1]
    # dist_range = [i for i in range(len(dist_mat))]
    # dist_range_filtered = [
    #     x for x in dist_range if x not in noise_points]
    # dist_mat_filtered = dist_mat[dist_range_filtered][:,
    #                                                   dist_range_filtered]

    # return cluster_filtered, dist_mat_filtered


def mydunn(labels, dist_mat, diameter_method='mean',
           cdist_method='mean'):

    clusters = np.unique(labels)
    n_clusters = len(clusters)
    diameters = np.zeros(n_clusters)

    if diameter_method == 'mean':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii]:

                    diameters[labels[i]] += dist_mat[i, ii]
        for i in range(len(diameters)):

            diameters[i] /= sum(labels == i)

    elif diameter_method == 'max':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii] and dist_mat[i, ii] > diameters[
                        labels[i]]:
                    diameters[labels[i]] = dist_mat[i, ii]

    if cdist_method == "mean":
        cluster_distances = np.zeros((n_clusters, n_clusters))
        counts = np.zeros((n_clusters, n_clusters))
        for i in np.arange(0, len(labels) - 1):
            for ii in np.arange(i, len(labels)):
                if labels[i] != labels[ii]:
                    cluster_distances[labels[i], labels[ii]] += dist_mat[i, ii]
                    cluster_distances[labels[ii], labels[i]
                                      ] += dist_mat[i, ii]
                    counts[labels[i], labels[ii]] += 1
                    counts[labels[ii], labels[i]] += 1

        for i in np.arange(0, len(cluster_distances)):
            for ii in np.arange(i, len(cluster_distances)):
                if i != ii:
                    cluster_distances[i, ii] /= counts[i, ii]
                    cluster_distances[ii, i] /= counts[ii, i]

    elif cdist_method == "min":
        cluster_distances = np.full(
            (n_clusters, n_clusters), np.inf)
        np.fill_diagonal(cluster_distances, 0)
        for i in np.arange(0, len(labels) - 1):
            for ii in np.arange(i, len(labels)):
                if labels[i] != labels[ii] and dist_mat[i, ii] < cluster_distances[labels[i], labels[ii]]:
                    cluster_distances[labels[i], labels[ii]] = cluster_distances[
                        labels[ii], labels[i]] = dist_mat[i, ii]

    # if len(cluster_distances[cluster_distances.nonzero()]) == 0:
    #     return 0
    # else:
    di = np.min(
        cluster_distances[cluster_distances.nonzero()]) / np.max(diameters)
    return (di)


import scipy.spatial as sp


# sublinear l2
# box = (0.96, 1.026)
# whiskers = (0.861, 1.125)


# geo
# box = (986, 2967)
# whiskers = (1000, 5934)
# # # no sublinear, l2
# # box = (0.767, 0.865)
# # whiskers = (0.65, 1.010)


def box_plt(gran, metric, distance_type):

    dist_mat_uni = _get_dist_mat(gran, metric, distance_type, "word")
    dist_list_uni = list(sp.distance.squareform(dist_mat_uni))

    dist_mat_gramword = _get_dist_mat(gran, metric, distance_type, "gramword")
    dist_list_gramword = list(sp.distance.squareform(dist_mat_gramword))

    dist_mat_char = _get_dist_mat(gran, metric, distance_type, "char")
    dist_list_char = list(sp.distance.squareform(dist_mat_char))

    fig = go.Figure()
    fig.add_trace(go.Box(y=dist_list_uni, name="Word Unigram"))
    fig.add_trace(go.Box(y=dist_list_gramword, name="1-4 Word grams"))
    fig.add_trace(go.Box(y=dist_list_char, name="3-3 Char grams"))

    fig.update_yaxes(title_text=distance_type + " distance")
    fig.update_traces(jitter=0.5)

    #gram_title = "Unigram word" if gram_type == "wrod" else "1-4 gram word"if gram_type == "gramword" else "2-5 gram char" if gram_type == "char" else ""
    fig.update_layout(
        title=distance_type + " distance distribution of " + str(len(dist_mat_uni)) + " US " + gran)

    plotly.offline.plot(fig, filename="lang_dist_box.html")


def dbscan_eval(gran, metric, distance_type):
    min_p_max = 50
    min_p_range = [i for i in np.arange(1, min_p_max)]

    dist_mat_uni = _get_dist_mat(gran, metric, distance_type, "word")
    dist_mat_gramword = _get_dist_mat(gran, metric, distance_type, "gramword")
    dist_mat_char = _get_dist_mat(gran, metric, distance_type, "char")

    dist_mat = [dist_mat_uni, dist_mat_gramword, dist_mat_char]

    eps_range_uni = [round(i, 3) for i in np.arange(0.811, 1.108, 0.005)]
    eps_range_gramword = [round(i, 3) for i in np.arange(0.865, 1.125, 0.005)]
    eps_range_char = [round(i, 3) for i in np.arange(1.6, 5934, 300)]

    eps_range = [eps_range_uni, eps_range_gramword, eps_range_char]

    result_mat_uni = np.zeros((len(eps_range_uni), len(min_p_range)))
    result_mat_gramword = np.zeros((len(eps_range_gramword), len(min_p_range)))
    result_mat_char = np.zeros((len(eps_range_char), len(min_p_range)))

    result_mat = [result_mat_uni, result_mat_gramword, result_mat_char]

    silhouette_mat_uni = np.zeros((len(eps_range_uni), len(min_p_range)))
    silhouette_mat_gramword = np.zeros(
        (len(eps_range_gramword), len(min_p_range)))
    silhouette_mat_char = np.zeros((len(eps_range_char), len(min_p_range)))

    silhouette_mat = [silhouette_mat_uni,
                      silhouette_mat_gramword, silhouette_mat_char]

    size_mat_uni = np.zeros((len(eps_range_uni), len(min_p_range)))
    size_mat_gramword = np.zeros((len(eps_range_gramword), len(min_p_range)))
    size_mat_char = np.zeros((len(eps_range_char), len(min_p_range)))

    size_mat = [size_mat_uni, size_mat_gramword, size_mat_char]

    noise_mat_uni = np.zeros((len(eps_range_uni), len(min_p_range)))
    noise_mat_gramword = np.zeros((len(eps_range_gramword), len(min_p_range)))
    noise_mat_char = np.zeros((len(eps_range_char), len(min_p_range)))

    noise_mat = [noise_mat_uni, noise_mat_gramword, noise_mat_char]

    mean_cluster_size_uni = np.zeros((len(eps_range_uni), len(min_p_range)))
    mean_cluster_size_gramword = np.zeros(
        (len(eps_range_gramword), len(min_p_range)))
    mean_cluster_size_char = np.zeros((len(eps_range_char), len(min_p_range)))

    mean_cluster_size = [mean_cluster_size_uni,
                         mean_cluster_size_gramword, mean_cluster_size_char]

    for n in range(len(eps_range)):
        print(n)

        if n == 2:
            print(len(eps_range[n]))
            for i, eps in enumerate(eps_range[n]):

                start = time.time()
                for j, min_p in enumerate(min_p_range):

                    clustering = DBSCAN(eps=eps, min_samples=min_p,
                                        metric='precomputed').fit_predict(dist_mat[n])
                    noise_ratio = np.count_nonzero(
                        clustering == -1) / len(clustering)
                    noise_mat[n][i, j] = noise_ratio
                    cluster_filtered = _filter_noise(clustering)

                    size_mat[n][i, j] = len(set(cluster_filtered))

                    subsets_per_cluster = list(
                        Counter(cluster_filtered).values())

                    mean_cluster_size[n][i, j] = 0 if len(
                        subsets_per_cluster) == 0 else np.mean(subsets_per_cluster)

                    if len(set(clustering)) == 1 and -1 in set(clustering):
                        result_mat[n][i, j] = 0
                        silhouette_mat[n][i, j] = -1
                    else:
                        if len(dist_mat[n]) > len(set(cluster_filtered)) > 1:
                            silhouette_result = silhouette_score(
                                X=dist_mat[n], labels=cluster_filtered, metric="precomputed", sample_size=None)
                            silhouette_mat[n][i, j] = silhouette_result
                            di = mydunn(labels=cluster_filtered,
                                        dist_mat=dist_mat[n])
                            result_mat[n][i, j] = di

                        else:
                            silhouette_mat[n][i, j] = -1
                            result_mat[n][i, j] = 0

                print(i)
                print(time.time() - start)

    fig = go.Figure(data=go.Heatmap(colorbar=dict(len=0.29, y=0.14), xgap=0.3, ygap=0.3, colorscale="reds",
                                    z=result_mat[2], x=min_p_range, y=eps_range[2], customdata=np.dstack((size_mat[2], noise_mat[2], mean_cluster_size[2])), hovertemplate='n_cluster: %{customdata[0]}<br>noise_ratio: %{customdata[1]:.2f}<br>average within cluster cities: %{customdata[2]:.0f}<br>min_samples: %{x}<br>eps: %{y:.3f}<br><b>dunn: %{z:.4f}</b>'))

    # fig = make_subplots(rows=3, cols=1, subplot_titles=[
    #                     'Color: Unigram', 'Color: 1-4 wordgram', 'Color: 3-3 char grams'], vertical_spacing=0.09)

    # fig.add_trace(go.Heatmap(colorbar=dict(len=0.29, y=0.86), xgap=0.3, ygap=0.3, colorscale="reds",
    #                          z=result_mat[0], x=min_p_range, y=eps_range[0], customdata=np.dstack((size_mat[0], noise_mat[0], mean_cluster_size[0])), hovertemplate='n_cluster: %{customdata[0]}<br>noise_ratio: %{customdata[1]:.2f}<br>average within cluster cities: %{customdata[2]:.0f}<br>min_samples: %{x}<br>eps: %{y:.3f}<br><b>dunn: %{z:.4f} </b>'),
    #               row=1, col=1)

    # fig.add_trace(go.Heatmap(colorbar=dict(len=0.29, y=0.50), xgap=0.3, ygap=0.3, colorscale="reds",
    #                          z=result_mat[1], x=min_p_range, y=eps_range[1], customdata=np.dstack((size_mat[1], noise_mat[1], mean_cluster_size[1])), hovertemplate='n_cluster: %{customdata[0]}<br>noise_ratio: %{customdata[1]:.2f}<br>average within cluster cities: %{customdata[2]:.0f}<br>min_samples: %{x}<br>eps: %{y:.3f}<br><b>dunn: %{z:.4f}</b>'),
    #               row=2, col=1)

    # fig.add_trace(go.Heatmap(colorbar=dict(len=0.29, y=0.14), xgap=0.3, ygap=0.3, colorscale="reds",
    #                          z=result_mat[2], x=min_p_range, y=eps_range[2], customdata=np.dstack((size_mat[2], noise_mat[2], mean_cluster_size[2])), hovertemplate='n_cluster: %{customdata[0]}<br>noise_ratio: %{customdata[1]:.2f}<br>average within cluster cities: %{customdata[2]:.0f}<br>min_samples: %{x}<br>eps: %{y:.3f}<br><b>dunn: %{z:.4f}</b>'),
    #               row=3, col=1)

    # fig = make_subplots(rows=3, cols=1, subplot_titles=[
    #                     'Color: Dunn index', 'Color: Silhouette', 'Color: Number of generated clusters'], vertical_spacing=0.09)

    # fig.add_trace(go.Heatmap(colorbar=dict(len=0.29, y=0.86), xgap=0.3, ygap=0.3, colorscale="reds",
    #                          z=result_mat, x=min_p_range, y=eps_range, customdata=np.dstack((size_mat, noise_mat, mean_cluster_size)), hovertemplate='n_cluster: %{customdata[0]}<br>noise_ratio: %{customdata[1]:.2f}<br>average within cluster cities: %{customdata[2]:.0f}<br>min_samples: %{x}<br>eps: %{y:.3f}<br><b>dunn: %{z:.4f} </b>'),
    #               row=1, col=1)

    # fig.add_trace(go.Heatmap(colorbar=dict(len=0.29, y=0.50), xgap=0.3, ygap=0.3, colorscale="reds",
    #                          z=silhouette_mat, x=min_p_range, y=eps_range, customdata=np.dstack((size_mat, noise_mat, mean_cluster_size)), hovertemplate='n_cluster: %{customdata[0]}<br>noise_ratio: %{customdata[1]:.2f}<br>average within cluster cities: %{customdata[2]:.0f}<br>min_samples: %{x}<br>eps: %{y:.3f}<br><b>Silhouette: %{z:.4f}</b>'),
    #               row=2, col=1)

    # fig.add_trace(go.Heatmap(colorbar=dict(len=0.29, y=0.14), xgap=0.3, ygap=0.3, colorscale="reds",
    #                          z=size_mat, x=min_p_range, y=eps_range, customdata=np.dstack((result_mat, noise_mat, mean_cluster_size)), hovertemplate='<b>n_cluster: %{z}</b><br>noise_ratio: %{customdata[1]:.2f}<br>average within cluster cities: %{customdata[2]:.0f}<br>min_samples: %{x}<br>eps: %{y:.3f}<br>dunn: %{customdata[0]:.4f}'),
    #               row=3, col=1)

    fig.update_layout(
        title="DBSCAN's dunn index based on different epsilon+min_samples params combinations. Epsilon chosen from inside whiskers \n")

    fig.update_xaxes(title_text="min_samples")
    fig.update_yaxes(title_text="eps")

    # fig.update_xaxes(title_text="min_samples", row=1, col=1)
    # fig.update_xaxes(title_text="min_samples", row=2, col=1)
    # fig.update_xaxes(title_text="min_samples", row=3, col=1)

    # fig.update_yaxes(title_text="eps", row=1, col=1)
    # fig.update_yaxes(title_text="eps", row=2, col=1)
    # fig.update_yaxes(title_text="eps", row=3, col=1)

    # fig.update_layout(
    #     autosize=False,
    #     width=1450,
    #     height=1600,)
    plotly.offline.plot(fig, filename="dbscan_eval_geo.html")


# dbscan_eval(gran="cities", metric="norm", distance_type="geo")


#box_plt("cities", "norm", "lang")
