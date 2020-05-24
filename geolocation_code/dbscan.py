import pickle
from sklearn.cluster import DBSCAN
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
from collections import Counter
from sklearn.metrics import silhouette_score
import time


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


def _filter_noise(lables, dist_mat):
    cluster_filtered = np.array([i for i in lables if i != -1])
    noise_points = [p for p, x in enumerate(lables) if x == -1]
    dist_range = [i for i in range(len(dist_mat))]
    dist_range_filtered = [
        x for x in dist_range if x not in noise_points]
    dist_mat_filtered = dist_mat[dist_range_filtered][:,
                                                      dist_range_filtered]

    return cluster_filtered, dist_mat_filtered


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


# print(len(dist_mat))

# dist_list = list(sp.distance.squareform(dist_mat))
# print(len(dist_list))
# fig = px.box(y=dist_list)
# fig.update_yaxes(title_text="language distance")
# fig.update_traces(jitter=0.5)
# fig.update_layout(
#     title="Language distance distribution of " + str(len(dist_mat)) + " US cities")

# plotly.offline.plot(fig, filename='lang_dist_box.html')


box = (0.757, 0.868)
whiskers = (0.624, 1.035)


def dbscan_eval(gran, metric, distance_type, eps_inside):

    dist_mat = _get_dist_mat('cities', 'norm', 'lang')

    if eps_inside == "whiskers":
        eps_min = whiskers[0]
        eps_max = whiskers[1]
        min_p_max = 110
    elif eps_inside == "box":
        eps_min = box[0]
        eps_max = box[1]
        min_p_max = 50

    eps_range = [round(i, 3) for i in np.arange(eps_min, eps_max, 0.005)]
    print(len(eps_range))
    min_p_range = [i for i in np.arange(1, min_p_max)]

    result_mat = np.zeros((len(eps_range), len(min_p_range)))
    silhouette_mat = np.zeros((len(eps_range), len(min_p_range)))

    size_mat = np.zeros((len(eps_range), len(min_p_range)))
    noise_mat = np.zeros((len(eps_range), len(min_p_range)))
    mean_cluster_size = np.zeros((len(eps_range), len(min_p_range)))

    for i, eps in enumerate(eps_range):

        start = time.time()
        for j, min_p in enumerate(min_p_range):
            clustering = DBSCAN(eps=eps, min_samples=min_p,
                                metric='precomputed').fit_predict(dist_mat)
            cluster_filtered, dist_mat_filtered = _filter_noise(
                clustering, dist_mat)
            noise_ratio = np.count_nonzero(clustering == -1) / len(clustering)
            noise_mat[i, j] = noise_ratio

            size_mat[i, j] = len(set(cluster_filtered))

            subsets_per_cluster = list(Counter(cluster_filtered).values())

            mean_cluster_size[i, j] = 0 if len(
                subsets_per_cluster) == 0 else np.mean(subsets_per_cluster)

            if len(set(clustering)) == 1 and -1 in set(clustering):
                result_mat[i, j] = 0
                silhouette_mat[i, j] = -1
            else:
                if len(dist_mat) > len(set(cluster_filtered)) > 1:
                    silhouette_result = silhouette_score(
                        X=dist_mat_filtered, labels=cluster_filtered, metric="precomputed", sample_size=None)
                    silhouette_mat[i, j] = silhouette_result
                    di = mydunn(labels=cluster_filtered,
                                dist_mat=dist_mat_filtered)
                    result_mat[i, j] = di

                else:
                    silhouette_mat[i, j] = -1
                    result_mat[i, j] = 0

        print(i)
        print(time.time() - start)

    fig = make_subplots(rows=3, cols=1, subplot_titles=[
                        'Color: Dunn index', 'Color: Silhouette', 'Color: Number of generated clusters'], vertical_spacing=0.09)

    fig.add_trace(go.Heatmap(colorbar=dict(len=0.29, y=0.86), xgap=0.3, ygap=0.3, colorscale="reds",
                             z=result_mat, x=min_p_range, y=eps_range, customdata=np.dstack((size_mat, noise_mat, mean_cluster_size)), hovertemplate='n_cluster: %{customdata[0]}<br>noise_ratio: %{customdata[1]:.2f}<br>average within cluster cities: %{customdata[2]:.0f}<br>min_samples: %{x}<br>eps: %{y:.3f}<br><b>dunn: %{z:.4f} </b>'),
                  row=1, col=1)

    fig.add_trace(go.Heatmap(colorbar=dict(len=0.29, y=0.50), xgap=0.3, ygap=0.3, colorscale="reds",
                             z=silhouette_mat, x=min_p_range, y=eps_range, customdata=np.dstack((size_mat, noise_mat, mean_cluster_size)), hovertemplate='n_cluster: %{customdata[0]}<br>noise_ratio: %{customdata[1]:.2f}<br>average within cluster cities: %{customdata[2]:.0f}<br>min_samples: %{x}<br>eps: %{y:.3f}<br><b>Silhouette: %{z:.4f}</b>'),
                  row=2, col=1)

    fig.add_trace(go.Heatmap(colorbar=dict(len=0.29, y=0.14), xgap=0.3, ygap=0.3, colorscale="reds",
                             z=size_mat, x=min_p_range, y=eps_range, customdata=np.dstack((result_mat, noise_mat, mean_cluster_size)), hovertemplate='<b>n_cluster: %{z}</b><br>noise_ratio: %{customdata[1]:.2f}<br>average within cluster cities: %{customdata[2]:.0f}<br>min_samples: %{x}<br>eps: %{y:.3f}<br>dunn: %{customdata[0]:.4f}'),
                  row=3, col=1)

    fig.update_layout(
        title="DBSCAN's dunn and silhouette index based on different epsilon+min_samples params combinations. Epsilon chosen from inside " + eps_inside + " \n")

    fig.update_xaxes(title_text="min_samples", row=1, col=1)
    fig.update_xaxes(title_text="min_samples", row=2, col=1)
    fig.update_xaxes(title_text="min_samples", row=3, col=1)

    fig.update_yaxes(title_text="eps", row=1, col=1)
    fig.update_yaxes(title_text="eps", row=2, col=1)
    fig.update_yaxes(title_text="eps", row=3, col=1)

    fig.update_layout(
        autosize=False,
        width=1450,
        height=1600,)
    plotly.offline.plot(fig, filename="dbscan_eval_" + eps_inside + ".html")


dbscan_eval(gran="cities", metric="norm", distance_type="lang",
            eps_inside="whiskers")
