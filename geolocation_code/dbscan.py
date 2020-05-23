import pickle
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from plotly.subplots import make_subplots
import scipy.spatial as sp
import plotly.graph_objects as go
import plotly
import plotly.express as px
from collections import Counter
import matplotlib
from sklearn.metrics import silhouette_score
from dunn import dunn
import time
matplotlib.use('GTK3Agg')


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


def mydunn(labels, dist_mat, diameter_method='mean',
           cdist_method='min'):

    inter_cluster_all = []
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

    for i in np.arange(0, len(clusters)):
        points_indices_i = [p for p, x in enumerate(
            labels) if x == i]
        for j in np.arange(i, len(clusters)):
            # for i in clusters:

            #     for j in clusters:
            if j != i:
                points_indices_j = [
                    p for p, x in enumerate(labels) if x == j]
                if cdist_method == 'mean':
                    inter_cluster = np.mean(
                        dist_mat[points_indices_i][:, points_indices_j])
                elif cdist_method == 'min':
                    inter_cluster = np.min(
                        dist_mat[points_indices_i][:, points_indices_j])
                inter_cluster_all.append(inter_cluster)

    di = np.min(inter_cluster_all) / np.max(diameters)
    # print(time.time() - start)
    return (di)


dist_mat = _get_dist_mat('cities', 'norm', 'lang')


# print(len(dist_mat))

# dist_list = list(sp.distance.squareform(dist_mat))
# print(len(dist_list))
# fig = px.box(y=dist_list)
# fig.update_yaxes(title_text="language distance")
# fig.update_traces(jitter=0.5)
# fig.update_layout(
#     title="Language distance distribution of " + str(len(dist_mat)) + " US cities")

# plotly.offline.plot(fig, filename='lang_dist_box.html')


# # #####################################
# # inside box old
# # eps_min = 0.736
# # eps_max = 0.83

# # inside box
# eps_min = 0.757
# eps_max = 0.868

# # # inside whiskers old
# # eps_min = 0.6
# #eps_max = 0.9

# inside whiskers
eps_min = 0.624
eps_max = 1.035


eps_range = [round(i, 3) for i in np.arange(eps_min, eps_max, 0.005)]
print(len(eps_range))
min_p_range = [i for i in np.arange(1, 110)]

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
        # print(set(clustering))
        cluster_filter = np.array([i for i in clustering if i != -1])
        noise_ratio = np.count_nonzero(clustering == -1) / len(clustering)
        noise_mat[i, j] = noise_ratio

        size_mat[i, j] = len(set(cluster_filter))

        subsets_per_cluster = list(Counter(cluster_filter).values())

        mean_cluster_size[i, j] = 0 if len(
            subsets_per_cluster) == 0 else np.mean(subsets_per_cluster)

        if len(set(clustering)) == 1 and -1 in set(clustering):
            result_mat[i, j] = 0
            silhouette_mat[i, j] = -1

        else:

            if len(dist_mat) > len(set(cluster_filter)) > 1:
                noise_points = [p for p, x in enumerate(clustering) if x == -1]
                dist_range = [i for i in range(len(dist_mat))]
                dist_range_filtered = [
                    x for x in dist_range if x not in noise_points]
                dist_mat_filtered = dist_mat[dist_range_filtered][:,
                                                                  dist_range_filtered]

                silhouette_result = silhouette_score(
                    X=dist_mat_filtered, labels=cluster_filter, metric="precomputed", sample_size=None)
                silhouette_mat[i, j] = silhouette_result
                di = mydunn(labels=cluster_filter, dist_mat=dist_mat_filtered)

                # di = dunn(labels=cluster_filter, distances=dist_mat_filtered,
                #           diameter_method='mean_cluster', cdist_method="nearest")

                result_mat[i, j] = di

            else:
                silhouette_mat[i, j] = -1
                result_mat[i, j] = 0

    print(i)
    print(time.time() - start)


def format_coord(x, y):

    return "min_p: {}, eps: {}, dunn: {}".format(min_p_range[int(x)], eps_range[int(y)], result_mat[int(y)][int(x)])


# fig = plt.figure()
# ax = fig.add_subplot()
x_range = range(1, len(min_p_range), 10)
y_range = range(1, len(eps_range), 10)

# custome_data = np.dstack((list(x_range), list(y_range)))
# fig.colorbar(cax)

# ax.set_xticks(x_range)
# ax.set_yticks(y_range)


# ax.set_xticklabels([min_p_range[i] for i in x_range])
# ax.set_yticklabels([eps_range[i] for i in y_range])

# ax.format_coord = format_coord
# plt.savefig('dbscan_dunn2.png', dpi=300, bbox_inches='tight')
# pickle.dump(ax, open('dbscan_dunn2.pickle', 'w'))


fig = make_subplots(rows=3, cols=1, subplot_titles=[
                    'Color: Dunn index', 'Color: Silhouette', 'Color: Number of generated clusters'], vertical_spacing=0.09)


# fig = go.Figure(data=go.Heatmap(
#     z=result_mat, x=min_p_range, y=eps_range, customdata=np.dstack((size_mat, noise_mat, mean_cluster_size)), hovertemplate='n_cluster: %{customdata[0]}<br>noise_ratio: %{customdata[1]:.3f}<br>average within cluster cities: %{customdata[2]:.0f}<br>min_p: %{x}<br>eps: %{y:.3f}<br><b>dunn: %{z:.3f} </b>'))
#cbarlocs = [.85, .5, .15]
fig.add_trace(go.Heatmap(colorbar=dict(len=0.29, y=0.86), xgap=0.3, ygap=0.3, colorscale="reds",
                         z=result_mat, x=min_p_range, y=eps_range, customdata=np.dstack((size_mat, noise_mat, mean_cluster_size)), hovertemplate='n_cluster: %{customdata[0]}<br>noise_ratio: %{customdata[1]:.2f}<br>average within cluster cities: %{customdata[2]:.0f}<br>min_samples: %{x}<br>eps: %{y:.3f}<br><b>dunn: %{z:.3f} </b>'),
              row=1, col=1)


fig.add_trace(go.Heatmap(colorbar=dict(len=0.29, y=0.50), xgap=0.3, ygap=0.3, colorscale="reds",
                         z=silhouette_mat, x=min_p_range, y=eps_range, customdata=np.dstack((size_mat, noise_mat, mean_cluster_size)), hovertemplate='n_cluster: %{customdata[0]}<br>noise_ratio: %{customdata[1]:.2f}<br>average within cluster cities: %{customdata[2]:.0f}<br>min_samples: %{x}<br>eps: %{y:.3f}<br><b>Silhouette: %{z:.3f}</b>'),
              row=2, col=1)


fig.add_trace(go.Heatmap(colorbar=dict(len=0.29, y=0.14), xgap=0.3, ygap=0.3, colorscale="reds",
                         z=size_mat, x=min_p_range, y=eps_range, customdata=np.dstack((result_mat, noise_mat, mean_cluster_size)), hovertemplate='<b>n_cluster: %{z}</b><br>noise_ratio: %{customdata[1]:.2f}<br>average within cluster cities: %{customdata[2]:.0f}<br>min_samples: %{x}<br>eps: %{y:.3f}<br>dunn: %{customdata[0]:.3f}'),
              row=3, col=1)


fig.update_layout(
    title="DBSCAN's dunn and silhouette index based on different epsilon+min_samples params combinations. Epsilon chosen from inside whiskers \n")

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
plotly.offline.plot(fig, filename='dbscan_eval_whiskers.html')
