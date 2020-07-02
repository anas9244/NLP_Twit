from sklearn.manifold import TSNE
import pickle

import plotly.express as px
import plotly.graph_objects as go
import plotly


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


dist_mat, city_ids = _get_dist_mat("cities", "norm", "lang", "char")


X_embedded = TSNE(n_components=2, metric="precomputed",
                  perplexity=15).fit_transform(dist_mat)

X = X_embedded[:, 0]
Y = X_embedded[:, 1]
# Z = X_embedded[:, 2]

fig = px.scatter(x=X, y=Y)

# fig = go.Figure(data=[go.Scatter3d(x=X, y=Y, z=Z,
#                                    mode='markers', hovertemplate=city_ids)])

# fig = go.Figure(data=[go.scatter(x=X, y=Y,
#                                    mode='markers', hovertemplate=city_ids)])
#fig = px.scatter_3d(x=X, y=Y, z=Z, hovertemplate=city_ids)
plotly.offline.plot(fig, filename='t-sne.html')
