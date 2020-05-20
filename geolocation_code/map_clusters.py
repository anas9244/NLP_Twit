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

city_coors_file = open("city_coors.pickle", "rb")
city_coors = pickle.load(city_coors_file)

city_id_file = open("city_id.pickle", "rb")
city_id = pickle.load(city_id_file)

dist_mat_file = open("norm_dist_mat.pickle", "rb")
dist_mat = pickle.load(dist_mat_file)

ordered_city_coords = []
for city in city_id:
  ordered_city_coords.append(city_coors[city])


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


def get_colors(labels):
  colors = []

  N = len(set(labels))
  HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
  RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
  for (r, g, b) in RGB_tuples:

    colors.append('#%02x%02x%02x' %
                  (int(r * 255), int(g * 255), int(b * 255)))
  return N, colors


def plot_map(n_clusters):
  labels = _cluster_lables(dist_mat, n_clusters, "kmed", "ward")

  N, colors = get_colors(labels)

  # print(labels)

  labels_colors = [colors[i] for i in labels]
  print(labels_colors)

  fig = go.Figure(data=go.Scattergeo(
      lon=[lon[1] for lon in ordered_city_coords],
      lat=[lon[0] for lon in ordered_city_coords],
      mode='markers', marker_color=labels_colors

  ))

  fig.update_layout(
      title='Number of clusters: ' + str(N),
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

  fig.show()


plot_map(2)
