import numpy as np
import matplotlib.pyplot as plt
import pickle
from operator import itemgetter
import math
from shapely.geometry import MultiPoint, Point, Polygon
import shapefile
# return a polygon for each state in a dictionary
from geopy.distance import geodesic
import os
import seaborn as sns
from scipy.stats import pearsonr
sns.set(color_codes=True)


def Plot_corr(gran,metric):

    if gran not in {"states", "cities"}:
        raise ValueError("'" + gran + "'" + " is invalid. Possible values are ('states' , 'cities')")
    if metric not in {'burrows_delta', 'jsd', 'tfidf', 'norm'}:
        raise ValueError("'" + metric + "'" + " is invalid. Possible values are ('burrows_delta', 'jsd', 'tfidf', 'norm')")

    gran_path = "data/" + gran

    if not os.path.exists(gran_path + "/dist_mats/"):
        raise Exception("Missing distance matrices data! Please run Burrows_delta(), JSD(), TF_IDF() and  Norm_mat() first.")
    elif len(os.listdir(gran_path + "/dist_mats/")) < 5:
        raise Exception("Missing distance matrices data! Please run Burrows_delta(), JSD(), TF_IDF() and  Norm_mat() first.")


    dist_mat_file = open(gran_path + "/dist_mats/" + metric + "_dist_mat.pickle", "rb")
    dist_mat = pickle.load(dist_mat_file)


    geo_mat_file = open("data/" + gran + "/dist_mats/geo_mat.pickle", "rb")
    geo_mat = pickle.load(geo_mat_file)


    state_deltas = {}
    state_deltas['dis'] = []
    state_deltas['km'] = []
    state_deltas['dis_all'] = []
    state_deltas['km_all'] = []

    state_deltas['names'] = []

    for index, x in enumerate(dist_mat):

        values = [val for val in x if val > 0]
        min_dist = min(values)
        # print(min_dist)
        min_dist_i = np.argmin(values)
        subset01 = index
        subset02 = min_dist_i

        state_deltas['dis'].append(min_dist)
        state_deltas['km'].append(geo_mat[subset01, subset02])

        state_deltas['dis_all'] += values
        state_deltas['km_all'] += [value for value in geo_mat[index] if value > 0]

    dis_values01 = [value for value in state_deltas['dis']]

    km_values02 = [value for value in state_deltas['km']]

    dis_values01_all = [value for value in state_deltas['dis_all']]

    km_values02_all = [value for value in state_deltas['km_all']]



    print(len(dis_values01))
    corr, _ = pearsonr(dis_values01, km_values02)
    corr_all, _ = pearsonr(dis_values01_all, km_values02_all)
    print(corr)
    plt.title("Correlation between language distance and georaphic distance of the "+str( len(dist_mat))+ " US " +gran+" pairs with the lowest language distance \n with Pearson coefficient= " +
              str(corr) + " , and Pearson coefficient of all pairs= " + str(corr_all))
    plt.xlabel("Language distance")
    plt.ylabel("KM")
    # for i, txt in enumerate(state_deltas['names']):
    #     plt.annotate(txt, (dis_values01[i], km_values02[i]))
    sns.regplot(x=dis_values01, y=km_values02)
    # plt.axis('off')
    plt.show()




Plot_corr('cities','norm')

