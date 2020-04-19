import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import scipy.spatial as sp
import scipy.cluster.hierarchy as hc
from scipy.cluster.hierarchy import dendrogram


def _alpha_sort(lables, mat):
    """ Sorts the given distance matrix and lables alphabetically """
    lables_sorted = sorted(lables)
    sorted_ind = []
    for label in lables_sorted:
        sorted_ind .append(lables.index(label))

    sorted_mat = np.zeros((len(mat), len(mat)))
    for i in range(len(sorted_ind)):
        for j in range(len(sorted_ind)):
            sorted_mat[i][j] = mat[sorted_ind[i]][sorted_ind[j]]
    return lables_sorted, sorted_mat


def _hrchy_sort(gran, sort_by, dist_mat, lables, method, show_lables):
    """ Sorts the given distance matrix and lables using hierarchical clustering, either by language or geographic distance """
    if sort_by == 'lang':
        linkage = hc.linkage(sp.distance.squareform(dist_mat), method=method)
    elif sort_by == 'geo':
        geo_mat_file = open("data/" + gran + "/dist_mats/geo_mat.pickle", "rb")
        geo_mat = pickle.load(geo_mat_file)
        linkage = hc.linkage(sp.distance.squareform(geo_mat), method=method)

    dendo = dendrogram(linkage, labels=lables)
    if not show_lables:
        plt.axis('off')
    leaves = dendo['leaves']

    sorted_lables = []
    for i in leaves:
        sorted_lables.append(lables[i])
    sorted_mat = np.empty(dist_mat.shape)

    # For highliting
    target_names = ['Lewisville, TX', 'Santa Clarita, CA',
                    'Yonkers, NY', 'Riverview, FL']

    target_ind = [i for i in range(len(sorted_lables))
                  if sorted_lables[i] in target_names]

    # highlight_names = []
    # for name in sorted_lables:
    #     if name in target_names:
    #         highlight_names.append(name)
    #     else:
    #         highlight_names.append(None)

    for i in range(len(leaves)):
        for j in range(len(leaves)):
            # if i in target_ind:
            #     sorted_mat[i][j] = 1.8
            # elif j in target_ind:
            #     sorted_mat[i][j] = 1.8
            # else:

            sorted_mat[i][j] = dist_mat[leaves[i]][leaves[j]]
    return sorted_lables, sorted_mat


def _show_mat(gran, measure, mat, lables, sort, method, show_lables):
    fig = plt.figure()
    ax = fig.add_subplot()
    cax = ax.matshow(mat, cmap='Reds')
    fig.colorbar(cax)
    ticks = np.arange(0, len(lables))

    ax.set_xticks(ticks,)
    ax.set_yticks(ticks)

    ax.set_xticklabels(lables, size=5, rotation=45)
    ax.set_yticklabels(lables, size=5)
    if sort == 'alpha':
        sort_by = "alphabetical order. "
    elif sort == 'lang':
        sort_by = "language similarity, method = " + method
    elif sort == 'geo':
        sort_by = "geographic distance, method = " + method

    if measure == 'norm':
        plt.title(gran + " language distance based on combination of 3 metrics (burrows_delta, TF-IDF, JSD) \n Sorted by " +
                  sort_by + ". Num of " + gran + ": " + str(len(lables)))
    else:
        plt.title(gran + " language distance based on " + measure +
                  ", sorted by " + sort_by + ". Num of " + gran + ": " + str(len(lables)))

    if not show_lables:
        plt.axis('off')

    #plt.title(" ")
    plt.savefig('lang.png', dpi=800, bbox_inches='tight')
    plt.show()


def Plot_Mat(gran, metric, sort, show_lables, method='ward'):
    """  Shows the distance matrix given granularity, metric and method with the ability to sort alphabetically, by language or geographic distance.

    Parameters:
    gran (str): granularity, can take 'cities' or 'states'
    sort (str): can take 'alpha' for alphabetic, 'geo' for geographic or 'lang' for language distance
    show_lables (bool): True to show lables
    method: linkage method. will be ignored if sort='alpha'

    """

    if gran not in {"states", "cities"}:
        raise ValueError(
            "'" + gran + "'" + " is invalid. Possible values are ('states' , 'cities')")
    if metric not in {'burrows_delta', 'jsd', 'tfidf', 'norm'}:
        raise ValueError(
            "'" + metric + "'" + " is invalid. Possible values are ('burrows_delta', 'jsd', 'tfidf', 'norm')")
    if sort not in {'alpha', 'geo', 'lang'}:
        raise ValueError(
            "'" + sort + "'" + " is invalid. Possible values are ('alpha', 'geo', 'lang')")

    gran_path = "data/" + gran

    if not os.path.exists(gran_path + "/dist_mats/"):
        raise Exception(
            "Missing distance matrices data! Please run Burrows_delta(), JSD(), TF_IDF() and  Norm_mat() first.")
    # elif len(os.listdir(gran_path + "/dist_mats/")) < 5:
    #     raise Exception(
    #         "Missing distance matrices data! Please run Burrows_delta(), JSD(), TF_IDF() and  Norm_mat() first.")

    labels_file = open(gran_path + "/labels.pickle", "rb")
    labels = pickle.load(labels_file)

    dist_mat_file = open(gran_path + "/dist_mats/" +
                         metric + "_dist_mat.pickle", "rb")
    dist_mat = pickle.load(dist_mat_file)

    if sort == 'alpha':
        sorted_labels, sorted_mat = _alpha_sort(labels, dist_mat)
    else:
        sorted_labels, sorted_mat = _hrchy_sort(
            gran, sort, dist_mat, labels, method, show_lables)

    _show_mat(gran, metric, sorted_mat, sorted_labels,
              sort, method, show_lables)


# if __name__ == "__main__":
#     # Possible values for metric are ('burrows_delta', 'jsd', 'tfidf', 'norm')
#     # Possible values for sort are ('alpha', 'geo', 'lang')
#     Plot_Mat(gran='states', metric='norm', sort='alpha', show_lables=False, method='ward')
