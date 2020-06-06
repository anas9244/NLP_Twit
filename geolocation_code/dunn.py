import numpy as np
from sklearn.preprocessing import LabelEncoder

DIAMETER_METHODS = ['mean_cluster', 'farthest']
CLUSTER_DISTANCE_METHODS = ['nearest', 'farthest']


def inter_cluster_distances(labels, distances, method='nearest'):
    """Calculates the distances between the two nearest points of each cluster.

    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param method: `nearest` for the distances between the two nearest points in each cluster, or `farthest`
    """
    if method not in CLUSTER_DISTANCE_METHODS:
        raise ValueError(
            'method must be one of {}'.format(CLUSTER_DISTANCE_METHODS))

    if method == 'nearest':
        return __cluster_distances_by_points(labels, distances)
    elif method == 'farthest':
        return __cluster_distances_by_points(labels, distances, farthest=True)


def __cluster_distances_by_points(labels, distances, farthest=False):
    n_unique_labels = len(np.unique(labels))
    cluster_distances = np.full((n_unique_labels, n_unique_labels),
                                float('inf') if not farthest else 0)

    np.fill_diagonal(cluster_distances, 0)

    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i, len(labels)):
            if labels[i] != labels[ii] and (
                (not farthest and
                 distances[i, ii] < cluster_distances[labels[i], labels[ii]])
                    or
                (farthest and
                 distances[i, ii] > cluster_distances[labels[i], labels[ii]])):
                cluster_distances[labels[i], labels[ii]] = cluster_distances[
                    labels[ii], labels[i]] = distances[i, ii]
    return cluster_distances


def diameter(labels, distances, method='farthest'):
    """Calculates cluster diameters

    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param method: either `mean_cluster` for the mean distance between all elements in each cluster, or `farthest` for the distance between the two points furthest from each other
    """
    if method not in DIAMETER_METHODS:
        raise ValueError('method must be one of {}'.format(DIAMETER_METHODS))

    n_clusters = len(np.unique(labels))
    diameters = np.zeros(n_clusters)

    if method == 'mean_cluster':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii]:
                    diameters[labels[i]] += distances[i, ii]

        for i in range(len(diameters)):
            diameters[i] /= sum(labels == i)

    elif method == 'farthest':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii] and distances[i, ii] > diameters[
                        labels[i]]:
                    diameters[labels[i]] = distances[i, ii]
    return diameters


def dunn(labels, distances, diameter_method='farthest',
         cdist_method='nearest'):
    """
    Dunn index for cluster validation (larger is better).

    .. math:: D = \\min_{i = 1 \\ldots n_c; j = i + 1\ldots n_c} \\left\\lbrace \\frac{d \\left( c_i,c_j \\right)}{\\max_{k = 1 \\ldots n_c} \\left(diam \\left(c_k \\right) \\right)} \\right\\rbrace

    where :math:`d(c_i,c_j)` represents the distance between
    clusters :math:`c_i` and :math:`c_j`, and :math:`diam(c_k)` is the diameter of cluster :math:`c_k`.

    Inter-cluster distance can be defined in many ways, such as the distance between cluster centroids or between their closest elements. Cluster diameter can be defined as the mean distance between all elements in the cluster, between all elements to the cluster centroid, or as the distance between the two furthest elements.

    The higher the value of the resulting Dunn index, the better the clustering
    result is considered, since higher values indicate that clusters are
    compact (small :math:`diam(c_k)`) and far apart (large :math:`d \\left( c_i,c_j \\right)`).

    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param diameter_method: see :py:function:`diameter` `method` parameter
    :param cdist_method: see :py:function:`diameter` `method` parameter

    .. [Kovacs2005] Kovács, F., Legány, C., & Babos, A. (2005). Cluster validity measurement techniques. 6th International Symposium of Hungarian Researchers on Computational Intelligence.
    """

    labels = LabelEncoder().fit(labels).transform(labels)

    ic_distances = inter_cluster_distances(labels, distances, cdist_method)
    min_distance = min(ic_distances[ic_distances.nonzero()])
    max_diameter = max(diameter(labels, distances, diameter_method))

    return min_distance / max_diameter
# from nltk import everygrams
# import numpy as np
# from sklearn.feature_extraction.text import TfidfTransformer


# def _get_word_vec(sample):
#     """ Generates a directory of word occurrences for subsets in a given sample """
#     word_vec = {}
#     for tweet in sample:
#         # grams = everygrams(tweet.split(), max_len=3)
#         # hash_grams = [str(int(sha256(

#            # "".ajoin(gram).encode('utf-8')).hexdigest(), 16) % 10**8) for gram in grams]
#         #char_tokens = [c for c in tweet if c != " "]
#         #char_grams = everygrams(char_tokens, min_len=2, max_len=5)
#         # for gram in char_grams:
#         for word in tweet.split():
#             #joined_gram = "".join(gram)
#             if word not in word_vec:
#                 word_vec[word] = 1
#             else:
#                 word_vec[word] += 1
#     return word_vec


# def _get_word_set(subsets_words):
#     """ Generates a set of word types that are common across all subsets """
#     word_set = set()
#     for index, subset in enumerate(subsets_words):
#         if index == 0:
#             for word in subsets_words[subset]:
#                 word_set.add(word)
#         else:
#             set2 = set()
#             for word in subsets_words[subset]:
#                 set2.add(word)
#             word_set = word_set.intersection(set2)
#     return word_set


# sample1 = ["hi there man, you good", "not much man, how are you",
#            "ok good i love you", "I love you too nigga"]
# sample2 = ["wtf man", "ay squeeze it nigga",
#            "ay ay what you said nigga", "man fuck you man", "hi man nigga"]
# subsets_words = {}

# vec1 = _get_word_vec(sample1)
# vec2 = _get_word_vec(sample2)
# subsets_words["sample1"] = vec1
# subsets_words["sample2"] = vec2
# all_types = {
#     char for subset in subsets_words for char in subsets_words[subset]}
# print(all_types)
# count_mat = np.zeros((len(subsets_words), len(all_types)))
# for i, subset in enumerate(subsets_words):
#     print(subset)
#     count_mat[i] = [subsets_words[subset][c]
#                     if c in subsets_words[subset] else 0 for c in all_types]


# vec_tf = TfidfTransformer(norm="l1")
# X_tf = vec_tf.fit_transform(count_mat)
# print(count_mat.shape)
# print(X_tf.toarray())
