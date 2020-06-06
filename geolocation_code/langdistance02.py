import numpy as np
import random
import math
import pickle
import os
import time
from scipy.spatial import distance
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import shutil
from statistics import mean, stdev
import math
from collections import Counter

-- Helper functions --#


def _prepend(files, dir_path):
    """ Prepend the full directory path to files, so they can be used in open() """
    dir_path += '{0}'
    files = [dir_path.format(i) for i in files]
    return(files)


def _get_files(dirr):
    """ Generates a list of files with full paths inside the given folder name """
    fileDir = os.path.dirname(os.path.abspath(__file__))
    path_dir = fileDir + "/" + dirr + "/"
    files = os.listdir(path=path_dir)
    files_paths = _prepend(files, path_dir)
    files_paths.sort(key=os.path.getmtime)
    return(files_paths)


def _getResamplData(pickle_file):
    """ Extracts sample of the dataset given a pickle file """
    resample_file = open(pickle_file, "rb")
    resample_data = pickle.load(resample_file)
    resample_file.close()

    return resample_data


def _save_results(gran, iter_results, metric):
    """ Saves a pickle file of distance matrix for given granularity and metric after averaging the samples results """

    avr_mat = sum(iter_results) / len(iter_results)
    output_path = "data_test/" + gran + "/dist_mats/" + metric + "_dist_mat.pickle"
    save_avr_result = open(output_path, "wb")
    pickle.dump(avr_mat, save_avr_result, -1)
    save_avr_result.close()

    file_path = os.path.abspath(output_path)
    print(metric + " distance matrix is stored in ", file_path)


def _translate(value, leftMin, leftMax):
    """ Translates a value in a given range into 0-1 range """
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = 1 - 0

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return 0 + (valueScaled * rightSpan)


def _get_word_vec(sample):
    """ Generates a directory of word occurrences for subsets in a given sample """
    word_vec = {}
    for tweet in sample:
        for word in tweet.split():
            if word not in word_vec:
                word_vec[word] = 1
            else:
                word_vec[word] += 1
    return word_vec


#-- End of helper functions --#


#-- Main functions --#
def resample(gran, dataset):
    """ Generates samples given granularity and dataset. A pickle for each iteration """
    if gran not in {"states", "cities"}:
        raise ValueError(
            "'" + gran + "'" + " is invalid. Possible values are ('states' , 'cities')")

    data_path = "data_test/" + gran
    if not os.path.exists(data_path):
        raise Exception("Missing dataset data for " + gran +
                        "! Please run build_data() first.")
    elif len(os.listdir(data_path)) == 0:
        raise Exception("Missing dataset data for " + gran +
                        "! Please run build_data() first.")

    resample_path = data_path + "/resampling"
    min_subset = min([len(dataset[subset]) for subset in dataset])
    max_subset = max([len(dataset[subset]) for subset in dataset])
    iters = int(round(max_subset / min_subset, 0))

    print("Creating random-resampling data....")
    print("Number of subsets: ", len(dataset))
    print("Smallest subset: ", min_subset, " tweets")
    print("Largest subset: ", max_subset, " tweets")
    print("Num. of iterations: ", iters)

    if os.path.exists(resample_path):
        shutil.rmtree(resample_path)
        os.makedirs(resample_path)
    else:
        os.makedirs(resample_path)

    for i in range(1, iters + 1):
        start_time = time.time()

        subsets_words = {}
        corpus = []
        for subset in dataset:

            start_index = random.randint(0, len(dataset[subset]) - min_subset)
            end_index = start_index + min_subset
            sample = dataset[subset][start_index:end_index]

            # word_vec = _get_word_vec(sample)
            # subsets_words[subset] = word_vec

        # word_set = {word
        #             for subset in subsets_words for word in subsets_words[subset]}

        # print(len(word_set))
        # print(type(word_set))
            sample_corpus = " ".join(sample)
            corpus.append(sample_corpus)

        vec = CountVectorizer(preprocessor=lambda x: x, ngram_range=(1, 3))
        X = vec.fit_transform(corpus)
        print(X.shape)

        time_elapsed = time.time() - start_time
        print("Finished " + str(i) + "/" + str(iters) + " iteration ")
        print("Estimated time left: ", int(
            time_elapsed * (iters - i)), " sec.")
        print("")

        save_resampling_iter = open(
            resample_path + "/iter_" + str(i) + ".pickle", "wb")
        pickle.dump(X, save_resampling_iter, -1)


def _get_delta(u, v):
    diff = abs(u - v)
    return np.mean(diff)


def burrows_delta(gran):
    """ Generates burrows_delta matrix given a granularity and store in a pickle file, This must be run after resample() """
    if gran not in {"states", "cities"}:
        raise ValueError(
            "'" + gran + "'" + " is invalid. Possible values are ('states' , 'cities')")
    resample_path = "data_test/" + gran + "/resampling"
    if not os.path.exists(resample_path):
        raise Exception(
            "No resampling data found! Please run resample() first.")
    elif len(os.listdir(resample_path)) == 0:
        raise Exception(
            "No resampling data found! Please run resample() first.")
    else:
        print("Starting Burrows_delta...")
        iter_results = []
        for res_index, file in enumerate(_get_files(resample_path)):
            start_time = time.time()

            X = _getResamplData(file)
            vec_tf = TfidfTransformer(use_idf=False, norm="l1")
            X_tf = vec_tf.fit_transform(X).toarray()
            common = X_tf[:, np.where(X_tf.all(axis=0))[0]]
            means = np.mean(common, 0)
            stds = np.std(common, 0)

            diff_common = common - means
            z_scores = diff_common / stds

            result_mat = np.zeros((len(z_scores), len(z_scores)))

            for i, target in enumerate(z_scores):
                deltas = []
                for j in z_scores:
                    #print(_get_delta(target, j))
                    deltas.append(_get_delta(target, j))

                result_mat[i] = deltas
            iter_results.append(result_mat)

            time_elapsed = time.time() - start_time
            print("Finished " + str(res_index + 1) + "/" +
                  str(len(_get_files(resample_path))) + " iteration ")
            print("Estimated time left: ", int(time_elapsed *
                                               (len(_get_files(resample_path)) - (res_index + 1))), " sec.")
            print("")

    _save_results(gran, iter_results, "burrows_delta")


def JSD(gran):
    """ Generates JSD matrix given a granularity and store in a pickle file, This must be run after resample() """
    if gran not in {"states", "cities"}:
        raise ValueError(
            "'" + gran + "'" + " is invalid. Possible values are ('states' , 'cities')")
    resample_path = "data_test/" + gran + "/resampling"
    if not os.path.exists(resample_path):
        raise Exception(
            "No resampling data found! Please run resample() first.")
    elif len(os.listdir(resample_path)) == 0:
        raise Exception(
            "No resampling data found! Please run resample() first.")
    else:
        print("Starting JSD...")
        iter_results = []
        for res_index, file in enumerate(_get_files(resample_path)):
            start_time = time.time()

            X = _getResamplData(file)
            vec_tf = TfidfTransformer(use_idf=False, norm="l1")
            X_tf = vec_tf.fit_transform(X).toarray()
            #common = X_tf[:, np.where(X_tf.all(axis=0))[0]]

            result_mat = np.zeros((len(X_tf), len(X_tf)))

            for index, subset in enumerate(X_tf):
                subset_jsds = []
                for other_subset in X_tf:
                    jsd_unfilter = distance.jensenshannon(
                        subset, other_subset, 2.0)
                    subset_jsds.append(jsd_unfilter)
                    print("unfiltered: ", jsd_unfilter)
                result_mat[index] = subset_jsds

            iter_results.append(result_mat)

            time_elapsed = time.time() - start_time
            print("Finished " + str(res_index + 1) + "/" +
                  str(len(_get_files(resample_path))) + " iteration ")
            print("Estimated time left: ", int(time_elapsed *
                                               (len(_get_files(resample_path)) - (res_index + 1))), " sec.")
            print("")
        _save_results(gran, iter_results, "jsd")




######## TEST ########################

# iter_results.append(result_mat)
# time_elapsed = time.time() - start_time
# print("Finished " + str(res_index + 1) + "/" +
#       str(len(_get_files(resample_path))) + " iteration ")
# print("Estimated time left: ", int(time_elapsed *
#                                    (len(_get_files(resample_path)) - (res_index + 1))), " sec.")
# print("")

#_save_results(gran, iter_results, "burrows_delta")
corpus = ["what Hi there man", "what is goind on man", "fuck you what man"]
vec = CountVectorizer(preprocessor=lambda x: x)
vec_tf = TfidfTransformer(use_idf=False, norm="l1", sublinear_tf=True)
X = vec.fit_transform(corpus)
print(vec.get_feature_names())
X_tf = vec_tf.fit_transform(X)
print(X.toarray())
print(X_tf.toarray())
# X_arr = X.toarray()

# print(np.mean(X_arr, 0))
# print(X_arr[:, np.where(X_arr.all(axis=0))[0]])
#print(a[np.all(a < 10, axis=1)])
# for r in range(np.size(X_arr, 1)):
#     ax = [X_arr[x, r] for x in range(len(X_arr))]
#     if all(v != 0 for v in ax):
#         print(mean(ax))
#         ax_stdev = 0
#         for i in ax:
#             diff = i - mean(ax)
#             ax_stdev += diff * diff
#         ax_stdev /= len(X_arr) - 1
#         ax_stdev = math.sqrt(ax_stdev)
#         print(stdev(ax))
#         print(ax_stdev)
#4472136