import pickle
import os
import matplotlib.pyplot as plt
#--- Custom modules ---#
from build_data import Build_data
from langdistance import Resample, Burrows_delta, JSD, TF_IDF, Norm_mat
from plot_mat import Plot_Mat
from clustering import Clustering


# Path to a folder containing json file/s of tweets where each line is a tweet object
RAW_PATH = "/media/data/twitter_geolocation/json/"


def _get_dataset(gran):
    if gran not in {"states", "cities"}:
        raise ValueError(
            "'" + gran + "'" + " is invalid. Possible values are ('states' , 'cities')")
    data_path = "data02/" + gran
    if not os.path.exists(data_path):
        raise Exception("Missing dataset data for " + gran +
                        "! Please run build_data() first.")
    elif len(os.listdir(data_path)) == 0:
        raise Exception("Missing dataset data for " + gran +
                        "! Please run build_data() first.")

    dataset_file = open(data_path + "/dataset.pickle", "rb")
    dataset = pickle.load(dataset_file)
    dataset_file.close()

    return dataset


def Plot_Subset_Freq(gran):
    """ Plots the distribution of given dataset, i.e. number of tweets per subset """
    if gran not in {"states", "cities"}:
        raise ValueError(
            "'" + gran + "'" + " is invalid. Possible values are ('states' , 'cities')")

    dataset = _get_dataset(gran)
    tweets_num = sum([len(dataset[subset]) for subset in dataset])
    X = [subset for subset in dataset]
    Y = [len(dataset[subset]) for subset in dataset]

    X = [x for _, x in sorted(zip(Y, X), reverse=True)]
    Y = sorted(Y, reverse=True)

    plt.bar(X, Y)
    plt.title("Dataset distribution." + " Number of " + gran + ": " +
              str(len(dataset)) + "\n Number of overall tweets: " + str(tweets_num))
    plt.xlabel(gran)
    plt.ylabel("Number of tweets")
    plt.xticks(rotation='vertical', fontsize=8)
    plt.show()


def CREATS_MATS(gran):
    """  Resamples the dataset and creates distance matrices as per multiple metrics. Then generate the norm matrix of all metrics """

    dataset = _get_dataset(gran)
    Resample(gran, dataset)
    # Burrows_delta(gran)
    # JSD(gran)
    TF_IDF(gran)
    Norm_mat(gran)


if __name__ == "__main__":

    # Accepted values for gran: 'states', 'cities'
    # Recommneded maxsubset for gran='states' to be >1000000 for better representation
    # Recommneded minsubset for gran='cities' to be >5000 since less will create very few common word types accros subsets

    # Build_data(raw_data_path=RAW_PATH, gran="states",
               # minsubset=5000, maxsubset=2000000)

    # Plot_Subset_Freq('states')

    # CREATS_MATS('states')

    #dataset = _get_dataset('cities')

    #print (sum([len(dataset[subset]) for subset in dataset]))

    #----------- Uncomment the following block and comment out CREATS_MATS() and Build_data() if you want to repeat running for different args --------------#

    Plot_Mat(gran='cities', metric='norm', sort='lang',
             show_lables=False, method='ward')
    #Clustering(gran='cities', metric='norm', n_clusters=8, algo='hrchy', method="complete")
