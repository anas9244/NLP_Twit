import pickle
import ujson as json
import os
import csv
import build_data as d
from datetime import datetime
from dateutil.parser import parse
from time import time

RAW_PATH = "/media/data/twitter_geolocation/json_combined/"

fieldnames = ['tweet_id', 'raw_tweet', 'norm_tweet',
              'place_name', 'place_type', 'state_code', 'user_id', 'user_display_name',
              'user_screen_name', 'user_profile_location', 'timestamp']


def _tweet_data(tweet, clean_tweet):
    # coords = tweet['place']['bounding_box']['coordinates'][0]

    # south = coords[0][1]
    # north = coords[1][1]
    # west = coords[0][0]
    # east = coords[2][0]

    # location = [south, north, west, east]

    tweet_id = tweet['id']
    raw_tweet = tweet['extended_tweet']['full_text'] if tweet['truncated'] else tweet['text']
    norm_tweet = clean_tweet
    #place_name = tweet['place']['full_name']
    #place_location = tweet['place']['bounding_box']['coordinates'][0]
    #place_type = tweet['place']['place_type']
    state_code = d._extract_state(tweet)
    user_id = tweet['user']['id'] if 'id' in tweet['user'] else None
    user_display_name = tweet['user']['name'] if 'name' in tweet['user'] else None
    user_screen_name = tweet['user']['screen_name'] if 'screen_name' in tweet['user'] else None
    user_profile_location = tweet['user']['location'] if 'location' in tweet['user'] else None
    timestamp = tweet['timestamp_ms']
    source = tweet['source']

    place = {'id': tweet['place']['id'], 'full_name': tweet['place']['full_name'], 'bounding_box': {
        'coordinates': tweet['place']['bounding_box']['coordinates']}, 'place_type': tweet['place']['place_type'], 'country_code': tweet['place']['country_code']}
    data_record = {'id': tweet_id,
                   'raw_tweet': raw_tweet,
                   'norm_tweet': norm_tweet,
                   'place': place,
                   'state_code': state_code,
                   'user_id': user_id,
                   'user_display_name': user_display_name,
                   'user_screen_name': user_screen_name,
                   'user_profile_location': user_profile_location,
                   'timestamp': timestamp,
                   'source': source}

    return data_record


def _set_file(file_index, gran):
    gran_path = "data/" + gran
    info_path = gran_path + "/infos_test/"

    if not os.path.exists(info_path):
        os.mkdir(info_path)
    path = info_path + str(file_index) + ".tsv"
    file = open(path, 'a', encoding="utf-8", newline='')
    return file


def subset(tweet):
    return tweet['place']['full_name']


def save_info(gran):
    """ Store more information about the selected dataset for clustering """
    if gran not in {"states", "cities"}:
        raise ValueError(
            "'" + gran + "'" + " is invalid. Possible values are ('states' , 'cities')")

    data_path = "data/" + gran
    if not os.path.exists(data_path):
        raise Exception("Missing dataset data for " + gran +
                        "! Please run build_data() first.")
    elif len(os.listdir(data_path)) == 0:
        raise Exception("Missing dataset data for " + gran +
                        "! Please run build_data() first.")

    ids_file = open(data_path + "/tweet_ids.pickle", "rb")
    ids = pickle.load(ids_file)

    dataset_file = open(data_path + "/dataset.pickle", "rb")
    dataset = pickle.load(dataset_file)

    file_out = open(data_path + "/dataset_infos02.tsv",
                    'a', encoding="utf-8", newline='')
    writer = csv.DictWriter(file_out, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()

    files = [file for file in d._get_files(RAW_PATH)]
    for file in files:

        opened_file = open(file, 'r', encoding="utf-8")
        start_time = time()
        tweets = [json.loads(line) for line in opened_file]
        print("finshed loading tweets ", time() - start_time)

        start_time = time()
        tweets_rows = [_tweet_data(tweet, dataset[subset(tweet)][ids[subset(tweet)].index(
            tweet['id'])]) for tweet in tweets if subset(tweet) in ids if tweet['id'] in ids[subset(tweet)]]
        print("finshed filtering tweets", time() - start_time)

        start_time = time()
        writer.writerows(tweets_rows)
        print("finshed storing tweets", time() - start_time)
        # for line in opened_file:
        #     tweet = json.loads(line)
        #     #subset = tweet['place']['full_name']
        #     if subset in ids:
        #         if tweet['id'] in ids[subset]:
        #             id_index = ids[subset].index(tweet['id'])
        #             clean_tweet = dataset[subset][id_index]
        #             writer.writerow(_tweet_data(tweet, clean_tweet))
        #             # if len(tweets_rows) % 1000000 == 0:
        #             # writer.writerows(tweets_rows)
        #             # print(len(tweets_rows))
        #             #tweets_rows = []
        #             # file_out.close()
        #             #file_index += 1
        #             #file_out = _set_file(file_index, gran)

        time_elapsed = time() - start_time
        print("file time done: ", time_elapsed, " sec.")

    print("Dataset TSV infos stored in ", os.path.abspath(
        data_path + "/dataset_infos.tsv"))


if __name__ == "__main__":
    save_info('cities')
