import pickle
import ujson as json
import os
import csv
import build_data as d
from datetime import datetime
from dateutil.parser import parse
from time import time
from tweets_infos import _tweet_data

RAW_PATH = "/media/data/json_combined/"

start = 391

tweet_count = 0


def _set_file(file_index):
    #gran_path = "data/" + gran
    path = "/media/data/twitter_geolocation/clean_tweets/"

    if not os.path.exists(path):
        os.mkdir(path)
    path = path + str(file_index) + ".ndjson"
    file = open(path, 'w', encoding="utf-8", newline='')
    return file


for index, file in enumerate(d._get_files(RAW_PATH)):
    file_index = index
    if index >= start:
        print(tweet_count)

        print(file)
        file_out = _set_file(file_index)

        opened_file = open(file, 'r', encoding="utf-8")
        for index, line in enumerate(opened_file):
            tweet = json.loads(line)
            try:

                if (tweet['source'] in d.source_whitelist) and tweet['place']['country_code'] == 'US' and tweet['place']['place_type'] in ('city', 'admin'):

                    norm_tweet = d._clean_text(tweet)

                    if norm_tweet:
                        tweet_count += 1

                        clean_tweet = _tweet_data(tweet, norm_tweet)
                        json.dump(clean_tweet, file_out, ensure_ascii=False)
                        file_out.write("\n")
            except Exception as e:
                print(e)
                print(line.encode())
