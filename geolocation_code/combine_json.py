import ujson as json
import build_data as d
import os

RAW_PATH = "/media/data/twitter_geolocation/tweets_clean/"


def _set_file(file_index):
    path = "/media/data/twitter_geolocation/"
    full_path = path + "/json_combined/"

    if not os.path.exists(full_path):
        os.mkdir(full_path)
    path = full_path + str(file_index) + ".ndjson"
    file = open(path, 'a', encoding="utf-8", newline='')
    return file


file_index = 0
file_out = _set_file(file_index)
files = [file for file in d._get_files(RAW_PATH)]

tweets_count = 0
for file in files:
    opened_file = open(file, 'r', encoding="utf-8")
    print(file)
    for line in opened_file:
        tweet = json.loads(line)
        tweets_count += 1
        if tweets_count % 1000000 == 0:
            print(tweets_count)
            json.dump(tweet, file_out, ensure_ascii=False)
            file_out.close()
            file_index += 1
            file_out = _set_file(file_index)
        else:
            json.dump(tweet, file_out, ensure_ascii=False)
            file_out.write('\n')

    # print(len(tweets))
    # if len(tweets) % 1000000 == 0:
    #     for tweet in tweets:
    #         json.dump(tweet, file_out, ensure_ascii=False)
    #         file_out.write('\n')
    #     file_out.close()
    #     file_index += 1
    #     file_out = _set_file(file_index)
    #     tweets = []
