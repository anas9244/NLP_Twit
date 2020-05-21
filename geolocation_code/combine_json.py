import ujson as json
import build_data as d
import os

new_files = d._get_files("/media/data/new_tweets/")
#old_files = d._get_files("/media/data/twitter_geolocation/json/")
#combined_files = old_files + new_files


def _set_file(file_index):
    path = "/media/data/"
    full_path = path + "/json_combined/"

    if not os.path.exists(full_path):
        os.mkdir(full_path)
    path = full_path + str(file_index) + ".ndjson"
    file = open(path, 'a', encoding="utf-8", newline='')
    return file


start = 38
end = 198
last = 286

file_index = 391
file_out = _set_file(file_index)
files = [file for file in new_files]

tweets_count = 0

# ids=set()
for index, file in enumerate(files):
    if index < start or index > end:
        if index == 286:
            break

        opened_file = open(file, 'r', encoding="utf-8")
        print(file)
        ids = set()
        for index, line in enumerate(opened_file):
            try:
                tweet = json.loads(line)
                if tweet['id'] not in ids:
                    ids.add(tweet['id'])
                    tweets_count += 1
                    if tweets_count % 100000 == 0:
                        print(tweets_count)
                        json.dump(tweet, file_out, ensure_ascii=False)
                        file_out.close()
                        file_index += 1
                        file_out = _set_file(file_index)
                    else:
                        json.dump(tweet, file_out, ensure_ascii=False)
                        file_out.write('\n')
            except:
                # pass
                print(index)

    # print(len(tweets))
    # if len(tweets) % 1000000 == 0:
    #     for tweet in tweets:
    #         json.dump(tweet, file_out, ensure_ascii=False)
    #         file_out.write('\n')
    #     file_out.close()
    #     file_index += 1
    #     file_out = _set_file(file_index)
    #     tweets = []
