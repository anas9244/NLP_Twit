import ujson as json
import build_data as d
import os

files = d._get_files("/media/data/new_tweets/")


def _set_file(file_index):
    path = "/media/data/"
    full_path = path + "/tweets_for_training/"

    if not os.path.exists(full_path):
        os.mkdir(full_path)
    path = full_path + str(file_index) + ".ndjson"
    file = open(path, 'a', encoding="utf-8", newline='')
    return file


file_index = 0
file_out = _set_file(file_index)
files = [file for file in files]


tweets_count = 0

start = 38
end = 198
finined = False
for index, file in enumerate(files):
    if finined is True:
        break
    if index >= start:
        opened_file = open(file, 'r', encoding="utf-8")
        print(file)
        ids = set()
        for index, line in enumerate(opened_file):
            try:
                tweet = json.loads(line)
                if tweet['id'] not in ids:
                    ids.add(tweet['id'])

                    text = ""
                    if tweet['truncated']:
                        if "extended_tweet" in tweet:
                            text = tweet['extended_tweet']['full_text']
                        else:

                            print("no extended_tweet :(")
                            continue
                    else:
                        text = tweet['text']

                    tweets_count += 1

                    if tweets_count % 1000000 == 0:
                        print(tweets_count)

                        record = {'id': tweet['id'], 'text': text, 'place_name': tweet['place']['full_name'], 'place_type': tweet['place']
                                  ['place_type'], 'country': tweet['place']['country_code'], 'bounding_box': tweet['place']['bounding_box']}
                        json.dump(record, file_out, ensure_ascii=False)
                        file_out.close()
                        file_index += 1
                        file_out = _set_file(file_index)
                    else:
                        record = {'id': tweet['id'], 'text': text, 'place_name': tweet['place']['full_name'], 'place_type': tweet['place']
                                  ['place_type'], 'country': tweet['place']['country_code'], 'bounding_box': tweet['place']['bounding_box']}
                        json.dump(record, file_out, ensure_ascii=False)
                        file_out.write('\n')

                    if tweets_count >= 15000000:
                        finined = True
                        break
            except:
                # pass
                print(index)
