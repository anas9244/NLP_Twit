import pickle
import os
import csv


fieldnames = ['tweet_id', 'raw_tweet', 'norm_tweet',
              'place_name', 'place_location', 'place_type', 'state_code', 'user_id', 'user_display_name',
              'user_screen_name', 'user_profile_location']


def _get_stats_file(gran):
    dist_path = "data/" + gran

    if not os.path.exists(dist_path):
        print(
            "Missing dataset data! Please run build_data.py first.")
        exit()
    elif len(os.listdir(dist_path)) == 0:
        print(
            "Missing dataset data! Please run build_data.py first.")
        exit()

    else:
        stats_file = open(dist_path + "/stats.pickle", "rb")
        stats = pickle.load(stats_file)
        stats_file.close()

        print("Number of tweets: ",
              len(stats))

        return stats


def save_stats(gran):
    dist_path = "data/" + gran
    stats = _get_stats_file(gran)

    file_out = open(dist_path + "/stats.csv", 'w',
                    encoding="utf-8", newline='')

    writer = csv.DictWriter(
        file_out, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()

    for stat in stats:

        csv_row = {'tweet_id': stat['tweet_id'],
                   'raw_tweet': stat['raw_tweet'],
                   'norm_tweet': stat['norm_tweet'],
                   'place_name': stat['place_name'],
                   'place_location': stat['place_location'],
                   'place_type': stat['place_type'],
                   'state_code': stat['state_code'],
                   'user_id': stat['user_id'],
                   'user_display_name': stat['user_display_name'],
                   'user_screen_name': stat['user_screen_name'],
                   'user_profile_location': stat['user_profile_location']}
        writer.writerow(csv_row)

    file_path = os.path.abspath(dist_path + "/stats.csv")

    print(gran + " dataset stats stored in ", file_path)


if __name__ == "__main__":
    save_stats('cities')
