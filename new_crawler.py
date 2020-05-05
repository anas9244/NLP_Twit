import json
import os.path
import tweepy
import glob
import os
import sys
# import langdetect
import datetime
print("crawling..")
sys.stdout.flush()
access_token = "588509788-1z07dMNOMhOCw4OyLJwxSfse31TR7Aywj6h2uZgd"
access_token_secret = "WPlTT3cpVXREXWuGvjdbxD8ie92e61vadWFlzxcoesjHe"
consumer_key = "Em7YjncUOkyjxzZhP3hWWUDJL"
consumer_secret = "IXkMkxVh1eFq9FJpo5vjI1NsTlzAscsEezVRjxhHZIBAnJiaEO"


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Dir of jsons files in relation to the current file path
fileDir = os.path.dirname(os.path.abspath(__file__)) + "/json/"

new_fileDir = "/media/data/new_tweets/"


# Set the name of the next file accrding to the last stored file in the json dir; if empty start with 0
def start_file():
    list_of_files = glob.glob(new_fileDir + "/*")
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        start_file_name = int(os.path.basename(
            os.path.splitext(latest_file)[0])) + 1
    else:
        start_file_name = 0
    return (start_file_name)


def set_file(file_index):

    path = new_fileDir + str(file_index) + ".jsonl"
    file = open(path, 'a', encoding="utf-8")
    return file


whitelist = [
    "Twitter for iPhone",
    "Twitter for Android",
    "Twitter Web Client",
    "Twitter for iPad"]

insta_source = "Instagram"


# Check the validity of the comming tweets; allow only those has location tags (geo and non-geo) from white-listed sources
def valid_chk(status):
    valid = False
    if (status.place is not None):
        if (status.source in whitelist):
            valid = True
            # if status.source == insta_source:
            #     if status.text.startswith("Just posted a"):
            #         valid = False
            #     else:
            #         valid = True
            # else:
            #     valid = True
        else:
            valid = False
    return valid


# Write status of crawling periodically and on exceptions to a logfile for email notification
def log_stuff(msg):
    with open("log.txt", 'w') as file:
        file.write(msg)


class TwitterListener(tweepy.StreamListener):

    def __init__(self):
        super(TwitterListener, self).__init__()
        self.num_tweets = 0
        self.valid = False

        self.file_index = start_file()
        self.file = set_file(self.file_index)

    def on_status(self, status):
        # print  (status._json['source'])
        try:
            if valid_chk(status):

                self.num_tweets += 1

                # Write to new file every 1000K tweet
                if self.num_tweets % 100000 == 0:
                    currentDT = datetime.datetime.now()
                    # log_stuff("number of tweets: " +
                    # str(self.num_tweets) + ", Time: " + str(currentDT))

                    # Write last tweet out of 100K to the current file; otherwise it would be 99999
                    json.dump(status._json, self.file, ensure_ascii=False)

                    # Close current file
                    self.file.close()
                    # increment filename for next 100K tweets
                    self.file_index += 1
                    self.file = set_file(self.file_index)
                    #print(self.num_tweets)
                    sys.stdout.flush()

                # elif self.num_tweets <= 1000000:
                #     # print(status._json)
                else:
                    json.dump(status._json, self.file, ensure_ascii=False)
                    self.file.write("\n")

                # else:
                #     print("Finished!", self.num_tweets)
                #     return False
                return True
        except Exception as ex:
            pass
           # log_stuff("Unhandled exception from on_status: " + str(ex))

    def on_error(self, status_code):
        if status_code == 420:
            print(" rate limit is reached")
            with open("log.txt", 'w') as file:
                file.write("rate limit is reached")
            return False
        else:
            #log_stuff("status code error: " + str(status_code))

            print("Error: ", status_code)
            return True

    def on_exception(self, exception):
        print("Error:", exception)

        #log_stuff("on_exception error: " + str(exception))


#
listener = TwitterListener()
twitterStream = tweepy.Stream(auth, listener)


try:
    twitterStream.filter(locations=[-180, -80, 180, 80],
#locations=[-124, 25, -66, 49, -168, 54, -141, 71, -163, 16, -151, 23],
                         languages=['en'], stall_warnings=True)
except Exception as ex:
    pass
    #log_stuff("Unhandled exception: " + str(ex))
