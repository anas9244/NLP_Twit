# import tweepy

# #import json
# access_token = "588509788-1z07dMNOMhOCw4OyLJwxSfse31TR7Aywj6h2uZgd"
# access_token_secret = "WPlTT3cpVXREXWuGvjdbxD8ie92e61vadWFlzxcoesjHe"
# consumer_key = "Em7YjncUOkyjxzZhP3hWWUDJL"
# consumer_secret = "IXkMkxVh1eFq9FJpo5vjI1NsTlzAscsEezVRjxhHZIBAnJiaEO"
# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_token_secret)


# class Listener(tweepy.StreamListener):
#     def on_status(self, status):
#         print(status._json['text'])


# listener = Listener()
# twitterStream = tweepy.Stream(auth, listener)
# twitterStream.filter(track=['trump'])


# print(tweepy.__file__)

import os


# def prepend(list, str):

#     # Using format()
#     str += '{0}'
#     list = [str.format(i) for i in list]
#     return(list)


# def get_files():

#     fileDir = os.path.dirname(__file__)
#     path_dir = fileDir + "/json/"
#     files = os.listdir(path=path_dir)
#     files_paths = prepend(files, path_dir)

#     files_paths.sort(key=os.path.getmtime)

#     return(files_paths)


# print(get_files())
# fileDir = os.path.dirname(os.path.abspath(__file__))

# print(__main__)


import smtplib
import string
import time

path="nohup.out"
def follow(thefile):
    thefile.seek(0, 2)
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1)
            continue
        else:
            return line


logfile = open(path, "r", encoding="utf-8")

error = follow(logfile)


def sendemail(from_addr, to_addr_list,
              subject, message,
              login, password,
              smtpserver='smtp.gmail.com:587'):
    header = 'From: %s\n' % from_addr
    header += 'To: %s\n' % ','.join(to_addr_list)
    header += 'Subject: %s\n\n' % subject
    message = header + message

    server = smtplib.SMTP(smtpserver)
    server.starttls()
    server.login(login, password)
    problems = server.sendmail(from_addr, to_addr_list, message)
    server.quit()
    return problems


change_date = os.stat(path)[8]
while True:
    #change_date = os.stat(logfile)[8]

    if change_date != os.stat(path)[8]:
        print("changed, sent email")

        change_date = os.stat(path)[8]
        with open(path) as hupfile:
            content = hupfile.read().replace('\n', ' ')

    # if error:

        sendemail(from_addr='anasnayef1@gmail.com',
                  to_addr_list=['anas.alnayef@uni-weimar.de'],
                  subject='crawler update',
                  message=content,
                  login='anasnayef1@gmail.com',
                  password='Yeje_9244')
        with open("log.txt", "w") as file:
            file.truncate(0)

        #error = follow(logfile)
    time.sleep(10)
