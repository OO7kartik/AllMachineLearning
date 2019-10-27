import tweepy
from textblob import TextBlob

consumer_key = 'XdQ22AAZwqmrOf9OzXAcxsq0o '
consumer_secret = '7Ugth0TWcUN2BU0XE6qtmJMGyuvlvvrQ10r5BY0VFsu1oGWp2C'

access_token = '1073792948695838720-TaJIbI0USFGbeBlBv7WXmluRU59SUY'
access_token_secret = 'so9ZfChqXcnmV420h3zSDVDG7vi59bBnulVnPvWGzrUOI'

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#Step 3 - Retrieve Tweets
public_tweets = api.search('Trump')



#CHALLENGE - Instead of printing out each tweet, save each Tweet to a CSV file
#and label each one as either 'positive' or 'negative', depending on the sentiment 
#You can decide the sentiment polarity threshold yourself


for tweet in public_tweets:
    print(tweet.text)
    
    #Step 4 Perform Sentiment Analysis on Tweets
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
    print("")