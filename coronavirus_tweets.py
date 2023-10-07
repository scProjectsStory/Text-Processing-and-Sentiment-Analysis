# Part 3: Mining text data.

import random
import numpy as np
import csv
import re
import nltk
import urllib.request # for requesting a stop word
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer #Corpus Natural Language Processing Stem Processing
from sklearn.feature_extraction.text import CountVectorizer #vectorised processor
from sklearn.naive_bayes import MultinomialNB #Naive Bayes classifier
from sklearn.metrics import accuracy_score # Calculation classfication accuracy
import matplotlib.pyplot as plt # Drawing plot

# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_3(data_file):
	df = pd.read_csv(data_file, encoding='latin-1')  # Change read format to latin-1
	return df

# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
	sentiments_list = df['Sentiment'].unique().tolist()
	return sentiments_list

# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
	# Calculate the number of individual species in the Sentiment and get the index of the second row of data (the second ranked Sentiment)
	second_popular_sentiment = df['Sentiment'].value_counts().index[1]
	return second_popular_sentiment

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
	date = df[df['Sentiment'] == 'Extremely Positive']['TweetAt'].value_counts()
	return date.idxmax()

# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
	low_df = pd.DataFrame(df.OriginalTweet.str.lower())
	df['OriginalTweet'] = low_df['OriginalTweet']

# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
	df['OriginalTweet'] = pd.DataFrame(df.OriginalTweet.replace(to_replace=r'[^a-zA-Z]', value=' ', regex=True))

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
	df['OriginalTweet'] = pd.DataFrame(df.OriginalTweet.replace(to_replace=r' /s+ ', value=' ', regex=True))
# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
	df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: x.split())

# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
	with_repetitions_long = 0  # Variable for storing total length
	for i in tdf.OriginalTweet.values:
		with_repetitions_long = with_repetitions_long + len(i)
	return with_repetitions_long
# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
	without_repetitions_long = 0  # Variable for storing total length
	dic = {}  # staging dictionary Set non-repeating words as key, initialize repeat count to 0
	for i in tdf.OriginalTweet.values:
		for j in i:
			if j not in dic:
				dic[j] = 1
			else:
				dic[j] = dic[j] + 1
	without_repetitions_long = len(dic)
	return without_repetitions_long

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
	dic = {}  # Same as past func, staging dictionary Set non-repeating words as key, initialize repeat count to 0
	for i in tdf.OriginalTweet.values:
		for j in i:
			if j not in dic:
				dic[j] = 1
			else:
				dic[j] = dic[j] + 1
	temp_result = []
	Most_frequent_words = []  # Store the most commonly used words
	sorted_dic = sorted([(k, v) for k, v in dic.items()], reverse=True)  # Sort all elements by first letter
	sorted_value = set()  # define collections, store key-value pairs
	for i in sorted_dic:
		sorted_value.add(i[1])  # store the occurance number of each word in the sorted_dic
	# Iterate over all collection elements to sort and assign the top K key-value pairs in result
	for i in sorted(sorted_value, reverse=True)[:k]:
		for j in sorted_dic:
			# If the number of occurrences of the value of a key is equal to the number of occurrences of some word in the dictionary
			if j[1] == i:
				# then these words and their occurrences are stored in the temporary RESULT (to exclude juxtapositions)
				temp_result.append(j)
	for i in temp_result:
		Most_frequent_words.append(
			i[0])  # get the first letter of the key-value pair, i.e. the most frequently occurring word
	return Most_frequent_words

# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
	urllib.request.urlretrieve(
		'https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt', 'stop.txt')
	with open('stop.txt', 'r') as f:
		stop = set(f.read().split())
	# Use lambda to loop through all instances of tweets and keep only those strings that match the requirements
	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(
		lambda tweet: [i for i in tweet if (len(i) > 2 and i not in stop)])
	return tdf

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
	stemmer = PorterStemmer()
	tdf["OriginalTweet"] = tdf["OriginalTweet"].apply(lambda x: [stemmer.stem(i) for i in x])
	return tdf

# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
	vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(df['OriginalTweet'])
	y = df['Sentiment']
	clf = MultinomialNB()  # Polynomial NB Classifier
	clf.fit(X, y)
	y_pred = clf.predict(X)  # get predicted results
	return y_pred

# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
	temp = accuracy_score(y_true, y_pred)  # Accuracy_score method for score
	accuracy = round(temp, 3)
	return accuracy






