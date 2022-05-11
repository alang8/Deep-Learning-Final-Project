from multiprocessing import Condition
from pprint import pprint
import pandas as pd
from collections import Counter
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# this file does binary classification

def only_alpha(text):
    "Makes it so that the string entered only returns ascii, alphabetically characters or spaces"
    alpha = "".join([i for i in text if (i.isalpha() and i.isascii()) or i == ' '])
    return alpha.lower()

def get_data():
    webmd = pd.read_csv("webmd.csv") #362806 rows

    reviews = webmd["Reviews"].to_list()

    # Average sentence is 75-100 characters, filters out reviews that are too short
    review_list = []

    for review in reviews:
        if (type(review) is str) and (len(list(review)) >= 75):    
            review_list += [review]

    # print(len(review_list)) #271509 examples left

    # makes it so there's only initial_example number of reviews left
    review_list = review_list

    # filters webmd so there's only values with review_list in it
    webmd = webmd[webmd.Reviews.isin(review_list)]

    # Adding Ease of Use, Effectiveness, Satisfaction together
    scores =  np.array(webmd['EaseofUse'].to_list()) + np.array(webmd['Effectiveness'].to_list()) + np.array(webmd['Satisfaction'].to_list())

    # classifying the scores to negative 0, neutral 1, and positive 2
    # score = np.where(score < 7, 0, score) #score is 0 if less than 7 (negative)
    # score = np.where(score > 11, 2, score) # score is 2 if greater than 11 (positive)
    # score = np.where(score > 6, 1, score) # if score is greater than 6 and less than 12 (neutral )

    score = np.where(scores < 9, 0, 1) #score is 0 if less than 9 (negative), 1 if greater than equal to 9, (positive)

    # just to make sure that reviews are in the same order as the scores
    reviews = webmd['Reviews'].to_list()

    reviews = list(map(lambda x: re.sub(r"\s+", " ", x), reviews)) # replaces tabs/newlines with spaces
    reviews = list(map(lambda x: re.sub(' +', ' ', x), reviews)) # removes repeating spaces

    # makes it so it's only alphabetical characters
    reviews = list(map(only_alpha, reviews))

    ### This is just to see the reviews and for LIME ###
    # with open("reviews.txt", "w") as f:
    #     for review, s, sentiment  in zip(reviews, scores, score):
    #         f.write(review + '\n' + '\n')
    #         f.write("Score is " + s.astype(str) + '\n')

    #         if s < 7: 
    #             sent = "Negative"
    #         elif s > 11:
    #             sent = "Positive"
    #         else:
    #             sent = "Neutral"

    #         if sentiment == 1:
    #             sent2 = "Positive"
    #         else:
    #             sent2 = "Negative"

    #         f.write("Sentiment for Multi is " + sent + '\n')
    #         f.write("Sentiment for Binary is " + sent2 + '\n' + '\n')

    tokenizer = Tokenizer(num_words = 10000, oov_token="oov")
    tokenizer.fit_on_texts(reviews)

    encoded_docs = tokenizer.texts_to_sequences(reviews)

    padded_sequence = pad_sequences(encoded_docs, maxlen=200)

    return (padded_sequence, score, tokenizer)

def main():
    get_data()

if __name__ == "__main__":
    main()
