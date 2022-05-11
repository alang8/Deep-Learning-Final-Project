from multiprocessing import Condition
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# this implementation does binary classification (positive vs negative)

def only_alpha(text):
    "Makes it so that the string entered only returns ascii, alphabetically characters or spaces"
    alpha = "".join([i for i in text if (i.isalpha() and i.isascii()) or i == ' '])
    return alpha.lower()


def get_data():
    webmd = pd.read_csv("webmd.csv") #362806 rows

    reviews = webmd["Reviews"].to_list()

    # average sentence is 75-100 characters, filters out reviews that are too short
    review_list = []

    for review in reviews:
        if (type(review) is str) and (len(list(review)) >= 75):    
            review_list += [review]

    # makes it so there's only initial_example number of reviews left
    review_list = review_list

    # filters webmd so there's only values with review_list in it
    webmd = webmd[webmd.Reviews.isin(review_list)]

    # adding ease of use, effectiveness, satisfaction together
    scores =  np.array(webmd['EaseofUse'].to_list()) + np.array(webmd['Effectiveness'].to_list()) + np.array(webmd['Satisfaction'].to_list())

    # classifying the scores to negative and positive
    # score is 0 if less than 9 (negative), 1 if greater than equal to 9 (positive)

    score = np.where(scores < 9, 0, 1)

    # just to make sure that reviews are in the same order as the scores
    reviews = webmd['Reviews'].to_list()

    reviews = list(map(lambda x: re.sub(r"\s+", " ", x), reviews)) # replaces tabs and newlines with spaces
    reviews = list(map(lambda x: re.sub(' +', ' ', x), reviews)) # removes repeating spaces

    # adjust review so that it only includes alphabetical characters
    reviews = list(map(only_alpha, reviews))

    # tokenizes review
    tokenizer = Tokenizer(num_words = 20000, oov_token="oov")
    tokenizer.fit_on_texts(reviews)

    # encodes and pads review
    encoded_docs = tokenizer.texts_to_sequences(reviews)
    padded_sequence = pad_sequences(encoded_docs, maxlen=200)

    return (padded_sequence, score, tokenizer)


def main():
    get_data()

if __name__ == "__main__":
    main()