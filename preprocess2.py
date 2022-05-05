import csv
from multiprocessing import Condition
from pprint import pprint
import pandas as pd
from collections import Counter
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import functools



def only_alpha(text):
    "Makes it so that the string entered only returns ascii, alphabetically characters or spaces"
    alpha = "".join([i for i in text if (i.isalpha() and i.isascii()) or i == ' '])
    return alpha.lower()


def go_to_100(l):
    '''Makes it so that the list will always have a length of 100. If it's too large it'll be truncated.
    If it's too short then it will keep adding PADDING to the list until 100.'''

    if len(l) > 100:
        l = l[:100]
    else:
        l += ['PADDING'] * (100 - len(l))
    return l

def remove_empty(l):
    '''Removes  '' from the list. '''
    return list(filter(lambda x: x != '', l))

def get_data():
    webmd = pd.read_csv("webmd.csv") #362806 rows


    reviews = webmd["Reviews"].to_list()

    #Average sentence is 75-100 characters, filters out reviews that are too short
    review_list = []

    for review in reviews:
        if (type(review) is str) and (len(list(review)) >= 75):    
            review_list += [review]

    #print(len(review_list)) #271509 examples left

    #makes it so there's only initial_example number of reviews left
    review_list = review_list

    #filters webmd so there's only values with review_list in it
    webmd = webmd[webmd.Reviews.isin(review_list)]

    #Adding Ease of Use, Effectiveness, Satisfaction together
    score =  np.array(webmd['EaseofUse'].to_list()) + np.array(webmd['Effectiveness'].to_list()) + np.array(webmd['Satisfaction'].to_list())

    # classifying the scores to negative 0, neutral 1, and positive 2
    # score = np.where(score < 7, 0, score) #score is 0 if less than 7 (negative)
    # score = np.where(score > 11, 2, score) # score is 2 if greater than 11 (positive)
    # score = np.where(score > 6, 1, score) # if score is greater than 6 and less than 12 (neutral )

    score = np.where(score < 9, 0, 1) #score is 0 if less than 9 (negative), 1 if greater than equal to 9, (positive)


    #just to make sure that reviews are in the same order as the scores
    reviews = webmd['Reviews'].to_list()

    #print(len(reviews)) #271509

    # #makes it so it's only alphabetical characters
    reviews = list(map(only_alpha, reviews))

    reviews = reviews

    tokenizer = Tokenizer(num_words = 20000, oov_token="oov")
    tokenizer.fit_on_texts(reviews)

    encoded_docs = tokenizer.texts_to_sequences(reviews)

    padded_sequence = pad_sequences(encoded_docs, maxlen=200)

    #print(padded_sequence.shape) (271509, 200)  
    #print(score.shape)

    train_inputs = padded_sequence[:220000]
    train_labels = score[:220000]
    test_inputs = padded_sequence[220000:] #(51509, 200)
    test_labels = score[220000:] #(51509, )


    return (train_inputs, train_labels, test_inputs, test_labels, padded_sequence, score)

    # print(test_inputs.shape)
    # print(test_labels.shape)
    


    # reviews = [x.split(" ") for x in reviews]

    # reviews = list(map(remove_empty, reviews))

    # reviews = list(map(go_to_100, reviews))

    # #flattens the list of list to a list
    # words = functools.reduce(lambda x,y: x + y, reviews)


    # #dictionary that stores the translataion of word to id
    # word_id = {} 

    # count = 0

    # for word in words:
    #     word_id[word] = count
    #     count += 1

    # corpus_id = np.empty((0, 100), int)

    # for review in reviews:
    #     for i, w in enumerate(review):
    #         review[i] = word_id[w]
    #     corpus_id = np.append(corpus_id, np.array([review]), axis=0)
            
    # pad_id = word_id['PADDING']


    # train_inputs = corpus_id[:train_num]
    # train_labels = score[:train_num]

    # test_inputs = corpus_id[train_num:]
    # test_labels = score[train_num:]


    # # print(test_inputs.shape)

    # # print(train_inputs.shape)  #(10000, 100)
    # # print(train_labels.shape) #(10000, 100)
    # #print(test_inputs.shape) (5204, 100)
    # #print(test_labels.shape) (5204,)
    # #print(pad_id)
    # #print(word_id)

    # return (train_inputs, train_labels, test_inputs, test_labels, pad_id, word_id)


def main():
	get_data()

if __name__ == "__main__":
    main()