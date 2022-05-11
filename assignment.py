from venv import create
from preprocess import *
from base64 import decode
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
import os
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt

# pip install lime and pip install ipython

# ensures that we run only on cpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# this file does binary classification

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.vocab_size = 20000
        self.batch_size = 128 
        self.embedding_size = 128
        self.optimizer = tf.keras.optimizers.Adam()
        self.hidden_size = 128
        self.num_units = 128 # lstm units
    

def create_model():
    m = Model()
    model = Sequential([tf.keras.layers.Embedding(m.vocab_size, m.embedding_size),
            tf.keras.layers.LSTM(m.num_units),
            tf.keras.layers.Dense(m.hidden_size, activation='sigmoid'),
            tf.keras.layers.Dense(1, activation='sigmoid')])
    return model


def main():

    all_inputs, all_labels, tokenizer = get_data()
    dir_exists = os.path.isdir('binary_model')

    # if model has been created, load from local directory
    if not dir_exists:
        model = create_model()

        model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        h = model.fit(all_inputs, all_labels, validation_split=0.2, epochs=1, batch_size=1024) # 
        # loss: 0.4719 - accuracy: 0.7735 - val_loss: 0.4315 - val_accuracy: 0.8003
        model.save("binary_model")
    else:
        model = tf.keras.models.load_model("binary_model")


    def predict_proba(arr):
            
        # tokenizes and pads
        encoded_docs = tokenizer.texts_to_sequences(arr)
        padded_sequence = pad_sequences(encoded_docs, maxlen=200)

        pred = model.predict(padded_sequence)

        ret = [] 
        for i in pred:
            val = i[0]
            ret.append(np.array([1 - val, val]))

        return np.array(ret)


    class_names=['negative','positive']
    explainer= LimeTextExplainer(class_names=class_names)

    # counter = 0

    # with open('binary_reviews.txt', 'r') as f:
    #     r_list = f.readlines()

    # for i in range(0, 1000, 2):
    #     review = r_list[i]
        
    #     review = ' '.join(review.split())
    #     exp = explainer.explain_instance(review, predict_proba)
        
    #     html = "LIME_pictures/" + str(counter) + r_list[i + 1].strip() + ".html"
    #     exp.save_to_file(html)

    #     print(counter)

    #     counter += 1

    ### Lime Example ###
    x = 'my grandfather was prescribed this medication coumadin to assist in blood thinning due to a heart and thyroid condition his primary doctor was aware that he was on an aspirin regiment and still prescribed this medicine it caused his blood to thin out to much and he ended up internally bleeding to death if you are going to take this medicine please ask your doctors about possible side effects or drug interactions'
    x = ' '.join(x.split())
    exp = explainer.explain_instance(x, predict_proba)
    exp.save_to_file('lime1.html') # has to be html 

    ### This is just to check that the data loads properly. Can ignore ###
    # indicies = np.random.choice(len(all_inputs), 54302, replace=False)

    # test_inputs = all_inputs[indicies]
    # test_labels = all_labels[indicies]

    # model = tf.keras.models.load_model("binary_model")

    # score = model.evaluate(test_inputs, test_labels, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

if __name__ == "__main__":
    main()
