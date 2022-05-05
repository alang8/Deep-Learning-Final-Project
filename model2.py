from preprocess2 import *
from base64 import decode
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
import os

# ensures that we run only on cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
		

def main():
	train_inputs, train_labels, test_inputs, test_labels, all_inputs, all_labels = get_data()

	#m = Model(20000)
	vocab_size = 20000
	
	#batch_size = 100 
	embedding_size = 100 
	#optimizer = tf.keras.optimizers.Adam()
	hidden_size = 100
	num_units = 100 # lstm units


	m = Sequential([tf.keras.layers.Embedding(vocab_size, embedding_size),
				tf.keras.layers.LSTM(num_units),
				tf.keras.layers.Dense(hidden_size, activation='sigmoid'),
				tf.keras.layers.Dense(1, activation='sigmoid')])

	#m.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
	m.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
	print(m.summary())

	history = m.fit(all_inputs, all_labels, validation_split=0.2, epochs=1, batch_size=1000)
	


	# # # # TODO: Set up the testing steps

	# train(m, train_inputs, train_labels)


	# print(test(m, test_inputs, test_labels))



if __name__ == "__main__":
	main()
