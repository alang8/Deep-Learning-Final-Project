import math
import numpy as np
import tensorflow as tf
from preprocess import get_data

class Feed_Forwards(tf.keras.layers.Layer):
	def __init__(self, emb_sz, ff_dim):
		super(Feed_Forwards, self).__init__()
		self.layer_1 = tf.keras.layers.Dense(ff_dim, activation='relu')
		self.layer_2 = tf.keras.layers.Dense(emb_sz)

	@tf.function
	def call(self, inputs):
		layer_1_out = self.layer_1(inputs)
		layer_2_out = self.layer_2(layer_1_out)
		return layer_2_out

class Transformer_Block(tf.keras.layers.Layer):
    def __init__(self, emb_sz, num_heads, ff_dim, rate=0.3):
        super(Transformer_Block, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=emb_sz)
        self.ff_layer = Feed_Forwards(emb_sz, ff_dim)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    @tf.function
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(inputs + attn_output)
        ffl_output = self.ff_layer(out1)
        ffl_output = self.dropout2(ffl_output, training=training)
        return self.layer_norm2(out1 + ffl_output)

class Position_Encoding_Layer(tf.keras.layers.Layer):
	def __init__(self, emb_sz):
		super(Position_Encoding_Layer, self).__init__()
		self.token_embeddings = tf.keras.layers.Embedding(emb_sz, emb_sz)
		self.pos_embeddings = tf.keras.layers.Embedding(emb_sz, emb_sz)

	@tf.function
	def call(self, x):
		positional_encoding = self.positional_embeddings(x)
		token_encoding = self.token_embeddings(x)
		return positional_encoding + token_encoding

class Decoding_Layer(tf.keras.layers.Layer):
	def __init__(self, emb_sz, num_heads, ff_dim, rate=0.3):
		super(Decoding_Layer, self).__init__()
		self.decoder_layer = Transformer_Block(emb_sz, num_heads, ff_dim)
		self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.dropout = tf.keras.layers.Dropout(rate)

def train(model, train_inputs, train_labels, pad_id):
	for i in range(0, len(train_inputs), model.batch_size):
		input_batch = train_inputs[i:i+model.batch_size]
		label_batch = train_labels[i:i+model.batch_size]
		mask = np.where(label_batch == pad_id, 0, 1)
		mask = tf.convert_to_tensor(mask, dtype=tf.float32)
		if (len(train_labels) != model.batch_size):
			return
		with tf.GradientTape() as tape:
			probabilities = model.call(input_batch, training=True)
			loss = model.loss(probabilities, label_batch, mask)
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels, pad_id):
	total_loss = 0
	total_accuracy = 0
	total_non_padded = 0
	for i in range(0, len(test_inputs), model.batch_size):
		input_batch = test_inputs[i:i+model.batch_size]
		label_batch = test_labels[i:i+model.batch_size]
		mask = np.where(label_batch == pad_id, 0, 1)
		non_padded = np.sum(mask)
		mask = tf.convert_to_tensor(mask, dtype=tf.float32)
		total_non_padded += non_padded
		if (len(test_labels) != model.batch_size):
			return
		probabilities = model.call(input_batch, training=False)
		loss = model.loss(probabilities, label_batch, mask)
		total_loss += loss
		total_accuracy += model.accuracy(probabilities, label_batch, mask)
	return math.exp(total_loss / total_non_padded), (total_accuracy / total_non_padded)	

class Model(tf.keras.Model):
	def __init__(self, ):
		super(Model, self).__init__()
		self.batch_size = 100
		self.embedding_size = 1000
		self.num_heads = 10
		self.ff_dim = 2000
		
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

		self.feed_forwards = Feed_Forwards(self.embedding_size, self.ff_dim)
		self.encoder_layer = Position_Encoding_Layer(self.embedding_size)
		self.decoder_layer = Decoding_Layer(self.embedding_size, self.num_heads, self.ff_dim)

	@tf.function
	def call(self, encoder_input):
		encoder_output = self.encoder_layer(encoder_input)
		decoder_output = self.decoder_layer(encoder_output)
		ffl_output = self.feed_forwards(decoder_output)
		return tf.nn.softmax(ffl_output)

	def accuracy_function(self, prbs, labels, mask):
		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy

	def loss_function(self, prbs, labels, mask):
		loss = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs)
		loss = tf.boolean_mask(loss, mask)
		loss = tf.reduce_sum(loss)
		return loss

	def __call__(self, *args, **kwargs):
		return super(Model, self).__call__(*args, **kwargs)

def main():
	epochs = 10

	model = Model()

	data = get_data()
	train_inputs = data[0]
	train_labels = data[1]
	test_inputs = data[2]
	test_labels = data[3]
	pad_id = data[4]
	word_id = data[5]
	
	for i in range(0, epochs):
		train(model, train_inputs, train_labels, pad_id)

	print(test(model, test_inputs, test_labels, pad_id))

if __name__ == '__main__':
    main()