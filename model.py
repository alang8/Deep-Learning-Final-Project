import numpy as np
import tensorflow as tf

class Feed_Forwards(tf.keras.layers.Layer):
	def __init__(self, emb_sz, ff_dim):
		super(Feed_Forwards, self).__init__()

		self.layer_1 = tf.keras.layers.Dense(ff_dim,activation='relu')
		self.layer_2 = tf.keras.layers.Dense(emb_sz)

	@tf.function
	def call(self, inputs):
		"""
		This functions creates a feed forward network as described in 3.3
		https://arxiv.org/pdf/1706.03762.pdf
		Requirements:
		- Two linear layers with relu between them
		:param inputs: input tensor [batch_size x window_size x embedding_size]
		:return: tensor [batch_size x window_size x embedding_size]
		"""
		layer_1_out = self.layer_1(inputs)
		layer_2_out = self.layer_2(layer_1_out)
		return layer_2_out

class Transformer_Block(tf.keras.layers.Layer):
    def __init__(self, emb_sz, num_heads, ff_dim, rate=0.1):
        super(Transformer_Block, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads = num_heads, key_dim = emb_sz)
        self.ff_layer = Feed_Forwards(emb_sz, ff_dim)
        #episilon is 1e-3 by defaut, use default?
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    @tf.function
    def call(self, inputs, training):
        """
		This functions calls a transformer block.
		There are two possibilities for when this function is called.
		    - if self.is_decoder == False, then:
		        1) compute unmasked attention on the inputs
		        2) residual connection and layer normalization
		        3) feed forward layer
		        4) residual connection and layer normalization
		    - if self.is_decoder == True, then:
		        1) compute MASKED attention on the inputs
		        2) residual connection and layer normalization
		        3) computed UNMASKED attention using context
		        4) residual connection and layer normalization
		        5) feed forward layer
		        6) residual layer and layer normalization
		If the multi_headed==True, the model uses multiheaded attention (Only 2470 students must implement this)
		:param inputs: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ]
		:context: tensor of [BATCH_SIZE x FRENCH_WINDOW_SIZE x EMBEDDING_SIZE ] or None
			default=None, This is context from the encoder to be used as Keys and Values in self-attention function
		"""
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(inputs + attn_output)
        ffl_output = self.ff_layer(out1)
        ffl_output = self.dropout2(ffl_output, training=training)
        return self.layer_norm2(out1 + ffl_output)

class Position_Encoding_Layer(tf.keras.layers.Layer):
	def __init__(self, window_sz, emb_sz):
		super(Position_Encoding_Layer, self).__init__()
		self.positional_embeddings = self.add_weight("pos_embed",shape=[window_sz, emb_sz])

	@tf.function
	def call(self, x):
		"""
		Adds positional embeddings to word embeddings.
		:param x: [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ] the input embeddings fed to the encoder
		:return: [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ] new word embeddings with added positional encodings
		"""
		return x+self.positional_embeddings

class Transformer_Seq2Seq(tf.keras.Model):
	def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):

		######vvv DO NOT CHANGE vvv##################
		super(Transformer_Seq2Seq, self).__init__()

		self.french_vocab_size = french_vocab_size # The size of the French vocab
		self.english_vocab_size = english_vocab_size # The size of the English vocab

		self.french_window_size = french_window_size # The French window size
		self.english_window_size = english_window_size # The English window size
		######^^^ DO NOT CHANGE ^^^##################


		# TODO:
		# 1) Define any hyperparameters
		# 2) Define embeddings, encoder, decoder, and feed forward layers

		# Define batch size and optimizer/learning rate
		self.batch_size = 100
		self.embedding_size = 200
		self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

		# Define English and French embedding layers:
		self.emb_e = tf.keras.layers.Embedding(self.english_vocab_size, self.embedding_size)
		self.emb_f = tf.keras.layers.Embedding(self.french_vocab_size, self.embedding_size)

		# Create positional encoder layers
		self.pos_e = Position_Encoding_Layer(self.english_window_size, self.embedding_size)
		self.pos_f = Position_Encoding_Layer(self.french_window_size, self.embedding_size)

		# Define encoder and decoder layers:
		self.encoder = Transformer_Block(self.embedding_size, False)
		self.decoder = Transformer_Block(self.embedding_size, True)

		# Define dense layer(s)
		self.ffl = tf.keras.layers.Dense(self.english_vocab_size)

	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to French sentences
		:param decoder_input: batched ids corresponding to English sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""

		# TODO:
		#1) Add the positional embeddings to French sentence embeddings
		#2) Pass the French sentence embeddings to the encoder
		#3) Add positional embeddings to the English sentence embeddings
		#4) Pass the English embeddings and output of your encoder, to the decoder
		#5) Apply dense layer(s) to the decoder out to generate probabilities

		f_input = self.pos_f(self.emb_f(encoder_input))
		output = self.encoder(f_input)
		e_input = self.pos_e(self.emb_e(decoder_input))
		output = self.decoder(e_input, output)
		output = self.ffl(output)
		probabilities = tf.nn.softmax(output)

		return probabilities

	def accuracy_function(self, prbs, labels, mask):
		"""
		DO NOT CHANGE
		Computes the batch accuracy

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy


	def loss_function(self, prbs, labels, mask):
		"""
		Calculates the model cross-entropy loss after one forward pass
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""

		# Note: you can reuse this from rnn_model.

		loss = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs)
		loss = tf.boolean_mask(loss, mask)
		loss = tf.reduce_sum(loss)

		return loss

	def __call__(self, *args, **kwargs):
		return super(Transformer_Seq2Seq, self).__call__(*args, **kwargs)