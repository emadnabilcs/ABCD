"""
Created on 10/10/2021
@author: Abdullah M. Alshanqiti (a.m.alshanqiti@gmail.com)
"""
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
import random as rn
import os
##
##
##
##
##
##
##
##
############################################################
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(0)
rn.seed(0)
tf.random.set_seed(0)
##
##
##
##
##
##
##
##
############################################################
class TransformerBlock(layers.Layer):
    
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        #
        #
        super(TransformerBlock, self).__init__()
        #
        #
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        #
        #
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        #
        #
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        #
        #
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        #
        #
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        #
        #
        out1 = self.layernorm1(inputs + attn_output)
        #
        #
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        #
        return self.layernorm2(out1 + ffn_output)
    
##
##
##
############################################################   
class mLayers():

    def __init__(self, Xtraining, isForNarrators=False):        
        #
        # If inputs are based on Matn       then Max_Tokens: 56205 & Max-sequence-length: 832
        # If inputs are based on Narrators  then Max_Tokens: 5266 & Max-sequence-length: 21
        max_tokens = 56205
        sequence_length = 832
        #
        #
        if isForNarrators:
            max_tokens = 5270
            sequence_length = 30
        #
        #
        LSTM_Dim = 64
        embed_dim = 128
        num_heads = 2  # Number of attention heads
        ff_dim = 32    # Hidden layer size in feed forward network inside transformer
        #
        #
        self.xIn = tf.keras.layers.Input(shape=(1,), dtype="string")    
        #
        #
        # Input Text-Vectorization layer        
        self.vectorize_layer = TextVectorization(
            # Any words outside of the max_tokens will be treated as "out of vocabulary" (OOV) tokens.
            max_tokens=max_tokens,
            # Output integer indices, one per string token
            output_mode="int",
            # Always pad or truncate to exactly this many tokens
            output_sequence_length=sequence_length, name="vectorizeLayer"
        )
        self.vectorize_layer.adapt(Xtraining)  
        #
        #
        # Embedding layer :
        # since there's an out-of-vocabulary (OOV) token that gets added to the vocab, we use (max_tokens + 1) .
        self.EmbLayer = tf.keras.layers.Embedding(max_tokens + 1, embed_dim, name="EmbLayer")
        #
        #   
        # Recurrent layer: #64 is the dimensionality of the output space.   
        self.lstmLayer = tf.keras.layers.LSTM(LSTM_Dim, name="lstmLayer")
        #
        #
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        #
        #
        # a standard fully-connected (Dense) layer
        self.standardDense = tf.keras.layers.Dense(ff_dim, activation=tf.nn.relu, name="standardDense")
        self.denseOuput = tf.keras.layers.Dense(3, activation=tf.nn.softmax, name="denseOuput")
        #
        #
        self.rnn = None
        
        

    def doBuild(self, TransformerLayers=False):
        
        self.rnn = keras.Sequential()      
        self.rnn.add(self.xIn)
        self.rnn.add(self.vectorize_layer)    
        self.rnn.add(self.EmbLayer)
        #
        #
        if TransformerLayers:
            self.rnn.add(self.transformer_block)
            self.rnn.add(layers.GlobalAveragePooling1D())
        else:
            self.rnn.add(self.lstmLayer)
        #
        #
        self.rnn.add(self.standardDense)
        self.rnn.add(self.denseOuput)  
        
        #
        #
        # Compiling ...         
        self.rnn.compile(
            optimizer='adam',
            loss='CategoricalCrossentropy',
            metrics=['accuracy'],
        )
        
        #
        #
        # End building the Rnn model
        #        
        return self.rnn
    
##
##
##
##
##
##
############################################################