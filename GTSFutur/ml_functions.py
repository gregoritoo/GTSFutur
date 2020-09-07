# -*- coding: utf-8 -*-


import os
import scipy.signal
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  LSTM, Dropout
from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate, Input ,RepeatVector ,TimeDistributed,Conv1D,MaxPooling1D,Flatten,Layer
from tensorflow.keras import Model
import numpy as np
import tensorflow
from matplotlib.widgets import SpanSelector
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import EarlyStopping, Callback
import sys
from sklearn.preprocessing import PowerTransformer ,MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
from statsmodels.tsa.seasonal import STL
import threading
import queue
from tensorflow.keras import backend as K
import xgboost as xgb
from xgboost import plot_importance, plot_tree
import pandas as pd
from datetime import datetime
import joblib
import pywt
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Queue
from sklearn.impute import KNNImputer


def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)

    def show(j):
        x = int(size * j / count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x), j, count))
        file.flush()

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    file.write("\n")
    file.flush()

class EarlyStoppingByUnderVal(Callback):
    '''
    Class to stop model's training earlier if the value we monitor(monitor) goes under a threshold (value)
    replace usual callbacks functions from keras
    '''

    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            print("error")

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array * 1

    def inverse_transform(self, input_array, y=None):
        return input_array * 1



def decoupe_dataframe(df, look_back):
    dataX, dataY = [], []
    for i in range(len(df) - look_back - 1):
        a = df[i:(i + look_back)]
        dataY = dataY + [df[i + look_back]]
        dataX.append(a)
    return (np.asarray(dataX), np.asarray(dataY).flatten())



def attention_block(hidden_states):
    hidden_size = int(hidden_states.shape[2])
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return attention_vector

class attention(Layer):
    
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(attention,self).__init__()
        
    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        
        super(attention,self).build(input_shape)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
        })
        return config
        
    def call(self, x):
        
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        
        return K.sum(output, axis=1)
