# -*- coding: utf-8 -*-
import os
import scipy.signal
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
import numpy as np
import tensorflow
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
import pandas as pd
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Queue
from sklearn.impute import KNNImputer
from ml_functions import decoupe_dataframe,EarlyStoppingByUnderVal,IdentityTransformer,attention_block


class Thread_train_model(threading.Thread):

    def __init__(self, model, q, x_train, y_train, nb_epochs, nb_batch, name_ts, name='', verbose=0,seq2seq=False):
        threading.Thread.__init__(self)
        self.name = name
        self.model = model
        self.cnn=False
        self.seq2seq=seq2seq
        self.x_train = x_train
        self.y_train = y_train
        self.nb_epochs = nb_epochs
        self.nb_batch = nb_batch
        self.name_ts = name_ts
        self.verbose = verbose
        self._stopevent = threading.Event()
        self.q = q
        print("The new thread with the name : " + self.name + " start running")

    def run(self):
        model = self.train_model(self.model, self.x_train, self.y_train, self.nb_epochs, self.nb_batch, self.name_ts)
        self.q.put(model)
        self.stop()
        print("The thread " + self.name + " ended ")
        return 0

    def stop(self):
        self._stopevent.set()

    def train_model(self, model, x_train, y_train, nb_epochs, nb_batch, name):
        '''
        Train the model 

        Parameters
        ----------
        model : Sequential object
            model.
        x_train : array
            training data inputs.
        y_train : array
            training data ouputs.
        nb_epochs : int.
            nb of training repetitions.
        nb_batch : int
            size of batch of element which gonna enter in the model before doing a back propagation.
        trend : bool
            Distinguish trend signal from others (more complicated to modelise).

        Returns
        -------
        model : Sequential object
            model.
        '''
        a=3
        if name == "trend":
            nb_epochs = self.nb_epochs * 2
            try:
                if self.cnn :
                    x_train = x_train.reshape((x_train.shape[0], a, int(x_train.shape[1]/a), 1))
                else :
                    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
                if self.seq2seq :
                    y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1]) 
            except Exception as e:
                if e:
                    print(e)
                    print(
                        "************Not enought data in the input compared to the look back size. Data has size :" + str(
                            np.shape(x_train)) + "*****")
            es = EarlyStopping(monitor='mse', mode='min', min_delta=0.01, patience=100)
            hist = model.fit(x_train, y_train, epochs=nb_epochs, batch_size=self.nb_batch, verbose=self.verbose,
                             callbacks=[es])
            i = 0
            while hist.history["loss"][-1] > 10 and i < 5:
                i = i + 1
                epochs = 50
                hist = model.fit(x_train, y_train, epochs=epochs, batch_size=100, verbose=self.verbose, callbacks=[es])
            print("model_trained")
        elif name == "residual":
            nb_epochs = self.nb_epochs * 5
            try:
                if self.cnn :
                    x_train = x_train.reshape((x_train.shape[0], a,int(x_train.shape[1]/a), 1))
                else :
                    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
                if self.seq2seq :
                    y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1]) 
            except Exception as e:
                if e:
                    print(e)
                    print(
                        "************Not enought data in the input compared to the look back size. Data has size :" + str(
                            np.shape(x_train)) + "*****")
            es = EarlyStopping(monitor='loss', mode='min', min_delta=0.1, patience=100)
            hist = model.fit(x_train, y_train, epochs=nb_epochs, batch_size=self.nb_batch, verbose=self.verbose,
                             callbacks=[es])
            i = 0
        else:
            nb_epochs = self.nb_epochs * 3
            es = EarlyStoppingByUnderVal(monitor="loss", value=0.1, verbose=self.verbose)
            try:
                if self.cnn :
                    x_train = x_train.reshape((x_train.shape[0], a,int(x_train.shape[1]/a), 1))
                else :
                    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
                if self.seq2seq :
                    y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1]) 
            except Exception as e:
                if e:
                    print(e)
                    print(
                        "***Not enought data in the input compared to the look back size. Data has size :" + str(
                            np.shape(x_train)) + "*****")
            model.fit(x_train, y_train, epochs=nb_epochs, batch_size=self.nb_batch, verbose=self.verbose)
            print("model_trained")
        return model
    