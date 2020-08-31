# -*- coding: utf-8 -*-
import os
import scipy.signal
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  LSTM, Dropout
from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate, Input ,RepeatVector ,TimeDistributed,Conv1D,MaxPooling1D,Flatten
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
import xgboost as xgb
from xgboost import plot_importance, plot_tree
import pandas as pd
from datetime import datetime
import joblib
from piecewise.regressor import piecewise
import pywt
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Queue
from sklearn.impute import KNNImputer
from ml_functions import progressbar
from Thread_train_model import Thread_train_model
from ml_functions import decoupe_dataframe,EarlyStoppingByUnderVal,IdentityTransformer,attention_block


class Thread_genetic_train_model(threading.Thread):

    def __init__(self,ind,que,train_df,test_df,look_back,freq_period,verbose,name="genetic"):
        threading.Thread.__init__(self)
        self.ind=ind
        self.q=que
        self.train_df=train_df
        self.test_df=test_df
        self.look_back=look_back
        self.freq_period=freq_period
        self.verbose=verbose
        self._stopevent = threading.Event()
        print("The Father thread with the name : " + self.name + " start running")

    def run(self):
        score = self.fitness_ind(self.ind)
        self.q.put(score)
        self.stop()
        print("The Father thread " + self.name + " ended ")
        return 0

    def stop(self):
        self._stopevent.set()

    def fitness_ind(self, ind):
        Loss = ["mse"]*4
        self.learning_rate = ind[4]
        self.fit(self.train_df, self.look_back, self.freq_period, verbose=self.verbose, nb_epochs=int(ind[0]),
                 nb_batch=int(ind[2]), nb_layers=int(ind[1]), metric=Loss[int(ind[3])])
        pred = self.predict(steps=len(self.test_df["y"]), frame=False)
        #self.plot_prediction(self.test_df, pred)
        score = self.score(np.array(self.test_df["y"]), pred)
        print(score)
        return score
    
    
    def fit(self, df, look_back, freq_period, verbose=0, nb_features=1, nb_epochs=200, nb_batch=100, nb_layers=50,
            attention=True,cnn=False, loss="mape", metric="mse", optimizer="Adamax", directory=r"."):
        if not os.path.isdir(directory) and directory != r".":
            os.makedirs(directory)
        self.loss = loss
        self.cnn=cnn
        self.metric = metric
        self.optimizer = optimizer
        self.nb_features = nb_features
        self.nb_epochs = nb_epochs
        self.nb_batch = freq_period
        maxi = min(9000, len(df))
        self.df = df[-1 * maxi:]
        self.nb_layers = nb_layers
        self.freq_period = freq_period
        self.look_back = look_back
        self.verbose = verbose
        self.directory = directory
        trend_x, trend_y, seasonal_x, seasonal_y, residual_x, residual_y = self.prepare_data(df, look_back,
                                                                                             self.freq_period, first=1)
        if attention and not cnn :
            model_trend = self.make_models_with(True, self.df)
            model_seasonal = self.make_models_with(False, self.df)
            model_residual = self.make_models_with(False, self.df)
        elif cnn:
            model_trend = self.make_model_cnn(True, self.df)
            model_seasonal = self.make_model_cnn(False, self.df)
            model_residual = self.make_model_cnn(False, self.df)
        else:
            model_trend = self.make_models(True, self.df)
            model_seasonal = self.make_models(False, self.df)
            model_residual = self.make_models(False, self.df)
        que = queue.Queue()
        threads_list = list()
        thread = Thread_train_model(model_trend, que, trend_x, trend_y, nb_epochs, nb_batch, "trend", self.name +" Trend Thread",
                                    self.verbose)
        thread.start()
        threads_list.append(thread)
        thread_1 = Thread_train_model(model_seasonal, que, seasonal_x, seasonal_y, nb_epochs, nb_batch, "seasonal",
                                      self.name + " Seasonal Thread", self.verbose)
        thread_1.start()
        threads_list.append(thread_1)
        thread_2 = Thread_train_model(model_residual, que, residual_x, residual_y, nb_epochs, nb_batch, "residual",
                                      self.name+ " Residual Thread", self.verbose)
        thread_2.start()
        threads_list.append(thread_2)
        for t in threads_list:
            t.join()
        self.model_trend = que.get(block=True)
        self.model_seasonal = que.get(block=True)
        self.model_residual = que.get(block=True)
        print("Models fitted")
        
    def predict(self, steps=1, frame=False):
        prediction = self.make_prediction(steps)
        if frame:
            return prediction[0]
        else:
            return prediction[0]
        
    def make_prediction(self, len_prediction):
        '''
        This function make the prediction :
            reshape data to fit with the model
            make one prediction
            crush outdated data with prediction
            repeat len_prediction times
        Parameters
        ----------
        len_prediction : int
            number of value we want to predict.
        Returns
        -------
        prediction : array
            values predicted.
        pred_trend : array
            values trend predicted (recherches prupose).
        pred_seasonal : array
            values seasonal predicted (recherches prupose).
        pred_residual : array
            values residual predicted (recherches prupose).
        '''
        data_residual = self.residual[-1 * self.look_back:]
        data_trend = self.trend[-1 * self.look_back:]
        data_seasonal = self.seasonal[-1 * self.look_back:]
        prediction = np.zeros((1, len_prediction))
        for i in progressbar(range(len_prediction), "Computing: ", 40):
            dataset = np.reshape(data_residual, (1, 1, self.look_back))
            prediction_residual = (self.model_residual.predict(dataset))
            data_residual = np.append(data_residual[1:], [prediction_residual]).reshape(-1, 1
                                                                                        )
            dataset = np.reshape(data_trend, (1, 1, self.look_back))
            prediction_trend = (self.model_trend.predict(dataset))
            data_trend = np.append(data_trend[1:], [prediction_trend]).reshape(-1, 1)

            dataset = np.reshape(data_seasonal, (1, 1, self.look_back))
            prediction_seasonal = (self.model_seasonal.predict(dataset))
            data_seasonal = np.append(data_seasonal[1:], [prediction_seasonal]).reshape(-1, 1)

            prediction[0, i] = prediction_residual + prediction_trend + prediction_seasonal
        prediction = self.scaler2.inverse_transform(np.reshape(prediction, (-1, 1)))       
        return np.reshape(prediction, (1, -1))
    
    def prepare_data(self, df, look_back, freq_period, first=0):
        '''
        Parameters
        ----------
        df : DataFrame
            datafrmae contening historical data .
        look_back : int
            length entry of the model .
        Decompose the signal in three sub signals, trend,seasonal and residual in order to work separetly on each signal
        Returns
        -------
        trend_x : array
             values of the trend of the signal, matrix of dimention X array of dimension (1,length entry of model) X= length(dataframe)/look_back.
        trend_y : array
            vaklues to be predicted during the training
        seasonal_x : array
            same as trend_x but with the seasonal part of the signal.
        seasonal_y : array
            same as trend_y but with the seasonal part of the signal.
        residual_x : array
            same as trend_x but with the residual part of the signal.
        residual_y : array
            same as trend_y but with the residual part of the signal.
        '''
        imputer = KNNImputer(n_neighbors=2, weights="uniform")
        df["y"]=imputer.fit_transform(np.array(df["y"]).reshape(-1, 1))
        if look_back%2==0:
            window=freq_period+1
        else:
            window=freq_period
       # plt.plot(df["y"])
       # plt.show()
        #df["y"] = scipy.signal.savgol_filter(df["y"], window, 3)
       # plt.plot(df["y"])
       # plt.show()
        scalerfile = self.directory + '/scaler_pred.sav'
        if not os.path.isfile(scalerfile) or os.path.isfile(scalerfile) and first == 1:
            if (df["y"].max() - df["y"].min()) > 100:
                if self.verbose == 1:
                    print("PowerTransformation scaler used")
                scaler = PowerTransformer()
            else:
                if self.verbose == 1:
                    print("Identity scaler used")
                scaler = IdentityTransformer()
            self.scaler2 = scaler.fit(np.reshape(np.array(df["y"]), (-1, 1)))
            Y = self.scaler2.transform(np.reshape(np.array(df["y"]), (-1, 1)))
            pickle.dump(self.scaler2, open(scalerfile, 'wb'))
        elif os.path.isfile(scalerfile) and first == 0:
            self.scaler2 = pickle.load(open(scalerfile, "rb"))
            Y = self.scaler2.transform(np.reshape(np.array(df["y"]), (-1, 1)))
        if freq_period % 2 == 0:
            freq_period = freq_period + 1
        decomposition = STL(Y, period=freq_period)
        decomposition = decomposition.fit()
        df.loc[:, 'trend'] = decomposition.trend
        df.loc[:, 'seasonal'] = decomposition.seasonal
        df.loc[:, 'residual'] = decomposition.resid
        df_a = df
        df = df.dropna()
        df = df.reset_index(drop=True)
        df["trend"] = df["trend"].fillna(method="bfill")
        self.trend = np.asarray(df.loc[:, 'trend'])
        self.seasonal = np.asarray(df.loc[:, 'seasonal'])
        self.residual = np.asarray(df.loc[:, 'residual'])
        trend_x, trend_y = decoupe_dataframe(df["trend"], look_back)
        seasonal_x, seasonal_y = decoupe_dataframe(df["seasonal"], look_back)
        residual_x, residual_y = decoupe_dataframe(df["residual"], look_back)
        if self.verbose == 1:
            print("prepared")
        return trend_x, trend_y, seasonal_x, seasonal_y, residual_x, residual_y
    
    def make_models_with(self, trend, df):
        '''
        Create an LSTM model depending on the parameters selected by the user
        Parameters
        ----------
        nb_layers : int
            nb of layers of the lstm model.
        loss : str
            loss of the model.
        metric : str
            metric to evaluate the model.
        nb_features : int
            size of the ouput (one for regression).
        optimizer : str
            gradient descend's optimizer.
        trend : bool
              Distinguish trend signal from others (more complicated to modelise).
        Returns
        -------
        model : Sequential object
            model
        '''
        i = Input(shape=(self.nb_features, self.look_back))
        x = LSTM(self.nb_layers, return_sequences=True, activation='relu',
                 input_shape=(self.nb_features, self.look_back))(i)
        x = Activation("relu")(x)
        x = Dropout(0.2)(x)
        x = LSTM(self.nb_layers, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        if not trend:
            x = attention_block(x)
        x = Dense(int(self.nb_layers / 2), activation='relu')(x)
        output = Dense(1)(x)
        model = Model(inputs=[i], outputs=[output])
        optimizer = tensorflow.keras.optimizers.Adamax(lr=self.learning_rate)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.metric])
        if self.verbose == 1:
            print(model.summary())
        return model
    def score(self, prediction, real_data):
        '''
        This function returns the mean squared error
        Parameters
        ----------
        prediction : array
            predicted values.
        real_data : array
            real data.
        Returns
        -------
        None.
        '''
        from sklearn.metrics import mean_squared_error as mse
        self.mse = mse(np.reshape(np.array(real_data), (1, -1)), np.reshape(prediction, (1, -1)))
        return self.mse
 
    
class Seq2seq():
       
    
    def __init__(self,look_back,y,directory,len_pred=2):

        print("New Seq2Seq is being created")
        self.look_back=look_back
        
        self.directory=directory

        self.model=self.make_model(look_back,len_pred)
        
        self.len_pred=len_pred

        self.scaler_fitted,self.history=self.train_model(y,self.look_back,self.model,len_pred)

        self.model_save(self.model,self.scaler_fitted)

    def decoupe_dataframe(self,y, look_back,len_pred):
        dataX, dataY = [], []
        for i in range(len(y) - look_back - 1):
            a = y[i:(i + look_back)]
            dataY = dataY + [y[i + look_back:i+look_back+len_pred]]
            dataX.append(a)
        return (np.asarray(dataX), np.asarray(dataY).flatten())

    def make_model(self, look_back,len_pred):

        model = Sequential()

        model.add(LSTM(units=look_back, input_shape=(1, look_back), return_sequences=True))

        model.add(Dropout(0.3))

        model.add(LSTM(units=int(look_back/2), input_shape=(1, look_back)))

        model.add(RepeatVector(n=1))

        model.add(LSTM(units=int(look_back/2), return_sequences=True))

        model.add(Dropout(0.3))

        model.add(LSTM(units=look_back, input_shape=(1, look_back), return_sequences=True))

        model.add(TimeDistributed(Dense(units=len_pred)))

        model.compile(loss='mse', optimizer='adam')

        return model


    def train_model(self, y, look_back, model,len_pred):

        scaler = MinMaxScaler()


         #Y = scaler.fit_transform(np.reshape(np.array(y),(-1,1)))
        

        x_train, y_train = decoupe_dataframe(y, look_back)

        print(np.shape(x_train))

        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])

        history = model.fit(

            x_train, y_train,

            epochs=1000,

            batch_size=32,

            validation_split=0.1,

            shuffle=False

        )

        return scaler, history
    
    def make_prediction(self,x_real):
        
        model = self.model
        #scaler = self.scaler
        #Y = scaler.transform(np.reshape(np.array(x_real), (-1, 1)))
        x_real,_=self.decoupe_dataframe(x_real,self.look_back,self.len_pred)
        x_real = x_real.reshape(x_real.shape[0], 1, x_real.shape[1])
        prediction=model.predict(x_real)
        return prediction


    def model_save(self, model,scaler, name="./"):
        path = self.directory + "/" + name + ".h5"
        if name =="var" :
            with open(self.directory+'/params_var_model.txt', 'w') as txt_file:
                    txt_file.write(str(self.look_back))
                    print("saved")

        #scalerfile = self.directory + "/" +'scaler.pkl'
        #joblib.dump(scaler, scalerfile)
        save_model(model, path)