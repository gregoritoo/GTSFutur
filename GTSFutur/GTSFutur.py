# -*- coding: utf-8 -*-
import os
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
import scipy.signal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  LSTM, Dropout
from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate, Input ,RepeatVector ,TimeDistributed,Conv1D,MaxPooling1D,Flatten,BatchNormalization
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
from Thread_train_model import Thread_train_model
from Thread_genetic_train_model import Thread_genetic_train_model
from ml_functions import decoupe_dataframe,EarlyStoppingByUnderVal,IdentityTransformer,attention_block,attention
import xgboost as xgb
from xgboost import plot_importance, plot_tree
import pandas as pd
from datetime import datetime
import joblib
import pywt
from tensorflow.keras.optimizers import Adam
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Queue
from sklearn.impute import KNNImputer
import warnings 
from LSTM import LSTM_PAST
from statsmodels.tsa.holtwinters import ExponentialSmoothing
results = []



def sequence_dataframe(df,look_back,len_pred):
    '''
     Cut dataframe into training sequences 
    Parameters
    ----------
    df : array
        array with data.
    look_back : int
        size of lsmt input.
    len_pred : int
        length of prediction.

    Returns
    -------
    tuple
        X_train and Y_train.

    '''
    dataX, dataY = [], []
    for i in range(len(df) - look_back -len_pred- 1):
        a = df[i:(i + look_back)]
        b=df[(i + look_back):(i + look_back+len_pred)]
        dataY.append(b)
        dataX.append(a)
    return (np.asarray(dataX), np.asarray(dataY))

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



class GTSPredictor():

    def __init__(self, verbose=0):
        self.learning_rate = 0.001
        if verbose == 1:
            print("Model is being created")
        else:
            pass

    def fit_with_UI(self, df, verbose=0, nb_features=1, nb_epochs=300, nb_batch=32, nb_layers=50,
                    attention=True, loss="mape", metric="mse", optimizer="Adamax", directory=r"."):
        '''
        open a simple matplotlib user interface for selecting the period size  , make the preprocessing steps and train the models

        Parameters
        ----------
        df : dataframe
            datframe containing "y" column with the data .
        verbose : int, optional
            same verbose as keras. The default is 0.
        nb_features : int, optional
            dimension of input. The default is 1.
        nb_epochs : int, optional
           same as keras . The default is 300.
        nb_batch : int, optional
            same as keras . The default is 32.
        nb_layers : int, optional
            same as keras . The default is 50.
        attention : bool, optional
            use attention layer or not. The default is True.
        loss : str, optional
            choose the loss. The default is "mape".
        metric : str, optional
            choose the metric. The default is "mse".
        optimizer : str, optional
            choose the metric. The default is "Adamax".
        directory : str, optional
            choose the directory where the models will be saved. The default is r".".

        Returns
        -------
        None.

        '''
        if not os.path.isdir(directory) and directory != r".":
            os.makedirs(directory)
        np.random.seed(19680801)

        fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
        ax1.set(facecolor='#FFFFCC')

        x = np.arange(0, len(df["y"]), 1)
        y = np.array(df["y"])

        ax1.plot(x, y, '-')
        ax1.set_title('Press left mouse button and drag to select the pattern')

        ax2.set(facecolor='#FFFFCC')
        line2, = ax2.plot(x, y, '-')

        def onselect(xmin, xmax):
            indmin, indmax = np.searchsorted(x, (xmin, xmax))
            indmax = min(len(x) - 1, indmax)

            thisx = x[indmin:indmax]
            thisy = y[indmin:indmax]
            print(thisx.min(), thisx.max())
            line2.set_data(thisx, thisy)
            ax2.set_xlim(thisx[0], thisx[-1])
            ax2.set_ylim(thisy.min(), thisy.max())
            fig.canvas.draw()
            pattern_size = int(abs(thisx.min() - thisx.max())) + 1
            with open('pattern_size.txt', 'w') as f:
                f.write('%d' % pattern_size)

        span = SpanSelector(ax1, onselect, 'horizontal', useblit=True,
                            rectprops=dict(alpha=0.5, facecolor='red'))
        plt.show(block=True)
        plt.close()
        f = open('pattern_size.txt', 'r')
        freq_period = int(f.readline())
        f.close()
        os.remove('pattern_size.txt')
        print(freq_period)
        self.fit(df, int(freq_period * 1.3) + 1, freq_period, verbose=verbose, nb_features=nb_features,
                 nb_epochs=nb_epochs, nb_batch=nb_batch, nb_layers=nb_layers,
                 attention=attention, loss=loss, metric=metric, optimizer=optimizer, directory=directory)

    def fit_without(self, df, look_back, freq_period, verbose=0, nb_features=1, nb_epochs=200, nb_batch=100,
                    nb_layers=50,
                    attention=True, loss="mape", metric="mse", optimizer="Adamax", directory=r"."):
        '''
        This function take the data , make the preprocessing steps and train the models
        Parameters
        ----------
        df : dataframe
            with a time columns (string) and y the columns with the values to forecast.
        look_back : int
            size of inputs .
        freq_period : int
            size in point of the seasonal pattern (ex 24 if daily seasonality for a signal sampled at 1h frequency).
        verbose : 0 or 1
            If 0 no information printed during the creation and training of models.
        nb_features :int, optional
             output size. The default is 1.
        nb_epochs : int, optional
             The default is 50.
        nb_batch : int, optional
            DESCRIPTION. The default is 100.
        nb_layers : int, optional
             The default is 50.
        attention : Bool, optional
            Either use or not the attention layer. The default is True.
        loss : str, optional
            loss for evaluation error between input and output . The default is "mape".
        metric : str, optional
            metrics for evaluation the training predictions. The default is "mse".
        optimizer : str, optional
            Keras optimizers. The default is "Adamax".
        directory : str, optional
            directory path which need to end by "/". The default is r".".
        Returns
        -------
        None.
        '''
        if not os.path.isdir(directory) and directory != r".":
            os.makedirs(directory)
        self.loss = loss
        self.metric = metric
        self.optimizer = optimizer
        self.nb_features = nb_features
        self.nb_epochs = nb_epochs
        self.nb_batch = nb_batch
        self.df = df
        self.nb_layers = nb_layers
        self.freq_period = freq_period
        self.look_back = look_back
        self.verbose = verbose
        self.directory = directory
        trend_x, trend_y, seasonal_x, seasonal_y, residual_x, residual_y = self.prepare_data(df, look_back,
                                                                                             self.freq_period, first=1)
        if attention:
            model_trend = self.make_models_with(True, self.df)
            model_seasonal = self.make_models_with(False, self.df)
            model_residual = self.make_models_with(False, self.df)
        else:
            model_trend = self.make_models(True, self.df)
            model_seasonal = self.make_models(False, self.df)
            model_residual = self.make_models(False, self.df)
        self.model_trend = self.train_model(model_trend, trend_x, trend_y, "trend")
        self.model_seasonal = self.train_model(model_seasonal, seasonal_x, seasonal_y, "seasonal")
        self.model_residual = self.train_model(model_residual, residual_x, residual_y, "residual")
        print("Modele fitted")
        
    

    def fit(self, df, look_back, freq_period, verbose=0, nb_features=1, nb_epochs=200, nb_batch=100, nb_layers=64,
            attention=True, loss="mape", metric="mse", optimizer="Adamax", directory=r".",len_pred=1,seq2seq=False,deep_layers=2):
        '''
        Parameters
        ----------
        df : dataframe
            with a time columns (string) and y the columns with the values to forecast.
        look_back : int
            size of inputs .
        freq_period : int
            size in point of the seasonal pattern (ex 24 if daily seasonality for a signal sampled at 1h frequency).
        verbose : 0 or 1
            If 0 no information printed during the creation and training of models.
        nb_features :int, optional
             output size. The default is 1.
        nb_epochs : int, optional
             The default is 50.
        nb_batch : int, optional
            DESCRIPTION. The default is 100.
        nb_layers : int, optional
             The default is 50.
        attention : Bool, optional
            Either use or not the attention layer. The default is True.
        loss : str, optional
            loss for evaluation error between input and output . The default is "mape".
        metric : str, optional
            metrics for evaluation the training predictions. The default is "mse".
        optimizer : str, optional
            Keras optimizers. The default is "Adamax".
        directory : str, optional
            directory path which need to end by "/". The default is r".".
        len_pred : int, optional
            length of prediction is seq2seq model selected. The default is 1.
        seq2seq : bool, optional
            use seq2seq model or not . The default is False.
        deep_layers : int, optional
            number of LSTM layers. The default is 2.
        testing : bool, optional
            use the CNN-LSTM with manual attention. The default is False.

        Returns
        -------
        None.

        '''
        if len(df["y"]) < 1000 :
        	warnings.warn('Attention this function needs more data to be efficient. Please use fit_predict_ES instead') 
        	print('Attention this function needs more data to be efficient. Please use fit_predict_ES instead')
        if not os.path.isdir(directory) and directory != r".":
            os.makedirs(directory)
        self.loss = loss
        self.metric = metric
        self.optimizer = optimizer
        self.nb_features = nb_features
        self.nb_epochs = nb_epochs
        self.nb_batch = freq_period
        self.seq2seq=seq2seq
        maxi = min(9000, len(df))
        self.df = df[-1 * maxi :]
        self.deep_layers=deep_layers
        self.nb_layers = nb_layers
        self.freq_period = freq_period
        self.look_back = look_back
        self.verbose = verbose
        self.len_pred=len_pred
        self.directory = directory
        if not seq2seq :
            trend_x, trend_y, seasonal_x, seasonal_y, residual_x, residual_y = self.prepare_data(df, look_back,
                                                                                                 self.freq_period, first=1)
        else :
            trend_x, trend_y, seasonal_x, seasonal_y, residual_x, residual_y = self.prepare_data(df, look_back,
                                                                                                 self.freq_period, first=1, seq2seq=True)

        if seq2seq :
            model_trend =self.make_seq2seq_models(trend_x,trend_y)
            model_residual = self.make_seq2seq_models(trend_x,trend_y)
        elif attention  :
            model_trend = self.make_models_att(False, self.df)
            model_residual = self.make_models_att(False, self.df)
      
        else :
            model_trend = self.make_models_with(True, self.df)
            model_residual = self.make_models_with(False, self.df)
        que = queue.Queue()
        threads_list = list()
        if seq2seq :
            
            thread = Thread_train_model(model_trend, que, trend_x, trend_y, nb_epochs, nb_batch, "trend", "Trend Thread",
                                        self.verbose,seq2seq=True)
            thread.start()
            threads_list.append(thread)

            thread_2 = Thread_train_model(model_residual, que, residual_x, residual_y, nb_epochs, nb_batch, "residual",
                                          "Residual Thread", self.verbose,seq2seq=True)
            thread_2.start()
            threads_list.append(thread_2)
        else :
            thread = Thread_train_model(model_trend, que, trend_x, trend_y, nb_epochs, nb_batch, "trend", "Trend Thread",
                                        self.verbose)
            thread.start()
            threads_list.append(thread)

            thread_2 = Thread_train_model(model_residual, que, residual_x, residual_y, nb_epochs, nb_batch, "residual",
                                          "Residual Thread", self.verbose)
            thread_2.start()
            threads_list.append(thread_2)
        for t in threads_list:
            t.join()
        self.model_trend = que.get(block=True)
        self.model_save(self.model_trend, "trend")
        self.model_residual = que.get(block=True)
        self.model_save(self.model_residual, "residual")
        print("Models fitted")

    def predict(self, steps=1, frame=True, seq2seq=False):
        '''
        Launch prediction

        Parameters
        ----------
        steps : int , optional
            lenght of prediction. The default is 1.
        frame : bool, optional
            frame the prediction or not(if true the function will return 3 arrays). The default is True.
        seq2seq : bool, optional
            use seq2seq models . The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if not seq2seq :
            prediction, lower, upper = self.make_prediction(steps)
            if frame:
                return prediction[0], lower[0], upper[0]            
            else :
                return prediction[0]
        else :
            prediction,_,_=self.make_prediction_seq2seq()
            prediction=prediction[0]
            return prediction

    def predict_past(self, df, freq_period, steps):
        scalerfile = self.directory + '/scaler_pred.sav'
        if not os.path.isfile(scalerfile) or os.path.isfile(scalerfile):
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
        elif os.path.isfile(scalerfile):
            self.scaler2 = pickle.load(open(scalerfile, "rb"))
            Y = self.scaler2.transform(np.reshape(np.array(df["y"]), (-1, 1)))
        if freq_period % 2 == 0:
            freq_period = freq_period + 1
        decomposition = STL(Y, period=freq_period + 1)
        decomposition = decomposition.fit()
        decomposition.plot()
        plt.show()
        df.loc[:, 'trend'] = decomposition.trend
        df.loc[:, 'seasonal'] = decomposition.seasonal
        df.loc[:, 'residual'] = decomposition.resid
        df= df.fillna(method="bfill")
        self.trend = np.asarray(df.loc[:, 'trend'])
        self.seasonal = np.asarray(df.loc[:, 'seasonal'])
        self.residual = np.asarray(df.loc[:, 'residual'])
        prediction, _, _ = self.make_prediction(steps)
        return prediction[0]

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
        x = LSTM(int(self.nb_layers), return_sequences=True, activation='relu',
                 input_shape=(self.nb_features, self.look_back))(i)
        x = Dropout(0.2)(x)
        x = LSTM(int(self.nb_layers / 2), return_sequences=True,activation='relu' ,input_shape=(self.nb_features, self.look_back))(x)
        x = Dropout(0.2)(x)
        x = attention_block(x)
        x = Dense(int(self.nb_layers / 2), activation='relu')(x)
        output = Dense(1)(x)
        model = Model(inputs=[i], outputs=[output])
        optimizer = tensorflow.keras.optimizers.Adamax(lr=self.learning_rate)
        model.compile(loss=self.loss, optimizer=optimizer, metrics=[self.metric])
        if self.verbose == 1:
            print(model.summary())
        return model
    

    def make_models_att(self, seasonal, df):
        model=LSTM_PAST(self.look_back,self.freq_period,is_seasonal=seasonal)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.metric])
        return model 

    def prediction_eval(self, prediction, real_data):
        '''
        This functino compute and print four differents metrics (mse ,mae ,r2 and median) to evaluate accuracy of the model
        prediction and real_data need to have the same size
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
        from sklearn.metrics import mean_absolute_error as mae
        from sklearn.metrics import mean_squared_error as mse
        from sklearn.metrics import median_absolute_error as medae
        from sklearn.metrics import r2_score as r2

        print("mean_absolute_error : ", mae(real_data, prediction))
        print("mean_squared_error : ", mse(real_data, prediction))
        print("median_absolute_error : ", medae(real_data, prediction))
        print("r2_score : ", r2(real_data, prediction))

    def prepare_data(self, df, look_back, freq_period, first=0,seq2seq=False):
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
        self.seq2seq=seq2seq
        imputer = KNNImputer(n_neighbors=2, weights="uniform")
        df.loc[:,"y"]=imputer.fit_transform(np.array(df["y"]).reshape(-1, 1))
        if look_back%2==0:
            window=freq_period+1
        else:
            window=freq_period

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
        self.trend = np.asarray(df.loc[:, 'trend'])
        self.seasonal = np.asarray(df.loc[:, 'seasonal'])
        self.residual = np.asarray(df.loc[:, 'residual'])
        if not self.seq2seq :
            trend_x, trend_y = decoupe_dataframe(df["trend"], look_back)
            seasonal_x, seasonal_y = decoupe_dataframe(df["seasonal"], look_back)
            residual_x, residual_y = decoupe_dataframe(df["residual"], look_back)
        else :
            trend_x, trend_y = sequence_dataframe(df["trend"], look_back,self.len_pred)
            seasonal_x, seasonal_y = sequence_dataframe(df["seasonal"], look_back,self.len_pred)
            residual_x, residual_y = sequence_dataframe(df["residual"], look_back,self.len_pred)
        if self.verbose == 1:
            print("prepared")
        return trend_x, trend_y, seasonal_x, seasonal_y, residual_x, residual_y

    def train_model(self, model, x_train, y_train, name):
        '''
        Train the model and save it in the right file
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
        if name == "trend":
            nb_epochs = int(self.nb_epochs * 5)
            try:
                x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
            except Exception as e:
                if e:
                    raise Exception(
                        "************Not enought data in the input compared to the look back size. Put more data as input or decrease look_back ******")
            es = EarlyStopping(monitor='loss', mode='min', min_delta=1, patience=100)
            hist = model.fit(x_train, y_train, epochs=nb_epochs, batch_size=self.nb_batch, verbose=self.verbose,
                             callbacks=[es])
            i = 0
            while hist.history["loss"][-1] > 10 and i < 5:
                i = i + 1
                epochs = 50
                hist = model.fit(x_train, y_train, epochs=epochs, batch_size=100, verbose=self.verbose, callbacks=[es])
            print("model_trained")
            self.model_save(model, name)
        elif name == "residual":
            nb_epochs = self.nb_epochs * 6
            x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
            es = EarlyStopping(monitor='loss', mode='min', min_delta=0.1, patience=100)
            hist = model.fit(x_train, y_train, epochs=nb_epochs, batch_size=self.nb_batch, verbose=self.verbose,
                             callbacks=[es])
            i = 0
            self.model_save(model, name)
        else:
            es = EarlyStoppingByUnderVal(monitor="loss", value=0.1, verbose=self.verbose)
            x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
            model.fit(x_train, y_train, epochs=self.nb_epochs, batch_size=self.nb_batch, verbose=self.verbose)
            print("model_trained")
            self.model_save(model, name)
        return model
       


    def model_save(self, model, name):
        path = self.directory + "/" + name 
        if name == "trend":
            with open(self.directory + '/look_back.txt', 'w') as f:
                f.write('%d' % self.look_back)
            with open(self.directory + '/freq_period.txt', 'w') as f:
                f.write('%d' % self.freq_period)
            with open(self.directory + '/loss.txt', 'w') as f:
                f.write('%s' % self.loss)
            with open(self.directory + '/metric.txt', 'w') as f:
                f.write('%s' % self.metric)
            with open(self.directory + '/optimizer.txt', 'w') as f:
                f.write('%s' % self.optimizer)
        model.save_weights(path,save_format='tf')

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
        prediction = np.zeros((1, len_prediction))
        self.TREND = np.zeros((1, len_prediction))
        self.RESIDUAL = np.zeros((1, len_prediction))
        for i in progressbar(range(len_prediction), "Computing: ", 40):
            dataset = np.reshape(data_residual, (1, 1, self.look_back))
            prediction_residual = (self.model_residual.predict(dataset))
            data_residual = np.append(data_residual[1:], [prediction_residual]).reshape(-1, 1
                                                                                        )
            dataset = np.reshape(data_trend, (1, 1, self.look_back))
            prediction_trend = (self.model_trend.predict(dataset))
            data_trend = np.append(data_trend[1:], [prediction_trend]).reshape(-1, 1)          
            prediction[0, i] = prediction_residual + prediction_trend 
            self.TREND[0, i] = prediction_trend
            self.RESIDUAL[0, i] = prediction_residual     
        fit_sea = ExponentialSmoothing(self.seasonal, seasonal_periods=self.freq_period, seasonal='add').fit()
        prediction_sea = fit_sea.forecast(len_prediction)
        prediction=[prediction[0, i] + prediction_sea[i] for i in range(len_prediction)]
        prediction = self.scaler2.inverse_transform(np.reshape(prediction, (-1, 1)))
        lower, upper = self.frame_prediction(np.reshape(prediction, (1, -1)))
        return np.reshape(prediction, (1, -1)), lower, upper


    def make_prediction_seq2seq(self):
        '''
        This function make the prediction :
            reshape data to fit with the model
            make one prediction
            crush outdated data with prediction
            repeat len_prediction times
        Parameters
        ----------

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
        
        dataset = np.reshape(data_residual, (1, 1, self.look_back))
        prediction_residual = (self.model_residual.predict(dataset))

        dataset = np.reshape(data_trend, (1, 1, self.look_back))
        prediction_trend = (self.model_trend.predict(dataset))


        fit_sea = ExponentialSmoothing(self.seasonal, seasonal_periods=self.freq_period, seasonal='add').fit()
        prediction_sea = fit_sea.forecast(self.len_pred)

        prediction = [prediction_residual[i] + prediction_trend[i] + prediction_seasonal[i] for i in range(len(prediction_seasonal))]
        prediction = self.scaler2.inverse_transform(np.reshape(prediction, (-1, 1)))
        lower, upper = self.frame_prediction(np.reshape(prediction, (1, -1)))
        return np.reshape(prediction, (1, -1)), lower, upper

    def frame_prediction(self, prediction):
        '''
        This function compute the 95% confident interval by calculating the standard deviation and the mean of the residual(Gaussian like distribution)
        and return yestimated +- 1.96*CI +mean (1.96 to have 95%)
        Parameters
        ----------
        prediction : array
            array contening the perdicted vector (1,N) size.
        Returns
        -------
        lower : array
            array contening the lowers boundry values (1,N) size.
        upper : array
            array contening the upper boundry values (1,N) size.
        '''
        mae = -1 * np.mean(self.scaler2.inverse_transform(np.reshape(self.residual, (-1, 1))))
        std_deviation = np.std(self.scaler2.inverse_transform(np.reshape(self.residual, (-1, 1))))
        sc = 1.96  # 1.96 for a 95% accuracy
        margin_error = mae + sc * std_deviation
        lower = prediction - margin_error
        upper = prediction + margin_error
        return lower, upper

    def fit_predict_without_dec(self, df, look_back, freq_period, len_prediction, verbose, nb_features=1, nb_epochs=50,
                                nb_batch=100, nb_layers=50, attention=True, loss="mape", metric="mse",
                                optimizer="Adamax"):
        df = df.reset_index()
        i = Input(shape=(nb_features, look_back))
        x = LSTM(nb_layers, return_sequences=True, activation='relu', input_shape=(nb_features, look_back))(i)
        x = Activation("relu")(x)
        x = Dropout(0.2)(x)
        x = LSTM(nb_layers, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = Dense(int(nb_layers / 2), activation='relu')(x)
        output = Dense(1)(x)
        model = Model(inputs=[i], outputs=[output])
        model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
        x_train, y_train = decoupe_dataframe(df["y"], look_back)
        nb_epochs = nb_epochs * 7
        try:
            x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
        except Exception as e:
            if e == IndexError:
                print("Not enought data in the input compared to the look back size. Put more data as input")
        hist = model.fit(x_train, y_train, epochs=nb_epochs, batch_size=nb_batch, verbose=verbose)
        data_residual = np.array(df["y"][-1 * look_back:])
        prediction = np.zeros((1, len_prediction))
        for i in progressbar(range(len_prediction), "Computing: ", 40):
            dataset = np.reshape(data_residual, (1, 1, look_back))
            prediction_residual = (model.predict(dataset))
            data_residual = np.append(data_residual[1:], [prediction_residual]).reshape(-1, 1)
            prediction[0, i] = prediction_residual
        return prediction, prediction, prediction

    def retrain(self, df, nb_features=1, nb_epochs=10, nb_batch=10):
        self.df = df
        self.nb_epochs = nb_epochs
        self.nb_batch = nb_batch
        self.model_trend = self.train_model(self.model_trend, self.trend_x, self.trend_y, "trend")
        self.model_residual = self.train_model(self.model_residual, self.residual_x, self.residual_y, "residual")
        self.model_save(self.model_trend, "trend")
        self.model_save(self.model_residual, "residual")
        return self

    def load_models(self, directory="."):
        self.directory = r"" + directory
        f = open(self.directory + '/look_back.txt', 'r')
        self.look_back = int(f.readline())
        f.close()
        f = open(self.directory + '/freq_period.txt', 'r')
        self.freq_period = int(f.readline())
        f.close()
        f = open(self.directory + '/loss.txt', 'r')
        self.loss = str(f.readline())
        f.close()
        f = open(self.directory + '/optimizer.txt', 'r')
        self.optimizer = str(f.readline())
        f.close()
        f = open(self.directory + '/metric.txt', 'r')
        self.metric = str(f.readline())
        f.close()
        model_trend =LSTM_PAST(self.look_back,self.freq_period)
        model_residual =LSTM_PAST(self.look_back,self.freq_period)
        model_trend.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.metric])
        model_residual.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.metric])
        model_trend.load_weights(r"" + directory + "/" + "trend" )
        model_residual.load_weights(r"" + directory + "/" + "residual" )
        print("loaded")
        self.model_trend = model_trend
        self.model_residual = model_residual
        

    def reuse(self, df, directory=".", verbose=0):
        self.directory = r"" + directory
        print(self.directory)
        self.load_models(directory=self.directory)
        self.verbose = verbose
        print("loaded")
        f = open(self.directory + '/look_back.txt', 'r')
        self.look_back = int(f.readline())
        f.close()
        f = open(self.directory + '/freq_period.txt', 'r')
        self.freq_period = int(f.readline())
        f.close()
        f = open(self.directory + '/loss.txt', 'r')
        self.loss = str(f.readline())
        f.close()
        f = open(self.directory + '/optimizer.txt', 'r')
        self.optimizer = str(f.readline())
        f.close()
        f = open(self.directory + '/metric.txt', 'r')
        self.metric = str(f.readline())
        f.close()
        self.trend_x, self.trend_y, self.seasonal_x, self.seasonal_y, self.residual_x, self.residual_y = self.prepare_data(
            df, self.look_back,
            self.freq_period, first=0)
        return self
        

    def plot_prediction(self, df, prediction, lower=0, upper=0):
        x = np.arange(0, len(df), 1)
        y = df["y"]
        x_2 = np.arange(len(df), len(df) + len(prediction), 1)
        y_2 = prediction
        plt.plot(x, y, label="real data", color="blue")
        plt.plot(x_2, y_2, label="predicted values", color="green")
        if type(lower) != int:
            x_2 = np.arange(len(df), len(df) + len(lower), 1)
            plt.fill_between(x_2, lower[:], upper[:], color="red", alpha=0.3, label="95% CI")
        plt.legend()
        plt.show()
        
    def plot_subsignal(self,subsignal='trend'):
        if subsignal=='trend':
            y=self.trend
            y_2=self.TREND[0]
        elif subsignal=='seasonal':
            y=self.seasonal
            y_2=self.SEASONAL[0]
        elif subsignal=='residual':
            y=self.residual
            y_2=self.RESIDUAL[0]
        x = np.arange(0, len(y), 1)
        x_2 = np.arange(len(y), len(y) + len(y_2), 1)
        plt.plot(x, y, label="real data", color="blue")
        plt.plot(x_2, y_2, label="predicted values", color="green")
        plt.legend()
        plt.show()

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

    def fitness_ind(self, ind):
        self.Loss = ["mse"]*4 
        self.learning_rate = ind[4]
        self.fit(self.train_df, self.look_back, self.freq_period, verbose=self.verbose, nb_epochs=int(ind[0]),
                 nb_batch=int(ind[2]), nb_layers=int(ind[1]), metric=self.Loss[int(ind[3])])
        pred = self.predict(steps=len(self.test_df["y"]), frame=False)
        score = self.score(np.array(self.test_df["y"]), pred)
        return score

    def fitness_pop(self, Pop):
        scores = np.zeros((1, self.pop))
        i = 0
        for ind in Pop:
            scores[0, i] = self.fitness_ind(ind)
            i = i + 1
        print(scores)
        return scores
    
    

    def selecting(self, Pop, scores):
        old_pop = np.zeros((self.mat_pool, self.nb_charac))
        for i in range(self.mat_pool):
            pos_max = int(np.where(scores == np.amin(scores))[0])
            scores[0, pos_max] = -1
            old_pop[i, :] = Pop[pos_max]
        return old_pop

    def crossover(self, old_pop):
        import random
        import scipy.special
        new_pop = np.zeros((int(scipy.special.factorial(old_pop.shape[0] - 1)), self.nb_charac))
        t = 0
        for i in range(self.mat_pool):
            for j in range(i + 1, self.mat_pool):
                new_pop[t, :2] = old_pop[i, :2]
                new_pop[t, 2:] = old_pop[j, 2:]
                b = random.randint(0, 2)
                new_pop[t, b] = int(new_pop[t, b] / 3)
                t = t + 1
        return new_pop

    def genetic_fit(self, df, train_ratio, look_back, freq_period, pop=3, gen=3,multi_thread=True):
        import random
        self.train_df = df[: int(train_ratio * len(df))]
        self.test_df = df[-1 * int((1 - train_ratio) * len(df)):]
        self.look_back = look_back
        self.freq_period = freq_period
        self.pop = pop
        self.mat_pool = int(pop / 2) + 1
        self.verbose = 0
        self.gen = gen
        self.Nb_epochs = [100, 150, 200, 180, 120, 250]
        self.Nb_layers = [30, 35, 40, 45, 50, 55]
        self.Batch_size = [15, 20, 25, 30, 35, 40]
        self.Learning_rate = [0.001, 0.0001, 0.005, 0.01, 0.05, 0.007]
        self.Loss = ["mse"]*4
        Pop = [[self.Nb_epochs[random.randint(0, len(self.Nb_epochs)-1)], self.Nb_layers[random.randint(0, len(self.Nb_layers)-1)], self.Batch_size[random.randint(0, len(self.Batch_size)-1)],
                random.randint(0, 3), self.Learning_rate[random.randint(0, len(self.Learning_rate)-1)]] for i in range(pop)]
        self.nb_charac = 5
        for i in range(1, self.gen):
            print("==================== Generation " + str(i) + "/" + str(gen) + " ========================")
            print(Pop)
            if multi_thread :
                scores = self.fitness_pop_multi(Pop)
            else :
                scores = self.fitness_pop(Pop)
            old_pop = self.selecting(Pop, scores)
            new_pop = self.crossover(old_pop)
            for p in range(old_pop.shape[0]):
                Pop[p] = old_pop[p, :]
            for j in range(new_pop.shape[0]):
                Pop[j] = new_pop[j, :]
        scores = self.fitness_pop(Pop)
        pos_max = np.where(scores == np.amin(scores))[0]
        final_ind = Pop[int(pos_max)]
        self.learning_rate = final_ind[4]
        self.fit(self.df, self.look_back, self.freq_period, verbose=self.verbose, nb_epochs=int(final_ind[0]),
                 nb_batch=int(final_ind[2]), nb_layers=int(final_ind[1]), metric=self.Loss[int(final_ind[3])],attention=True)

    def fit_predict_XGBoost(self, df, freq, date_format, steps=1, early_stopping_rounds=300, test_size=0.01,
                            nb_estimators=3000):
        df = df.dropna()
        TIME = [None] * len(df['ds'])
        # modification du timestamp en format compréhensible par le modèle
        for i in range(len(df['ds'])):
            dobj_a = datetime.strptime(df['ds'][i], date_format)
            TIME[i] = dobj_a
        df["ds"] = TIME
        df = df.reset_index(drop=True)
        df = df.set_index("ds")
        df.index = pd.to_datetime(df.index)
        X, Y = self.create_features(df=df, label='y')
        reg = xgb.XGBRegressor(n_estimators=nb_estimators)
        x_train = X[: int((1 - test_size) * len(X))]
        y_train = Y[: int((1 - test_size) * len(Y))]
        x_test = X[int((test_size) * len(X)):]
        y_test = Y[int((test_size) * len(Y)):]
        reg.fit(x_train, y_train,
                eval_set=[(x_train, y_train), (x_test, y_test)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=True)
        dates = pd.date_range(df.index[len(df) - 1], freq=freq, periods=steps)
        df2 = pd.DataFrame({"ds": dates})
        df2 = df2.set_index("ds")
        df2.index = pd.to_datetime(df2.index)
        X2 = self.create_features(df=df2)
        pred = reg.predict(X2)
        return pred

    def create_features(self, df, label=None):

        df['date'] = df.index
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['dayofmonth'] = df['date'].dt.day
        df['weekofyear'] = df['date'].dt.weekofyear
        X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
                'dayofyear', 'dayofmonth', 'weekofyear']]
        if label:
            y = df[label]
            return X, y
        return X



    def Exp_smooth(self, df, freq_period, steps=1):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        fit1 = ExponentialSmoothing(df["y"], seasonal_periods=freq_period, trend='add', seasonal='add').fit()
        pred = fit1.forecast(steps)
        return pred

    def fit_predict_ES(self, df, freq_period, steps=1):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        self.freq_period = freq_period
        decomposition = STL(df["y"], period=freq_period + 1)
        decomposition = decomposition.fit()
        df.loc[:, 'trend'] = decomposition.trend
        df.loc[:, 'seasonal'] = decomposition.seasonal
        df.loc[:, 'residual'] = decomposition.resid
        fit_tres = ExponentialSmoothing(df["trend"], seasonal_periods=freq_period, seasonal='add').fit()
        prediction_trend = fit_tres.forecast(steps)
        fit_res = ExponentialSmoothing(df["residual"], seasonal_periods=freq_period, seasonal='add').fit()
        prediction_res = fit_res.forecast(steps)
        fit_sea = ExponentialSmoothing(df["seasonal"], seasonal_periods=freq_period, seasonal='add').fit()
        prediction_sea = fit_sea.forecast(steps)
        prediction = prediction_trend + prediction_res + prediction_sea
        return prediction


    def denoise(self, df, threshold=0.2):
        '''
            Apply wavelet decomposition and reconstruction in order to denoise the signal, use the sym mother wavelet with a default threshold value of 0.2
        '''
        data=df["y"]
        w = pywt.Wavelet('sym4')
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)
        threshold = threshold
        coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
        datarec = pywt.waverec(coeffs, 'sym4')
        df["y"] = datarec[: len(df["y"])]
        return df

    def collect_result(result):
        global results
        results.append(result)
        

    def fitness_pop_multi(self, Pop):
        scores = np.zeros((1, self.pop))
        que = queue.Queue()
        threads_list = list()
        for ind in Pop:
            thread = Thread_genetic_train_model(ind,que,self.train_df,self.test_df,self.look_back,self.freq_period,self.verbose,name="genetic")
            thread.start()
            threads_list.append(thread)
        for i,t in enumerate(threads_list):
            t.join()
            scores[0, i]=que.get(block=True)
        print("end of generation")
        return scores

         
    def make_seq2seq_models(self,x_train,y_train):
        input=Input(shape=(1, x_train.shape[1]))
        output=Input(shape=(1, y_train.shape[1]))
        n_hidden = 50
        encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(
        n_hidden, activation='elu', dropout=0.2,
        return_sequences=True, return_state=True)(input)
        encoder_last_h = BatchNormalization(momentum=0.1)(encoder_last_h)
        encoder_last_c = BatchNormalization(momentum=0.1)(encoder_last_c)
        decoder = RepeatVector(self.len_pred)(encoder_last_h)
        decoder_stack_h, decoder_last_h, decoder_last_c = LSTM(n_hidden, activation='elu', dropout=0.2,return_state=True, return_sequences=True)(decoder, initial_state=[encoder_last_h, encoder_last_c])
        attention = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
        attention = Activation('softmax')(attention)
        context = dot([attention, encoder_stack_h], axes=[2,1])
        context = BatchNormalization(momentum=0.6)(context)
        decoder_combined_context = concatenate([context, decoder_stack_h])
        out = TimeDistributed(Dense(1))(decoder_combined_context)
        model = Model(inputs=input, outputs=out)
        opt = Adam(lr=0.001)
        model.compile(loss='mae', optimizer=opt, metrics=['mse'])
        print(model.summary())
        return model
