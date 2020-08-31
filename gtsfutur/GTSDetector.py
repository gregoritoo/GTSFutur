# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt 
from tensorflow.keras.models import save_model,Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout,TimeDistributed,RepeatVector
from sklearn.preprocessing import MinMaxScaler,PowerTransformer
from tensorflow.keras.models import load_model
import os
from sklearn.preprocessing import MinMaxScaler
import joblib
import pywt
import shutil
from sklearn.base import BaseEstimator, TransformerMixin
import scipy
from scipy.signal import argrelextrema

class MinMax():
    def __init__(self):
        pass
    
    def train(self,y):
        self.y=y
        self.max=np.max(y)
        self.min=np.min(y)

    def apply(self):
        y_norm=[(self.y[i]-self.min)/(self.max-self.min) for i in range(len(self.y))]
        return y_norm 
    
    def inverse(self,y):
        y=[(self.max-self.min)*y[i]+self.min for i in range(len(y))]
        return y 



def decoupe_dataframe(y, look_back):
    dataX, dataY = [], []
    for i in range(len(y) - look_back - 1):
        a = y[i:(i + look_back)]
        dataY = dataY + [y[i + look_back]]
        dataX.append(a)
    return (np.asarray(dataX), np.asarray(dataY).flatten())

def moving_windows(y,window_size,i):
    w_i=y[i:i+window_size]
    return w_i

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]

def do_plot(anomaly,y):
    plt.plot(np.array(y),label=" data",color="blue")
    begin=0
    for i in range(len(anomaly)):
        if anomaly[i]==True and begin==0 :
            plt.scatter(i,y[i],label="anomaly",color="red")
            begin=begin+1
        elif anomaly[i] == True:
            plt.scatter(i, y[i], color="red")
    plt.tight_layout()
    plt.legend()
    plt.show()
    
def span_plot(anomaly,y,period):
    plt.plot(np.array(y),label=" data")
    begin=0
    for i in range(0,len(anomaly)-period,period):
      if anomaly[i]==True and begin==0:
         plt.axvspan(i, i+period, facecolor='red', alpha=0.5,label="anomalous period")
         begin=begin+1
      elif anomaly[i]==True :
         plt.axvspan(i, i + period, facecolor='red', alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()
    
def mae(y1,y2):
   if len(y1)==len(y2) :
      diff=abs(np.array(y1)-np.array(y2))
      mae=np.max(diff)/len(y1)
   else :
      print("Vectors need to have the same size")
      raise Exception
   return mae


def derivation(y):
    derive=np.zeros((1,len(y)))
    for i in range(1,len(y)-1):
        derive[0,i+1]=y[i+1]-y[i]
    return derive

class IEQ_Detector():
    '''
    This detector detect a percentage of points as abnomalous.
    '''
    def __init__(self):
        pass
    
    def fit(self,y):
        self.mean=np.mean(y)
        self.Q1=np.percentile(y, 25)
        self.Q3=np.percentile(y, 75)
        self.range=(self.Q3-self.Q1)
    	
    def detect(self,y,alpha=1.96) :
        anomalies=[y[i] > self.mean+alpha*self.range or y[i] < self.mean-alpha*self.range for i in range(len(y))]
        return anomalies
    
    def rolling_fit_detect(self,y,past_lenght=24,alpha=1.96):
        anomaly=[None]*len(y)
        for i in range(0,len(y)-past_lenght,past_lenght):
            w_i=moving_windows(y,past_lenght,i)
            self.fit(w_i)
            anomaly[0,i:i+past_lenght]=self.detect(w_i,alpha)
        return anomaly
    
    def plot_anomalies(self,anomaly,y):
        do_plot(anomaly,y)
       
          
    	
class Mad_Detector():
    '''
    This detector compare the distance form the median and class point above a threshold distance as abnomalus 
    '''
    def __init__(self):
       pass
    
    def fit(self,y):
        self.MAD=np.median([abs(y[i]-np.median(y)) for i in range(len(y))])
 
    def detect(self,y,alpha=0.6745):
        self.median=np.median(y)
        anomalies=[alpha*(y[i]-self.median)/self.MAD > 1.5 for i in range(len(y))]
        return anomalies
    	
    def rolling_fit_detect(self,y,past_lenght=24,alpha=1.96):
        if len(y) > past_lenght:
            anomaly=[None]*len(y)
            for i in range(0,len(y)-past_lenght,past_lenght):
                w_i=moving_windows(y,past_lenght,i)
                self.fit(w_i)
                anomaly[0,i:i+past_lenght]=self.detect(w_i,alpha)
        else :
            print('The vector lenght need to be superior than the past_length parameter')
        return anomaly
    def plot_anomalies(self,anomaly,y):
        do_plot(anomaly,y)
        
class Variation_Detector():
     '''
     This class detect variation of somes mesure on a slidding windows.
     It can compare median, mean and standard deviation of every windows
     By comparing the mean it can detect a variation of trend.
     By comparing the median it can detect a variation of level
     By comparing th standart deviation it can detect a change in the stability of the signal 
     '''
     def __init__(self,method="mean"):
         self.method=method
         pass 
         
     def fit(self,y):
         print("This detector do not need to by trained")
         
     def detect(self,y,threshold=1,windows_size=2):
        transformed_vector=[]
        for i in range(0,len(y)-windows_size):
            w_i=moving_windows(y,windows_size,i)
            if self.method=="mean":
                transformed_vector=transformed_vector+[np.mean(w_i)]
            elif self.method=="median":
                transformed_vector=transformed_vector+[np.median(w_i)]
            elif self.method=="std":
                transformed_vector = transformed_vector + [np.std(w_i)]
            else :
                print("This method does not exist")
                raise Exception
        peaks, _ = find_peaks(transformed_vector, height=threshold*(np.max(y)-np.min(y))/100)
        anomalies=[False]*len(y)
        for i in range(len(peaks)):
            anomalies[peaks[i]]=True
        return anomalies

     def plot_anomalies(self, anomaly, y):
         do_plot(anomaly, y)

     def load_fitted(self):
        print("still need to be done ")

def moving_windows(y,window_size,i):
    w_i=y[i:i+window_size]
    return w_i
          
class Period_Detector():
      '''
      This class detect a non respect in the pattern learned during the training.
      '''
      
      
      def __init__(self):
        self.PEAK=[]
        pass 
        
      def detect_periodicity_length(self,y):
        for i in range(11,3*int(len(y)/100*10),int(len(y)/100*10)):
            y=scipy.signal.savgol_filter(y,i,polyorder=10)
            auto_correlation=autocorr(y)
            peaks = argrelextrema(auto_correlation, np.greater)
            self.PEAK.append([peaks])
        print(self.PEAK)
        for i in range (1,len(self.PEAK)) :
            for element in self.PEAK[-1*i] :
                if (len(list(filter (lambda x : x == element, self.PEAK[-1*i-1]))) > 0) :
                    print(element)
                    return element
        if len(peaks) < 1 :
            print("The signal does not show a strong enought seasonality")
            raise Exception
        if auto_correlation[(peaks[1])] > 0.60 :
            return peaks[1]
        else :
          return -1
          
      def fit(self,y):
         self.period=self.detect_periodicity_length(y)
         print(self.period)
         if self.period <= 0 :
             print("The signal does not show a strong seasonality")
             raise Exception
         j=0
         self.shape = np.zeros((1, self.period))
         for i in range(0,len(y)-self.period,self.period):
             self.shape=self.shape+np.array(y[i:i+self.period])
             j=j+1
         self.shape=self.shape[0]/j
         if self.period < 0 :
            print("The time series does not have a stong enought seasonality")
            raise Exception
        
      def detect(self,y,threshold=1):
         min_error=1000
         pos=0
         anomaly=np.zeros((1,len(y)))
         for i in range(self.period):
             value=mae(self.shape,y[i:i+self.period])
             if value < min_error :
                pos=i
                min_error=value
         for i in range(pos,len(y)-self.period,self.period):
            distance=mae(self.shape,y[i:i+self.period])
            anomaly[0,i:i+self.period]=[distance]*self.period
         anomaly=[anomaly[0,i]>threshold*(np.max(y)-np.min(y))/100 for i in range(len(y))]
         return anomaly
         
      def plot_anomalies(self,anomaly,y) :
          span_plot(anomaly,y,self.period)
         

class Autoencoder_Detector():
     def __init__(self):
         pass
     
     def fit(self,look_back,y,directory):
         self.directory=directory
         VAR=New_VAR_LSTM(look_back,y,directory)
         
     def detect(self,y,directory,threshold=5):
         Trained_var=Existing_VAR_LSTM(directory)
         anomaly=Trained_var.make_prediction(y,threshold)
         anomaly = np.append(anomaly[0], [anomaly[i][-1] for i in range(1, len(anomaly))])
         print(anomaly)
         return anomaly

     def plot_anomalies(self,anomaly,y):
         do_plot(anomaly,y)

class Wave_Detector():
    def __init__(self):
        pass

    def denoise(self, y, threshold=0.2):
        '''
            Apply wavelet decomposition and reconstruction in order to denoise the signal, use the sym mother wavelet with a default threshold value of 0.2
        '''
        data=y
        w = pywt.Wavelet('sym4')
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)
        threshold = threshold
        coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
        datarec = pywt.waverec(coeffs, 'sym4')
        return datarec[: len(y)]

    def fit(self,y,threshold=0.3):
        self.y=self.denoise(y,threshold=threshold)

    def detect(self,y,alpha=10):
        anomaly = [False] * len(y)
        peaks, _ = find_peaks(self.y, height=alpha * (np.max(y) - np.min(y)) / 100)
        for i in range(len(peaks)):
            anomaly[peaks[i]]=True
        return anomaly

    def plot_anomalies(self,anomaly,y):
        do_plot(anomaly, y)

class VAR_LSTM():

    def __init__(self,look_back,directory):

        print("Creating variable encoder object ")

        self.look_back=look_back

        self.directory=directory
        if not os.path.exists(directory):
            os.mkdir(directory)



    def make_prediction(self,x_real,threshold):
        model = self.model
        scaler = self.scaler
        Y = scaler.transform(np.reshape(np.array(x_real), (-1, 1)))
        x_real,_=decoupe_dataframe(Y,self.look_back)
        print(self.look_back)
        x_real = x_real.reshape(x_real.shape[0], 1, x_real.shape[1])
        prediction=model.predict(x_real)
        test_mae_loss = np.mean(np.abs(prediction - x_real), axis=1)
        anomalies=[test_mae_loss[i] > threshold/10 for i in range(len(test_mae_loss))]
        return anomalies


    def model_save(self, model,scaler, name="var"):
        path = self.directory + "/" + name + ".h5"
        if name =="var" :
            with open(self.directory+'/params_var_model.txt', 'w') as txt_file:
                    txt_file.write(str(self.look_back))
                    print("saved")

        scalerfile = self.directory + "/" +'scaler.pkl'
        joblib.dump(scaler, scalerfile)
        save_model(model, path)






class New_VAR_LSTM(VAR_LSTM):

    def __init__(self,look_back,y,directory):

        print("New VAR_LSTM is being created")

        VAR_LSTM.__init__(self,look_back,directory)

        self.model=self.make_model(look_back)

        self.scaler_fitted,self.history=self.train_model(y,self.look_back,self.model)

        self.model_save(self.model,self.scaler_fitted)


    def make_model(self, look_back):

        model = Sequential()

        model.add(LSTM(units=128, input_shape=(1, look_back), return_sequences=True))

        model.add(Dropout(0.3))

        model.add(LSTM(units=64, input_shape=(1, look_back)))

        model.add(RepeatVector(n=1))

        model.add(LSTM(units=64, return_sequences=True))

        model.add(Dropout(0.3))

        model.add(LSTM(units=128, input_shape=(1, look_back), return_sequences=True))

        model.add(TimeDistributed(Dense(units=1)))

        model.compile(loss='mse', optimizer='adam')

        return model


    def train_model(self, y, look_back, model):

        scaler = MinMaxScaler()


        Y = scaler.fit_transform(np.reshape(np.array(y),(-1,1)))
        

        x_train, y_train = decoupe_dataframe(np.reshape(Y,(-1,1)), look_back)

        print(np.shape(x_train))

        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])

        history = model.fit(

            x_train, y_train,

            epochs=200,

            batch_size=32,

            validation_split=0.1,

            shuffle=False

        )

        return scaler, history


class Existing_VAR_LSTM(VAR_LSTM):

    def __init__(self,directory):

        self.directory=directory

        with open(self.directory +'/params_var_model.txt', 'r') as txt_file:

            dic=txt_file.read()

        self.look_back=int(dic)

        VAR_LSTM.__init__(self,self.look_back,self.directory)
        
        self.model,self.scaler=self.load_models()

        



    def load_models(self):

        file=self.directory+"/var.h5"

        model_trend=load_model(file)

        scalerfile = self.directory + "/" + 'scaler.pkl'

        scaler = joblib.load(scalerfile)

        return model_trend,scaler    





        	
          
        
       
