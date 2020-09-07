import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras import backend as K
import os
from tensorflow.keras.layers import  LSTM, Dropout
from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate, Input ,RepeatVector ,TimeDistributed,Conv1D,MaxPooling1D,Flatten,BatchNormalization








class attention(tk.layers.Layer):
    
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
    
class Moy_layer(tk.layers.Layer):
    
    def __init__(self):
        super(Moy_layer,self).__init__()
        self.filter= tf.convert_to_tensor(tf.constant([0.05,0.10,0.7,0.10,0.05]), dtype=tf.float32)
        self.filter=tf.reshape(self.filter, [self.filter.shape[-1],1], name="mean")
    
        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
        })
        return config
    
    def call(self,inputs):
        e=K.dot(inputs,self.filter)
        return e
    
class Neigh(tk.layers.Layer):
    
    def __init__(self):
        super(Neigh,self).__init__()
        
        
    def call(self,inputs):
        scores=tf.math.exp(-(inputs))
        scoresT=K.transpose(scores)
        return K.dot(scoresT,inputs)
    
    def get_config(self):
        config=super().get_config().copy()
        config.update({
            })
        return config
    
        
class Own_dense(tk.layers.Layer):

    def __init__(self,units=32):
        super(Own_dense,self).__init__()
        self.units=units
        
    def build(self,input_shape):
        self.w=self.add_weight(shape=(input_shape[-1], self.units),initializer="random_normal", trainable=True)
        self.b=self.add_weight(shape=(self.units,),initializer="zeros", trainable=True)
        
    def call(self,inputs):
        return K.matmul(inputs,self.w)+self.b
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        })
        return config
        


class LSTM_PAST(tk.Model):

    def __init__(self,look_back,period,units=50,nb_features=1,is_seasonal=False):
        super(LSTM_PAST,self).__init__()
        self.units=units
        self.period=period
        self.is_seasonal=is_seasonal
        self.nb_features=nb_features
        self.look_back=look_back
        self.Conv1D=Conv1D(int(look_back/period), (1), activation='relu', input_shape=(1, look_back),padding="same",name="conv1")
        self.Conv=Moy_layer()
        self.Conv1=Moy_layer()
        self.Conv2=Moy_layer()
        self.MaxPooling1D=MaxPooling1D(pool_size=1,strides=1, padding='valid')
        self.LSTM_1=LSTM(self.units, return_sequences=True, activation='relu',
                 input_shape=(self.nb_features, int(self.look_back/self.period)))
        self.LSTM_2=LSTM(int(self.units / 2), return_sequences=True)
        self.LSTM_NN=LSTM(int(self.units / 2), return_sequences=True)
        self.neigh=Neigh()
        self.Normal=BatchNormalization(momentum=0.6)
        self.Normal_1=BatchNormalization(momentum=0.6)
        self.att = attention(return_sequences=False)
        self.fc_1=Dense(int(self.units / 2), activation='relu')
        self.fc_2=Dense(1)
        self.fc_3=Dense(1)
        self.fc_4=Dense(1,name='final')

    
    def call(self,inputs):
        x_1=self.Conv1D(inputs)
        x_1=self.MaxPooling1D(x_1)
        x_1=Flatten()(x_1)          
        x_1=tf.reshape(x_1, [-1,1,x_1.shape[-1]], name="1")
        x_1=self.Normal(x_1)
        x_1=self.LSTM_NN(x_1)
        x_1=Dropout(0.2)(x_1)
        x=self.LSTM_2(x_1)
        x=Dropout(0.2)(x)
        x=self.att(x)
        x=self.Normal_1(x)
        x=self.fc_1(x)
        x=self.fc_2(x)
        return x

    def get_config(self):

        config = super().get_config().copy()
        config.update({"units":self.units,
        "features":self.features
        })
        return config

