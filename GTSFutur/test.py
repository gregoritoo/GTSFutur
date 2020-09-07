import pandas as pd 
import numpy as np 
from GTSFutur import GTSPredictor
import matplotlib.pyplot as plt 
from fbprophet import Prophet
import time 


past=-3*168

look_back=168
len_pred=200
df=pd.read_csv("test.csv",sep=",")
df=df[: past]
period=24


m=Prophet(changepoint_prior_scale=0.001)
m.fit(df)
future = m.make_future_dataframe(periods=period,freq='H')
forecast = m.predict(future)
#model=GTSPredictor(
#model=model.fit(df,look_back=look_back,freq_period=period,seq2seq=True,len_pred=len_pred)
#pred=model.make_prediction(df["y"])
"""
model=GTSPredictor()
prediction_3= model.fit_predict_ES(df,period,len_pred)"""
#model=GTSPredictor()
#prediction_xgb=model.fit_predict_XGBoost( df, "D", "%Y-%m-%d",steps=len_pred, early_stopping_rounds=300, test_size=0.01,nb_estimators=1000)
"""model=GTSPredictor()
model.fit(df,look_back=look_back,freq_period=period)
prediction_2,lower,upper=model.predict(steps=len_pred)"""

model=GTSPredictor()
model.fit(df,look_back=168,freq_period=24,directory="My_directory_name")
prediction,lower,upper=model.predict(steps=len_pred)

model=GTSPredictor()
prediction_3= model.fit_predict_ES(df,period,len_pred)



plt.plot(np.array(prediction),label="prediction lstm+att")
plt.plot(np.array(prediction_3),label="prediction es")
plt.plot(np.array(forecast["yhat"][-1*len_pred :]),label="fbprophet")
plt.plot(np.array(df['y'][past : past+len(prediction) ]),label="vrai data ")
plt.legend()

"""model.fit(df,look_back=look_back,freq_period=period,attention=False,directory="testing")
prediction_4,lower_4,upper_4=model.predict(steps=len_pred)



#model.plot_subsignal(subsignal="trend")
#plt.plot(np.array(prediction_2),label="prediction lstm+att")
plt.plot(np.array(prediction),label="prediction lstm")
plt.plot(np.array(prediction_3),label="prediction ExpS")
plt.plot(np.array(prediction_4),label="prediction simple lstm")
#plt.plot(np.array(prediction_xgb),label="prediction XGBOOST")
plt.plot(np.array(df['y'][past : past+len(prediction) ]),label="vrai data ")
plt.plot(np.array(forecast["yhat"][-1*len_pred :]),label="fbprophet")
plt.legend()
plt.show()

x=150
#plt.plot(np.array(prediction_2)[: x],label="prediction lstm +att")
plt.plot(np.array(prediction)[: x],label="prediction lstm")
plt.plot(np.array(prediction_3)[: x],label="prediction  ExpS")
plt.plot(np.array(prediction_4)[: x],label="prediction simple lstm")
#♣plt.plot(np.array(pred[0,0,:]),label="Seq2seq")
plt.plot(np.array(df['y'][past : ])[: x],label="vrai data ")
#plt.plot(np.array(prediction_xgb)[: x],label="prediction XGBOOST")
plt.plot(np.array(forecast["yhat"][-1*len_pred :])[: x],label="fbprophet")
plt.legend()
plt.show()



#print("*************************************** RES LSTM + ATT *********************************")
#model.prediction_eval(prediction_2,df["y"][past : min(len(df["y"])-1,past+len(prediction_2))])
print("*************************************** RES LSTM *********************************")
model.prediction_eval(prediction,df["y"][past : min(len(df["y"])-1,past+len(prediction))])
print("*************************************** RES EXP *********************************")
model.prediction_eval(prediction_3,df["y"][past : min(len(df["y"])-1,past+len(prediction_3))])
print("*************************************** RES simple lstm *********************************")
model.prediction_eval(prediction_4,df["y"][past : min(len(df["y"])-1,past+len(prediction_3))])
print("*************************************** RES FB *********************************")
model.prediction_eval(forecast["yhat"][-1*len_pred :],df["y"][past : past+len_pred])
#print("*************************************** RES SEQ2SEQ *********************************")
#model.prediction_eval(pred,df["y"][past : past+len_pred])"""
