import pandas as pd 
import numpy as np 
from GTSFutur import GTSPredictor
import matplotlib.pyplot as plt 
#from fbprophet import Prophet
import time 

past=-5*168

look_back=400
len_pred=200
df=pd.read_csv("test.csv")
df=df[: past]
period=24
#m=Prophet(changepoint_prior_scale=0.001)
#m.fit(df)
#future = m.make_future_dataframe(periods=period,freq='H')
#forecast = m.predict(future)
model=GTSPredictor()
model=model.fit(df,look_back=look_back,freq_period=period,seq2seq=True,len_pred=len_pred)
pred=model.make_prediction(df["y"])

model=GTSPredictor()
prediction= model.fit_predict_ES(df=df, freq_period=period, steps=len_pred)
#model=GTSPredictor()
#prediction_xgb=model.fit_predict_XGBoost( df, "D", "%Y-%m-%d",steps=len_pred, early_stopping_rounds=300, test_size=0.01,nb_estimators=1000)
model=GTSPredictor()
model.fit(df,look_back=look_back,freq_period=period)
prediction_2,lower,upper=model.predict(steps=len_pred)



model.plot_subsignal(subsignal="trend")
plt.plot(np.array(prediction_2),label="prediction lstm")
plt.plot(np.array(prediction),label="prediction ExpS")
#plt.plot(np.array(prediction_xgb),label="prediction XGBOOST")
plt.plot(np.array(df['y'][past : past+len_pred]),label="vrai data ")
#plt.plot(np.array(forecast["yhat"][-1*len_pred :]),label="fbprophet")
plt.legend()
plt.show()

x=40
plt.plot(np.array(prediction_2)[: x],label="prediction lstm")
plt.plot(np.array(prediction)[: x],label="prediction ExpS")
#♣plt.plot(np.array(pred[0,0,:]),label="Seq2seq")
plt.plot(np.array(df['y'][past : past+len_pred])[: x],label="vrai data ")
#plt.plot(np.array(prediction_xgb)[: x],label="prediction XGBOOST")
#plt.plot(np.array(forecast["yhat"][-1*len_pred :])[: x],label="fbprophet")
plt.legend()
plt.show()



print("*************************************** RES LSTM *********************************")
model.prediction_eval(prediction_2,df["y"][past : past+len_pred])
print("*************************************** RES EXPS *********************************")
model.prediction_eval(prediction,df["y"][past : past+len_pred])
print("*************************************** RES FB *********************************")
#model.prediction_eval(forecast["yhat"][-1*len_pred-13 :-13],df["y"][past : past+len_pred])
print("*************************************** RES SEQ2SEQ *********************************")
model.prediction_eval(pred,df["y"][past : past+len_pred])
