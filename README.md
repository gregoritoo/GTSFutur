# General Time Serie Prediction and Anomalies Detection 
Exemple of LSTM + DECOMPOSITION prédiction :

![LSTM PRED](/Images/gif_lstm.gif)



Summary
-------

-[GSFutur object](#gsfutur-object) </br>
-[Example use case ](#example-use-case ) </br>
-[Requirements](#quick-methods-explanation ) </br>
-[Some results](#results) </br>


GTSPredictor object
--------------

This object simplify prediction by hidding all the paramaters for the user.
Let's focus on the most important methods : </br>
-fit()  </br>
-predict() </br>
-retrain() </br>
-prediction_eval() </br>
-load_models() </br>

Install
-------
```python
pip install GTSFutur
```


Use
---

**Example use case (The dataframe needs to have a columns named "time" and one named "y" in order to work)** </br>
my dataframe (df) is like below and have a 200 points seasonal pattern :</br>
"time","y"</br>
"1749-01",58.0</br>
"1749-02",62.6</br>
"1749-03",70.0</br>
...</br>
**Code for prediction 365 steps ahead**
```python
import  GTSFutur.GTSFutur as gt
model=gt.GTSPredictor()
model.fit(df,look_back=400,freq_period=200,directory="My_directory_name")
prediction,lower,upper=model.predict(steps=365)
```

**Code for prediction 365 steps ahead using the user interface**
```python
import  GTSFutur.GTSFutur as gt
model=gt.GTSPredictor()
model.fit_with_UI(df,directory="My_directory_name")
prediction,lower,upper=model.predict(steps=365)
```
This will open a matplotlib figure and you will be able to select the seasonal pattern with the mouse

**Use Holt-Winters models for quicker prediction (also better to use if less than 1000 training points)**
```python
import  GTSFutur.GTSFutur as gt
model=gt.GTSPredictor()
prediction= model.fit_predict_ES(df=df, freq_period=200, steps=365)
```

**Use genetic algorithm to find best hyper-parameters**
```python
import  GTSFutur.GTSFutur as gt
model=gt.GTSPredictor()
model.genetic_fit(df=df,look_back=365,freq_period=200,train_ratio=0.90,pop=3,gen=3,multi_thread=True)
prediction,lower,upper=model.predict(steps=365)
```

**plot**
```python
model.plot_prediction(df,prediction,lower,upper)
```

**plot prediction of the separate signals (trend,seasonal and residual)**
```python
model.plot_subsignal(subsignal='trend')
```

**Generate new dataframe with date and predicted values**
```python
df=model.generate_dataframe(prediction=prediction,start_date="2020-07-05 14:00:00",freq="H",lower=lower,upper=upper)
```

**reuse a saved model**
```python
model=model.reuse(df,directory="My_directory_name")
```

**Retrain the model (on new data for example in order to do incremental learning)**
```python
# before the model need to be loaded either it's just after the first training or used reuse function
model=model.retrain(df)
```


 Requirements 
------------
pandas </br>
numpy </br>
statsmodels</br>
tensorflow</br>
matplotlib</br>
scipy</br>
sklearn</br>
pickle</br>
xgboost </br>



Results
--------
![LSTM PRED](/Images/figures.png)
Theses are "out of the box" results. The only parameter to determine was the size of the seasonal pattern which is easy to find as he can be determine through visual inspection


GTSDetection object
-------------------

Many differents models for anomaly detection.



List of models
--------------

IEQ_Detector

Mad_Detector

Variation_Detector

Period_Detector

Autoencoder_Detector

Wave_Detector



Use
---

**Detect disruption in the seasonal shape for contextual anomalies**:
```python
    import  GTSFutur.GTSDetector as gd
    detector=gd.Period_Detector()
    detector.fit(data)
    anomaly=detector.detect(data,threshold=0.2)
    detector.plot_anomalies(anomaly,data)
```

**Detect points far from the average statical distribution**:
```python
    import  GTSFutur.GTSDetector as gd
    detector=gd.IEQ_Detector()
    detector.fit(data)
    #alpha a severity parameter
    anomaly=detector.detect(data,alpha=1.96)
    detector.plot_anomalies(anomaly,data)
```


**Detect points far from the median**:
```python
    import  GTSFutur.GTSDetector as gd
    detector=gd.Mad_Detector()
    detector.fit(data)
    anomaly=detector.detect(data,alpha=0.6785)
    detector.plot_anomalies(anomaly,data)
```
**Detect important changes of values for measurements like mean,standard deviation and median on slidding windows**:
```python
    import  GTSFutur.GTSDetector as gd
    detector=gd.Variation_Detector(method="std")
    detector.fit(data)
    anomaly=detector.detect(data,threshold=0.5,windows_size=25)
    detector.plot_anomalies(anomaly,data)
```

**Detect peaks once the signal denoised**:
```python
    import  GTSFutur.GTSDetector as gd
    detector=gd.Wave_Detector()
    detector.fit(data,threshold=0.3)
    anomaly=detector.detect(data,alpha=10)
    detector.plot_anomalies(anomaly,data)
```
**Use a LSTM auto encoder to detect more complexe, global and contextual, anomalies**:
```python
    import  GTSFutur.GTSDetector as gd
    detector=gd.Autoencoder_Detector()
    detector.fit(100,data,"My_directory_name")
    anomaly=detector.detect(data,"My_directory_name",threshold=5)
    detector.plot_anomalies(anomaly,data)
```



