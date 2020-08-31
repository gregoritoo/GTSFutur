# General Time Serie Prediction
[![PyPI version](https://badge.fury.io/py/GTSFutur.svg)](https://badge.fury.io/py/GTSFutur) </br>
Exemple of LSTM + DECOMPOSITION prédiction :

![LSTM PRED](/Images/gif_lstm.gif)

Please refer to [https://www.influxdata.com/blog/how-supralog-built-an-online-incremental-machine-learning-pipeline-with-influxdb-for-capacity-planning/] for further informations.</br>

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
from GTSFutur.GTSPredictor import GTSPredictor
model=GTSPredictor()
model.fit(df,look_back=400,freq_period=200,directory="My_directory_name")
prediction,lower,upper=model.predict(steps=365)
```

**Code for prediction 365 steps ahead using the user interface**
```python
from GTSFutur.GTSPredictor import GTSPredictor
model=GTSPredictor()
model.fit_with_UI(df,directory="My_directory_name")
prediction,lower,upper=model.predict(steps=365)
```
This will open a matplotlib figure and you will be able to select the seasonal pattern with the mouse


**plot**
```python
model.plot_prediction(df,prediction,lower,upper)
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
