General Time Serie Prediction
=============================




GTSPredictor object
-------------------

This object simplify prediction by hidding all the paramaters for the user. Let's focus on the most important features :


Requirements
------------

pandas

numpy

statsmodels

tensorflow

matplotlib

scipy

sklearn

pickle

xgboost

pyWavelets

Install
-------

::

    pip install GTSFutur

Use
---

**Example use case (The dataframe needs to have a columns named "time" and one named "y" in order to work)** my dataframe (df) is like below and have a 200 points seasonal pattern :
::

    "time","y"
    "1749-01",58.0
    "1749-02",62.6
    "1749-03",70.0
    ...

**Code for prediction 365 steps ahead**

::

    from GTSFutur.GTSPredictor import GTSPredictor
    model=GTSPredictor()
    model.fit(df,look_back=400,freq_period=200,directory="My_directory_name")
    prediction,lower,upper=model.predict(steps=365)

**Code for prediction 365 steps ahead using the user interface**

::

    from GTSFutur.GTSPredictor import GTSPredictor
    model=GTSPredictor()
    model.fit_with_UI(df,directory="My_directory_name")
    prediction,lower,upper=model.predict(steps=365)

This will open a matplotlib figure and you will be able to select the seasonal pattern with the mouse

**plot**

::

    model.plot_prediction(df,prediction,lower,upper)

**reuse a saved model**

::

    model=model.reuse(df,directory="My_directory_name")

**Retrain the model (on new data for example in order to do incremental learning)**

::

    # before the model need to be loaded either it's just after the first training or used reuse function
    model=model.retrain(df)

**Train and predict using XGBOOST for quicker but less accurate prediction**

::

    # You need to know the format of the time column (here 2020M02 for februray 2020)
    prediction=model.fit_predict_XGBoost(df,"m","%YM%m",steps=100)
    
**Train and predict using Holt Winters**

::
    prediction= model.fit_predict_ES(df=df, freq_period=period, steps=len_pred)


**Train and hyper parameters tunning using genetic algorithm**

::
    # pop = number of people in the first generation / gen = number of generation to do 
    # WARNING = multi_thread mode consume 100% of the CPU during the training.
    prediction= model.genetic_fit(df, train_ratio=0.95, look_back=168, freq_period=24, pop=3, gen=3,multi_thread=True)
    
**Train and prediction using a seq2seq model**

::
    # pop = number of people in the first generation / gen = number of generation to do 
    # WARNING = multi_thread mode consume 100% of the CPU during the training.
    model.fit(df,look_back=400,freq_period=200,directory="My_directory_name",len_pred=10,seq2seq=True)
    prediction=model.predict()
Results
-------


.. figure::  https://raw.githubusercontent.com/gregoritoo/GTSFutur/master/Images/figures.png


 Theses are "out of the box" results. The only parameter to determine was the size of the seasonal pattern which is easy to find as he can be determine through visual inspection


Quick methods explanation
-------------------------

**fit needs only three inputs**
   -df : dataframe with a time columns (string) and y the columns with the values to forecast.

   -look\_back : int size of inputs (generaly freq\_period \*2 but always more than freq\_period).

   -freq\_period : int size in point of the seasonal pattern (ex 24 if daily seasonality for a signal sampled at 1h frequency).

   -directory : str optionnal Directory where the models are going to be saved, by default at the root (r".").

**Once the model fitted it can by used by applying the predict function which need only two inputs**:
   -steps : int number of points you want to forecast, by default 1.

   -frame : Bool *If frame == True , compute an 95% intervalle and retruns 3 arrays* \| if frame == False return an array with the predicted values

**Retrain allows your model to do incremental learning by retraining yours models with new data :**\
   -df : dataframe with a time columns (string) and y the columns with the values to forecast.

   -look\_back : int size of inputs (generaly freq\_period \*2 but always more than freq\_period).

   -freq\_period : int size in point of the seasonal pattern (ex 24 if daily seasonality for a signal sampled at 1h frequency).

**load\_models allows to reuse saved model by loading it in the class** :
   -directory : str name of directory contaning trend.h5,seasonal.h5,residual.h5 by default (r".") ie root of project

**prediction\_eval : once the prediction made**\  This function compute and print four differents metrics (mse ,mae ,r2 and median) to evaluate accuracy of the model prediction and real\_data need to have the same size
   -prediction : array predicted values.

   -real\_data : array real data.\n


Why decomposition ?
-------------------

As describe in the article above, the aim of the project is to create a module able to forecast values of severals time series that could deferred in nature. One of the main problem in Deep Neural Network is to tune hyper-parameters (as for example the number of neurones ...) especially for multi-step ahead predictions. Decomposing the signal allow us to apply a single model for all the time series without spending time on hyper parameters tunning. Here below the results of this pre-processing process on differents signals :

.. figure:: https://raw.githubusercontent.com/gregoritoo/GTSFutur/master/Images/res_1.PNG



.. figure:: https://raw.githubusercontent.com/gregoritoo/GTSFutur/master/Images/res_2.PNG



.. figure:: https://raw.githubusercontent.com/gregoritoo/GTSFutur/master/Images/res_3.PNG



For the experiments above, the same LSTM model was applied on three differents signals with the same hyper parameters. For the first two signals the accuracy is almost the same (except a one point delay for the cpu signal that appears for the LSTM + DECOMPOSITION model after one weak ahead prediction (which explain the difference of accuracy on the table below)).

But for the third signal, the model without decomposition seems to reach a local minimum during the training and then the forecated values converge to the mean value while the model with decomposition is way more accurate. (the dataset of the third experiment is the Minimum Daily Temperatures Dataset available here : [https://machinelearningmastery.com/time-series-datasets-for-machine-learning/]) Here the results of the three experiments :

.. figure:: https://raw.githubusercontent.com/gregoritoo/GTSFutur/master/Images/table_res.PNG


 Note : this method also seems to disminuish the variance of the predicted values.( ie for the same dataset, the LSTM with decomposition is more likely to give the same forecasted value)

.. |PyPI version| image:: https://badge.fury.io/py/GTSFutur.svg
   :target: https://badge.fury.io/py/GTSFutur
