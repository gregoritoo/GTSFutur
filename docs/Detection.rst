General Time Serie Anomalies Detection
=============================




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
::
    from GTSFutur.GTSDetector import Wave_Detector,Period_Detector,Mad_Detector,Variation_Detector,Autoencoder_Detector,IEQ_Detector
    detector=Period_Detector()
    detector.fit(data)
    anomaly=detector.detect(data,threshold=0.2)
    detector.plot_anomalies(anomaly,data)


**Detect points far from the average statical distribution**:
::
    from GTSFutur.GTSDetector import Wave_Detector,Period_Detector,Mad_Detector,Variation_Detector,Autoencoder_Detector,IEQ_Detector
    detector=IEQ_Detector()
    detector.fit(data)
    #alpha a severity parameter
    anomaly=detector.detect(data,alpha=1.96)
    detector.plot_anomalies(anomaly,data)



**Detect points far from the median**:
::
    from GTSFutur.GTSDetector import Wave_Detector,Period_Detector,Mad_Detector,Variation_Detector,Autoencoder_Detector,IEQ_Detector
    detector=Mad_Detector()
    detector.fit(data)
    anomaly=detector.detect(data,alpha=0.6785)
    detector.plot_anomalies(anomaly,data)

**Detect important changes of values for measurements like mean,standard deviation and median on slidding windows**:
::
    from GTSFutur.GTSDetector import Wave_Detector,Period_Detector,Mad_Detector,Variation_Detector,Autoencoder_Detector,IEQ_Detector
    detector=Variation_Detector(method="std")
    detector.fit(data)
    anomaly=detector.detect(data,threshold=0.5,windows_size=25)
    detector.plot_anomalies(anomaly,data)


**Detect peaks once the signal denoised**:
::
    from GTSFutur.GTSDetector import Wave_Detector,Period_Detector,Mad_Detector,Variation_Detector,Autoencoder_Detector,IEQ_Detector
    detector=Wave_Detector()
    detector.fit(data,threshold=0.3)
    anomaly=detector.detect(data,alpha=10)
    detector.plot_anomalies(anomaly,data)

**Use a LSTM auto encoder to detect more complexe, global and contextual, anomalies**:
::
    from GTSFutur.GTSDetector import Wave_Detector,Period_Detector,Mad_Detector,Variation_Detector,Autoencoder_Detector,IEQ_Detector
    detector=Autoencoder_Detector()
    detector.fit(100,data,"My_directory_name")
    anomaly=detector.detect(data,"My_directory_name",threshold=5)
    detector.plot_anomalies(anomaly,data)


Some Results
------------

.. figure:: https://raw.githubusercontent.com/gregoritoo/GTSFutur/master/Images/global_anomalies.png

.. figure::  https://raw.githubusercontent.com/gregoritoo/GTSFutur/master/Images/contextual_anomalies.png
