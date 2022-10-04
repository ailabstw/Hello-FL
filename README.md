# Hello FL

Easiest way for ML people to learn FL framework. Because it is the simply integration of Ailabs FL framework and the well know Mnist.

## Getting started

Hello FL consist of three mainly parts: `operator` and `fl_edge.py` and `fl_train.py`.
But one who wants to fit there model into Ailabs framework only need to replace `fl_train.py` with their training script.

* Operator : is the one communicate with our edge and the centrol aggregator. 
While it works, it will follow the `lifecycle of FL` (will be introduced later）to call `the four GRPC API` within fl_edge.py. 
The **lifecycle of FL** and `the four GRPC API` will be introduced later.


* fl_edge.py : is the example of grpc handler for Ailabs FL framework. It handles all the grpc calls while doing FL training.

* fl_train.py : is mainly consist of `Mnist` training code. Some changes need to be implemented in fl_train.py to fit Ailabs FL framework. We will introduce this later.


## The four GRPC API

In the last section, we have said that hello FL consist of 3 parts : `Operator`, `fl_edge.py` and `fl_train.py`. Grpc implementation have been done in `Operator` and `fl_edge.py` by Ailabs. So one who wants to integrate their model into Ailabs FL framwork no need to implement `The four GRPC API` theirself . But one need to know how it works, then they will how to operatate within the  `fl_train.py`.

* TrainInit : When a FL get started, `Operator` will call the TrainInit （GRPC）in `fl_edge.py`, which directly means make the train initialized. Users need to initialize their training after this call have been trigger. `fl_edge.py` helps to handle this event, and will call the `init` in `fl_train.py` （which will be replace with one's script）. So if user need to implement `init` in their `fl_train.py` and do initiation in this function. After user have done the initialization in `init`, should do trainInitDoneEvent.set() to infer `fl_edge.py` that initialization have done and `fl_edge.py` will help to reply `Operator` that initialization have done and go to next section : LocalTrain.

* LocalTrain : After user send trainInitDoneEvent.set(), `Operator` will later call LocalTrain in `fl_edge.py`. LocalTrain means to trigger one epoch of training. `fl_edge.py` have help to handle this GPPC and will do `trainStartedEvent.set()` event to `fl_train.py`, and user need to handle this event in `fl_train.py` to lauch a new epoch of training.And after the new epoch of training have done, do `trainFinishedEvent.set()` to inform `fl_edge.py` that this new epoch of training have done.

One will do `trainStartedEvent.wait()` in the begining of their training loop and reply with  `trainStartedEvent.clear()`.


* TrainInterrupt : This GPPC has not be implemented currently.

* TrainFinish : This GRPC will be call after FL training has done. And `fl_edge.py` has help to close the training process.


## MSC of Ailabs FL framework

* TrainInit Fhase
![image](uploads/24a488d7ba78a5abbeba054447667eff/image.png)

* LocalTrain Fhase
![image](uploads/97545714b82ce65f1729680518a09702/image.png)

## FL Logging system in Ailabs FL framework


## The most important things while replace fl_train.py ?

* Must to be done in fl_train.py.
  * Do `trainInitDoneEvent.set()` after initialization has done. To inform that your training initialization has done （before entering training loop）
  * Do `trainStartedEvent.wait()` and `trainStartedEvent.clear()` in the begining of training loop. To blocking the new epoch of training until it recevie trainStartedEvent event.
  * Do `trainFinishedEvent.set()` in the end of training loop to inforn that your one epoch of training has done. 

* Good to have
  * using `FL Logging system` to log some customized message whihe in FL triaing will help to realize more when bugs or training problem has occured.

