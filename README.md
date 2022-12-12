# Hello FL

Easiest way for ML people to learn FL framework. Because it is the simply integration of Ailabs FL framework and the well know Mnist.

## Getting started

Hello FL consist of three mainly parts: `operator` and `fl_edge.py` and `fl_train.py`.
One who wants to fit their 3rd party model into Ailabs framework only need to replace `fl_train.py` with their training script. But should implement some machanism in their own `fl_train.py`.

## Role introduction

* **Operator** : is the one communicate with our edge and the centrol aggregator.
While it works, it will follow the `lifecycle of FL` to call `the four GRPC API` within fl_edge.py.
The `lifecycle of FL` and `the four GRPC API` will be introduced later..


* **fl_edge.py** : is the example of grpc handler for Ailabs FL framework. It handles all the grpc calls while doing FL training. This script is provided by Ailabs if 3rd party model training code is python base. We provide this script to help handling the FL GRPC call.

* **fl_train.py** : is now mainly consist of `Mnist` training code. Users with 3rd party training code will replace fl_train.py with their own training code(should also be renamed fl_train.py). And implement the machism need to be implement in current `Mnist`  of fl_train.py example to fit Ailabs FL framework. We will introduce this later.


## The five GRPC API

In the last section, we have introduced the 3 parts of Ailabs FL framework : `Operator`, `fl_edge.py` and `fl_train.py`. Grpc implementation have been done in `Operator` and `fl_edge.py` by Ailabs. So one who wants to integrate their model with Ailabs FL framework don't need to implement `The four GRPC API` by theirself but should know how it works. Then they will know when to implement machanism should be implemented in `fl_train.py`.

* **DataValidation** : Before a FL really get started, we will need to do data's validation. When receive this grpc call, fl_edge.py will help to lauch a process which will call the function in the script file you have provided. Then after the validation process have been done. Return this grpc call with OK. If there is something wrong, one should use our log system to report error.


* **TrainInit** : After data's validation have been done, FL get started, `Operator` will call the TrainInit（GRPC call) in `fl_edge.py`, which directly means make the training initialized. Users need to initialize their training after this call have been triggered. `fl_edge.py` helps to handle this event, and will call the `init` in `fl_train.py` （the script which will be replaced with 3rd's training script if users want their 3rd party training to be in FL framwork). So users need to implement `init` in their own `fl_train.py` like example `Mnist` and do initiation in this `init` function. After users have done the initialization in `init` function, users should do trainInitDoneEvent.set() to inform outside that  `fl_edge.py` initialization have done and `fl_edge.py` will help to reply `Operator` this and we will go to next section : LocalTrain.

* **LocalTrain** : After users send trainInitDoneEvent.set(), `Operator` will later call LocalTrain（GRPC call)  in `fl_edge.py`. LocalTrain means to trigger one epoch of training. `fl_edge.py` have help to handle this GPPC and will do `trainStartedEvent.set()` event to `fl_train.py`, and users need to handle this event in `fl_train.py` to lauch a new epoch of training.And after the new epoch of training have done, do `trainFinishedEvent.set()` to inform `fl_edge.py` that this new epoch of training have done.One will do `trainStartedEvent.wait()` in the beginning of their training loop and reply with  `trainStartedEvent.clear()`.


* **TrainInterrupt** : This GPPC has not be implemented currently.

* **TrainFinish** : This GRPC will be call after FL training has done. And `fl_edge.py` has help to close the training process.


## Lifecycle of Ailabs FL framework

<div align="center"><img src="./assets/msc_1.png" style="width:75%"></img></div>

Here we begin with DataValidation phase,

```plaintext
  DataValidation: This is the first phase of FL's lifecycle, we need to check whether our training
  dataset is valid or not in this phase. In this phase, a validation function will be called, and
  you can do validation in this function. After you have validated your dataset, if it is correct
  return grpc call with OK. If something wrong, report error through our log system.
```

After DataValidation phase done,

```plaintext
  TrainInit: In the beginning of Ailabs FL training, we will enter a phase called TrainInit.
  This is also the GRPC call name from Operator. We will do initialization at this phase
  like preprocessing and load pretrained model weight. Also any other things needed to be
  done before training can be done at this phase.

  In this phase, our training communicate outside with 1 event : trainInitDoneEvent.
```

```plaintext
  LocalTrain: In the second phase of Ailabs FL training, we enter a phase called LocalTrain.
  This is also the GRPC call name from Operator. This phase will loop 250-1000 times or
  even more. Every LocalTrain phase means an epoch of FL training. How many LocalTrain
  phases we run throug depends on how many epochs(rounds) a FL project set.

  In this phase, our training communicate outside with python event system, there are 2 events in this phase: trainStartedEvent and trainFinishedEvent.
```

<div align="center"><img src="./assets/msc_2.png" style="width:75%"></img></div>

## FL Logging system in Ailabs FL framework

```python
class LogLevel(Enum):
    INFO = 1
    WARNING = 2
    ERROR = 3


def PackageLogMsg(loglevel: LogLevel, message: string)-> object:
    return {"level":loglevel.name, "message":message}

def UnPackageLogMsg(log :object):
    return log["level"] , log["message"]
```

Totally we have 3 types of logging: `INFO`,`WARNING` and `ERROR`.
Users should pack their message with provided `PackageLogMsg` and put it to the provided queue. Like

```python
logQueue.put(PackageLogMsg(LogLevel.INFO,'Training :trained finished. Start saving model weight'))
```

## The most important things while replace fl_train.py ?

* Must to be done in fl_train.py.
  * Do `trainInitDoneEvent.set()` after initialization has done. To inform that your training initialization has done （before entering training loop）
  * Do `trainStartedEvent.wait()` and `trainStartedEvent.clear()` in the beginning of training loop. To blocking the new epoch of training until it recevie trainStartedEvent event.
  * Do `trainFinishedEvent.set()` in the end of training loop to inform that your one epoch of training has done.

* Good to have
  * using `FL Logging system` to log some customized message whihe in FL triaing will help to realize more when bugs or training problem has occured.


## 3rd that not using Python base, how to do ?

* First, that's see another version msc of this situation. We will not have fl_edge.py. Instead, all GRPC function calls have been handled by fl_edge.py will now handled by 3rd it self.

<div align="left"><img src="./assets/3rd_without_python.png" style="width:100%"></img></div>

* Must to be know to replace fl_edge.py
    * Should at least have a thread （process）[called GRPC handler] handles 3 GRPC funcaiton calls.
    * GRPC handler should be independent, not be in same thread/process of training process.
    * GRPC handler can launch training thread/process after receiving TrainInit GRPC call.
    * Training thread/process will do N epochs training.
    * Training thread/process will wait（blocking）in the beginning of every epoch, wait until GRPC handler receiving LocalTrain GRPC call, and start this epoch of training then finish.
    * At the end of every epoch, Training thread/process will call LocallTrainFinish via GRPC handler.
    * GRPC handler received TrainFinsh GRPC call will end all threads/processes.
    * GRPC handler should also implement logs interface and let Training thread/process send logs.
