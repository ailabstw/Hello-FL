# Taiwan AI Labs Federated Framework

*Read this in other languages: [English](README.md), [正體中文](README.zh-tw.md).*


## 什麼是 Taiwan AI Labs Federated Framework

Taiwan AI Labs Federated Framework (以下簡稱 AILabs FL Framework) 是由台灣人工智慧實驗室推出的聯邦式學習框架; 透過此框架, 可以簡單的實現聯邦式學習. AILabs FL Framework 包含三個組件 : `Operator`, `gRPC handler` 與 `Training Process`, 並定義了一組 gRPC 介面作為溝通使用. `Operator` 本身已經包含在 AILabs FL framework 內, 而 `gRPC handler` 與 `Training Process` 需要由使用者來實作.


在 AILabs FL Framework 中, `Operator` 負責發號司令, 在訓練過程中會發送 RPC (Remote Procedure Call) 給 `gRPC handler`. 當 `gRPC handler` 接收到 RPC 後, `gRPC handler` 必須根據收到的 RPC 執行對應的操作. `gRPC handler` 必須要能夠即時的接收並處理 RPC, 因此建議以一個獨立的 thread / process 來實作. 而 `Training Process` 則是另一個 thread / process, 用以處理長時間的運算.

<div align="left"><img src="./assets/3rd_without_python.png" style="width:100%"></img></div>

* 圖1. AILabs FL Framework 運作過程

## AILabs FL Framework 流程與 gRPC Server 介面

在聯邦式學習的訓練過程中, `Operator` 會在各個階段向 `gRPC handler` 發送不同的 gRPC, 此時 `Operator` 為 gRPC client 而 `gRPC handler` 為 gRPC server.

1. **DataValidate** (圖1 階段 A)
  這是聯邦式學習的第一個階段, `Operator` 會向 `gRPC handler` 發送 `DataValidate` gRPC. `gRPC handler` 收到後, 要驗證訓練資料是否有效. 如果資料有效則回傳 OK; 如果資料無效則必須透過 Logging 介面回報錯誤.
  <br/>

2. **TrainInit** (圖1 階段 B)
  在 `DataValidate` 完成後, `Operator` 會向 `gRPC handler` 發送 `TrainInit` gRPC. `gRPC handler` 收到後, 要進行訓練初始化, 諸如資料前處理, 載入模型初始權重等等需要在正式開始訓練前完成的操作.
  <br/>

3. **LocalTrain** (圖1 階段 C)
  在 `TrainInit` 完成後, `Operator` 會向 `gRPC handler` 發送多次 `LocalTrain` gRPC, 而次數取決於訓練計畫 (Training Plan) 的設定. 每一個 `LocalTrain` gRPC 代表訓練過程中的一個 epoch, 而 `gRPC handler` 收到 `LocalTrain` gRPC 後, 會在 `Training Process` 啟動一次本地訓練, 訓練完成後產生一個本地模型權重以及一些驗證資料. 這些驗證資料要透過 `LocalTrainFinish` gRPC 介面回傳給 `Operator`.
  <br/>

4. **TrainFinish** (圖1 階段 D)
  當所有訓練回合完成後, `Operator` 會向 `gRPC handler` 發送 `TrainFinish` gRPC. `gRPC handler` 收到後, 則需要關閉 `Training Process` 以及 `gRPC handler`.
  <br/>

5. **TrainInterrupt**
  在一些特殊情況下, `Operator` 會向 `gRPC handler` 發送 `TrainInterrupt` gRPC. `gRPC handler` 收到後, 則需要關閉 `Training Process` 以及 `gRPC handler`.


## gRPC Client Interface
  除了上節所述的五個 gRPC server 介面, `gRPC handler` 還需要實作兩個 gRPC client 介面:

1. **LocalTrainFinish**
    當每一輪的本地訓練完成後, `gRPC handler` 要向 `Operator` 發送 `LocalTrainFinish` gRPC 並將驗證資料回傳給 `Operator`.
    `LocalTrainFinish` 的定義如下:
    ```proto
    service EdgeOperator {
      rpc LocalTrainFinish (LocalTrainResult) returns (Empty) {}
    }

    message LocalTrainResult {
      enum Error {
        SUCCESS = 0;
        FAIL = 1;
      }
      Error error = 13;
      int32 datasetSize = 14;
      map<string, string> metadata = 15; // FUTURE:
      map<string, double> metrics = 18;
    }
    ```

2. **LogMessage** (logging interface in AILabs FL Framework)
    在 AILabs FL Framework 中, 定義了 `LogMessage` gRPC 介面做為 `gRPC handler` 與 `Training Process` 向 `Operator` 回傳 log 的管道.
    `LogMessage` 的定義如下:
    ```proto
      service EdgeOperator {
        rpc LogMessage (Log) returns (Empty) {}
      }

      message Log {
        string level = 1;
        string message = 2;
      }
    ```
    Log 有三個等級, 分別是 `INFO`,`WARNING` 以及 `ERROR`:
    * `INFO`: 回傳一般的資訊.
    * `WARNING`: 回傳需要注意但不嚴重的錯誤訊息.
    * `ERROR`: 當訓練過程有嚴重錯誤而無法繼續進行, 回傳 `ERROR` log; `Operator` 收到 `ERROR` log 後會中止整個訓練.

--------------------------------------------------------------------------------------------------------------------

# Hello-FL

Hello-FL 專案由 python 撰寫, 使用 MNIST 作示範, 讓開發者能夠快速上手 Ailabs's FL Framework.

在 Hello-FL 之中, 我們實作了一個 docker image, 裡面實作了 `gRPC handlder`, `Training Process`, 5 個 gRPC server interfaces 以及 2 個 gRPC client interfaces.

`gRPC handlder` 的實作為 `fl_edge.py`, 而 `Training Process` 實作為 `fl_train.py`.
* `fl_edge.py`: 實作一個 gRPC server 負責接收 gRPC.
* `fl_train.py`: 實作一個 process 來處理資料前處理以及訓練等耗時工作.


## 訓練流程


### 階段 A 與階段 B
<div align="left"><img src="./assets/msc_1.png" style="width:75%"></img></div>

#### 階段 A
`Operator` 會發送 `DataValidate` gRPC 給 `gRPC handler` (fl_edge.py 內的 `EdgeAppServicer`), `gRPC handler` 收到後應進行資料驗證 (在我們的範例中並沒有實作), 預設的處理時限為1個小時. 完成資料驗證後, 不論資料是有效或無效, `gRPC handler` 都會回傳 OK, 代表驗證已完成. 若驗證結果為有效, 將不再有一步的動作; 若是資料無效, `gRPC handler` 會透過 logging interface 回傳 `ERROR` log 給 `Operator`.

#### 階段 B
`Operator` 在收到驗證完成的回報後, 會發送 `TrainInit` gRPC 給 `gRPC handler`, `gRPC handler` 收到後進行訓練初始化, 預設處理時限為1個小時. 這時 `EdgeAppServicer` 會執行訓練前的
   1. 載入預訓練模型權重
   2. 以獨立的 process 啟動 `Training Process`
   3. 若有其他初始化操作也可在這時執行

`Training Process` 被叫起後, 會進入一個迴圈 (fl_train.py, #167), 並透過 python event 來做同步 (註A), 等待進一步指令. `gRPC handler` 處理完訓練的初始化後, 會回傳 OK 給`Operator`. 若是初始化過程有錯誤, 則應透過 logging interface 傳送 `ERROR` log 給 `Operator` (fl_train.py, #121).

#### 階段 C
<div align="left"><img src="./assets/msc_2.png" style="width:75%"></img></div>

`Operator` 在收到訓練初始化完成的回報後, 會發送 `LocalTrain` gRPC 給 `gRPC handler`, `gRPC handler` 收到後會透過 python event (註B) 通知 `Training Process` 開始訓練. 每一輪訓練結束後會產出本地模型權重 (weight.ckpt) 以及validation metrics; Validation metrics 會透過 `LocalTrainFinish` gRPC 回傳給 `Operator`, 而本地模型權重則是透過由 `Operator` 與 `Appication` mount 同資料夾的方式, 讓 `Operator` 能直接存取產出. `Operator`收 `LocalTrainFinish` gRPC 後, 會對本地模型權重與 validating metrics 進行處理, 處理完後會再次發送 `LocalTrain` gRPC 給 `gRPC handler`, 開始下一輪的本地訓練. 如此反覆, 直到達到指定的訓練輪數（註C).

#### 階段 D
 訓練輪數達到訓練計畫指定的數量後, `Operator` 會發送 `TrainFinish` gRPC 給 `gRPC handler`, 此時 `gRPC handler` 會把 `Training Process` 關閉, 再將自己關閉.
* 註 A: `Training Process` 在完成初始化後, 會執行 `trainInitDoneEvent.set()` 來告知 `gRPC Handler` 初始化完成 (fl_train.py, #158), 並進入 for loop 中, 等候 `trainStartedEvent` 通知 (fl_train.py, #167).
<br/>
* 註 B: `gRPC handler` 會呼叫 `trainStartedEvent.set()` 來通知 `Training Process` 開始訓練. `Training Process` 收到通之後, 會呼叫 `trainStartedEvent.clear()` 通知 `gRPC handler` 訓練已經開始, 並在訓練結束時呼叫 `trainFinishedEvent.set()` 來通知 `gRPC handler` 已完成訓練.
<br/>
* 註 C: 實際執行的次數取決於訓練計畫中指定的訓練輪數.


## Log System in Hello-FL

在 Hello-FL 專案中, 使用 process 與 queue 來實作 log 發送機制. 要發送的 log 會被送入 queue 中, 再由 log process 取出, 透過 `LogMessage` gRPC client interface 發給 `Operator`.

```python
def logEventLoop(logQueue):
    while True:
        obj = logQueue.get()
        channel = grpc.insecure_channel(OPERATOR_URI)
        stub = service_pb2_grpc.EdgeOperatorStub(channel)
        level, message = UnPackageLogMsg(obj)
        logging.info(f"Send log level: {level} message: {message}")
        message = service_pb2.Log(
            level = level,
            message = message
        )
        try:
            response = stub.LogMessage(message)
            logging.info(f"Log sending succeeds, response: {response}")
        except grpc.RpcError as rpc_error:
            logging.error(f"grpc error: {rpc_error}")
        except Exception as err:
            logging.error(f"got error: {err}")
        if level == LogLevel.ERROR:
            global loop
            loop = False
            return
```

## 如何建立 docker image

在 dockerfile 資料夾內提供了 Dockerfile, 可用來建立 docker image; 只要執行以下指令, 即可建立名為 `hello-fl:1.0` 的 docker image.

```bash
docker build --tag hello-fl:1.0 -f ./dockerfile/Dockerfile.fl.edge .
```