from concurrent import futures
from grpcpb import service_pb2, service_pb2_grpc
from multiprocessing import Process
import argparse
import fl_train
import grpc
import logging
import multiprocessing
import os
import shutil
import time
from fl_enum import UnPackLogMsg,LogLevel

OPERATOR_URI = os.getenv("OPERATOR_URI") or "127.0.0.1:8787"
APPLICATION_URI = "0.0.0.0:7878"
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
trainStartedEvent = multiprocessing.Event()
trainFinishedEvent = multiprocessing.Event()
mgr = multiprocessing.Manager()
namespace = mgr.Namespace()
loop = True
cofig_path = ""

logQueue = multiprocessing.Queue()
trainingProcess = None

def logEventLoop(logQueue):
    while True:
        obj = logQueue.get()
        channel = grpc.insecure_channel(OPERATOR_URI)
        stub = service_pb2_grpc.EdgeOperatorStub(channel)
        level, message = UnPackLogMsg(obj)
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


def train_model(yml_path, namespace, trainStartedEvent, trainFinishedEvent, epochCount):

    logging.info(f"pretrained (global) model path: [{namespace.pretrainedModelPath}]")
    logging.info(f"local model path: [{namespace.localModelPath}]")
    logging.info(f"epoch count: [{epochCount}]")

    logging.info("trainer has been called to start training.")
    trainStartedEvent.set()

    logging.info("wait until the training has done.")
    trainFinishedEvent.wait()
    logging.info("training finished event clear.")
    trainFinishedEvent.clear()

    logging.info(f"model last epoch path: [{namespace.epoch_path}]")
    shutil.copyfile(namespace.epoch_path, namespace.localModelPath)

    logging.info(f"model datasetSize: {namespace.dataset_size}")
    logging.info(f"model metrics: {namespace.metrics}")
    logging.info(f"config.GRPC_CLIENT_URI: {OPERATOR_URI}")
    try:
        channel = grpc.insecure_channel(OPERATOR_URI)
        logging.info(f"grpc.insecure_channel: {OPERATOR_URI} Done.")
        stub = service_pb2_grpc.EdgeOperatorStub(channel)
        logging.info("service_pb2_grpc.EdgeOperatorStub Done.")
        result = service_pb2.LocalTrainResult(
            error=0, datasetSize=namespace.dataset_size, metadata=namespace.metadata, metrics=namespace.metrics
        )
        logging.info("service_pb2.LocalTrainResult Done.")
        response = stub.LocalTrainFinish(result, timeout=30)
        logging.info("stub.LocalTrainFinish Done.")
        logging.info(f"namespace: {namespace}")
        logging.debug(f"sending grpc message succeeds, response: {response}")
        channel.close()
        logging.info("channel.close() Done.")
    except grpc.RpcError as rpc_error:
        logging.error(f"grpc error: {rpc_error}")
    except Exception as err:
        logging.error(f"got error: {err}")


class EdgeAppServicer(service_pb2_grpc.EdgeAppServicer):
    def DataValidate(self, request, context):
        resp = service_pb2.Empty()
        logging.info(f"Sending response: {resp}")
        return resp

    def TrainInit(self, request, context):
        logging.info("TrainInit, reset the current epoch, increase the version")
        namespace.localModelPath = os.environ['LOCAL_MODEL_PATH']
        namespace.pretrainedModelPath = os.environ['GLOBAL_MODEL_PATH']

        global trainingProcess
        global trainStartedEvent
        global trainFinishedEvent
        global logQueue
        trainInitDoneEvent = multiprocessing.Event()
        trainingProcess = Process(
            target=fl_train.init, args=(cofig_path, namespace, trainInitDoneEvent, trainStartedEvent, trainFinishedEvent, logQueue, None)
        )
        trainingProcess.start()
        # trainInitDoneEvent.wait() to wait until trainingProcess's initialization has been done.
        trainInitDoneEvent.wait()
        resp = service_pb2.Empty()
        logging.info(f"Sending response: {resp}")
        return resp

    def LocalTrain(self, request, context):
        global trainStartedEvent
        global trainFinishedEvent
        global trainingProcess
        global namespace
        global cofig_path
        logging.info("LocalTrain")
        p = Process(
            target=train_model,
            args=(
                cofig_path,
                namespace,
                trainStartedEvent,
                trainFinishedEvent,
                request.EpR,
            ),
        )
        p.start()
        resp = service_pb2.Empty()
        logging.info(f"Sending response: {resp}")
        return resp

    def TrainInterrupt(self, request, context):
        # Not Implemented
        return service_pb2.Empty()

    def TrainFinish(self, _request, _context):
        logging.info("TrainFinish")
        global loop
        loop = False
        return service_pb2.Empty()

def serve():
    logging.basicConfig(level=logging.DEBUG)
    logging.info(f"Start server... {APPLICATION_URI}")

    global logQueue
    p = Process(
        target=logEventLoop, args=(logQueue,)
    )
    p.start()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_EdgeAppServicer_to_server(EdgeAppServicer(), server)
    server.add_insecure_port(APPLICATION_URI)
    server.start()
    while loop:
        time.sleep(10)
    server.stop(None)
    time.sleep(200)
    os._exit(os.EX_OK)


if __name__ == "__main__":
    # Parse Yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out', type=str, help='Yaml name')
    args, unparsed = parser.parse_known_args()

    cofig_path = args.out

    serve()
