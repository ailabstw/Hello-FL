from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
import logging
import importlib
from fl_enum import PackLogMsg,LogLevel


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def init(cofig_path, namespace, trainInitDoneEvent, trainStartedEvent, trainFinishedEvent, logQueue, hasPretrainedWeight=None):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args, unparsed = parser.parse_known_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset1 = None
    dataset2 = None
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    try:
        logQueue.put(PackLogMsg(LogLevel.INFO,"Loading training dataset .."))
        dataset1 = datasets.MNIST('/data', train=True, download=False, transform=transform)
        dataset2 = datasets.MNIST('/data', train=False, download=False, transform=transform)
    except Exception as err:
        logQueue.put(PackLogMsg(LogLevel.ERROR,"Loading datasets failed .. reason: " + str(err)))
        return

    if namespace is not None:
        namespace.dataset_size = len(dataset1)  #c

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # ------------------------
    # 1.Load pretrained weight if existed before traininig.
    # ------------------------
    logQueue.put(PackLogMsg(LogLevel.INFO,"Initialization :Check if pretrained weight exist."))
    if hasPretrainedWeight != None:
        try:
            model.load_state_dict(torch.load(namespace.pretrainedModelPath)["state_dict"])
            logQueue.put(PackLogMsg(LogLevel.INFO,"Initialization :Pretrained weight is being loaded."))

        except Exception as err:
            logQueue.put(PackLogMsg(LogLevel.ERROR,"Loading pretrained failed ..reason: " + str(err)))
            return
    else:
        logQueue.put({"level":"info", "message":"Initialization :Pretrained weight not found."})
    # ------------------------
    # 2.Inform main process (fl_edge.py) that training initialization has been done.
    #   Set this event before entering training loop
    # ------------------------
    print("training initialization has done !")
    trainInitDoneEvent.set()

    print("first training epoch started ...")

    logQueue.put(PackLogMsg(LogLevel.INFO,"Initialization :initalization finished."))


    logQueue.put(PackLogMsg(LogLevel.WARNING,"Training :test warning."))

    for epoch in range(args.epochs):
        if epoch > 0:
            # ------------------------
            # 3.Wait starting event: Wait trainStartedEvent to be set and start to new epoch of local training.
            # ------------------------
            print("wait until new epoch training started ...")

            logQueue.put(PackLogMsg(LogLevel.INFO, "Wait until new epoch training started ..."))

            trainStartedEvent.wait()
            trainStartedEvent.clear()

            logQueue.put(PackLogMsg(LogLevel.INFO, "Epoch " + str(epoch) + " : training starting !"))
            print("Epoch ", epoch ," training starting !")

            # ------------------------
            # 4.Load global model as pretraind: Load the newest global model weight as pretrained model weight.
            # ------------------------

            # 4-1:See information of global model
            # weight = torch.load(namespace.pretrained_path)
            # print("pretrained keys :", weight.keys())
            # print("pretrained state_dict :", weight["state_dict"].keys())

            # 4-2:Load pretrained weight from global model

            try:
                model.load_state_dict(torch.load(namespace.pretrainedModelPath)["state_dict"])
                logQueue.put(PackLogMsg(LogLevel.INFO, "Loading global model weight.."))
            except Exception as err:
                logQueue.put(PackLogMsg(LogLevel.ERROR,"loading global model is failed ..: " + str(err)))
                return

        # ------------------------
        # 5.Local training started: Start this epoch of training and validaiton.
        # ------------------------
        try:
            train(args, model, device, train_loader, optimizer, epoch)
            logQueue.put(PackLogMsg(LogLevel.INFO, "Local train is ongoing..."))
        except Exception as err:
            logQueue.put(PackLogMsg(LogLevel.ERROR,"Local train is failed ..: " + str(err)))
            return

        try:
            test(model, device, test_loader)
            logQueue.put(PackLogMsg(LogLevel.INFO, "Testing(validation) is ongoing..."))
        except Exception as err:
            logQueue.put(PackLogMsg(LogLevel.ERROR,"Testing(validation) is is failed ..: " + str(err)))
            return

        scheduler.step()

        # ------------------------
        # 6.Saving local model : should save the weight of this epoch.
        # Saved model path will be epoch_{number of epoch}.ckpt and set namespace.epoch_path = epoch_{number of epoch}.ckpt
        # (in order to make main process(fl_edge.py) knows where the weight of this epoch is by namespace)
        # ------------------------

        y_pred = []
        y_true = []

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            output = model(inputs) # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

            confu = [[0,0],[0,0]]

            cf_matrix = metrics.multilabel_confusion_matrix(y_true, y_pred)
            for matrix in cf_matrix:
                confu[0][0] = confu[0][0] + matrix[0][0]
                confu[0][1] = confu[0][1] + matrix[0][1]
                confu[1][0] = confu[1][0] + matrix[1][0]
                confu[1][1] = confu[1][1] + matrix[1][1]

            namespace.metrics = {
                "epoch":epoch,
                "basic/confusion_tn":confu[0][0],
                "basic/confusion_fp":confu[0][1],
                "basic/confusion_fn":confu[1][0],
                "basic/confusion_tp":confu[1][1],
                "datasetSize":len(test_loader.dataset)*10,
                "basic/weight":1.0,
            }


        # Save and Namespace
        if namespace is not None:
            logQueue.put(PackLogMsg(LogLevel.INFO,"Training :trained finished. Start saving model weight"))
            namespace.epoch_path = f'epoch_{epoch}.ckpt'
            try:
                torch.save({'state_dict': model.state_dict()}, namespace.epoch_path)  # original
            except Exception as err:
                logQueue.put(PackLogMsg(LogLevel.ERROR,"Saving model failed ..: " + str(err)))
                return

            logging.info(f"save to : [{namespace.epoch_path}]")

            # namespace.metrics = valid_metrics['0'].copy()

        # ------------------------
        # 5.Set trainFinishedEvent to inform main process (fl_edge.py) that this epoch of local training has done.
        # ------------------------
        if epoch == 0:
            print("first training epoch end ...")
            trainStartedEvent.wait()
            trainStartedEvent.clear()
        # clear start event anyway
        trainFinishedEvent.set()
