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



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    y_pred = []
    y_true = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # output

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

            confu = [[0,0],[0,0]]

            cf_matrix = metrics.multilabel_confusion_matrix(y_true, y_pred)
            for matrix in cf_matrix:
                confu[0][0] = confu[0][0] + matrix[0][0]  # TN = True Negative
                confu[0][1] = confu[0][1] + matrix[0][1]  # FP = False Positive
                confu[1][0] = confu[1][0] + matrix[1][0]  # FN = False Negative
                confu[1][1] = confu[1][1] + matrix[1][1]  # TP = True Positive

            precision = confu[1][1] / (confu[0][1]+confu[1][1])
            recall = confu[1][1] / (confu[1][1] + confu[1][0])
            fi_score = 2 * ( precision * recall ) / ( precision + recall )


            result = {
                "dataNum":len(test_loader.dataset),
                "classification":{
                    "statistics": {
                        "mean": {
                            "table":{
                                "cols": ["f1","precision","recall"],
                                "rows": [fi_score,precision,recall]
                            }
                        }
                    },
                    "classNames": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                    "0":{
                        "draw":{
                            "roc":{
                                "x-label" : "fpr",
                                "y-label": "tpr",
                                "x-values":[],
                                "y-values":[],
                            }
                        },
                        "table": {
                            "cols": ["f1", "precision", "recall"],
                            "rows": [[0.68, 0.87, 0.79]]
                        }
                    },
                    "1":{
                        "draw":{
                            "roc":{
                                "x-label" : "fpr",
                                "y-label": "tpr",
                                "x-values":[],
                                "y-values":[],
                            }
                        },
                        "table": {
                            "cols": ["f1", "precision", "recall"],
                            "rows": [[0.68, 0.87, 0.79]]
                        }
                    },
                    "2":{
                        "draw":{
                            "roc":{
                                "x-label" : "fpr",
                                "y-label": "tpr",
                                "x-values":[],
                                "y-values":[],
                            }
                        },
                        "table": {
                            "cols": ["f1", "precision", "recall"],
                            "rows": [[0.68, 0.87, 0.79]]
                        }
                    },
                    "3":{
                        "draw":{
                            "roc":{
                                "x-label" : "fpr",
                                "y-label": "tpr",
                                "x-values":[],
                                "y-values":[],
                            }
                        },
                        "table": {
                            "cols": ["f1", "precision", "recall"],
                            "rows": [[0.68, 0.87, 0.79]]
                        }
                    },
                    "4":{
                        "draw":{
                            "roc":{
                                "x-label" : "fpr",
                                "y-label": "tpr",
                                "x-values":[],
                                "y-values":[],
                            }
                        },
                        "table": {
                            "cols": ["f1", "precision", "recall"],
                            "rows": [[0.68, 0.87, 0.79]]
                        }
                    },
                    "5":{
                        "draw":{
                            "roc":{
                                "x-label" : "fpr",
                                "y-label": "tpr",
                                "x-values":[],
                                "y-values":[],
                            }
                        },
                        "table": {
                            "cols": ["f1", "precision", "recall"],
                            "rows": [[0.68, 0.87, 0.79]]
                        }
                    },
                    "6":{
                        "draw":{
                            "roc":{
                                "x-label" : "fpr",
                                "y-label": "tpr",
                                "x-values":[],
                                "y-values":[],
                            }
                        },
                        "table": {
                            "cols": ["f1", "precision", "recall"],
                            "rows": [[0.68, 0.87, 0.79]]
                        }
                    },
                    "7":{
                        "draw":{
                            "roc":{
                                "x-label" : "fpr",
                                "y-label": "tpr",
                                "x-values":[],
                                "y-values":[],
                            }
                        },
                        "table": {
                            "cols": ["f1", "precision", "recall"],
                            "rows": [[0.68, 0.87, 0.79]]
                        }
                    },
                    "8":{
                        "draw":{
                            "roc":{
                                "x-label" : "fpr",
                                "y-label": "tpr",
                                "x-values":[],
                                "y-values":[],
                            }
                        },
                        "table": {
                            "cols": ["f1", "precision", "recall"],
                            "rows": [[0.68, 0.87, 0.79]]
                        }
                    },
                    "9":{
                        "draw":{
                            "roc":{
                                "x-label" : "fpr",
                                "y-label": "tpr",
                                "x-values":[],
                                "y-values":[],
                            }
                        },
                        "table": {
                            "cols": ["f1", "precision", "recall"],
                            "rows": [[0.68, 0.87, 0.79]]
                        }
                    },
                }
            }


    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def validate(cofig_path, namespace, trainInitDoneEvent, trainStartedEvent, trainFinishedEvent, logQueue, hasPretrainedWeight=None):
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




