# -*- coding: utf-8 -*-

"""
Data loader: MNIST
"""

import os
import shutil
import numpy as np
import logging
import tqdm

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
#from skewedmnist import SkewedMnist
import matplotlib.pyplot as plt


def mnist(data_path='data/MNIST_data/'):
    # normalize the data
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])

    # download and load the data
    trainset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    testset = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    return trainset, testset, classes


def cifar10(data_path='data/CIFAR10_data/'):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, testset, classes


"""
def skewedmnist(data_path='data/MNIST_data', skew_list = [1.0] * 10):
    # normalize the data
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])

    # download and load the data
    trainset = SkewedMnist(data_path, train=True, download=True, transform=transform, skewList=skew_list)
    testset = SkewedMnist(data_path, train=False, download=True, transform=transform, skewList=skew_list)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    return trainset, testset, classes
"""


def get_dataloader(trainset, testset, batch_size, eval_batch_size, seed):
    torch.manual_seed(seed)
    init_fn = np.random.seed(seed)

    num_train = len(trainset)
    indices = list(range(num_train))
    split = len(testset) # dev set with the same size as test set

    np.random.seed(seed)
    np.random.shuffle(indices)

    train_idx, dev_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    dev_sampler = SubsetRandomSampler(dev_idx)
    #test_batch_size = len(testset) # evaluate all test examples at once
    #test_batch_size = int(len(testset)/100) # = 100

    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, worker_init_fn=init_fn)
    devloader = DataLoader(trainset, batch_size=eval_batch_size, sampler=dev_sampler, worker_init_fn=init_fn)
    testloader = DataLoader(testset, batch_size=eval_batch_size, shuffle=False, worker_init_fn=init_fn)
    return trainloader, testloader, devloader


def show_image(dataloader):
    images, labels = iter(dataloader).next()
    plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')


def make_log_dir(log_dir, overwrite=False):
    if os.path.isdir(log_dir):
        if not overwrite:
            raise FileExistsError(
                "Model directory exists and overwriting is disabled.")
        # delete previous directory to start with empty dir again
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    return log_dir

def plot_confusion_matrix(cm, classes, output_path=None):
    # normalize confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    cmap = plt.cm.Blues # colormap
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label', xlabel='Predicted label')

    if max([len(n) for n in classes]) > 5:
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    if output_path is not None:
        assert output_path.endswith(".png")
        plt.savefig(output_path)

    plt.close()

    return fig

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)