from Models.ANN.BaseSCNN import BaseSCNN
from Models.CustomDataset import CustomDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt
import logging


def load_and_run_eval():
    pass
    # Create Model and Load Model
    # load training set
    # Generate average spike frequencies
    # set model to eval mode
    # load eval files
    # for eval data
        # generate output
        # transform to labels via average spike frequency
        # generate stastics
    # return statistics


def run_binary_classification():
    pass
    # set parameters
    # get train val and eval files
    # create custom datasets
    # prepare data loaders
    # create model, optimizer and loss function, prepare best epoch and min valid loss as well as statistics, prepare spike average array

    # generate average spike frequncy rates for class and non class
        # load all samples, sum all 1 and 0 spike trains and generate averages, save as comparison value

    # Beginn of Training for each epoch

        # loss and acc

        # for each batch
            # transform data
            # generate outputs
            # convert outputs to closest class based on average spike frequncy
            # compute loss
            # backward loss
            # optimzer step
        # train_loss = train acc

        # generate average spike frequncy rates for class and non class
            # load all samples, sum all 1 and 0 spike trains and generate averages, save as comparison value

        # validation phase
            # same as training steps but without optimizer step
        # val loss and acc

        # log info
        # append statistics
        # check if val loss is reduced
            # save model

    # Beginn of Eval
        # Load Eval set
        # Same as above
        # generate eval acc and print\log
    # save last model
    #combine and return statistics
    #

