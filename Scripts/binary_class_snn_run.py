from Models.SNN.BinaryEEGClassifierLIF import BinaryEEGClassifierLIF
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


def run_binary_classification(
    batch_size, shuffle, workers, max_epochs,
        train_data_file_path, train_label_file_path,
        val_data_file_path, val_label_file_path,
        eval_data_file_path, eval_label_file_path,
        model_channels, model_classes, model_dropout,
        model_learning_rate, model_weight_decay, data_cut_front, data_cut_back, save_model, model_name, device
):
    pass
    # set parameters
    params = {'batch_size': batch_size,
              'shuffle': shuffle,
              'num_workers': workers}
    max_epochs = max_epochs

    # get train val and eval files
    train_data_file = train_data_file_path
    train_label_file = train_label_file_path

    val_data_file = val_data_file_path
    val_label_file = val_label_file_path

    eval_data_file = eval_data_file_path
    eval_label_file = eval_label_file_path

    # create custom datasets
    training_data = CustomDataset(train_label_file, train_data_file)
    validate_data = CustomDataset(val_label_file, val_data_file)
    evaluate_data = CustomDataset(eval_label_file, eval_data_file)

    # prepare data loaders
    training_generator = DataLoader(training_data, **params)
    validation_generator = DataLoader(validate_data, **params)
    evaluation_generator = DataLoader(evaluate_data, batch_size=evaluate_data.__len__())

    # create model, optimizer and loss function, prepare best epoch and min valid loss as well as statistics, prepare spike average array
    model = BinaryEEGClassifierLIF(channels=model_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=model_learning_rate, weight_decay=model_weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    min_valid_loss = np.inf
    best_val_epoch = -1

    epoch_statistics = []
    train_loss_statistics = []
    train_acc_statistics = []
    validation_loss_statistics = []
    validation_acc_statistics = []

    # generate average spike frequncy rates for class and non class
    spike_frequencies = [0 for x in model_classes]
    sample_amount = [0 for x in model_classes]

    # load all samples, sum all 1 and 0 spike trains and generate averages, save as comparison value
    for data, labels in training_generator:
        pass
        data = data[:, :, :, data_cut_front:data_cut_back].float()
        data = data.to(device)
        labels = labels.long()
        labels = labels.to(device)
        outputs = model(data)
        outputs = outputs.sum() # add dim?
        for i, x in enumerate(labels):
            c_label = int(x)
            spikes = outputs[i]
            spike_frequencies[c_label] =+ spikes
            sample_amount[c_label] =+ 1

    spike_frequencies = (np.array(spike_frequencies)/np.array(sample_amount)).tolist()

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

