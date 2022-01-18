from Models.SNN.BinaryEEGClassifierLIF import BinaryEEGClassifierLIF
from Models.CustomDataset import CustomDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt
import logging


def load_and_run_eval(model_path, train_data_file_path, train_label_file_path, eval_data_file_path, eval_label_file_path, data_cut_front, data_cut_back, model_channels, model_classes, model_dropout, device):
    pass
    # Create Model and Load Model
    model = BinaryEEGClassifierLIF(channels=model_channels).to(device)
    model.load_state_dict(torch.load(model_path))
    # load training set
    train_data_file = train_data_file_path
    train_label_file = train_label_file_path
    training_data = CustomDataset(train_label_file, train_data_file)
    training_generator = DataLoader(training_data,  batch_size=training_data.__len__())

    # generate average spike frequncy rates for class and non class
    spike_frequencies = [0 for x in range(model_classes)]
    sample_amount = [0 for x in range(model_classes)]

    # Generate average spike frequencies
    # load all samples, sum all 1 and 0 spike trains and generate averages, save as comparison value
    for data, labels in training_generator:
        data = data[:, :, data_cut_front:data_cut_back].float()
        data = torch.swapaxes(data, 0, 2)
        data = torch.swapaxes(data, 1, 2)
        data = data.to(device)
        labels = labels.long()
        labels = labels.to(device)
        outputs = model(data)
        outputs = outputs[0].sum(dim=0)  # batch size, spikes
        for i, x in enumerate(labels):
            c_label = int(x)
            spikes = outputs[i]
            spike_frequencies[c_label] += spikes
            sample_amount[c_label] += 1


    spike_frequencies = (np.array(spike_frequencies) / np.array(sample_amount))
    # set model to eval mode
    model.eval()

    # load eval files
    eval_data_file = eval_data_file_path
    eval_label_file = eval_label_file_path
    evaluate_data = CustomDataset(eval_label_file, eval_data_file)
    evaluation_generator = DataLoader(evaluate_data, batch_size=evaluate_data.__len__())
    criterion = torch.nn.L1Loss()

    # for eval data
    with torch.set_grad_enabled(False):
        for data, labels in evaluation_generator:
            # transform data
            data = data[:, :, data_cut_front:data_cut_back].float()
            data = torch.swapaxes(data, 0, 2)
            data = torch.swapaxes(data, 1, 2)
            data = data.to(device)
            labels = labels.long()
            # Convert class labels to target spike frequencies
            s_labels = [spike_frequencies[i] for i in labels]
            s_labels = s_labels.to(device)
            # generate output
            outputs = model(data)
            outputs = outputs[0].sum(dim=0)  # batch size, spikes
            e_loss = criterion(outputs, s_labels)

            # transform to labels via average spike frequency
            distances = np.array([x - spike_frequencies for x in outputs])
            distances = np.absolute(distances)
            out_labels = np.argmin(distances, axis=1)
            diff_l = [0 if out_labels[i] == labels[i] else 1 for i in range(len(labels))]
            eval_c_1 = sum(labels)

            # generate stastics
            mae_acc = sum(diff_l) / len(diff_l)
            eval_acc = 1 - mae_acc
            chance_acc = eval_c_1 / len(labels)
            eval_kappa = (eval_acc - chance_acc) / (1-chance_acc)
            # generate eval acc and print\log
            print(f'Eval Loss: {e_loss:.6f} \t Eval Acc: {eval_acc} \t Eval C1: {eval_c_1} \t Evak kappa: {eval_kappa}')
    # return statistics
    return e_loss, eval_acc, eval_kappa





def run_binary_classification(
    batch_size, shuffle, workers, max_epochs,
        train_data_file_path, train_label_file_path,
        val_data_file_path, val_label_file_path,
        eval_data_file_path, eval_label_file_path,
        model_channels, model_classes,
        model_learning_rate, model_weight_decay, data_cut_front, data_cut_back, save_model, model_name, device
):
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
    criterion = torch.nn.L1Loss()

    min_valid_loss = np.inf
    best_val_epoch = -1

    epoch_statistics = []
    train_loss_statistics = []
    train_acc_statistics = []
    validation_loss_statistics = []
    validation_acc_statistics = []

    # generate average spike frequncy rates for class and non class
    spike_frequencies = [0 for x in range(model_classes)]
    sample_amount = [0 for x in range(model_classes)]

    # load all samples, sum all 1 and 0 spike trains and generate averages, save as comparison value
    for data, labels in training_generator:
        data = data[:, :, data_cut_front:data_cut_back].float()
        data = torch.swapaxes(data, 0, 2)
        data = torch.swapaxes(data, 1, 2)
        data = data.to(device)
        labels = labels.long()
        labels = labels.to(device)
        outputs = model(data)
        outputs = outputs[0].sum(dim=0) # batch size, spikes
        outputs = torch.squeeze(outputs)
        for i, x in enumerate(labels):
            c_label = int(x)
            spikes = outputs[i]
            spike_frequencies[c_label] += spikes.item()
            sample_amount[c_label] += 1
    spike_frequencies = (np.array(spike_frequencies)/np.array(sample_amount))


    # Beginn of Training for each epoch
    for epoch in range(max_epochs):
        # loss and acc
        train_loss = 0.0
        train_mae_acc = 0.0
        # for each batch
        for data, labels in training_generator:
            # transform data
            data = data[:, :, data_cut_front:data_cut_back].float()
            data = torch.swapaxes(data, 0, 2)
            data = torch.swapaxes(data, 1, 2)
            data = data.to(device)
            labels = labels.long()
            # Convert class labels to target spike frequencies
            s_labels = [spike_frequencies[i] for i in labels]
            s_labels = torch.tensor(s_labels)
            s_labels = s_labels.to(device)
            # generate outputs
            optimizer.zero_grad()
            outputs = model(data)
            # convert spike trains to sum of spikes
            outputs = outputs[0].sum(dim=0)  # batch size, spikes
            outputs = torch.squeeze(outputs)
            # compute loss
            loss = criterion(outputs, s_labels)
            # backward loss
            loss.backward()
            # optimzer step
            optimizer.step()
            train_loss += loss.item()

            # convert spike trains to closest label for acc prediction
            distances = np.array([x.cpu() - spike_frequencies for x in outputs])
            distances = np.absolute(distances)
            out_labels = np.argmin(distances, axis=1)
            diff_l = [0 if out_labels[i] == labels[i] else 1 for i in range(len(labels))]
            train_mae_acc += sum(diff_l)
            print(distances, out_labels)
            quit()

        # train_loss = train acc
        l = len(training_generator)*params['batch_size']
        train_mae_acc = 1 - (train_mae_acc/l)

        # reset spike frequencies after training
        spike_frequencies = [0 for x in range(model_classes)]
        sample_amount = [0 for x in range(model_classes)]
        for data, labels in training_generator:
            data = data[:, :, data_cut_front:data_cut_back].float()
            data = torch.swapaxes(data, 0, 2)
            data = torch.swapaxes(data, 1, 2)
            data = data.to(device)
            labels = labels.long()
            labels = labels.to(device)
            outputs = model(data)
            outputs = outputs[0].sum(dim=0)  # batch size, spikes
            for i, x in enumerate(labels):
                c_label = int(x)
                spikes = outputs[i][0]
                spike_frequencies[c_label] += spikes.item()
                sample_amount[c_label] += 1
        spike_frequencies = (np.array(spike_frequencies) / np.array(sample_amount))

        # validation phase
        val_loss = 0.0
        val_mae_acc = 0.0
        # same as training steps but without optimizer step
        with torch.set_grad_enabled(False):
            for data, labels in validation_generator:
                # transform data
                data = data[:, :, data_cut_front:data_cut_back].float()
                data = torch.swapaxes(data, 0, 2)
                data = torch.swapaxes(data, 1, 2)
                data = data.to(device)
                labels = labels.long()
                # Convert class labels to target spike frequencies
                s_labels = [spike_frequencies[i] for i in labels]
                s_labels = s_labels.to(device)
                outputs = model(data)
                outputs = outputs[0].sum(dim=0)  # batch size, spikes
                v_loss = criterion(outputs, s_labels)
                val_loss += v_loss.item()

                # convert spike trains to closest label for acc prediction
                distances = np.array([x - spike_frequencies for x in outputs])
                distances = np.absolute(distances)
                out_labels = np.argmin(distances, axis=1)
                diff_l = [0 if out_labels[i] == labels[i] else 1 for i in range(len(labels))]
                val_mae_acc += sum(diff_l)

        l = len(validation_generator) * params['batch_size']
        # val loss and acc
        val_mae_acc = 1 - (val_mae_acc / l)
        # log info
        logging.info(
            f'Epoch {epoch + 1} \t\t Training Loss: {train_loss / len(training_generator)} \t Training Acc: {train_mae_acc} \t\t Validation Loss: {val_loss / len(validation_generator)} \t Validation Acc: {val_mae_acc}')
        epoch_statistics.append(epoch)
        # append statistics
        train_loss_statistics.append(train_loss / len(training_generator))
        train_acc_statistics.append(train_mae_acc)
        validation_loss_statistics.append(val_loss / len(validation_generator))
        validation_acc_statistics.append(val_mae_acc)
        # check if val loss is reduced
        if min_valid_loss > val_loss / len(validation_generator):
            logging.info(
                f'Validation Loss Decreased({min_valid_loss:.6f}--->{val_loss / len(validation_generator):.6f}')
            min_valid_loss = val_loss / len(validation_generator)
            # save model
            if save_model:
                torch.save(model.state_dict(), f'{model_name}.pth')
                best_val_epoch = epoch

    # Beginn of Eval
    model.eval()
    # Load Eval set
    with torch.set_grad_enabled(False):
        for data, labels in evaluation_generator:
            # transform data
            data = data[:, :, data_cut_front:data_cut_back].float()
            data = torch.swapaxes(data, 0, 2)
            data = torch.swapaxes(data, 1, 2)
            data = data.to(device)
            labels = labels.long()
            # Convert class labels to target spike frequencies
            s_labels = [spike_frequencies[i] for i in labels]
            s_labels = s_labels.to(device)
            outputs = model(data)
            outputs = outputs[0].sum(dim=0)  # batch size, spikes
            e_loss = criterion(outputs, s_labels)

            # convert spike trains to closest label for acc prediction
            distances = np.array([x - spike_frequencies for x in outputs])
            distances = np.absolute(distances)
            out_labels = np.argmin(distances, axis=1)
            diff_l = [0 if out_labels[i] == labels[i] else 1 for i in range(len(labels))]
            eval_c_1 = sum(labels)

            mae_acc = sum(diff_l) / len(diff_l)
            eval_acc = 1 - mae_acc
            chance_acc = eval_c_1 / len(labels)
            eval_kappa = (eval_acc - chance_acc) / (1-chance_acc)
            # generate eval acc and print\log
            print(f'Eval Loss: {e_loss:.6f} \t Eval Acc: {eval_acc} \t Eval C1: {eval_c_1} \t Evak kappa: {eval_kappa}')

    # save last model
    torch.save(model.state_dict(), f'{model_name}_last.pth')

    #combine and return statistics
    statistics = [epoch_statistics,train_loss_statistics, train_acc_statistics ,validation_loss_statistics, validation_acc_statistics]
    return statistics, e_loss, eval_acc, eval_kappa, best_val_epoch