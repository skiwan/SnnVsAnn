from Models.ANN.BaseSCNN import BaseSCNN
from Models.CustomDataset import CustomDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt

def load_and_run_eval(model_path, eval_data_file_path, eval_label_file_path, data_cut_front, data_cut_back, model_channels, model_classes, model_dropout, device):
    model = BaseSCNN(channels=model_channels, base_filters=8, classes=model_classes, image_height=44, dropout_value=model_dropout).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    eval_data_file = eval_data_file_path
    eval_label_file = eval_label_file_path
    evaluate_data = CustomDataset(eval_label_file, eval_data_file)
    evaluation_generator = DataLoader(evaluate_data, batch_size=evaluate_data.__len__())
    criterion = torch.nn.CrossEntropyLoss()
    softmax_l = torch.nn.LogSoftmax(dim=1)
    with torch.set_grad_enabled(False):
        for data, labels in evaluation_generator:
            data = data[:, :, :,data_cut_front:data_cut_back].float()
            data = data.to(device)
            labels = labels.long()
            labels = labels.to(device)
            outputs = model(data)
            outputs = softmax_l(outputs)
            e_loss = criterion(outputs, labels)
            calcs = torch.argmax(outputs, dim=1)
            diff_l = [0 if calcs[i] == labels[i] else 1 for i in range(len(labels))]
            diff_s = sum(diff_l)
            eval_c_1 = sum(labels)
            mae = diff_s / len(diff_l)
            eval_acc = 1 - mae
            chance_acc = eval_c_1/len(labels)
            eval_kappa = (eval_acc - chance_acc) / (1-chance_acc)
            print(f'Eval Loss: {e_loss:.6f} \t Eval Acc: {eval_acc} \t Eval C1: {eval_c_1} \t Evak kappa: {eval_kappa}')

    return e_loss, eval_acc, eval_kappa

def run_binary_classification(
        batch_size, shuffle, workers, max_epochs,
        train_data_file_path, train_label_file_path,
        val_data_file_path, val_label_file_path,
        eval_data_file_path, eval_label_file_path,
        model_channels, model_classes, model_dropout,
        model_learning_rate, model_weight_decay, data_cut_front, data_cut_back, save_model, model_name, device
):
    params = {'batch_size': batch_size,
              'shuffle': shuffle,
              'num_workers': workers}
    max_epochs = max_epochs

    train_data_file = train_data_file_path #'/home/merlinsewina/MaWork/SnnVsAnn/prep/A01T_Prep_train_data.npy'
    train_label_file = train_label_file_path# '/home/merlinsewina/MaWork/SnnVsAnn/prep/A01T_Prep_train_labels.npy'

    val_data_file = val_data_file_path# '/home/merlinsewina/MaWork/SnnVsAnn/prep/A01T_Prep_validate_data.npy'
    val_label_file = val_label_file_path# '/home/merlinsewina/MaWork/SnnVsAnn/prep/A01T_Prep_validate_labels.npy'

    eval_data_file = eval_data_file_path# '/home/merlinsewina/MaWork/SnnVsAnn/prep/A01E_7_30_class1_CWT.npy'
    eval_label_file = eval_label_file_path# '/home/merlinsewina/MaWork/SnnVsAnn/prep/A01E_7_30_Features_class1_labels.npy'

    training_data = CustomDataset(train_label_file, train_data_file)
    validate_data = CustomDataset(val_label_file, val_data_file)
    evaluate_data = CustomDataset(eval_label_file, eval_data_file)

    training_generator = DataLoader(training_data, **params)
    validation_generator = DataLoader(validate_data, **params)
    evaluation_generator = DataLoader(evaluate_data, batch_size=evaluate_data.__len__())


    model = BaseSCNN(channels=model_channels, base_filters=8, classes=model_classes, image_height=44, dropout_value=model_dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=model_learning_rate, weight_decay=model_weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    softmax_l = torch.nn.LogSoftmax(dim=1)
    min_valid_loss = np.inf
    best_val_epoch = -1


    statistics = []
    epoch_statistics = []
    train_loss_statistics = []
    train_acc_statistics = []
    validation_loss_statistics = []
    validation_acc_statistics = []


    for epoch in range(max_epochs):
        train_loss = 0.0
        train_mae_acc = 0.0
        # Training
        for data, labels in training_generator:
            data = data[:,:,:,data_cut_front:data_cut_back].float()
            data = data.to(device)
            labels = labels.long()
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            outputs = softmax_l(outputs)
            t_loss = criterion(outputs, labels)
            t_loss.backward()
            optimizer.step()
            train_loss += t_loss.item()

            calcs = torch.argmax(outputs, dim=1)
            diff_l = [0 if calcs[i] == labels[i] else 1 for i in range(len(labels))]
            train_mae_acc += sum(diff_l)
            # print statistics
        l = len(training_generator)*params['batch_size']
        train_mae_acc = 1 - (train_mae_acc/l)

        val_loss = 0.0
        val_mae_acc = 0.0
        # Validation
        with torch.set_grad_enabled(False):
            for data, labels in validation_generator:
                data = data[:, :, :,data_cut_front:data_cut_back].float()
                data = data.to(device)
                labels = labels.long()
                labels = labels.to(device)
                outputs = model(data)
                outputs = softmax_l(outputs)
                v_loss = criterion(outputs, labels)
                val_loss += v_loss.item()

                calcs = torch.argmax(outputs, dim=1)
                diff_l = [0 if calcs[i] == labels[i] else 1 for i in range(len(labels))]
                val_mae_acc += sum(diff_l)
        l = len(validation_generator) * params['batch_size']
        val_mae_acc = 1 - (val_mae_acc/l)
        print(f'Epoch {epoch + 1} \t\t Training Loss: {train_loss / len(training_generator)} \t Training Acc: {train_mae_acc} \t\t Validation Loss: { val_loss / len(validation_generator)} \t Validation Acc: {val_mae_acc}')
        epoch_statistics.append(epoch)
        train_loss_statistics.append(train_loss / len(training_generator))
        train_acc_statistics.append(train_mae_acc)
        validation_loss_statistics.append(val_loss / len(validation_generator))
        validation_acc_statistics.append(val_mae_acc)
        if min_valid_loss > val_loss / len(validation_generator):
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{val_loss / len(validation_generator):.6f}')
            min_valid_loss = val_loss / len(validation_generator)
            # save model
            if save_model:
                torch.save(model.state_dict(), f'{model_name}.pth')
                best_val_epoch = epoch

        #        loss = criterion(outputs, local_labels)
    model.eval()
    with torch.set_grad_enabled(False):
        for data, labels in evaluation_generator:
            data = data[:, :, :,data_cut_front:data_cut_back].float()
            data = data.to(device)
            labels = labels.long()
            labels = labels.to(device)
            outputs = model(data)
            outputs = softmax_l(outputs)
            e_loss = criterion(outputs, labels)
            calcs = torch.argmax(outputs, dim=1)
            diff_l = [0 if calcs[i] == labels[i] else 1 for i in range(len(labels))]
            diff_s = sum(diff_l)
            eval_c_1 = sum(labels)
            mae = diff_s / len(diff_l)
            eval_acc = 1 - mae
            chance_acc = eval_c_1/len(labels)
            eval_kappa = (eval_acc - chance_acc) / (1-chance_acc)
            print(f'Eval Loss: {e_loss:.6f} \t Eval Acc: {eval_acc} \t Eval C1: {eval_c_1} \t Evak kappa: {eval_kappa}')

    y = range(max_epochs)

    # save last epoch model
    torch.save(model.state_dict(), f'{model_name}_last.pth')


    #plt.plot(y, [statistics[i][0] for i in range(len(statistics))], label='train_loss')
    #plt.plot(y, [statistics[i][1] for i in range(len(statistics))], label='train_acc')
    #plt.plot(y, [statistics[i][2] for i in range(len(statistics))], label='val_loss')
    #plt.plot(y, [statistics[i][3] for i in range(len(statistics))], label='val_acc')
    #plt.legend()
    #plt.show()
    #print("Honey")
    statistics = [epoch_statistics,train_loss_statistics, train_acc_statistics ,validation_loss_statistics, validation_acc_statistics]
    return statistics, e_loss, eval_acc, eval_kappa, best_val_epoch