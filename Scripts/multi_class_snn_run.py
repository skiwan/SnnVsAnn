from Models.SNN.BinaryEEGClassifierLIF import BinaryEEGClassifierLIF
from Models.CustomDataset import CustomDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
import os




# Binary_SNN_A01_class1_model.pth
# Binary_SNN_A01_class1_model_last.pth
# normalized_eval_class1.npy
# raw_eval_labels.npy

def main(base_path, base_model_name, class_amount
         ,data_cut_front, data_cut_back, model_channels, model_classes, device):

    best_models = []
    last_models = []
    eval_label_file = f'{base_path}raw_eval_labels.npy'
    eval_labels = torch.from_numpy(np.load(eval_label_file)).to(device)
    eval_labels = eval_labels - 1
    eval_sets = []
    # load models for each class
    for i in range(class_amount):
        best_model_path = os.path.join(base_path, f'{base_model_name}_class{i + 1}_model.pth')
        last_model_path = os.path.join(base_path, f'{base_model_name}_class{i + 1}_model_last.pth')

        best_model = BinaryEEGClassifierLIF(channels=model_channels).to(device)
        best_model.load_state_dict(torch.load(best_model_path))
        best_models.append(best_model)

        last_model = BinaryEEGClassifierLIF(channels=model_channels).to(device)
        last_model.load_state_dict(torch.load(last_model_path))
        last_models.append(last_model)
        eval_sets.append(torch.from_numpy(np.load(f'{base_path}normalized_eval_class{i+1}.npy')).to(device))

    best_predicts = []
    last_predicts = []
    best_models_convidence = []
    last_models_convidence = []

    for c in range(class_amount):
        b_model = best_models[c]
        l_model = last_models[c]
        data = eval_sets[c]
        data = data[:,:,data_cut_front:data_cut_back].float()
        data = torch.swapaxes(data, 0, 2)
        data = torch.swapaxes(data, 1, 2)
        outputs = b_model(data)
        outputs = outputs[0].sum(dim=0)  # batch size, spikes
        outputs = torch.squeeze(outputs) # spikes per sample
        best_models_convidence.append(outputs)
        outputs = l_model(data)
        outputs = outputs[0].sum(dim=0)  # batch size, spikes
        outputs = torch.squeeze(outputs)  # spikes per sample
        last_models_convidence.append(outputs)

    best_models_convidence = [torch.tensor(best_models_convidence[i]) for i in range(class_amount)]
    best_models_convidence = [torch.tensor([best_models_convidence[0][i],
                                            best_models_convidence[1][i],
                                            best_models_convidence[2][i],
                                            best_models_convidence[3][i]]) for i in range(best_models_convidence[0].shape[0])]
    last_models_convidence = [torch.tensor(last_models_convidence[i]) for i in range(class_amount)]
    last_models_convidence = [torch.tensor([last_models_convidence[0][i],
                                            last_models_convidence[1][i],
                                            last_models_convidence[2][i],
                                            last_models_convidence[3][i]]) for i in range(last_models_convidence[0].shape[0])]

    best_models_labels = [torch.argmax(best_models_convidence[i], dim=0) for i in range(len(best_models_convidence))]
    last_models_labels = [torch.argmax(last_models_convidence[i], dim=0) for i in range(len(last_models_convidence))]
    diff_b = [1 if best_models_labels[i] == eval_labels[i] else 0 for i in range(len(eval_labels))]
    diff_l = [1 if last_models_labels[i] == eval_labels[i] else 0 for i in range(len(eval_labels))]
    diff_b_s = sum(diff_b)
    diff_l_s = sum(diff_l)

    best_acc = diff_b_s / len(best_models_labels)
    last_acc = diff_l_s / len(last_models_labels)

    best_kappa = (best_acc - 0.25) / (1-0.25)
    last_kappa = (last_acc - 0.25) / (1-0.25)

    return best_acc, best_kappa, last_acc, last_kappa


def main_return_data(base_path, base_model_name, class_amount
         ,data_cut_front, data_cut_back, model_channels, model_classes, device):

    best_models = []
    last_models = []
    eval_label_file = f'{base_path}raw_eval_labels.npy'
    eval_labels = torch.from_numpy(np.load(eval_label_file)).to(device)
    eval_labels = eval_labels - 1
    eval_sets = []
    # load models for each class
    for i in range(class_amount):
        best_model_path = os.path.join(base_path, f'{base_model_name}_class{i + 1}_model.pth')
        last_model_path = os.path.join(base_path, f'{base_model_name}_class{i + 1}_model_last.pth')

        best_model = BinaryEEGClassifierLIF(channels=model_channels).to(device)
        best_model.load_state_dict(torch.load(best_model_path))
        best_models.append(best_model)

        last_model = BinaryEEGClassifierLIF(channels=model_channels).to(device)
        last_model.load_state_dict(torch.load(last_model_path))
        last_models.append(last_model)
        eval_sets.append(torch.from_numpy(np.load(f'{base_path}normalized_eval_class{i+1}.npy')).to(device))

    best_train = []
    last_train = []
    best_models_convidence = []
    last_models_convidence = []

    # classes -> last/best -> labels
    model_outputs = [
        [[[],[],[],[]],[[],[],[],[]]],
        [[[],[],[],[]],[[],[],[],[]]],
        [[[],[],[],[]],[[],[],[],[]]],
        [[[],[],[],[]],[[],[],[],[]]]
    ]

    for c in range(class_amount):
        b_model = best_models[c]
        l_model = last_models[c]
        data = eval_sets[c]
        data = data[:,:,data_cut_front:data_cut_back].float()
        data = torch.swapaxes(data, 0, 2)
        data = torch.swapaxes(data, 1, 2)
        outputs = b_model(data)
        best_train.append(list(outputs))
        outputs = outputs[0].sum(dim=0)  # batch size, spikes
        outputs = torch.squeeze(outputs) # spikes per sample
        best_models_convidence.append(outputs)
        outputs = l_model(data)
        last_train.append(list(outputs))
        outputs = outputs[0].sum(dim=0)  # batch size, spikes
        outputs = torch.squeeze(outputs)  # spikes per sample
        last_models_convidence.append(outputs)

        for i, l in enumerate(eval_labels):
            print(best_train)
            model_outputs[c][0][l].append(best_train[c][i])
            model_outputs[c][1][l].append(last_train[c][i])


    best_models_convidence = [torch.tensor(best_models_convidence[i]) for i in range(class_amount)]
    best_models_convidence = [torch.tensor([best_models_convidence[0][i],
                                            best_models_convidence[1][i],
                                            best_models_convidence[2][i],
                                            best_models_convidence[3][i]]) for i in range(best_models_convidence[0].shape[0])]
    last_models_convidence = [torch.tensor(last_models_convidence[i]) for i in range(class_amount)]
    last_models_convidence = [torch.tensor([last_models_convidence[0][i],
                                            last_models_convidence[1][i],
                                            last_models_convidence[2][i],
                                            last_models_convidence[3][i]]) for i in range(last_models_convidence[0].shape[0])]

    best_models_labels = [torch.argmax(best_models_convidence[i], dim=0) for i in range(len(best_models_convidence))]
    last_models_labels = [torch.argmax(last_models_convidence[i], dim=0) for i in range(len(last_models_convidence))]
    diff_b = [1 if best_models_labels[i] == eval_labels[i] else 0 for i in range(len(eval_labels))]
    diff_l = [1 if last_models_labels[i] == eval_labels[i] else 0 for i in range(len(eval_labels))]
    diff_b_s = sum(diff_b)
    diff_l_s = sum(diff_l)

    best_acc = diff_b_s / len(best_models_labels)
    last_acc = diff_l_s / len(last_models_labels)

    best_kappa = (best_acc - 0.25) / (1-0.25)
    last_kappa = (last_acc - 0.25) / (1-0.25)

    return best_acc, best_kappa, last_acc, last_kappa, model_outputs