from Models.ANN.BaseSCNN import BaseSCNN
from Models.CustomDataset import CustomDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
import os




# Binary_ANN_A01_class1_model.pth
# Binary_ANN_A01_class1_model_last.pth
# cwt_eval_class1.npy
# raw_eval_labels.npy

def main(base_path, base_model_name, class_amount
         ,data_cut_front, data_cut_back, model_channels, model_classes, model_dropout, device):

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

        best_model = BaseSCNN(channels=model_channels, base_filters=8, classes=model_classes, image_height=44,
                              dropout_value=model_dropout).to(device)
        best_model.load_state_dict(torch.load(best_model_path))
        best_models.append(best_model)

        last_model = BaseSCNN(channels=model_channels, base_filters=8, classes=model_classes, image_height=44,
                              dropout_value=model_dropout).to(device)
        last_model.load_state_dict(torch.load(last_model_path))
        last_models.append(last_model)
        eval_sets.append(torch.from_numpy(np.load(f'{base_path}cwt_eval_class{i+1}.npy')).to(device))

    best_predicts = []
    last_predicts = []
    best_models_convidence = []
    last_models_convidence = []

    for c in range(class_amount):
        b_model = best_models[c]
        l_model = last_models[c]
        data = eval_sets[c]
        data = data[:,:,:,data_cut_front:data_cut_back].float()
        outputs = b_model(data)
        outputs_sum = torch.sum(torch.abs(outputs), dim=1)
        best_models_convidence.append(outputs[:,1]/outputs_sum)
        outputs = l_model(data)
        outputs_sum = torch.sum(torch.abs(outputs), dim=1)
        last_models_convidence.append(outputs[:,1]/outputs_sum)

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
    diff_b = [0 if best_models_labels[i] == eval_labels[i] else 1 for i in range(len(eval_labels))]
    diff_l = [0 if last_models_labels[i] == eval_labels[i] else 1 for i in range(len(eval_labels))]
    diff_b_s = sum(diff_b)
    diff_l_s = sum(diff_l)

    best_acc = diff_b_s / len(best_models_labels)
    last_acc = diff_l_s / len(last_models_labels)

    best_kappa = (best_acc - 0.25) / (1-0.25)
    last_kappa = (last_acc - 0.25) / (1-0.25)

    return best_acc, best_kappa, last_acc, last_kappa
        # select highest confidence as class
        # calculate evaluation accuracy and kappa
        # return statistics