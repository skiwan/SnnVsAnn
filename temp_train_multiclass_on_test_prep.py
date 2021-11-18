from Models.ANN.BaseSCNN import BaseSCNN
from Models.CustomDataset import CustomDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt

params = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 3}
max_epochs = 100

train_data_file = '/home/merlinsewina/MaWork/SnnVsAnn/prep/A01T_Prep_whole_train_data.npy'
train_label_file = '/home/merlinsewina/MaWork/SnnVsAnn/prep/A01T_Prep_whole_train_labels.npy'

val_data_file = '/home/merlinsewina/MaWork/SnnVsAnn/prep/A01T_Prep_whole_validate_data.npy'
val_label_file = '/home/merlinsewina/MaWork/SnnVsAnn/prep/A01T_Prep_whole_validate_labels.npy'

eval_data_file = '/home/merlinsewina/MaWork/SnnVsAnn/prep/A01E_Prep_whole_set.npy'
eval_label_file = '/home/merlinsewina/MaWork/SnnVsAnn/prep/A01E_Prep_whole_labels.npy'

training_data = CustomDataset(train_label_file, train_data_file)
validate_data = CustomDataset(val_label_file, val_data_file)
evaluate_data = CustomDataset(eval_label_file, eval_data_file)

training_generator = DataLoader(training_data, **params)
validation_generator = DataLoader(validate_data, **params)
evaluation_generator = DataLoader(evaluate_data, batch_size=evaluate_data.__len__())


model = BaseSCNN(channels=32, base_filters=8, classes=4, image_height=44, dropout_value=0.3).to('cpu')
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = torch.nn.CrossEntropyLoss()

running_train_loss = 0.0
running_val_loss = 0.0
softmax_l = torch.nn.LogSoftmax(dim=1)
min_valid_loss = np.inf
save = False
model_name = 'someModelname.pth'

statistics = []

for epoch in range(max_epochs):
    train_loss = 0.0
    train_mae_acc = 0.0
    # Training
    for data, labels in training_generator:
        data = data[:,:,:,:200].float()
        labels = labels.long()
        labels = labels - 1
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
            data = data[:, :, :, :200].float()
            labels = labels.long()
            labels = labels - 1
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
    statistics.append([train_loss / len(training_generator),train_mae_acc,val_loss / len(validation_generator),val_mae_acc])
    if min_valid_loss > val_loss / len(validation_generator):
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{val_loss / len(validation_generator):.6f}')
        min_valid_loss = val_loss / len(validation_generator)
        if save:
            torch.save(model.state_dict(), model_name)

    #        loss = criterion(outputs, local_labels)
with torch.set_grad_enabled(False):
    for data, labels in evaluation_generator:
        data = data[:, :, :, :200].float()
        labels = labels.long()
        labels = labels - 1
        outputs = model(data)
        outputs = softmax_l(outputs)
        e_loss = criterion(outputs, labels)
        calcs = torch.argmax(outputs, dim=1)
        diff_l = [0 if calcs[i] == labels[i] else 1 for i in range(len(labels))]
        diff_s = sum(diff_l)
        mae = diff_s / len(diff_l)
        eval_acc = 1 - mae
        chance_acc = 0.25
        eval_kappa = (eval_acc - chance_acc) / (1-chance_acc)
        print(f'Eval Loss: {e_loss:.6f} \t Eval Acc: {eval_acc} \t  Eval kappa: {eval_kappa}')



y = range(max_epochs)

[statistics[i][0] for i in range(len(statistics))]

plt.plot(y, [statistics[i][0] for i in range(len(statistics))], label='train_loss')
plt.plot(y, [statistics[i][1] for i in range(len(statistics))], label='train_acc')
plt.plot(y, [statistics[i][2] for i in range(len(statistics))], label='val_loss')
plt.plot(y, [statistics[i][3] for i in range(len(statistics))], label='val_acc')
plt.legend()
plt.show()
print("Honey")