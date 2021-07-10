from Models.ANN.BaseSCNN import BaseSCNN
import torch
import numpy as np
import os
import time

def train(model, data, labels, optimizer, epochs=200):
    losses = []
    start = time.time()
    loss = torch.nn.NLLLoss()
    log = torch.nn.LogSoftmax(dim=1)
    for i in range(epochs): # repeat this epoch amount of times
        print(f'epoch {i} out of {epochs}')
        epoch_losses = []
        for i in range(len(data)):
            xs = data[i].unsqueeze(0)
            label = labels[i]
            out, _ = model(xs)
            out = log(out)
            loss_v = loss(out, torch.tensor(label-1).unsqueeze(0))
            epoch_losses.append(loss_v)
        epoch_loss = torch.stack(epoch_losses).mean(0)
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()
        losses.append(epoch_loss.detach())
        print(f'min epoch loss {min(losses)} min average epoch loss {min(epoch_losses)}')
        print(f'max epoch loss {max(losses)} max average epoch loss {max(epoch_losses)}')
    print(f"Training for {epochs} took a total of {time.time()-start} seconds")
    return losses


if __name__ == '__main__':
    current_wd = os.getcwd()

    model = BaseSCNN(channels=25, base_filters=8, classes=4, image_height=44).to('cpu')
    model.float()

    dataset = np.load(os.path.join(current_wd, os.path.join('Raw_Preprocessed_CWT', 'BCI4_2a_A01T_car_7_30.npy')))
    dataset = dataset[:,:,:,:200]
    dataset = torch.from_numpy(dataset).float()
    ev_set = np.load(os.path.join(current_wd, os.path.join('Raw_Preprocessed_CWT', 'BCI4_2a_A01E_car_7_30.npy')))
    ev_set = ev_set[:, :, :, :200]
    ev_set = torch.from_numpy(ev_set).float()

    with open(os.path.join(current_wd, os.path.join('Raw_Preprocessed_CWT', 'BCI4_2a_A01T_car_labels.txt'))) as labelfile:
        lines = labelfile.readlines()
        data_labels = [int(l.replace('\n','')) for l in lines]
    ev_labels = np.load(os.path.join(os.path.join(current_wd, os.path.join('Raw_Preprocessed', 'A01E_labels.npy'))))

    print(dataset.shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    m1_losses = train(model, dataset, data_labels, optimizer, epochs=50)
    print(m1_losses)
    print(min(m1_losses))

    softmax = torch.nn.Softmax(dim=1)
    predictions = []
    for x in ev_set:
        pre, _ = model(x.unsqueeze(0))
        pre = softmax(pre)
        pre = torch.argmax(pre)
        predictions.append(pre)

    acc = 0
    for p in range(len(predictions)):
        pre = predictions[p]
        truth = int(ev_labels[p])-1
        if int(pre) == truth:
            acc += 1
    print(acc, acc/len(predictions))
    print('done')
