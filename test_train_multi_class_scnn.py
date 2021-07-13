from Models.ANN.BaseSCNN import BaseSCNN
import torch
import numpy as np
import os
import time
from random import randint
from sklearn.metrics import cohen_kappa_score

def binary_kappa_value(predictions, truth_labels):
    a = b = c = d = 0
    n = len(predictions)
    for i in range(n):
        p = predictions[i]
        t = truth_labels[i]
        if int(p) == 0:
            if int(p) == int(t):
                d += 1
            else:
                c += 1
        else:
            if int(p) == int(t):
                a += 1
            else:
                b += 1
    p_acc = (a + d) / n
    r_acc = (((a+b)/n) * ((a+c)/n)) + (((c+d)/n) * ((b+d)/n))
    return ((p_acc - r_acc) / (1 - r_acc))

def random_split(data, labels, splitnr):
    train_d = []
    train_l = []

    val_d = []
    val_l = []

    d = data.detach().clone()
    l = list(labels)
    r = list(range(len(l)))
    for i in range(splitnr):
        pick = randint(0,len(r)-1)
        pick = r[pick]
        train_d.append(d[pick])
        train_l.append(l[pick])
        r.remove(pick)

    for n in r:
        val_d.append(d[n])
        val_l.append(l[n])

    return train_d, train_l, val_d, val_l


def train(model, train_data, train_labels, val_data, val_labels, optimizer, epochs=200, weights=None):
    losses = []
    val_losses = []
    start = time.time()
    loss = torch.nn.NLLLoss(weight=weights)
    log = torch.nn.LogSoftmax(dim=1)

    last_min_val_loss = None

    for i in range(epochs): # repeat this epoch amount of times
        print(f'epoch {i} out of {epochs}')
        epoch_losses = []
        for i in range(len(train_data)):
            xs = train_data[i].unsqueeze(0)
            label = train_labels[i]
            out, _ = model(xs)
            out = log(out)
            loss_v = loss(out, torch.tensor(label).unsqueeze(0))
            epoch_losses.append(loss_v)
        epoch_loss = torch.stack(epoch_losses).mean(0)
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()
        losses.append(epoch_loss.detach())

        val_epoch_losses = []
        for i in range(len(val_data)):
            xs = val_data[i].unsqueeze(0)
            label = val_labels[i]
            out, _ = model(xs)
            out = log(out)
            loss_v = loss(out, torch.tensor(label).unsqueeze(0))
            val_epoch_losses.append(loss_v)
        val_epoch_loss = torch.stack(val_epoch_losses).mean(0)
        val_losses.append(val_epoch_loss)
        if last_min_val_loss is None:
            last_min_val_loss = val_epoch_loss
        if val_epoch_loss < last_min_val_loss:
            last_min_val_loss = val_epoch_loss
            torch.save(model, 'best_val_model.pth')

        print(f'min average epoch loss {min(losses)} min current epoch loss {min(epoch_losses)}')
        print(f'min average val epoch loss {min(val_losses)} min current val epoch loss {min(val_epoch_losses)}')
        print(f'max average epoch loss {max(losses)} max current epoch loss {max(epoch_losses)}')
        print(f'max average val epoch loss {max(val_losses)} max val current epoch loss {max(val_epoch_losses)}')
        print(f'average epoch loss {epoch_loss}, average val epoch loss {val_epoch_loss}')
    print(f"Training for {epochs} took a total of {time.time()-start} seconds")

    print(f'Minimum overall average epoch val loss was {min(val_epoch_losses)} at epoch {val_epoch_losses.index(min(val_epoch_losses))}')
    return losses


if __name__ == '__main__':
    current_wd = os.getcwd()

    model = BaseSCNN(channels=8, base_filters=8, classes=2, image_height=44).to('cpu')
    model.float()

    dataset = np.load(os.path.join(current_wd, os.path.join('raw_normalized_CSP_CWT', 'BCI4_2a_A01T_car_7_30_full.npy')))
    dataset = dataset[:,:,:,:200]
    #dataset = dataset[:144]
    dataset = torch.from_numpy(dataset).float()
    ev_set = np.load(os.path.join(current_wd, os.path.join('raw_normalized_CSP_CWT', 'BCI4_2a_A01E_car_7_30_full.npy')))
    ev_set = ev_set[:, :, :, :200]
    #ev_set = ev_set[:144]
    ev_set = torch.from_numpy(ev_set).float()

    #with open(os.path.join(current_wd, os.path.join('Raw_Preprocessed_CWT', 'BCI4_2a_A01T_car_labels.txt'))) as labelfile:
    #    lines = labelfile.readlines()
    #    data_labels = [int(l.replace('\n','')) for l in lines]
    #ev_labels = np.load(os.path.join(os.path.join(current_wd, os.path.join('Raw_Preprocessed', 'A01E_labels.npy'))))

    labels = [1 if i < 72 else 0 for i in range(288)]
    #labels = labels[:144]
    data_labels = ev_labels = labels
    print(dataset.shape)

    train_data, train_labels, test_data, test_labels = random_split(dataset, data_labels, 240)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    weights = torch.tensor([1,sum(labels[:72]) / len(labels[72:])])

    m1_losses = train(model, train_data,train_labels, test_data, test_labels, optimizer, epochs=50, weights=weights)
    print(m1_losses)
    print(min(m1_losses))

    model = torch.load('best_val_model.pth')

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
        truth = int(ev_labels[p])
        if int(pre) == truth:
            acc += 1
    print(acc, acc/len(predictions))
    print(sum(predictions[:72]))
    print(binary_kappa_value(predictions, ev_labels))
    print('done')
