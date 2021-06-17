from Models.SNN.MulticlassEEGClassififer import MulticlassEEGClassififer

import torch
from norse.torch.functional.izhikevich import IzhikevichSpikingBehaviour, IzhikevichState, IzhikevichParameters


def train_spike(model, data, labels, optimizer, epochs=200):
    losses = []
    outs = []
    for i in range(epochs): # repeat this epoch amount of times
        print(f'epoch {i} out of {epochs}')
        epoch_losses = []
        loss = torch.nn.NLLLoss()
        for i in range(len(data)): # take a single sample of shape classes x timesteps x channels (4,100,C)
            xs = data[i]
            label = labels[i]
            out, _ = model(xs)
            loss_v = loss(out.unsqueeze(0), label)
            epoch_losses.append(loss_v)
        epoch_loss = torch.stack(epoch_losses).mean(0)
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()
        losses.append(epoch_loss.detach())
        print(f'min epoch loss {min(losses)} min average epoch loss {min(epoch_losses)}')
        print(f''max(losses), max(epoch_losses))
    return losses


if __name__ == '__main__':
    gaining_factor = 70
    timesteps = 100
    all_channel_info = [4,3,5,4]
    samples = 40


    initial_params = IzhikevichParameters(a=0.02, b=0.2, c=-65, d=2)
    initial_state = IzhikevichState(v=torch.tensor(-65.0, requires_grad=True), u=torch.tensor(-65) * initial_params.b)
    behaviour = IzhikevichSpikingBehaviour(initial_params, initial_state)

    all_iz_params = [behaviour, behaviour, behaviour, behaviour]

    mceeg = MulticlassEEGClassififer(all_channel_info=all_channel_info, all_class_iz_params=all_iz_params, gaining_factor=gaining_factor)


    # create 50 samples each with 100 60 30 and 0 activity for class 4 to 1
    class_1_freq = [0.] * timesteps

    class_2_freq = [1.,0.,0.,1.,0.] * timesteps
    class_2_freq = class_2_freq[:timesteps]

    class_3_freq = [1.,1.,0.] * timesteps
    class_3_freq = class_3_freq[:timesteps]

    class_4_freq = [1.] * timesteps

    class_freq = {
        'class_1_freq' : class_1_freq,
        'class_2_freq' : class_2_freq,
        'class_3_freq' : class_3_freq,
        'class_4_freq' : class_4_freq
    }

    class_freq_inputs = {}
    for c in range(4):
        class_freq_inputs[f'{c}'] = []

    # samples * classes * timesteps * channels
    single_channel_inputs = []
    for i in range(len(all_channel_info)):
        channel_amount = all_channel_info[i]
        for c in range(4):
            single_data = [class_freq[f'class_{c+1}_freq'] for x in range(channel_amount)]
            single_data = torch.tensor(single_data)
            single_data = torch.swapaxes(single_data, 0, 1)
            class_freq_inputs[f'{i}'].append(single_data)

    inputs = []
    for i in range(4):
        single = []
        for x in range(4):
            single.append(class_freq_inputs[f'{x}'][i])
        inputs.append(single)

    data = [inputs[i] for i in range(4)] * (samples//4)
    labels = [torch.tensor([i]) for i in range(4)] * (samples//4)

    optimizer = torch.optim.Adam(mceeg.parameters(), lr=0.01)
    torch.autograd.set_detect_anomaly(True)

    m1_losses = train_spike(mceeg, data, labels, optimizer, epochs=2)
    print(m1_losses)
    print(min(m1_losses))
