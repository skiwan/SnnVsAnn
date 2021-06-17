from Models.SNN.BinaryEEGClassifier import BinaryEEGClassifier

import torch
from norse.torch.functional.izhikevich import IzhikevichSpikingBehaviour, IzhikevichState, IzhikevichParameters


def train_spike(model, data, optimizer, epochs=200):
    losses = []
    outs = []
    frac = 0.2 # goal of average spikes per trial
    for i in range(epochs): # repeat this epoch amount of times
        print(f'epoch {i} out of {epochs}')
        epoch_losses = []
        for xs in data: # take a single sample of shape timesteps x channels (100,4)
            out = model(xs, None)
            spike_frac = out.mean(0)
            frac_loss =  (frac - spike_frac) ** 2 * 7.5 # 5 makes it equivalent to loss, I want it to be slioghtly more important
            n_loss = frac_loss
            epoch_losses.append(n_loss)
        epoch_loss = torch.stack(epoch_losses).mean(0)
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()
        losses.append(epoch_loss.detach())
        print(min(losses))
        print(max(losses))
    print(out)
    return losses


if __name__ == '__main__':
    gaining_factor = 70

    initial_params = IzhikevichParameters(a=0.02, b=0.2, c=-65, d=2)
    initial_state = IzhikevichState(v=torch.tensor(-65.0, requires_grad=True), u=torch.tensor(-65) * initial_params.b)
    behaviour = IzhikevichSpikingBehaviour(initial_params, initial_state)

    bec = BinaryEEGClassifier(4, behaviour, gaining_factor).cuda()
    data = (torch.ones(10, 100, 4)).cuda() # I have 4 channels of EEG as input, for 100 timesteps, and I want 10 samples
    pattern = [0.0, 0.0, 0.0, 0.0, 1.0] * 2

    optimizer = torch.optim.Adam(bec.parameters(), lr=0.01)
    torch.autograd.set_detect_anomaly(True)

    m1_losses = train_spike(bec, data, optimizer, epochs=100)
    print(m1_losses)
    print(min(m1_losses))
