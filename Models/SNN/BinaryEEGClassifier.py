from norse.torch.module import IzhikevichCell
import torch

class BinaryEEGClassifier(torch.nn.Module):
  def __init__(self, channels,p):
    super().__init__()
    self.input_weights = torch.nn.Linear(channels, 2) # 2 is number of IZ Neurons
    self.input_layer = IzhikevichCell(p)
    self.output_weights = torch.nn.Linear(2, 1) # 2 is number of iZ neurons above, 1 is target number of output neurons
    self.output_neuron = IzhikevichCell(p)

  def forward(self, xs, state=(None, None)): # xs has dim timesteps , channels
    input_state, output_state = state
    # multiply whole input by weight matrix to scale input
    xs = self.input_weights(xs) # transform the input from timesteps x channels into timesteps x neuron amount (100,4) to (100,2)
    # for each time step compute spike train of input layer
    input_spikes = []
    for x in xs: # run all timesteps and record spikes for each neuron (list of timesteps x spikes_per_neuron)
      spike, input_state = self.input_layer(x, input_state)
      input_spikes.append(spike)
    input_spikes = torch.stack(input_spikes) # make a tensor out of the list
    # multiply whole spike train by weight matrix to scale to new input current
    xs = self.output_weights(input_spikes) # multiply spikes with output weights to get new input current train for output neuron timesteps x 1
    # for each time step compute spike train for output neuron
    output_spikes = []
    for x in xs:
      spike, output_state = self.output_neuron(x, output_state)
      output_spikes.append(spike)
    output_spikes = torch.stack(output_spikes) # timesteps x 1 output spikes
    return output_spikes, (input_state, output_state)
