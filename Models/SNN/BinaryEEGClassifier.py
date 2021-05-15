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
    xs = self.input_weights(xs)
    # for each time step compute spike train of input layer
    input_spikes = []
    for x in xs:
      spike, input_state = self.input_layer(x, input_state)
      input_spikes.append(spike)
    input_spikes = torch.stack(input_spikes)
    # multiply whole spike train by weight matrix to scale to new input current
    xs = self.output_weights(input_spikes)
    # for each time step compute spike train for output neuron
    output_spikes = []
    for x in xs:
      spike, output_state = self.output_neuron(x, output_state)
      output_spikes.append(spike)
    output_spikes = torch.stack(output_spikes)
    return output_spikes, (input_state, output_state)
