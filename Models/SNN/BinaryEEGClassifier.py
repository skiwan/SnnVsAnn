from norse.torch.module import  IzhikevichCell
import torch

class BinaryEEGClassifier(torch.nn.Module):
  def __init__(self, p):
    super().__init__()
    self.input_weights = torch.nn.Linear(4,2) 
    self.input_layer = IzhikevichCell(p)
    self.output_weights = torch.nn.Linear(2,1)
    self.output_neuron = IzhikevichCell(p)

  def forward(self, x, state):
    input_state, output_state = state
    x = self.input_weights(x)
    x, new_input_state = self.input_layer(x, input_state)
    x = self.output_weights(x)
    x, new_output_state = self.output_neuron(x, output_state)
    return x, (new_input_state, output_state)
