from norse.torch import SequentialState, LIFCell
from norse.torch.module import IzhikevichCell
import torch

class BinaryEEGClassifier(torch.nn.Module):
  def __init__(self, channels,p, gaining_factor=1):
    super().__init__()
    gaining_layer_1 = torch.nn.Linear(channels,channels)
    gaining_layer_2 = torch.nn.Linear(2,2)
    gaining_layer_1.weight = torch.nn.Parameter(torch.ones_like(gaining_layer_1.weight)*gaining_factor)
    gaining_layer_1.requires_grad = False
    gaining_layer_2.weight = torch.nn.Parameter(torch.ones_like(gaining_layer_2.weight) * gaining_factor)
    gaining_layer_2.requires_grad = False

    self.model = SequentialState(
    gaining_layer_1,
    torch.nn.Linear(channels,2),
    IzhikevichCell(spiking_method=p),
    gaining_layer_2,
    torch.nn.Linear(2,1),
    IzhikevichCell(spiking_method=p)
    )

  def forward(self, xs, state=None):
    spikes = []
    for ts in xs:
       s, state = self.model.forward(ts, state)
       spikes.append(s)
    return torch.stack(spikes)
