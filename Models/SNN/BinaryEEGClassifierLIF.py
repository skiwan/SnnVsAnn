from norse.torch.module import Izhikevich
from norse.torch import SequentialState
import torch

class BinaryEEGClassifierLIF(torch.nn.Module):
  def __init__(self, p):
    super().__init__()
    self.model = SequentialState(
    torch.nn.Linear(100,100),
    Izhikevich(spiking_method=p),
    torch.nn.Linear(100,100),
    Izhikevich(spiking_method=p)
    )

  def forward(self, x):
    return self.model(x)
