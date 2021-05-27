from norse.torch.module import Izhikevich
from norse.torch import SequentialState, LIF
import torch

class BinaryEEGClassifierLIF(torch.nn.Module):
  def __init__(self, p):
    super().__init__()
    self.model = SequentialState(
    torch.nn.Linear(100,200),
    LIF(p=p),
    torch.nn.Linear(200,100),
    LIF(p=p)
    )

  def forward(self, x):
    return self.model(x)
