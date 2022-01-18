from norse.torch.module import LIFCell
from norse.torch import SequentialState, LIF
import torch

class BinaryEEGClassifierLIF(torch.nn.Module):
  def __init__(self, channels=8):
    super().__init__()
    self.linear_1 = torch.nn.Linear(channels,100)
    #self.linear_1.weight = torch.nn.Parameter(torch.abs(self.linear_1.weight))
    self.linear_2 = torch.nn.Linear(100,1)
    #self.linear_2.weight = torch.nn.Parameter(torch.abs(self.linear_2.weight))

    self.model = SequentialState(
    self.linear_1,
    #LIF(),
    self.linear_2,
    LIF()
    )

  def forward(self, x):
    return self.model(x)
