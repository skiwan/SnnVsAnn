from norse.torch.module import LIFCell
from norse.torch import SequentialState, LIF
import torch

class BinaryEEGClassifierLIF(torch.nn.Module):
  def __init__(self, channels=8):
    super().__init__()
    linear_1 = torch.nn.Linear(channels,2)
    linear_1.weight = torch.nn.Parameter(torch.abs(linear_1.weight))
    linear_2 = torch.nn.Linear(2,1)
    linear_2.weight = torch.nn.Parameter(torch.abs(linear_2.weight)

    self.model = SequentialState(
    linear_1,
    LIF(),
    linear_2,
    LIF()
    )

  def forward(self, x):
    return self.model(x)
