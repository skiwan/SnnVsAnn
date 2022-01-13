from norse.torch.module import LIFCell
from norse.torch import SequentialState
import torch

class BinaryEEGClassifierLIF(torch.nn.Module):
  def __init__(self, input_channels):
    super().__init__()
    self.model = SequentialState(
    torch.nn.Linear(input_channels,2),
    LIFCell(2,2),
    torch.nn.Linear(2,2),
    LIFCell(2,1)
    )

  def forward(self, x):
    return self.model(x)
