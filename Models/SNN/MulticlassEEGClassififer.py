from .FourClassSNNLayer import FourClassSNNLayer
from norse.torch import SequentialState
import torch


class MulticlassEEGClassififer(torch.nn.Module):
  def __init__(self, all_channel_info, all_class_iz_params, gaining_factor):
    super().__init__()
    self.four_class_snn_layer = FourClassSNNLayer(all_channel_info=all_channel_info, all_class_iz_params=all_class_iz_params, gaining_factor=gaining_factor)
    self.softmax = torch.nn.LogSoftmax(dim=0)

    self.model = SequentialState(
        self.four_class_snn_layer,
        self.softmax
    )

  def forward(self, xs):
      return self.model.forward(xs)

