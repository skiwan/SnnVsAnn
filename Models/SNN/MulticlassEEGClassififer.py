from .BinaryEEGClassififer import BinaryEEGClassififer
import torch


#Todo makes this abstract and nice code independent of the 4
class MulticlassEEGClassififer(torch.nn.Module):
  def __init__(self, iz_params):
    super().__init__()
    self.binary_classifiers = [BinaryEEGClassififer(iz_params[i]) for i in range(4)]

  def forward(self, x, state):
    c1_state, c2_state, c3_state, c4_state  = state

    c1_x, c1_n_state = self.binary_classifiers[0](x[0], c1_state)
    c2_x, c2_n_state = self.binary_classifiers[1](x[1], c2_state)
    c3_x, c3_n_state = self.binary_classifiers[2](x[2], c3_state)
    c4_x, c4_n_state = self.binary_classifiers[3](x[3], c4_state)

    return torch.stack([c1_x, c2_x, c3_x, c4_x]), (c1_n_state, c2_n_state, c3_n_state, c4_n_state)
