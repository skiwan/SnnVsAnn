from .BinaryEEGClassififer import BinaryEEGClassififer
import torch


#Todo makes this abstract and nice code independent of the 4
class MulticlassEEGClassififer(torch.nn.Module):
  def __init__(self, iz_params):
    super().__init__()
    self.binary_classifiers = [BinaryEEGClassififer(iz_params[i]) for i in range(4)]

  def forward(self, xs, state):
    c1_state, c2_state, c3_state, c4_state  = state
    spikes = [[],[],[],[]]

    for x in xs: 
        c1_x, c1_state = self.binary_classifiers[0](x[0], c1_state)
        c2_x, c2_state = self.binary_classifiers[1](x[1], c2_state)
        c3_x, c3_state = self.binary_classifiers[2](x[2], c3_state)
        c4_x, c4_state = self.binary_classifiers[3](x[3], c4_state)
        spikes[0].append(c1_x)
        spikes[1].append(c2_x)
        spikes[2].append(c3_x)
        spikes[3].append(c4_x)

    spikes_c = [sum(s) for s in spikes]

    return spikes_c.index(max(spikes_c))
