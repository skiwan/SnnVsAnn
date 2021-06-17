from .BinaryEEGClassifier import BinaryEEGClassifier
import torch


class FourClassSNNLayer(torch.nn.Module):
  def __init__(self, all_channel_info, all_class_iz_params, gaining_factor):
    super().__init__()
    self.class_one_classifier = BinaryEEGClassifier(channels=all_channel_info[0], p=all_class_iz_params[0], gaining_factor=gaining_factor)
    self.class_two_classifier = BinaryEEGClassifier(channels=all_channel_info[1], p=all_class_iz_params[1], gaining_factor=gaining_factor)
    self.class_three_classifier = BinaryEEGClassifier(channels=all_channel_info[2], p=all_class_iz_params[2], gaining_factor=gaining_factor)
    self.class_four_classifier = BinaryEEGClassifier(achannels=all_channel_info[3], p=all_class_iz_params[3], gaining_factor=gaining_factor)


  # input to the forward function should be a touple of 4 data samples and a touple of 4 states
  # a data sample should be timesteps x channels
  def forward(self, xs, state):
    c1_state, c2_state, c3_state, c4_state = state
    c1_in, c2_in, c3_in, c4_in = xs

    spikes_one = torch.sum(self.class_one_classifier.forward(c1_in, c1_state))
    spikes_two = torch.sum(self.class_two_classifier.forward(c2_in, c2_state))
    spikes_three = torch.sum(self.class_three_classifier.forward(c3_in, c3_state))
    spikes_four = torch.sum(self.class_four_classifier.forward(c4_in, c4_state))

    return torch.stack([spikes_one,spikes_two,spikes_three,spikes_four])
