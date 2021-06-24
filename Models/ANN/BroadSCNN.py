from .BaseSCNN import BaseSCNN
import torch

class BroadSCNN(torch.nn.Module):
  def __init__(self, channels, base_filters):
    super().__init__()
    self.class_one_classifier = BaseSCNN(channels=channels[0], base_filters=base_filters[0], classes=2)
    self.class_two_classifier = BaseSCNN(channels=channels[1], base_filters=base_filters[1], classes=2)
    self.class_three_classifier = BaseSCNN(channels=channels[2], base_filters=base_filters[2], classes=2)
    self.class_four_classifier = BaseSCNN(channels=channels[3], base_filters=base_filters[3], classes=2)


  # input to the forward function should be a touple of 4 data samples and a touple of 4 states
  # a data sample should be timesteps x channels
  def forward(self, xs):
    print(xs.size())
    c1_in, c2_in, c3_in, c4_in = xs
    spikes_one = self.class_one_classifier.forward(c1_in)[0]
    spikes_two = self.class_two_classifier.forward(c2_in)[0]
    spikes_three = self.class_three_classifier.forward(c3_in)[0]
    spikes_four = self.class_four_classifier.forward(c4_in)[0]

    class_convidences = torch.stack([spikes_one, spikes_two, spikes_three, spikes_four])
    class_convidences /= torch.sum(class_convidences)
    return class_convidences
