from norse.torch import SequentialState
import torch


class BaseSCNN(torch.nn.Module):
  def __init__(self, channels, base_filters, classes, image_width=200, image_height=44, padding=0, stride=0):
    super().__init__()
    self.model = SequentialState(
        torch.nn.Conv2d(channels, base_filters, kernel_size=(image_height, 1), padding=padding, stride=stride),
        torch.nn.BatchNorm2d(200),
        torch.nn.ReLU(),
        torch.nn.Conv2d(channels, base_filters*2, kernel_size=(1, 10), padding=padding, stride=stride),
        torch.nn.BatchNorm2d(200),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(1, classes),
        torch.nn.Softmax()
    )

  def forward(self, xs):
      return self.model.forward(xs)

