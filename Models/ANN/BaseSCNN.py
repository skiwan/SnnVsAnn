from norse.torch import SequentialState
import torch


class BaseSCNN(torch.nn.Module):
  def __init__(self, channels, base_filters, classes, image_height=44, padding=0, stride=1, dropout_value=0.3):
    super().__init__()
    self.model = SequentialState(
        torch.nn.Conv2d(channels, base_filters, kernel_size=(image_height, 1), padding=padding, stride=stride),
        torch.nn.BatchNorm2d(base_filters),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=dropout_value),
        torch.nn.Conv2d(base_filters, base_filters*2, kernel_size=(1, 10), padding=padding, stride=stride, dilation=(1,20)),
        torch.nn.BatchNorm2d(base_filters*2),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=dropout_value),
        torch.nn.Flatten(),
        torch.nn.Dropout(p=dropout_value),
        torch.nn.Linear(base_filters*40, base_filters*8),
        torch.nn.Dropout(p=dropout_value),
        torch.nn.Linear(base_filters*8, classes),
        #torch.nn.Softmax(dim=1) Excluded to be able to better use this in advanced models
    )

  def forward(self, xs):
      return self.model.forward(xs)

