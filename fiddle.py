from Models.ANN.BaseSCNN import BaseSCNN
from torchsummary import summary

model = BaseSCNN(channels=3, base_filters=8, classes=2, image_width=200, image_height=44).to('cpu')
print(summary(model, (3,44,200), device='cpu'))
print('Done')