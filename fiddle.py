from Models.ANN.BaseSCNN import BaseSCNN
from torchsummary import summary

model = BaseSCNN(channels=8, base_filters=8, classes=2, image_height=44).to('cpu')
print(summary(model, (8,44,200), device='cpu'))
model = BaseSCNN(channels=32, base_filters=8, classes=4, image_height=44).to('cpu')
print(summary(model, (32,44,200), device='cpu'))
print('Done')