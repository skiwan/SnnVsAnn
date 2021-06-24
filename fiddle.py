from Models.ANN.BaseSCNN import BaseSCNN
from torchsummary import summary

model = BaseSCNN(channels=3, base_filters=8, classes=4, image_width=200, image_height=44, padding=0, stride=0).to('cpu')
print(summary(model, (44,200,3)))
print('Done')