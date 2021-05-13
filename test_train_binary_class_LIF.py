from Models.SNN.BinaryEEGClassifierLIF import BinaryEEGClassifierLIF
from norse.torch.module import LIFParameters


import torch



def train_spike(model, data, labels, optimizer, epochs=200):
    losses = []
    for i in range(epochs):
      print(f'epoch {i} out of {epochs}')
      for xs in data:
          out, state = model(xs)
          loss = torch.nn.functional.mse_loss(out, labels)
          optimizer.zero_grad()
          loss.backward(retain_graph=True)
          optimizer.step()
          losses.append(loss)
      #print(model.output_weights.weight, model.output_weights.weight.grad) # Debug purposes
    return losses


if __name__ == '__main__':

  bec_lif = BinaryEEGClassifierLIF(LIFParameters())
  data = torch.ones(32,100,4)
  pattern = [0.0, 0.0, 0.0, 0.0, 1.0] * 2
  labels = torch.as_tensor(pattern * 10)
  labels = torch.reshape(labels,(100,1))


  optimizer = torch.optim.Adam(bec_lif.parameters(), lr=0.01)
  torch.autograd.set_detect_anomaly(True)

  m1_losses = train_spike(bec_lif, data, labels, optimizer, epochs=5)
  print(m1_losses)