from Models.SNN.BinaryEEGClassifier import BinaryEEGClassifier


import torch
from norse.torch.module import  IzhikevichCell
from norse.torch.functional.izhikevich import IzhikevichSpikingBehaviour, IzhikevichState, IzhikevichParameters 



def train_spike(model, data, labels, optimizer, epochs=200):
    losses = []
    for i in range(epochs):
      state = (None, None) # Make sure to reset state before every new epoch
      for x in data:
          out, state = model(x,state)
          loss = torch.nn.functional.mse_loss(out, labels)
          optimizer.zero_grad()
          loss.backward(retain_graph=True)
          optimizer.step()
          losses.append(loss)
          #print(model.output_weights.weight, model.output_weights.weight.grad) # Debug purposes
    return losses


if __name__ == '__main__':
  initial_params = IzhikevichParameters(0.02,0.2,-65,2)
  initial_state=IzhikevichState(v=torch.tensor(-70.0, requires_grad=True), u=torch.tensor(-70) * initial_params.b)
  behaviour = IzhikevichSpikingBehaviour(initial_params, initial_state)

  bec = BinaryEEGClassifier(behaviour)
  data = torch.ones(32,100,4)
  pattern = [0.0, 0.0, 0.0, 0.0, 1.0] * 2
  labels = torch.as_tensor(pattern * 10)
  labels = torch.reshape(labels,(100,1))

  optimizer = torch.optim.Adam(bec.parameters(), lr=0.1)

  m1_losses = train_spike(bec, data, labels, optimizer, epochs=1000)
  print(m1_losses)