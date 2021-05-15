from Models.SNN.BinaryEEGClassifier import BinaryEEGClassifier


import torch
from norse.torch.functional.izhikevich import IzhikevichSpikingBehaviour, IzhikevichState, IzhikevichParameters



def train_spike(model, data, labels, optimizer, epochs=200):
    losses = []
    outs = []
    for i in range(epochs):
      print(f'epoch {i} out of {epochs}')
      for xs in data:
          state = (None, None) # Make sure to reset state before every new epoch
          out, state = model(xs,state)
          loss = torch.nn.functional.mse_loss(out, labels)
          optimizer.zero_grad()
          loss.backward(retain_graph=True)
          optimizer.step()
          losses.append(loss)
          #print(model.output_weights.weight, model.output_weights.weight.grad) # Debug purposes
      print(min(losses))
    return losses


if __name__ == '__main__':
  initial_params = IzhikevichParameters(0.02,0.2,-65,2)
  initial_state=IzhikevichState(v=torch.tensor(-65.0, requires_grad=True), u=torch.tensor(-65) * initial_params.b)
  behaviour = IzhikevichSpikingBehaviour(initial_params, initial_state)

  bec = BinaryEEGClassifier(4, behaviour)
  data = torch.ones(32,100,4)*105
  pattern = [0.0, 0.0, 0.0, 0.0, 1.0] * 2
  labels = torch.as_tensor(pattern * 10)
  labels = torch.reshape(labels,(100,1))

  bec.input_weights.weight = torch.nn.Parameter(torch.ones_like(bec.input_weights.weight))
  bec.output_weights.weight = torch.nn.Parameter(torch.ones_like(bec.output_weights.weight)*210)

  optimizer = torch.optim.Adam(bec.parameters(), lr=0.1)
  torch.autograd.set_detect_anomaly(True)

  m1_losses = train_spike(bec, data, labels, optimizer, epochs=50)
  print(m1_losses)
  print(min(m1_losses))