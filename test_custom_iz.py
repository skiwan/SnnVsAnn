from Models.SNN.IZMerlin import CustomIzhikevichState, CustomIzhikevichCell, CustomIzhikevichSpikingBehaviour
from norse.torch.functional.izhikevich import IzhikevichParameters
import torch

IP = IzhikevichParameters(0.02, 0.2, -65, 2)
CIS = CustomIzhikevichState(v=torch.tensor(-65.0, requires_grad=True), u=torch.tensor(-65) * IP.b, e=torch.empty(0,1))
CIB = CustomIzhikevichSpikingBehaviour(p=IP, s=CIS)
CIZ = CustomIzhikevichCell(CIB)

# Single timestep
#out, n_state = CIZ.forward(torch.tensor([1,1]))
#loss = torch.nn.functional.mse_loss(out, torch.tensor([1,1]))
#grads = CIZ.backward(n_state,[n_state.v], [loss], IP)
#print(grads)





# Multi Timestep
n_state = None
input = torch.tensor([1,1])
outs = []
voltages = []
learning_s = [torch.tensor([1,1]) for x in range(10)]
for t in range(10):
    out, n_state = CIZ.forward(input, n_state)
    voltages.append(n_state.v)
    outs.append(out)
loss = torch.nn.functional.mse_loss(torch.stack(outs), torch.stack(learning_s))
grads = CIZ.backward(n_state,voltages, [loss], IP)
print(grads)