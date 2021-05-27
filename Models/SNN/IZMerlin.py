from typing import Tuple, NamedTuple, Optional, Any

import torch
from norse.torch import heaviside
from norse.torch.module.snn import SNNCell
from norse.torch.functional.izhikevich import (
    IzhikevichParameters,
)

class CustomIzhikevichState(NamedTuple):
    v: torch.Tensor
    u: torch.Tensor
    e: torch.Tensor # 2 dim -> basically one list of traces per neuron

# Uses new State
class CustomIzhikevichSpikingBehaviour(NamedTuple):
    p: IzhikevichParameters
    s: CustomIzhikevichState



class CustomIzhikevichCell(torch.nn.Module):

    def __init__(self, spiking_method: CustomIzhikevichSpikingBehaviour, **kwargs):
        super().__init__()
        self.activation = custom_izhikevich_step
        self.state_fallback = self.initial_state
        self.p = spiking_method.p
        self.dt = 0.001
        self.spiking_method = spiking_method

    def forward(self, input_tensor: torch.Tensor, state: CustomIzhikevichState = None):
        state = state if state is not None else self.state_fallback(input_tensor)
        return self.activation(input_tensor, state, self.p, self.dt)

    def initial_state(self, input_tensor: torch.Tensor) -> CustomIzhikevichState:
        state = self.spiking_method.s
        state.v.requires_grad = True
        return state

    def backward(self, state: CustomIzhikevichState, loss):
        random_weight = 0.5
        #TODO Q: Do I take first or last eligibility trace?
        while state.e.numel() > 0:
            trace = state.e[:, -1:]
            grad = loss * trace * random_weight
            state.e[:, :-1]
        print('No eligibility traces left')
        self.grad = grad
        return grad

def partial_hidden(iz_state: CustomIzhikevichState, p: IzhikevichParameters, dt: float): # autograd would solve this
    s = iz_state
    return 1 + p.tau_inv * dt * (2 * s.v * p.sq + p.mn)


def compute_eligibility_trace(iz_state: CustomIzhikevichState, p: IzhikevichParameters, dt: float):
    if iz_state.e.numel() == 0:
        new_e = partial_hidden(iz_state, p, dt)
    else:
        old_e = iz_state.e[:, -1:] # last eligibility trace
        new_e = torch.mul(old_e, partial_hidden(iz_state, p, dt))
    return new_e

def custom_izhikevich_step(
    input_current: torch.Tensor,
    s: CustomIzhikevichState,
    p: IzhikevichParameters,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, CustomIzhikevichState]:
    v_ = s.v + p.tau_inv * dt * (
        p.sq * s.v ** 2 + p.mn * s.v + p.bias - s.u + input_current
    )
    u_ = s.u + p.tau_inv * dt * p.a * (p.b * s.v - s.u)

    # Todo Q:should this happen before or after membrane reset?
    e_ = compute_eligibility_trace(s)
    e_ = torch.column_stack((s.e, e_))

    z_ = heaviside(v_ - p.v_th)
    v_ = (1 - z_) * v_ + z_ * p.c
    u_ = (1 - z_) * u_ + z_ * (u_ + p.d)
    return z_, CustomIzhikevichState(v_, u_, e_)

