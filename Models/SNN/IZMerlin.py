from typing import Tuple, NamedTuple, Optional, Any
import numpy as np
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

    def pseudo_derivative(self, v , p: IzhikevichParameters):
        v_th = torch.tensor([p.v_th for i in range(v.size()[0])])
        s = (1 - np.abs((v.detach() - v_th) / v_th))
        t = np.maximum(np.zeros_like(v.detach().numpy()), s)
        return 1 / v_th * 0.3 * t

    def compute_eligibility_trace(self, v, p: IzhikevichParameters, eligibility_vector):
        pseudo = self.pseudo_derivative(v, p)
        return torch.mul(pseudo, eligibility_vector)

    def backward(self, state: CustomIzhikevichState, voltages, learning_signals, p: IzhikevichParameters):
        random_weight = 0.5
        print(state.e.size())
        if state.e.size()[0] != len(learning_signals):
            print('Length of Learning Signals and eligbility vectors is not equal')
            return None
        if state.e.size()[0] != len(voltages):
            print('Length of voltages and eligbility vectors is not equal')
            return None
        grads = []
        states = state.e
        #TODO Q: Do I take first or last eligibility trace? > first one because of learning signal
        while states.numel() > 0:
            e_vec = state.e[:1, :]
            voltage = voltages[0]
            ls = learning_signals[0]
            trace = self.compute_eligibility_trace(voltage, p, e_vec)
            grads.append(ls * trace * random_weight)
            states = states[1:, :]
            learning_signals = learning_signals[1:]
            voltages = voltages[1:]
        print('No eligibility traces left')
        self.grad = grads[-1]

        return grads


def v_over_v(iz_state: CustomIzhikevichState, p: IzhikevichParameters, dt: float):
    return 1 + p.tau_inv * dt * (2*iz_state.v * p.sq + p.mn)

def v_over_u(p: IzhikevichParameters, dt: float):
    return - p.tau_inv * dt

def u_over_v(p: IzhikevichParameters, dt: float):
    return p.tau_inv * dt * p.a * p.b

def u_over_u(p: IzhikevichParameters, dt: float):
    return 1 + p.tau_inv * dt * p.a * (-1)

def v_over_w(p: IzhikevichParameters, dt: float):
    return p.tau_inv * dt

def u_over_w():
    return 0

def compute_eligibility_vector(iz_state: CustomIzhikevichState, p: IzhikevichParameters, dt: float, previous_eligibility_vector=None):
    sec = torch.tensor([v_over_w(p, dt), u_over_w()])
    a = sec.size()[0]
    x = iz_state.v.size()[0]
    sec = torch.cat(x*[sec])
    sec = torch.reshape(sec, (x, a))
    if previous_eligibility_vector is None:
        return sec
    else:
        vov = v_over_v(iz_state, p, dt)
        vou = v_over_u(p, dt)
        uov = u_over_v(p, dt)
        uou = u_over_u(p, dt)
        jacobian = torch.tensor([[vov, vou],[uov, uou]])
        fir = (jacobian * previous_eligibility_vector + sec)
        return fir




# TODO eligibility trace needs right dimensions for time and multi neurons
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

    z_ = heaviside(v_ - p.v_th)
    v_ = (1 - z_) * v_ + z_ * p.c
    u_ = (1 - z_) * u_ + z_ * (u_ + p.d)

    # Todo Q:should this happen before or after membrane reset?
    #  > given OSTL paper this should happen after Cell computation
    s_ = CustomIzhikevichState(v_, u_, s.e)
    if s_.e.numel() > 0:
        last_e = s_.e[:, -1:]  # last eligibility trace
    else:
        last_e = None

    e_ = compute_eligibility_vector(s_, p, dt, last_e)
    if s_.e.numel() > 0:
        e_ = torch.column_stack((s_.e, e_))
    else:
        e_ = torch.unsqueeze(e_,0)

    return z_, CustomIzhikevichState(v_, u_, e_)

