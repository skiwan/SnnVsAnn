from typing import Tuple, NamedTuple

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



class CustomIzhikevichCell(SNNCell):

    def __init__(self, spiking_method: CustomIzhikevichSpikingBehaviour, **kwargs):
        super().__init__(
            custom_izhikevich_step, self.initial_state, spiking_method.p, **kwargs
        )
        self.spiking_method = spiking_method

    def initial_state(self, input_tensor: torch.Tensor) -> CustomIzhikevichState:
        state = self.spiking_method.s
        state.v.requires_grad = True
        return state



# TODO Replace dummy implementation
# Will be partial of state given previous step * previous eligibility trace + partial hidden state over weight
# Only partial of hidden state over weight for t=1
def compute_eligibility_trace(iz_state: CustomIzhikevichState):
    if iz_state.e.numel() == 0:
        new_e = torch.ones(iz_state.u.size()[0]) # TODO replace with partial hidden state over weight function of same dimension as number of neurons
    else:
        old_e = iz_state.e[:, -1:] # last eligibility trace
        new_e = torch.add(old_e, torch.ones(iz_state.u.size()[0])) # Todo replace with correct eligibility trace function
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

    z_ = eprop_fn(v_ - p.v_th, torch.as_tensor(p.alpha))
    v_ = (1 - z_) * v_ + z_ * p.c
    u_ = (1 - z_) * u_ + z_ * (u_ + p.d)
    return z_, CustomIzhikevichState(v_, u_, e_)


# TODO Needs to forward eligibility traces to backward
# TODO 2 Needs to compute eligibility traces here and make them accesable without changing IZ state
class CustomEProp(torch.autograd.Function):
    @staticmethod
    @torch.jit.ignore
    def forward(ctx, input_tensor: torch.Tensor, alpha: float, eligibility_trace: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward((input_tensor, eligibility_trace)) # save trace for backward
        ctx.alpha = alpha
        return heaviside(input_tensor)

# TODO Needs to change to eprop equation
    @staticmethod
    @torch.jit.ignore
    def backward(ctx, grad_output):
        (inp, e_traces) = ctx.saved_tensors #Todo Q: retrive eligibility traces (check if this is done correctly or if you need to unpack better)
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        grad = grad_input / (alpha * torch.abs(inp) + 1.0).pow(
            2
        )  # section 3.3.2 (beta -> alpha)
        return grad, None


@torch.jit.ignore
def eprop_fn(x: torch.Tensor, alpha: float = 100.0) -> torch.Tensor:
    return CustomEProp.apply(x, alpha)