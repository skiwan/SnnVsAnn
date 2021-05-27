from typing import Tuple, NamedTuple

import torch
from norse.torch import heaviside
from norse.torch.module.snn import SNNCell
from norse.torch.functional.izhikevich import (
    IzhikevichParameters,
)

# TODO Needs to hold eligibility traces
class CustomIzhikevichState(NamedTuple):
    v: torch.Tensor
    u: torch.Tensor

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





# TODO needs to compute eligibility traces
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
    z_ = eprop_fn(v_ - p.v_th, torch.as_tensor(p.alpha))
    v_ = (1 - z_) * v_ + z_ * p.c
    u_ = (1 - z_) * u_ + z_ * (u_ + p.d)
    return z_, CustomIzhikevichState(v_, u_)


# TODO Needs to forward eligibility traces to backward
# TODO 2 Needs to compute eligibility traces here and make them accesable without changing IZ state
class CustomEProp(torch.autograd.Function):
    @staticmethod
    @torch.jit.ignore
    def forward(ctx, input_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.alpha = alpha
        return heaviside(input_tensor)

# TODO Needs to change to eprop equation
    @staticmethod
    @torch.jit.ignore
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        grad = grad_input / (alpha * torch.abs(inp) + 1.0).pow(
            2
        )  # section 3.3.2 (beta -> alpha)
        return grad, None


@torch.jit.ignore
def eprop_fn(x: torch.Tensor, alpha: float = 100.0) -> torch.Tensor:
    return CustomEProp.apply(x, alpha)