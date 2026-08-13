"""
    This file is the structure of HyperICNN, following the paper: 
    Hyper Input Convex Neural Networks for Shape Constrained Learning and Optimal Transport (2026). 

    Main codes copied from https://github.com/hundrieser/HyCNN

    Our HyCNN is strongly convex by adding a quadratic term to form a invertible map. 

    We add the estimator of log-det-Hessian and its gradient in the forward part (def forward_transform_stochastic). 
    
    We also add some basic settings for the estimators in __init__. 

"""

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


HESS_NORM_TRACER = list()

# Additional imports for log-det-Hessian estimation
from Sinkhorn.logdet_estimators import *
from functools import partial


def _hk_moments(fan_in: int) -> tuple[float, float, float]:
    """Hoedt-Klambauer ICNN moments."""
    n = float(fan_in)
    denom = 6.0 * (math.pi - 1.0) + (n - 1.0) * (3.0 * math.sqrt(3.0) + 2.0 * math.pi - 6.0)
    mu = math.sqrt(6.0 * math.pi / (n * denom))
    sigma2 = 1.0 / n
    mu_b = -math.sqrt(3.0 * n / denom)
    return mu, sigma2, mu_b


def _picnn_v_moments(fan_in: int) -> tuple[float, float]:
    """PICNN-V moments used for the V-matrices of HyCNN."""
    denom = fan_in * fan_in + (math.pi - 1.0 / math.pi) * fan_in
    return math.sqrt(2.0 * math.pi / denom), 1.0 / (3.0 * fan_in)


def _lognormal_params_from_mean_var(mean: float, var: float) -> tuple[float, float]:
    tau = mean * mean + var
    return (math.log(mean * mean / math.sqrt(tau)),
            math.log(tau / (mean * mean)))


class PositiveLinear(nn.Module):
    """Positive linear layer with softplus reparameterisation."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_raw = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            mu, sigma2 = _picnn_v_moments(self.in_features)
            mu_ln, sigma2_ln = _lognormal_params_from_mean_var(mu, sigma2)
            pos = torch.exp(torch.randn_like(self.weight_raw) * math.sqrt(sigma2_ln) + mu_ln)
            pos = pos.clamp_min(torch.finfo(pos.dtype).tiny)
            # invert softplus: W_raw = log(exp(W) - 1)
            self.weight_raw.copy_(torch.log(torch.expm1(pos)))
            if self.bias is not None:
                _, _, mu_b = _hk_moments(self.in_features)
                self.bias.fill_(mu_b)

    @property
    def weight(self) -> torch.Tensor:
        return F.softplus(self.weight_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class HyCNN(nn.Module):
    """HyCNN: two-branch log-sum-exp partially input-convex network."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim, 
                 activation: str = "logsumexp", tau: float = 10.0, rtol=0.0, atol=1e-3, quad_coefficient=0.2, unbiased=False):
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("HyCNN needs hidden layers")
        self.activation = activation.lower()
        if self.activation not in {"max", "logsumexp", "lse"}:
            raise ValueError("activation must be 'max' or 'logsumexp'")
        self.tau = float(tau)
        self.tail_tau = float(tau)
        self.input_dim = input_dim
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.num_gate_layers = len(self.hidden_dims)
        self.quad_coefficient = quad_coefficient

        first = self.hidden_dims[0]
        self.W0_gates = nn.ModuleList([nn.Linear(input_dim, first), nn.Linear(input_dim, first)])
        self.layers = nn.ModuleList()
        for i_dim, o_dim in zip(self.hidden_dims[:-1], self.hidden_dims[1:]):
            self.layers.append(nn.ModuleDict({
                "V_gates": nn.ModuleList([
                    PositiveLinear(i_dim, o_dim, bias=False),
                    PositiveLinear(i_dim, o_dim, bias=False),
                ]),
                "W_gates": nn.ModuleList([
                    nn.Linear(input_dim, o_dim),
                    nn.Linear(input_dim, o_dim),
                ]),
            }))
        self.V_out = PositiveLinear(self.hidden_dims[-1], output_dim, bias=False)
        self.W_out = nn.Linear(input_dim, output_dim)
        self._reset_parameters(input_dim)

        # A strongly convex coefficient
        self.raw_a = nn.Parameter(torch.tensor(0.2))

        # For log-det-Hessian estimation, we introduce two methods. 
        self.m1 = 10
        self.m2 = input_dim
        self.rtol = rtol
        self.atol = atol
        # For testing/evaluating process, we use unbiased or biased but accurate estimator.
        self.stochastic_estimate_fn = unbiased_logdet if unbiased else \
            partial(stochastic_lanczos_quadrature, m=min(self.m1, self.m2))
        # For training process, we use an estimator that has more stable gradient. 
        self.stochastic_grad_estimate_fn = partial(
            stochastic_logdet_gradient_estimator, m=min(self.m2, self.m2), rtol=self.rtol, atol=self.atol)

    @torch.no_grad()
    def _reset_parameters(self, input_dim: int) -> None:
        w0_std = math.sqrt(1.0 / input_dim)
        for gate in self.W0_gates:
            nn.init.normal_(gate.weight, 0.0, w0_std)
            gate.bias.zero_()
        wl_std = math.sqrt(1.0 / (4.0 * input_dim))
        for layer in self.layers:
            out_dim = layer["W_gates"][0].out_features
            bias_val = -math.sqrt(out_dim / (2.0 * math.pi * out_dim + (2.0 * math.pi - 2.0)))
            for gate in layer["W_gates"]:
                nn.init.normal_(gate.weight, 0.0, wl_std)
                gate.bias.fill_(bias_val)
            for v in layer["V_gates"]:
                v.reset_parameters()
        self.V_out.reset_parameters()
        nn.init.normal_(self.W_out.weight, 0.0, wl_std)
        self.W_out.bias.fill_(-math.sqrt(1.0 / (2.0 * math.pi + 2.0 * math.pi - 2.0)))

    def _logsumexp(self, a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
        if tau <= torch.finfo(a.dtype).eps:
            return torch.maximum(a, b)
        stacked = torch.stack((a, b), dim=0)
        return tau * torch.logsumexp(stacked / tau, dim=0)

    def _tau_for_layer(self, layer_idx: int) -> float:
        if layer_idx >= max(0, self.num_gate_layers - 2):
            return self.tail_tau
        return self.tau

    def _gate(self, a: torch.Tensor, b: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if self.activation == "max":
            return torch.maximum(a, b)
        return self._logsumexp(a, b, self._tau_for_layer(layer_idx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._gate(self.W0_gates[0](x), self.W0_gates[1](x), 0)
        for layer_idx, layer in enumerate(self.layers, start=1):
            a1 = layer["V_gates"][0](z) + layer["W_gates"][0](x)
            a2 = layer["V_gates"][1](z) + layer["W_gates"][1](x)
            z = self._gate(a1, a2, layer_idx)

        # return self.V_out(z) + self.W_out(x)

        # quad_coefficient = F.softplus(self.raw_a) + 1e-4
        quad_term = self.quad_coefficient * 0.5 * torch.sum(x ** 2, dim=1, keepdim=True)

        return self.V_out(z) + self.W_out(x) + quad_term
    
    # This forward process return the convex function and estimated log-det-Hessian
    def forward_transform_stochastic(self, x, logdet=0, context=None, extra=None):

        nt, *dims = x.shape
        dim = np.prod(dims)

        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            z = self._gate(self.W0_gates[0](x), self.W0_gates[1](x), 0)
            for layer_idx, layer in enumerate(self.layers, start=1):
                a1 = layer["V_gates"][0](z) + layer["W_gates"][0](x)
                a2 = layer["V_gates"][1](z) + layer["W_gates"][1](x)
                z = self._gate(a1, a2, layer_idx)

            # Quadratic term for strong convexity
            # quad_coefficient = Func.softplus(self.raw_a) + 1e-4
            quad_term = self.quad_coefficient * 0.5 * torch.sum(x ** 2, dim=1, keepdim=True)
            
        F = self.V_out(z) + self.W_out(x) + quad_term
        f = torch.autograd.grad(F.sum(), x, create_graph=True)[0]

        def hvp_fun(v):
            # v is (nt, dim)
            v = v.reshape(nt, *dims)
            hvp = torch.autograd.grad(f, x, v, create_graph=self.training, retain_graph=True)[0]

            HESS_NORM_TRACER.append((torch.norm(hvp) / torch.norm(v)).detach().cpu())

            if not torch.isnan(v).any() and torch.isnan(hvp).any():
                raise ArithmeticError("v has no nans but hvp has nans.")
            hvp = hvp.reshape(nt, dim)
            return hvp

        v2 = torch.nn.functional.normalize(sample_rademacher(nt, dim), dim=-1).to(f)
        exact_estimate = self.stochastic_estimate_fn(hvp_fun, v2)

        # return F, logdet + est1 if self.training else logdet + est2
        return F, logdet + exact_estimate


def set_hycnn_tau(module: nn.Module, tau: float) -> None:
    """Update `tau` on every HyCNN submodule."""
    for child in module.modules():
        if isinstance(child, HyCNN):
            child.tau = float(tau)
            child.tail_tau = float(tau)


def transport(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Brenier transport map T(x) = grad f(x)."""
    x_req = x.detach().clone().requires_grad_(True)
    y = net(x_req)
    grad = torch.autograd.grad(y.sum(), x_req, create_graph=False)[0]
    return grad.detach()


# Hutchinson’s estimation, sample an isotropic random vector (e.g., Rademacher or standard Gaussian)
def sample_rademacher(*shape, device='cpu', dtype=torch.float64):
    return (torch.rand(*shape, device=device) > 0.5).to(dtype) * 2.0 - 1.0