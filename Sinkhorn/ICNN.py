"""
    A smooth ICNN (use SoftPlus to satisfy non-negativity) structure, which is a special case of a HyCNN.
"""

import math
from functools import partial
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence, Union

from Sinkhorn.logdet_estimators import (
    stochastic_lanczos_quadrature,
    stochastic_logdet_gradient_estimator,
    unbiased_logdet,
)


HESS_NORM_TRACER = list()


def _hk_moments(fan_in: int) -> tuple[float, float, float]:
    n = float(fan_in)
    denom = 6.0 * (math.pi - 1.0) + (n - 1.0) * (
        3.0 * math.sqrt(3.0) + 2.0 * math.pi - 6.0
    )
    mu = math.sqrt(6.0 * math.pi / (n * denom))
    sigma2 = 1.0 / n
    mu_b = -math.sqrt(3.0 * n / denom)
    return mu, sigma2, mu_b


def _positive_weight_moments(fan_in: int) -> tuple[float, float]:
    denom = fan_in * fan_in + (math.pi - 1.0 / math.pi) * fan_in
    return math.sqrt(2.0 * math.pi / denom), 1.0 / (3.0 * fan_in)


def _lognormal_params_from_mean_var(mean: float, var: float) -> tuple[float, float]:
    tau = mean * mean + var
    return (
        math.log(mean * mean / math.sqrt(tau)),
        math.log(tau / (mean * mean)),
    )


class PositiveLinear(nn.Module):
    """Linear layer with non-negative weights via softplus reparameterization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_raw = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            mu, sigma2 = _positive_weight_moments(self.in_features)
            mu_ln, sigma2_ln = _lognormal_params_from_mean_var(mu, sigma2)
            pos = torch.exp(
                torch.randn_like(self.weight_raw) * math.sqrt(sigma2_ln) + mu_ln
            )
            pos = pos.clamp_min(torch.finfo(pos.dtype).tiny)
            self.weight_raw.copy_(torch.log(torch.expm1(pos)))
            if self.bias is not None:
                _, _, mu_b = _hk_moments(self.in_features)
                self.bias.fill_(mu_b)

    @property
    def weight(self) -> torch.Tensor:
        return F.softplus(self.weight_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ICNN(nn.Module):
    """Smooth ICNN using PositiveLinear layers and no maxout/log-sum-exp gates."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[Sequence[int], int],
        output_dim: int,
        rtol=0.0,
        atol=1e-3,
        quad_coefficient=0.2,
        unbiased=False,
    ):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        if len(hidden_dims) == 0:
            raise ValueError("ICNN needs at least one hidden layer.")

        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.quad_coefficient = quad_coefficient

        self.input_layer = nn.Linear(self.input_dim, self.hidden_dims[0])
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(self.hidden_dims[:-1], self.hidden_dims[1:]):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "positive": PositiveLinear(in_dim, out_dim, bias=False),
                        "input": nn.Linear(self.input_dim, out_dim),
                    }
                )
            )
        self.positive_out = PositiveLinear(self.hidden_dims[-1], output_dim, bias=False)
        self.input_out = nn.Linear(self.input_dim, output_dim)

        self._reset_input_weights()

        self.m1 = 10
        self.m2 = self.input_dim
        self.rtol = rtol
        self.atol = atol
        self.stochastic_estimate_fn = unbiased_logdet if unbiased else partial(
            stochastic_lanczos_quadrature, m=min(self.m1, self.m2)
        )
        self.stochastic_grad_estimate_fn = partial(
            stochastic_logdet_gradient_estimator,
            m=min(self.m2, self.m2),
            rtol=self.rtol,
            atol=self.atol,
        )

    @torch.no_grad()
    def _reset_input_weights(self) -> None:
        first_std = math.sqrt(1.0 / self.input_dim)
        nn.init.normal_(self.input_layer.weight, 0.0, first_std)
        self.input_layer.bias.zero_()

        passthrough_std = math.sqrt(1.0 / (4.0 * self.input_dim))
        for layer in self.layers:
            nn.init.normal_(layer["input"].weight, 0.0, passthrough_std)
            layer["input"].bias.zero_()
            layer["positive"].reset_parameters()

        self.positive_out.reset_parameters()
        nn.init.normal_(self.input_out.weight, 0.0, passthrough_std)
        self.input_out.bias.zero_()

    def _potential(self, x: torch.Tensor) -> torch.Tensor:
        z = F.softplus(self.input_layer(x))
        for layer in self.layers:
            z = F.softplus(layer["positive"](z) + layer["input"](x))

        quad_term = self.quad_coefficient * 0.5 * torch.sum(x ** 2, dim=1, keepdim=True)
        return self.positive_out(z) + self.input_out(x) + quad_term

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._potential(x)

    def forward_transform_stochastic(self, x, logdet=0, context=None, extra=None):
        nt, *dims = x.shape
        dim = np.prod(dims)

        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            output_func = self._potential(x)
            f = torch.autograd.grad(output_func.sum(), x, create_graph=True)[0]

            def hvp_fun(v):
                v = v.reshape(nt, *dims)
                hvp = torch.autograd.grad(
                    f,
                    x,
                    v,
                    create_graph=self.training,
                    retain_graph=True,
                )[0]
                HESS_NORM_TRACER.append((torch.norm(hvp) / torch.norm(v)).detach().cpu())
                if not torch.isnan(v).any() and torch.isnan(hvp).any():
                    raise ArithmeticError("v has no nans but hvp has nans.")
                return hvp.reshape(nt, dim)

        if self.training:
            v1 = sample_rademacher(nt, dim, device=x.device, dtype=x.dtype)
            est1 = self.stochastic_grad_estimate_fn(hvp_fun, v1)
        else:
            est1 = 0

        if not self.training or (extra is not None and len(extra) > 0):
            try:
                v2 = torch.nn.functional.normalize(
                    sample_rademacher(nt, dim, device=x.device, dtype=x.dtype),
                    dim=-1,
                )
                est2 = self.stochastic_estimate_fn(hvp_fun, v2)
            except Exception:
                import traceback

                print("stochastic_estimate_fn failed with the following error message:")
                print(traceback.format_exc(), flush=True)
                est2 = torch.zeros_like(logdet).fill_(float("nan"))
            if extra is not None and len(extra) > 0:
                extra[0] = extra[0] + est2.detach()
        else:
            est2 = 0

        return output_func, logdet + est1 if self.training else logdet + est2


def sample_rademacher(*shape, device="cpu", dtype=torch.float64):
    return (torch.rand(*shape, device=device) > 0.5).to(dtype) * 2.0 - 1.0
