from dataclasses import dataclass

import torch

from torch import nn


@dataclass(frozen=True)
class BlrParametersConfiguration:
    diagonal_floor: float
    sigma_floor: float
    sigma_max: float
    u_diagonal_initial_value: float
    v_diag_initial_value: float
    sigma_initial_value: float
    alpha_init: float


# Pytorch module to train parameterization of covariances thorugh backprop
class BlrParameters(nn.Module):

    def __init__(self, D: int, K: int, cfg: BlrParametersConfiguration) -> None:
        super().__init__()
        self.D = int(D)
        self.K = int(K)
        self.cfg = cfg
        # We actually do not use this as a forward pass in the classical sense. Because the covariances have no real outout in the functional sense.
        # Gradients are flowing to it because we manually register the parameters with the optimizer.
        self.u_raw = nn.Parameter(
            torch.full((self.D,), float(cfg.u_diagonal_initial_value))
        )
        self.v_raw = nn.Parameter(
            torch.full((self.K,), float(cfg.v_diag_initial_value))
        )
        self.sigma2_raw = nn.Parameter(torch.tensor(float(cfg.sigma_initial_value)))
        self.a = nn.Parameter(float(cfg.alpha_init) * torch.randn(self.D))

    def u(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.u_raw) + float(self.cfg.diagonal_floor)

    def v(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.v_raw) + float(self.cfg.diagonal_floor)

    def sigma2(self) -> torch.Tensor:
        sigma2 = torch.nn.functional.softplus(self.sigma2_raw) + float(
            self.cfg.sigma_floor
        )

        return torch.clamp(sigma2, max=float(self.cfg.sigma_max))

    # We are computing this inverse using the Sherman–Morrison formula https://en.wikipedia.org/wiki/Sherman–Morrison_formula
    def U_inverse_sm(self) -> torch.Tensor:
        u = self.u()
        dinv = 1.0 / u
        a = self.a
        dinv_a = dinv * a
        denom = 1.0 + torch.sum(a * dinv_a)
        corr = torch.outer(dinv_a, dinv_a) / denom
        return torch.diag(dinv) - corr

    def V_inverse_diag(self) -> torch.Tensor:
        v = self.v()
        return 1.0 / v
