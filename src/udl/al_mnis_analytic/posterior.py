import torch

from udl.al_mnis_analytic.covariances import BlrParameters


# We use this for implicitly calculating the inverse of lamda (which we need for the posterior).
# Otherwise expensive + Lambda is pos. def.
def _cholesky(A: torch.Tensor) -> torch.Tensor:
    A = 0.5 * (A + A.transpose(-1, -2))
    try:
        return torch.linalg.cholesky(A)
    except RuntimeError:
        eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
        return torch.linalg.cholesky(A + 1e-6 * eye)


def posterior_mean_and_choleskys(
    Phi,
    Y,
    lastLayerCovariances: BlrParameters,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    invU = lastLayerCovariances.U_inverse_sm()
    invV = lastLayerCovariances.V_inverse_diag()
    sigma2 = lastLayerCovariances.sigma2()
    alpha = 1.0 / sigma2  # sigma_y^{-1}_ii
    PhiTPhi = Phi.transpose(0, 1) @ Phi
    PhiYAlpha = (Phi.transpose(0, 1) @ Y) * alpha
    W_cols: list[torch.Tensor] = []
    choleskys: list[torch.Tensor] = []
    lambda_diags: list[torch.Tensor] = []
    for k in range(Y.shape[1]):
        Lambda_k = (invV[k] * invU) + (alpha * PhiTPhi)
        L = _cholesky(Lambda_k)
        # Essentially instead of mu =sigma_N vec(PhiTYSigmaY) we solve Lambda mu = vec(PhiTYSigmaY)
        wk = torch.cholesky_solve(PhiYAlpha[:, k : k + 1], L).squeeze(1)
        W_cols.append(wk)
        choleskys.append(L)
        lambda_diags.append(torch.diagonal(Lambda_k, 0))
    W_mean = torch.stack(W_cols, dim=1)
    return W_mean, choleskys, lambda_diags


@torch.no_grad()
def variance_diagonal_analytic(
    Phi_pool: torch.Tensor,
    chol_L: list[torch.Tensor],
) -> torch.Tensor:
    M, D = Phi_pool.shape
    K = len(chol_L)
    out = torch.zeros((M, K), device=Phi_pool.device, dtype=Phi_pool.dtype)
    rhs = Phi_pool.transpose(0, 1).contiguous()
    for k, L in enumerate(chol_L):
        X = torch.cholesky_solve(rhs, L)
        out[:, k] = torch.sum(rhs * X, dim=0)
    return out


@torch.no_grad()
def variance_diagonal_mfvi(
    Phi_pool: torch.Tensor,
    lambda_diag: list[torch.Tensor],
) -> torch.Tensor:
    phi2 = Phi_pool**2
    M = Phi_pool.shape[0]
    K = len(lambda_diag)
    out = torch.zeros((M, K), device=Phi_pool.device, dtype=Phi_pool.dtype)
    for k, d in enumerate(lambda_diag):
        out[:, k] = (phi2 / d[None, :]).sum(dim=1)
    return out


@torch.no_grad()
def score_trace(
    var_diag: torch.Tensor,
) -> torch.Tensor:
    return var_diag.sum(dim=1)
