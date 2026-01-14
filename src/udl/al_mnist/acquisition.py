from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


Method = Literal["random", "max_entropy", "var_ratios", "mean_std", "bald"]


@dataclass(frozen=True)
class AcquisitionResult:
    scores: np.ndarray
    selected_pos: np.ndarray


def _entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0)
    return -(p * torch.log(p)).sum(dim=-1)


@torch.no_grad()
def score_pool(
    model: nn.Module,
    pool_ds: Dataset,
    method: Method,
    device: torch.device,
    batch_size: int,
    mc_samples: int,
    deterministic: bool,
    seed: int,
) -> np.ndarray:
    if method == "random":
        rng = np.random.default_rng(seed)
        return rng.random(len(pool_ds), dtype=np.float64)

    loader = DataLoader(pool_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = model.to(device)

    scores_all: list[np.ndarray] = []

    if deterministic:
        model.eval()
        for xb, _, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.softmax(logits, dim=-1)
            if method == "max_entropy":
                s = _entropy(p)
            elif method == "var_ratios":
                s = 1.0 - p.max(dim=-1).values
            elif method == "mean_std":
                # According to definition
                s = torch.zeros(xb.shape[0], device=device)
            elif method == "bald":
                # According to definition
                s = torch.zeros(xb.shape[0], device=device)
            else:
                raise ValueError(f"Unknown method: {method}")
            scores_all.append(s.detach().cpu().numpy().astype(np.float64))
        return np.concatenate(scores_all, axis=0)
    model.train()

    for xb, _, _ in loader:
        xb = xb.to(device)

        probs_t: list[torch.Tensor] = []
        for _ in range(int(mc_samples)):
            logits = model(xb)
            probs_t.append(torch.softmax(logits, dim=-1))
        probs = torch.stack(probs_t, dim=0)

        p_mean = probs.mean(dim=0)

        if method == "max_entropy":
            s = _entropy(p_mean)
        elif method == "var_ratios":
            s = 1.0 - p_mean.max(dim=-1).values
        elif method == "mean_std":
            std_c = probs.std(dim=0)
            s = std_c.mean(dim=-1)
        elif method == "bald":
            h_mean = _entropy(p_mean)
            h_t = _entropy(probs)
            # In the deterministic case, this should be a constant zero. I am confused about how this is handeled in the paper
            s = h_mean - h_t.mean(dim=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        scores_all.append(s.detach().cpu().numpy().astype(np.float64))

    return np.concatenate(scores_all, axis=0)


def select_topk(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=np.int64)
    part = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
    order = np.lexsort((part, -scores[part]))
    return part[order].astype(np.int64)


def acquire(
    model: nn.Module,
    pool_ds: Dataset,
    method: Method,
    k: int,
    device: torch.device,
    batch_size: int,
    mc_samples: int,
    deterministic: bool,
    seed: int,
) -> AcquisitionResult:
    scores = score_pool(
        model=model,
        pool_ds=pool_ds,
        method=method,
        device=device,
        batch_size=batch_size,
        mc_samples=mc_samples,
        deterministic=deterministic,
        seed=seed,
    )

    if deterministic and method in ("bald", "mean_std"):
        if k <= 0:
            selected_pos = np.array([], dtype=np.int64)
        else:
            selected_pos = np.argsort(scores)[-min(k, len(scores)) :][::-1].astype(
                np.int64
            )
        return AcquisitionResult(scores=scores, selected_pos=selected_pos)

    selected_pos = select_topk(scores, k=k)
    return AcquisitionResult(scores=scores, selected_pos=selected_pos)
