from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import logging


logger = logging.getLogger("udl.al_mnist")


@dataclass(frozen=True)
class TrainConfig:
    device: str
    batch_size: int
    epochs: int
    lr: float
    optimizer: str
    weight_decay_grid: list[float]
    log_every_epochs: int = 0


def _resolve_device(device: str) -> torch.device:
    return "cpu"


def _make_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adadelta":
        return torch.optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer,
) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    n = 0
    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        bs = xb.shape[0]
        total_loss += float(loss.detach().cpu()) * bs
        n += bs
    return total_loss / max(n, 1)


@torch.no_grad()
def accuracy(
    model: nn.Module,
    ds: Dataset,
    device: torch.device,
    batch_size: int,
    deterministic: bool,
    mc_samples: int,
) -> float:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = model.to(device)
    correct = 0
    total = 0

    if deterministic:
        model.eval()
        for xb, yb, _ in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=-1)
            correct += int((pred == yb).sum().detach().cpu())
            total += int(xb.shape[0])
        return correct / max(total, 1)

    model.train()
    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        probs_t: list[torch.Tensor] = []
        for _ in range(int(mc_samples)):
            logits = model(xb)
            probs_t.append(torch.softmax(logits, dim=-1))
        p_mean = torch.stack(probs_t, dim=0).mean(dim=0)
        pred = p_mean.argmax(dim=-1)
        correct += int((pred == yb).sum().detach().cpu())
        total += int(xb.shape[0])
    return correct / max(total, 1)


def train_with_weight_decay_selection(
    get_model_lambda,
    train_ds: Dataset,
    val_ds: Dataset,
    cfg: TrainConfig,
    seed: int,
    deterministic: bool,
    mc_samples_eval: int,
) -> tuple[nn.Module, float, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _resolve_device(cfg.device)

    best_wd = None
    best_acc = -1.0
    best_state = None

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )

    for wd in cfg.weight_decay_grid:
        model = get_model_lambda().to(device)
        opt = _make_optimizer(
            cfg.optimizer, model.parameters(), lr=cfg.lr, weight_decay=float(wd)
        )
        logger.info(
            "WD search: training candidate weight_decay=%.3g on %d labeled examples for %d epochs",
            float(wd),
            len(train_ds),
            int(cfg.epochs),
        )
        for ep in range(int(cfg.epochs)):
            loss = _epoch(model, train_loader, device=device, optimizer=opt)
            if cfg.log_every_epochs and ((ep + 1) % int(cfg.log_every_epochs) == 0):
                logger.info(
                    "WD=%.3g epoch %d/%d train_loss=%.4f",
                    float(wd),
                    ep + 1,
                    int(cfg.epochs),
                    loss,
                )
        val_acc = accuracy(
            model,
            val_ds,
            device=device,
            batch_size=cfg.batch_size,
            deterministic=deterministic,
            mc_samples=mc_samples_eval,
        )
        logger.info(
            "WD search: weight_decay=%.3g val_acc=%.4f", float(wd), float(val_acc)
        )
        if val_acc > best_acc:
            best_acc = float(val_acc)
            best_wd = float(wd)
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_wd is None or best_state is None:
        raise RuntimeError("Failed to select weight decay")

    best_model = get_model_lambda().to(_resolve_device(cfg.device))
    best_model.load_state_dict(best_state)
    return best_model, best_wd, best_acc


def retrain_with_fixed_weight_decay(
    get_model_lambda,
    train_ds: Dataset,
    cfg: TrainConfig,
    seed: int,
    weight_decay: float,
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _resolve_device(cfg.device)
    model = get_model_lambda().to(device)
    opt = _make_optimizer(
        cfg.optimizer, model.parameters(), lr=cfg.lr, weight_decay=float(weight_decay)
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    logger.info(
        "Retrain: weight_decay=%.3g on %d labeled examples for %d epochs",
        float(weight_decay),
        len(train_ds),
        int(cfg.epochs),
    )
    for ep in range(int(cfg.epochs)):
        loss = _epoch(model, train_loader, device=device, optimizer=opt)
        if cfg.log_every_epochs and ((ep + 1) % int(cfg.log_every_epochs) == 0):
            logger.info(
                "Retrain: wd=%.3g epoch %d/%d train_loss=%.4f",
                float(weight_decay),
                ep + 1,
                int(cfg.epochs),
                loss,
            )
    return model
