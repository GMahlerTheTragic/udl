import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from udl.al_mnist.data import HFDatasetsMnist, load_mnist_hf, make_splits
from udl.al_mnist.model import ReferenceCNN
from udl.al_mnis_analytic.covariances import BlrParametersConfiguration, BlrParameters
from udl.al_mnis_analytic.posterior import (
    posterior_mean_and_choleskys,
    score_trace,
    variance_diagonal_mfvi,
    variance_diagonal_analytic,
)


logger = logging.getLogger("udl.al_mnist_analytic")

DATASET_NUM_CLASSES = 10
DATASET_INIT_LABELED = 20
DATASET_VAL_SIZE = 0
DATASET_SEED = 0

AL_STEPS = 100
AL_ACQUIRE_PER_STEP = 10
AL_REPEATS = 3
AL_BASE_SEED = 1337
AL_POOL_SUBSET = 20000

FEATURE_DEVICE = "cpu"

PRETRAIN_N = 1000
PRETRAIN_EPOCHS = 50
PRETRAIN_BATCH_SIZE = 128
PRETRAIN_LR = 1.0e-3
PRETRAIN_OPTIMIZER = "adam"
PRETRAIN_WEIGHT_DECAY = 0.0

TARGET_CENTER = True
TARGET_SCALE = False
TARGET_EPS = 1.0e-6

COV_DIAG_FLOOR = 1.0e-4
COV_SIGMA2_FLOOR = 1.0e-3
COV_SIGMA2_MAX = 100.0
COV_U_DIAG_INIT = -1.0
COV_U_LOW_RANK_INIT = 0.01
COV_V_DIAG_INIT = -1.0
COV_SIGMA2_INIT = -0.5

TRAIN_EPOCHS = 50
TRAIN_BATCH_SIZE_FEATURES = 128
TRAIN_LR = 1.0e-4
TRAIN_OPTIMIZER = "adam"
TRAIN_WEIGHT_DECAY = 0.0
TRAIN_REG_LAMBDA = 1.0e-4
TRAIN_LOG_EVERY_EPOCHS = 0

METHODS: list[str] = ["random", "blr_analytic", "blr_mfvi"]


@dataclass(frozen=True)
class StepRecord:
    method: str
    repeat: int
    seed: int
    step: int
    acquired_from_pool: int
    total_labeled: int
    test_rmse: float
    train_rmse: float
    test_acc: float
    train_acc: float
    sigma2: float


def _set_requires_grad(m: torch.nn.Module, flag: bool) -> None:
    for p in m.parameters():
        p.requires_grad_(flag)


def _one_hot(labels: np.ndarray, K: int) -> np.ndarray:
    y = np.zeros((len(labels), K), dtype=np.float32)
    y[np.arange(len(labels)), labels.astype(np.int64)] = 1.0
    return y


def _target_stats(
    labels: np.ndarray, K: int, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(labels.astype(np.int64), minlength=K).astype(np.float64)
    p = counts / max(counts.sum(), 1.0)
    std = np.sqrt(np.maximum(p * (1.0 - p), 0.0)) + float(eps)
    return p.astype(np.float32), std.astype(np.float32)


def _make_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def _extract_features(
    feature_net: ReferenceCNN,
    ds: Dataset,
    device: torch.device,
    batch_size: int,
    with_grad: bool,
) -> tuple[torch.Tensor, np.ndarray]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    feature_net = feature_net.to(device)
    feature_net.eval()

    feats: list[torch.Tensor] = []
    labels: list[np.ndarray] = []

    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        for xb, yb, _ in loader:
            xb = xb.to(device)
            z = feature_net.forward_features(xb, apply_dropout2=False)
            feats.append(z)
            labels.append(yb.detach().cpu().numpy().astype(np.int64))

    Phi = torch.cat(feats, dim=0)
    y = np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=np.int64)
    return Phi, y


def _rmse_and_acc(
    mu: torch.Tensor, y_true_onehot: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mse = torch.mean((mu - y_true_onehot) ** 2)
    rmse = torch.sqrt(mse + 1e-12)
    pred = mu.argmax(dim=1)
    true = y_true_onehot.argmax(dim=1)
    acc = torch.mean((pred == true).to(torch.float32))
    return rmse, acc


@torch.no_grad()
def _eval_with_W_mean(
    feature_net: ReferenceCNN,
    W_mean: torch.Tensor,
    ds: Dataset,
    device: torch.device,
    batch_size_features: int,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> tuple[float, float]:
    Phi, y_int = _extract_features(
        feature_net=feature_net,
        ds=ds,
        device=device,
        batch_size=batch_size_features,
        with_grad=False,
    )
    K = int(len(y_mean))
    y_mean_t = torch.tensor(y_mean, device=device, dtype=torch.float32)
    y_std_t = torch.tensor(y_std, device=device, dtype=torch.float32)
    Y = torch.tensor(_one_hot(y_int, K), device=device, dtype=torch.float32)
    Y = (Y - y_mean_t[None, :]) / y_std_t[None, :]
    mu = Phi @ W_mean
    rmse, acc = _rmse_and_acc(mu, Y)
    return float(rmse.detach().cpu()), float(acc.detach().cpu())


def _train_cov_rmse(
    cov: BlrParameters,
    Phi_l: torch.Tensor,
    Y_l: torch.Tensor,
    epochs: int,
    lr: float,
    optimizer: str,
    weight_decay: float,
    reg_lambda: float,
    opt_state: dict[str, Any] | None,
    log_every_epochs: int,
) -> dict[str, Any]:
    opt = _make_optimizer(
        optimizer, list(cov.parameters()), lr=lr, weight_decay=weight_decay
    )
    if opt_state is not None:
        opt.load_state_dict(opt_state)

    for ep in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        W_mean, _, _ = posterior_mean_and_choleskys(
            Phi=Phi_l, Y=Y_l, lastLayerCovariances=cov
        )
        mu = Phi_l @ W_mean
        rmse, _ = _rmse_and_acc(mu, Y_l)

        # Regulisation to avoid blow up. Did not give good results otherwise.
        reg = reg_lambda * (
            torch.sum(cov.u_raw**2)
            + torch.sum(cov.v_raw**2)
            + cov.sigma2_raw**2
            + torch.sum(cov.a**2)
        )
        loss = rmse + reg
        loss.backward()
        opt.step()

        if log_every_epochs and ((ep + 1) % int(log_every_epochs) == 0):
            logger.info(
                "Epoch %d/%d rmse=%.4f sigma2=%.4g",
                ep + 1,
                int(epochs),
                float(rmse.detach().cpu()),
                float(cov.sigma2().detach().cpu()),
            )
    return opt.state_dict()


def _pretrain_cnn(
    cnn: ReferenceCNN,
    train_ds: Dataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    optimizer: str,
    weight_decay: float,
    seed: int,
) -> ReferenceCNN:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cnn = cnn.to(device)
    opt = _make_optimizer(optimizer, cnn.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    for ep in range(int(epochs)):
        cnn.train()
        total = 0.0
        n = 0
        for xb, yb, _ in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = cnn(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            bs = xb.shape[0]
            total += float(loss.detach().cpu()) * bs
            n += bs
            logger.info(
                "Pretrain CNN epoch %d/%d loss=%.4f",
                ep + 1,
                int(epochs),
                total / max(n, 1),
            )
    return cnn


def run() -> list[StepRecord]:
    K = int(DATASET_NUM_CLASSES)
    steps = int(AL_STEPS)
    k_acq = int(AL_ACQUIRE_PER_STEP)
    repeats = int(AL_REPEATS)
    base_seed = int(AL_BASE_SEED)
    pool_subset = int(AL_POOL_SUBSET)

    device = str(FEATURE_DEVICE)

    train_split, test_split = load_mnist_hf()
    splits = make_splits(
        train_split,
        init_labeled=int(DATASET_INIT_LABELED),
        val_size=int(DATASET_VAL_SIZE),
        num_classes=K,
        seed=int(DATASET_SEED),
    )

    test_ds = HFDatasetsMnist(
        test_split, indices=np.arange(len(test_split), dtype=np.int64)
    )

    cov_cfg_obj = BlrParametersConfiguration(
        diagonal_floor=float(COV_DIAG_FLOOR),
        sigma_floor=float(COV_SIGMA2_FLOOR),
        sigma_max=float(COV_SIGMA2_MAX),
        u_diagonal_initial_value=float(COV_U_DIAG_INIT),
        v_diag_initial_value=float(COV_V_DIAG_INIT),
        sigma_initial_value=float(COV_SIGMA2_INIT),
        alpha_init=float(COV_U_LOW_RANK_INIT),
    )

    epochs = int(TRAIN_EPOCHS)
    batch_size_features = int(TRAIN_BATCH_SIZE_FEATURES)
    lr = float(TRAIN_LR)
    optimizer = str(TRAIN_OPTIMIZER)
    weight_decay = float(TRAIN_WEIGHT_DECAY)
    reg_lambda = float(TRAIN_REG_LAMBDA)
    log_every_epochs = int(TRAIN_LOG_EVERY_EPOCHS)

    y_center = bool(TARGET_CENTER)
    y_scale = bool(TARGET_SCALE)
    y_eps = float(TARGET_EPS)

    pre_n = int(PRETRAIN_N)
    pre_epochs = int(PRETRAIN_EPOCHS)
    pre_bs = int(PRETRAIN_BATCH_SIZE)
    pre_lr = float(PRETRAIN_LR)
    pre_opt = str(PRETRAIN_OPTIMIZER)
    pre_wd = float(PRETRAIN_WEIGHT_DECAY)
    pre_log_every = 0

    methods = list(METHODS)

    records: list[StepRecord] = []
    for repeat in range(repeats):
        # We pretrain once per seed the feature extractor and then freeze
        run_seed = base_seed + repeat
        rng = np.random.default_rng(run_seed)
        pool_idx = splits.pool_idx.copy()
        rng.shuffle(pool_idx)

        pre_n_eff = min(pre_n, len(pool_idx))
        pre_idx = pool_idx[:pre_n_eff]
        pool_idx = pool_idx[pre_n_eff:]
        pretrain_ds = HFDatasetsMnist(train_split, indices=pre_idx)
        feature_net = ReferenceCNN(num_classes=K)
        logger.info(
            "Repeat %d/%d: Pretraining CNN on %d examples for %d epochs",
            repeat + 1,
            repeats,
            int(pre_n_eff),
            int(pre_epochs),
        )
        feature_net = _pretrain_cnn(
            cnn=feature_net,
            train_ds=pretrain_ds,
            device=device,
            epochs=pre_epochs,
            batch_size=pre_bs,
            lr=pre_lr,
            optimizer=pre_opt,
            weight_decay=pre_wd,
            seed=run_seed + 999,
        )

        # Freeze
        _set_requires_grad(feature_net, False)
        feature_net.eval()

        for method in methods:
            labeled_idx = splits.init_labeled_idx.copy()
            cur_pool = pool_idx.copy()

            for step in range(0, steps + 1):
                labeled_labels = np.asarray(
                    train_split.select(labeled_idx.tolist())["label"], dtype=np.int64
                )
                p, std = _target_stats(labeled_labels, K=K, eps=y_eps)

                # Since we do not have an intercept we for sure need to center..
                y_mean = p if y_center else np.zeros_like(p)
                y_std = std if y_scale else np.ones_like(std)

                train_ds = HFDatasetsMnist(train_split, indices=labeled_idx)

                Phi_l, y_int = _extract_features(
                    feature_net=feature_net,
                    ds=train_ds,
                    device=device,
                    batch_size=batch_size_features,
                    with_grad=False,
                )
                Y_l = torch.tensor(
                    _one_hot(y_int, K), device=device, dtype=torch.float32
                )
                y_mean_t = torch.tensor(y_mean, device=device, dtype=torch.float32)
                y_std_t = torch.tensor(y_std, device=device, dtype=torch.float32)
                Y_l = (Y_l - y_mean_t[None, :]) / y_std_t[None, :]

                # Each active learning step relearns the covariances...
                cov = BlrParameters(D=128, K=K, cfg=cov_cfg_obj).to(device)
                _ = _train_cov_rmse(
                    cov=cov,
                    Phi_l=Phi_l,
                    Y_l=Y_l,
                    epochs=epochs,
                    lr=lr,
                    optimizer=optimizer,
                    weight_decay=weight_decay,
                    reg_lambda=reg_lambda,
                    opt_state=None,
                    log_every_epochs=log_every_epochs,
                )

                # Get mean and covariances for point estimate
                W_mean, chols, lambda_diag = posterior_mean_and_choleskys(
                    Phi=Phi_l, Y=Y_l, lastLayerCovariances=cov
                )

                train_rmse, train_acc = _eval_with_W_mean(
                    feature_net=feature_net,
                    W_mean=W_mean,
                    ds=train_ds,
                    device=device,
                    batch_size_features=batch_size_features,
                    y_mean=y_mean,
                    y_std=y_std,
                )
                test_rmse, test_acc = _eval_with_W_mean(
                    feature_net=feature_net,
                    W_mean=W_mean,
                    ds=test_ds,
                    device=device,
                    batch_size_features=batch_size_features,
                    y_mean=y_mean,
                    y_std=y_std,
                )

                records.append(
                    StepRecord(
                        method=str(method),
                        repeat=repeat,
                        seed=run_seed,
                        step=step,
                        acquired_from_pool=int(step * k_acq),
                        total_labeled=int(len(labeled_idx)),
                        test_rmse=float(test_rmse),
                        train_rmse=float(train_rmse),
                        test_acc=float(test_acc),
                        train_acc=float(train_acc),
                        sigma2=float(cov.sigma2().detach().cpu()),
                    )
                )
                logger.info(
                    "Repeat %d/%d method=%s step %d/%d labeled=%d test_acc=%.4f test_rmse=%.4f",
                    repeat + 1,
                    repeats,
                    method,
                    step,
                    steps,
                    int(len(labeled_idx)),
                    float(test_acc),
                    float(test_rmse),
                )

                if step == steps:
                    break

                if method == "random":
                    k_eff = min(k_acq, len(cur_pool))
                    picked = rng.choice(cur_pool, size=k_eff, replace=False).astype(
                        np.int64
                    )
                else:
                    # For speedup we can restrict the pool...
                    if 0 < pool_subset < len(cur_pool):
                        subset_pos = rng.choice(
                            len(cur_pool), size=int(pool_subset), replace=False
                        )
                        subset_idx = cur_pool[subset_pos]
                    else:
                        subset_idx = cur_pool

                    pool_ds = HFDatasetsMnist(train_split, indices=subset_idx)
                    Phi_pool, _ = _extract_features(
                        feature_net=feature_net,
                        ds=pool_ds,
                        device=device,
                        batch_size=batch_size_features,
                        with_grad=False,
                    )

                    # Score always using diagonal trace but compute diagonal differntly
                    if method == "blr_analytic":
                        var_posterior_score = variance_diagonal_analytic(
                            Phi_pool=Phi_pool, chol_L=chols
                        )
                    elif method == "blr_mfvi":
                        var_posterior_score = variance_diagonal_mfvi(
                            Phi_pool=Phi_pool, lambda_diag=lambda_diag
                        )

                    scores = (
                        score_trace(var_diag=var_posterior_score).detach().cpu().numpy()
                    )

                    k_eff = min(k_acq, len(subset_idx))
                    order = np.lexsort((subset_idx, -scores))
                    picked = subset_idx[order[:k_eff]]

                picked_set = set(int(i) for i in picked.tolist())
                labeled_idx = np.concatenate(
                    [labeled_idx, picked.astype(np.int64)], axis=0
                )
                cur_pool = np.array(
                    [i for i in cur_pool.tolist() if int(i) not in picked_set],
                    dtype=np.int64,
                )
    return records
