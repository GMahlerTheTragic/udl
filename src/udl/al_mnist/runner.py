from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import logging

from udl.al_mnist.acquisition import Method, acquire
from udl.al_mnist.data import HFDatasetsMnist, MnistSplits, load_mnist_hf, make_splits
from udl.al_mnist.model import ReferenceCNN
from udl.al_mnist.train_eval import (
    TrainConfig,
    _resolve_device,
    accuracy,
    retrain_with_fixed_weight_decay,
    train_with_weight_decay_selection,
)


Mode = Literal["bayesian", "deterministic"]

logger = logging.getLogger("udl.al_mnist")

DATASET_NUM_CLASSES = 10
DATASET_INIT_LABELED = 20
DATASET_VAL_SIZE = 100
DATASET_SEED = 0

AL_STEPS = 100
AL_ACQUIRE_PER_STEP = 10
AL_REPEATS = 3
AL_BASE_SEED = 1337

TRAIN_DEVICE = "cuda"
TRAIN_BATCH_SIZE = 128
TRAIN_EPOCHS = 50
TRAIN_LR = 1.0e-3
TRAIN_OPTIMIZER = "adam"
TRAIN_WEIGHT_DECAY_GRID = [1.0e-6, 1.0e-5, 1.0e-4, 5.0e-4, 1.0e-3]

MC_SAMPLES_ACQ = 100
MC_SAMPLES_EVAL = 100

BAYES_METHODS: list[str] = ["bald", "var_ratios", "max_entropy", "mean_std", "random"]
DET_METHODS: list[str] = ["bald", "var_ratios", "max_entropy"]


@dataclass(frozen=True)
class StepRecord:
    mode: Mode
    method: Method
    repeat: int
    seed: int
    step: int
    acquired_from_pool: int
    total_labeled: int
    weight_decay: float
    val_acc: float
    test_acc: float
    test_error_pct: float


def getModel():
    return ReferenceCNN(num_classes=int(DATASET_NUM_CLASSES))


def run() -> list[StepRecord]:
    train_split, test_split = load_mnist_hf()
    splits: MnistSplits = make_splits(
        train_split,
        init_labeled=int(DATASET_INIT_LABELED),
        val_size=int(DATASET_VAL_SIZE),
        num_classes=int(DATASET_NUM_CLASSES),
        seed=int(DATASET_SEED),
    )

    test_ds = HFDatasetsMnist(
        test_split, indices=np.arange(len(test_split), dtype=np.int64)
    )
    val_ds = HFDatasetsMnist(train_split, indices=splits.val_idx)

    train_cfg = TrainConfig(
        device=str(TRAIN_DEVICE),
        batch_size=int(TRAIN_BATCH_SIZE),
        epochs=int(TRAIN_EPOCHS),
        lr=float(TRAIN_LR),
        optimizer=str(TRAIN_OPTIMIZER),
        weight_decay_grid=[float(x) for x in TRAIN_WEIGHT_DECAY_GRID],
        log_every_epochs=0,
    )

    device = _resolve_device(train_cfg.device)

    steps = int(AL_STEPS)
    k = int(AL_ACQUIRE_PER_STEP)
    repeats = int(AL_REPEATS)
    base_seed = int(AL_BASE_SEED)

    bayes_methods = list(BAYES_METHODS)
    det_methods = list(DET_METHODS)

    records: list[StepRecord] = []

    for repeat in range(repeats):
        run_seed = base_seed + repeat
        logger.info("Repeat %d/%d (seed=%d)", repeat + 1, repeats, run_seed)
        rng = np.random.default_rng(run_seed)
        pool_idx = splits.pool_idx.copy()
        rng.shuffle(pool_idx)

        init_labeled_idx = splits.init_labeled_idx.copy()
        init_train_ds = HFDatasetsMnist(train_split, indices=init_labeled_idx)
        base_model, best_wd, best_val_acc_bayes = train_with_weight_decay_selection(
            get_model_lambda=getModel,
            train_ds=init_train_ds,
            val_ds=val_ds,
            cfg=train_cfg,
            seed=run_seed,
            deterministic=False,
            mc_samples_eval=int(MC_SAMPLES_EVAL),
        )
        base_state = {
            k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()
        }
        best_val_acc_det = accuracy(
            base_model,
            val_ds,
            device=device,
            batch_size=train_cfg.batch_size,
            deterministic=True,
            mc_samples=int(MC_SAMPLES_EVAL),
        )
        logger.info(
            "WD selection (global per repeat): best_wd=%.3g val_acc_bayes=%.4f val_acc_det=%.4f",
            float(best_wd),
            float(best_val_acc_bayes),
            float(best_val_acc_det),
        )

        for mode in ("bayesian", "deterministic"):
            deterministic = mode == "deterministic"
            methods = det_methods if deterministic else bayes_methods
            logger.info("Mode=%s methods=%s", mode, ",".join(methods))

            for method in methods:
                method_t = method
                logger.info("Starting method=%s (mode=%s)", method_t, mode)
                labeled_idx = init_labeled_idx.copy()
                cur_pool = pool_idx.copy()

                best_model = getModel().to(device)
                best_model.load_state_dict(base_state)
                best_val_acc = float(
                    best_val_acc_det if deterministic else best_val_acc_bayes
                )
                test_acc0 = accuracy(
                    best_model,
                    test_ds,
                    device=device,
                    batch_size=train_cfg.batch_size,
                    deterministic=deterministic,
                    mc_samples=int(MC_SAMPLES_EVAL),
                )
                records.append(
                    StepRecord(
                        mode=mode,
                        method=method_t,
                        repeat=repeat,
                        seed=run_seed,
                        step=0,
                        acquired_from_pool=0,
                        total_labeled=int(len(labeled_idx)),
                        weight_decay=float(best_wd),
                        val_acc=float(best_val_acc),
                        test_acc=float(test_acc0),
                        test_error_pct=float(100.0 * (1.0 - test_acc0)),
                    )
                )
                logger.info(
                    "Step 0: method=%s mode=%s labeled=%d wd=%.3g val_acc=%.4f test_acc=%.4f",
                    method_t,
                    mode,
                    int(len(labeled_idx)),
                    float(best_wd),
                    float(best_val_acc),
                    float(test_acc0),
                )

                for step in range(1, steps + 1):
                    pool_ds = HFDatasetsMnist(train_split, indices=cur_pool)
                    acq = acquire(
                        model=best_model,
                        pool_ds=pool_ds,
                        method=method_t,
                        k=k,
                        device=device,
                        batch_size=train_cfg.batch_size,
                        mc_samples=int(MC_SAMPLES_ACQ),
                        deterministic=deterministic,
                        seed=run_seed + step,
                    )
                    picked_pos = acq.selected_pos
                    picked_idx = cur_pool[picked_pos]

                    labeled_idx = np.concatenate([labeled_idx, picked_idx], axis=0)
                    keep_mask = np.ones(len(cur_pool), dtype=bool)
                    keep_mask[picked_pos] = False
                    cur_pool = cur_pool[keep_mask]

                    train_ds = HFDatasetsMnist(train_split, indices=labeled_idx)
                    best_model = retrain_with_fixed_weight_decay(
                        get_model_lambda=getModel,
                        train_ds=train_ds,
                        cfg=train_cfg,
                        seed=run_seed + 10000 * (step + 1),
                        weight_decay=float(best_wd),
                    )

                    test_acc = accuracy(
                        best_model,
                        test_ds,
                        device=device,
                        batch_size=train_cfg.batch_size,
                        deterministic=deterministic,
                        mc_samples=int(MC_SAMPLES_EVAL),
                    )
                    records.append(
                        StepRecord(
                            mode=mode,
                            method=method_t,
                            repeat=repeat,
                            seed=run_seed,
                            step=step,
                            acquired_from_pool=step * k,
                            total_labeled=int(len(labeled_idx)),
                            weight_decay=float(best_wd),
                            val_acc=float(best_val_acc),
                            test_acc=float(test_acc),
                            test_error_pct=float(100.0 * (1.0 - test_acc)),
                        )
                    )
                    logger.info(
                        "Step %d/%d: method=%s mode=%s acquired=%d labeled=%d test_acc=%.4f",
                        step,
                        steps,
                        method_t,
                        mode,
                        int(step * k),
                        int(len(labeled_idx)),
                        float(test_acc),
                    )
    return records
