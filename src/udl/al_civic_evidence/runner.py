from dataclasses import dataclass
from pathlib import Path
import numpy as np

from udl.al_civic_evidence.acquisition import (
    score_pool_with_mc_dropout,
    tokenize_pool_texts,
)
from udl.al_civic_evidence.data import load_or_doinload_dataset
from udl.al_civic_evidence.train_eval import (
    TrainConfig,
    eval_multilabel,
    train_multilabel_distilbert,
)
import logging
from udl.utils.utils import dir_atifacts


@dataclass(frozen=True)
class StepRecord:
    method: str
    repeat: int
    seed: int
    step: int
    acquired_from_pool: int
    total_labeled: int
    val_weighted_f1: float
    test_subset_accuracy: float
    test_macro_f1: float
    test_weighted_f1: float


DATASET_NAME = "civic-evidence"
DATASET_SUBDIR = "datasets"
TEXT_FIELD = "sourceAbstract"
MAX_LENGTH = 512

AL_STEPS = 20
AL_ACQUIRE_PER_STEP = 50
AL_REPEATS = 3
AL_BASE_SEED = 1337
AL_INIT_LABELED = 20

DEVICE = "cuda"

TRAIN_PRETRAINED = "distilbert-base-uncased"
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 1024
TRAIN_EPOCHS = 20.0
TRAIN_LR = 6e-6
TRAIN_WARMUP_RATIO = 0.0
TRAIN_LOGGING_STEPS = 1
TRAIN_MAX_STEPS = None

MC_SAMPLES_ACQ = 20

METHODS: list[str] = ["bald", "random"]


def run() -> list[StepRecord]:
    logger = logging.getLogger("udl.al_civic_evidence")
    dsd = load_or_doinload_dataset(
        artifacts_dir=dir_atifacts(),
        dataset_name=str(DATASET_NAME),
        dataset_subdir=str(DATASET_SUBDIR),
        rebuild=False,
    )
    train_split = dsd["train"]
    val_split = dsd["validation"] if "validation" in dsd else dsd["train"].select([])
    test_split = dsd["test"]

    levels = ("A", "B", "C", "D", "E")
    pos_rates = np.array(
        [float(np.mean(train_split[L])) for L in levels], dtype=np.float64
    )
    eps = 1e-6
    inv = 1.0 / np.clip(pos_rates, eps, None)
    label_weights = (inv / float(inv.sum())).astype(np.float32)

    steps = int(AL_STEPS)
    k = int(AL_ACQUIRE_PER_STEP)
    repeats = int(AL_REPEATS)
    base_seed = int(AL_BASE_SEED)
    init_labeled = int(AL_INIT_LABELED)

    methods = list(METHODS)

    text_field = str(TEXT_FIELD)
    max_length = int(MAX_LENGTH)

    bs_score = int(EVAL_BATCH_SIZE)
    bs_eval = int(EVAL_BATCH_SIZE)
    mc_samples_acq = int(MC_SAMPLES_ACQ)

    records: list[StepRecord] = []

    for repeat in range(repeats):
        # Each repetition gets differen seed
        run_seed = base_seed + repeat
        rng = np.random.RandomState(run_seed)

        pool_order = rng.permutation(len(train_split)).tolist()
        labeled: list[int] = pool_order[:init_labeled]
        pool: list[int] = pool_order[init_labeled:]

        logger.info("Repeat %d/%d (seed=%d)", repeat + 1, repeats, run_seed)

        for method in methods:
            method_t = str(method)

            logger.info("Starting method=%s", method_t)

            labeled_idx = list(labeled)
            cur_pool = list(pool)
            train_ds0 = train_split.select(sorted(labeled_idx))
            # We use HugginFace Trainer API for simplicity...
            train_cfg = TrainConfig(
                pretrained=str(TRAIN_PRETRAINED),
                text_field=str(text_field),
                max_length=int(max_length),
                learning_rate=float(TRAIN_LR),
                per_device_train_batch_size=int(TRAIN_BATCH_SIZE),
                per_device_eval_batch_size=int(EVAL_BATCH_SIZE),
                num_train_epochs=float(TRAIN_EPOCHS),
                warmup_ratio=float(TRAIN_WARMUP_RATIO),
                logging_steps=int(TRAIN_LOGGING_STEPS),
                max_steps=TRAIN_MAX_STEPS,
            )
            model, tokenizer, train_out, train_metrics = train_multilabel_distilbert(
                cfg=train_cfg,
                train_ds=train_ds0,
                val_ds=val_split,
            )
            test_metrics0 = eval_multilabel(
                model=model,
                tokenizer=tokenizer,
                ds=test_split,
                device=str(DEVICE),
                thresholds=train_out.val_thresholds,
                text_field=text_field,
                max_length=max_length,
                batch_size=bs_eval,
            )
            records.append(
                StepRecord(
                    method=str(method_t),
                    repeat=repeat,
                    seed=run_seed,
                    step=0,
                    acquired_from_pool=0,
                    total_labeled=len(labeled_idx),
                    val_weighted_f1=float(train_out.val_weighted_f1),
                    test_subset_accuracy=float(test_metrics0["subset_accuracy"]),
                    test_macro_f1=float(test_metrics0["macro_f1"]),
                    test_weighted_f1=float(test_metrics0["weighted_f1"]),
                )
            )
            for step in range(1, steps + 1):
                if not cur_pool:
                    break
                pool_ds = train_split.select(cur_pool)
                pool_ds_tokenized = tokenize_pool_texts(
                    ds=pool_ds,
                    tokenizer=tokenizer,
                    text_field=text_field,
                    max_length=max_length,
                )
                scored = score_pool_with_mc_dropout(
                    model=model,
                    pool_ds_tok=pool_ds_tokenized,
                    tokenizer=tokenizer,
                    device=str(DEVICE),
                    acq=method_t,
                    mc_samples=mc_samples_acq,
                    batch_size=bs_score,
                    seed=run_seed + step,
                    label_weights=label_weights,
                )
                scores = scored.scores
                k_eff = min(k, len(cur_pool))
                # Pick new points
                order = np.lexsort((np.array(cur_pool), -scores))
                picked = [cur_pool[i] for i in order[:k_eff]]

                labeled_idx.extend(picked)
                picked_set = set(picked)
                cur_pool = [i for i in cur_pool if i not in picked_set]
                train_ds = train_split.select(sorted(labeled_idx))
                # Train the model...
                model, tokenizer, train_out, train_metrics = (
                    train_multilabel_distilbert(
                        cfg=train_cfg,
                        train_ds=train_ds,
                        val_ds=val_split,
                    )
                )
                test_metrics = eval_multilabel(
                    model=model,
                    tokenizer=tokenizer,
                    ds=test_split,
                    device=str(DEVICE),
                    thresholds=train_out.val_thresholds,
                    text_field=text_field,
                    max_length=max_length,
                    batch_size=bs_eval,
                )
                records.append(
                    StepRecord(
                        method=str(method_t),
                        repeat=repeat,
                        seed=run_seed,
                        step=step,
                        acquired_from_pool=step * k,
                        total_labeled=len(labeled_idx),
                        val_weighted_f1=float(train_out.val_weighted_f1),
                        test_subset_accuracy=float(test_metrics["subset_accuracy"]),
                        test_macro_f1=float(test_metrics["macro_f1"]),
                        test_weighted_f1=float(test_metrics["weighted_f1"]),
                    )
                )
                logger.info(
                    "Step %d/%d: method=%s labeled=%d test_wF1=%.4f",
                    step,
                    steps,
                    method_t,
                    len(labeled_idx),
                    float(test_metrics["weighted_f1"]),
                )
    return records
